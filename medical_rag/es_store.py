"""Elasticsearch access layer.

Creates the two indices (documents + chunks), manages the client connection,
and provides bulk-indexing and kNN search helpers. Vectors live in a
`dense_vector` field with `similarity: cosine`; ES builds the HNSW graph
automatically because `index: true` is set in the mapping.

Document and chunk ids are deterministic hashes, so re-indexing the same
(source, source_url) is a no-op upsert rather than a duplicate.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Sequence

from elasticsearch import Elasticsearch

from .config import CONFIG

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _doc_id(source: str, source_url: str) -> str:
    return hashlib.sha256(f"{source}|{source_url}".encode("utf-8")).hexdigest()


def _chunk_id(document_id: str, chunk_index: int) -> str:
    return hashlib.sha256(f"{document_id}#{chunk_index}".encode("utf-8")).hexdigest()


def connect() -> Elasticsearch:
    """Build a client from CONFIG.es and verify the cluster is reachable."""
    cfg = CONFIG.es
    client = Elasticsearch(
        cfg.url,
        basic_auth=(cfg.user, cfg.password) if cfg.password else None,
        ca_certs=cfg.ca_cert_path,
        request_timeout=30,
    )
    if not client.ping():
        raise ConnectionError(
            f"cannot reach Elasticsearch at {cfg.url} "
            f"(check ES_URL / ES_USER / ES_PASSWORD / ES_CA_CERT)"
        )
    return client


def _documents_index_body() -> dict[str, Any]:
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "source": {"type": "keyword"},          # site key, e.g. 'who_zh'
                "source_name": {"type": "keyword"},
                "source_url": {"type": "keyword"},      # canonical page URL
                "title": {"type": "text"},
                "page_number": {"type": "integer"},     # for paged sources, else null
                "lang": {"type": "keyword"},
                "doc_metadata": {"type": "object"},     # crawler-specific extras
                "raw_text": {"type": "text"},           # full cleaned text (pre-chunk)
                "created_at": {"type": "date"},
            }
        },
    }


def _chunks_index_body(dim: int) -> dict[str, Any]:
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            # Document-level fields are denormalised into every chunk so search
            # needs no join.
            "properties": {
                "document_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "source_url": {"type": "keyword"},
                "title": {"type": "text"},
                "lang": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "chunk_text": {"type": "text"},
                "char_length": {"type": "integer"},
                "token_count": {"type": "integer"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dim,
                    "index": True,          # builds the HNSW graph
                    "similarity": "cosine",
                },
                "created_at": {"type": "date"},
            }
        },
    }


def ensure_indices(client: Elasticsearch, *, recreate: bool = False) -> None:
    """Create the documents/chunks indices, reading the embedding dim at call
    time (the model loads before this and may set the true dimensionality)."""
    dim = CONFIG.embedding.dim
    specs = [
        (CONFIG.es.documents_index, _documents_index_body()),
        (CONFIG.es.chunks_index, _chunks_index_body(dim)),
    ]
    for name, body in specs:
        exists = client.indices.exists(index=name)
        if recreate and exists:
            client.indices.delete(index=name)
            log.info("Deleted index %s", name)
            exists = False
        if not exists:
            client.indices.create(
                index=name, mappings=body["mappings"], settings=body["settings"]
            )
            log.info("Created index %s (dim=%d)", name, dim)


def insert_document(
    client: Elasticsearch,
    *,
    source: str,
    source_name: str,
    source_url: str,
    title: str | None,
    page_number: int | None,
    lang: str | None,
    doc_metadata: dict[str, Any] | None,
    raw_text: str | None,
) -> str:
    """Index a document under a deterministic id and return it."""
    doc_id = _doc_id(source, source_url)
    client.index(
        index=CONFIG.es.documents_index,
        id=doc_id,
        document={
            "source": source,
            "source_name": source_name,
            "source_url": source_url,
            "title": title,
            "page_number": page_number,
            "lang": lang,
            "doc_metadata": doc_metadata or {},
            "raw_text": raw_text,
            "created_at": _now(),
        },
    )
    return doc_id


def insert_chunks(
    client: Elasticsearch,
    document_id: str,
    doc_fields: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> int:
    """Bulk-index chunks for a single document.

    Each row needs: chunk_index, chunk_text, char_length, token_count,
    embedding (list[float]). `doc_fields` carries the denormalised document
    fields (source, source_url, title, lang).
    """
    if not rows:
        return 0
    now = _now()
    operations: list[dict[str, Any]] = []
    for r in rows:
        operations.append(
            {"index": {"_index": CONFIG.es.chunks_index,
                       "_id": _chunk_id(document_id, r["chunk_index"])}}
        )
        operations.append({
            **doc_fields,
            "document_id": document_id,
            "chunk_index": r["chunk_index"],
            "chunk_text": r["chunk_text"],
            "char_length": r["char_length"],
            "token_count": r.get("token_count"),
            "embedding": r["embedding"],
            "created_at": now,
        })
    resp = client.bulk(operations=operations)
    if resp.get("errors"):
        for item in resp["items"]:
            err = item.get("index", {}).get("error")
            if err:
                log.error("Bulk index error on chunk %s: %s",
                          item.get("index", {}).get("_id"), err)
        raise RuntimeError("bulk indexing of chunks failed")
    return len(rows)


def document_exists(
    client: Elasticsearch,
    *,
    source: str,
    source_url: str,
) -> str | None:
    """Uniqueness guard: find an existing document id for the same source url.

    Returns the existing id if found, else None. Used to make re-runs
    idempotent (we skip documents whose URL we already ingested).
    """
    resp = client.search(
        index=CONFIG.es.documents_index,
        query={"bool": {"filter": [
            {"term": {"source": source}},
            {"term": {"source_url": source_url}},
        ]}},
        size=1,
        _source=False,
    )
    hits = resp["hits"]["hits"]
    return hits[0]["_id"] if hits else None


def search_vectors(
    client: Elasticsearch,
    query_vector: Sequence[float],
    *,
    k: int = 5,
    sites: list[str] | None = None,
) -> list[dict[str, Any]]:
    """kNN cosine search over stored chunk embeddings.

    ES scores cosine similarity as (1 + cos) / 2 in [0, 1]; convert back to
    plain cosine (= 1 - pgvector's `<=>` cosine distance) so the reported
    similarity keeps its old meaning: 1.0 identical, ~0 unrelated.
    """
    knn: dict[str, Any] = {
        "field": "embedding",
        "query_vector": list(query_vector),
        "k": k,
        "num_candidates": max(10 * k, 100),
    }
    if sites:
        knn["filter"] = {"terms": {"source": list(sites)}}
    resp = client.search(
        index=CONFIG.es.chunks_index,
        knn=knn,
        size=k,
        _source=["source", "source_url", "title", "chunk_index", "chunk_text"],
    )
    out: list[dict[str, Any]] = []
    for hit in resp["hits"]["hits"]:
        s = hit["_source"]
        cos = 2.0 * float(hit["_score"]) - 1.0
        out.append({
            "source": s.get("source"),
            "url": s.get("source_url"),
            "title": s.get("title"),
            "chunk_index": s.get("chunk_index"),
            "text": s.get("chunk_text"),
            "similarity": round(cos, 4),
        })
    return out


def refresh(client: Elasticsearch) -> None:
    """Make indexed documents immediately visible to searches."""
    client.indices.refresh(index=[CONFIG.es.documents_index,
                                  CONFIG.es.chunks_index])
