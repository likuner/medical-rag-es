"""Central configuration for the medical RAG pipeline.

All tunable values live here so the rest of the code stays declarative.
Read from environment variables with sensible defaults for a local setup.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from ROOT/.env into os.environ (no override).

    Keeps secrets such as the GLM API key out of the repository while still
    exposing them as environment variables. Must run before the dataclass
    field defaults below snapshot the environment.
    """
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")


_load_dotenv()


def _ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Vector store (Elasticsearch)
# --------------------------------------------------------------------------- #
@dataclass
class ESConfig:
    # Base URL of the cluster. With security enabled the official image serves
    # HTTPS with a self-signed CA; point ca_cert at the exported http_ca.crt.
    url: str = os.getenv("ES_URL", "http://localhost:9200")
    user: str = os.getenv("ES_USER", "")
    # The password is a secret: keep it in the environment / a git-ignored
    # .env file, never in committed code.
    password: str = os.getenv("ES_PASSWORD", "")
    ca_cert: str = os.getenv("ES_CA_CERT", "")
    documents_index: str = os.getenv("ES_DOCS_INDEX", "medical_documents")
    chunks_index: str = os.getenv("ES_CHUNKS_INDEX", "medical_chunks")

    @property
    def ca_cert_path(self) -> str | None:
        """Resolve ca_cert against the project root (or return None if unset)."""
        if not self.ca_cert:
            return None
        p = Path(self.ca_cert)
        return str(p if p.is_absolute() else ROOT / p)


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
@dataclass
class EmbeddingConfig:
    # Backend to use: "sentence-transformers" | "tfidf" | "glm" | "auto".
    # "auto" prefers the GLM API when GLM_API_KEY is set, then tries the
    # sentence-transformers model, and finally falls back to TF-IDF.
    backend: str = os.getenv("EMBED_BACKEND", "auto")
    # The sentence-transformers model id (HuggingFace hub or local path).
    model_name: str = os.getenv(
        "EMBED_MODEL", "BAAI/bge-small-zh-v1.5"
    )
    # Output dimensionality. MUST match the model for sentence-transformers
    # (bge-small-zh-v1.5 -> 512). The tfidf backend always reduces to this,
    # and the glm backend passes it as the API `dimensions` parameter
    # (embedding-3 supports 256-2048; 1024 by default here).
    dim: int = int(os.getenv("EMBED_DIM", "1024"))
    # ---- GLM (Zhipu AI) embedding API ------------------------------------ #
    # The API key is a secret: keep it in the environment / a git-ignored
    # .env file, never in committed code.
    glm_api_key: str = os.getenv("GLM_API_KEY", "")
    glm_model: str = os.getenv("GLM_EMBED_MODEL", "embedding-3")
    glm_base_url: str = os.getenv(
        "GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/embeddings"
    )
    glm_batch_size: int = int(os.getenv("GLM_EMBED_BATCH", "32"))
    # For TF-IDF fallback: number of features and LSA components.
    tfidf_max_features: int = 200_000
    tfidf_lsa_components: int | None = None  # None -> use `dim`
    # bge-style models want a prefix on queries to signal retrieval intent.
    query_instruction: str = os.getenv(
        "EMBED_QUERY_INSTRUCTION", "为这个句子生成表示以用于检索相关文章："
    )
    normalize: bool = True  # L2-normalise vectors (cosine distance in pgvector)
    # "cpu" is the safe default (avoids intermittent MPS hangs); set "mps"/"cuda"
    # via EMBED_DEVICE if you want acceleration.
    device: str | None = os.getenv("EMBED_DEVICE", "cpu")


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
@dataclass
class ChunkConfig:
    # Units are "tokens" as counted by the chunker's tokenizer (CJK char == 1,
    # latin word == 1), passed to LlamaIndex SentenceSplitter as chunk_size /
    # chunk_overlap.
    size: int = int(os.getenv("CHUNK_SIZE", "512"))        # target tokens
    overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))  # overlapping tokens
    min_chunk: int = 80  # drop chunks shorter than this (chars) after cleaning


# --------------------------------------------------------------------------- #
# Crawling
# --------------------------------------------------------------------------- #
@dataclass
class CrawlConfig:
    # Delay between requests (seconds) - be polite to source servers.
    delay: float = float(os.getenv("CRAWL_DELAY", "1.0"))
    timeout: int = int(os.getenv("CRAWL_TIMEOUT", "25"))
    max_retries: int = 2
    # Per-site cap on number of pages to crawl in one run (0 = unlimited).
    max_pages_per_site: int = int(os.getenv("CRAWL_MAX_PAGES", "15"))
    user_agent: str = (
        "MedicalRAG-Crawler/1.0 (+https://example.org/crawler; "
        "educational research) requests/2.31"
    )
    # Respect robots.txt.
    respect_robots: bool = True


@dataclass
class PipelineConfig:
    es: ESConfig = field(default_factory=ESConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    crawl: CrawlConfig = field(default_factory=CrawlConfig)

    def __post_init__(self) -> None:
        _ensure_dirs()


CONFIG = PipelineConfig()
