"""Embedding backends.

Primary backend: GLM (Zhipu AI) embedding API — remote semantic vectors, no
local ML stack needed. Alternative local backends: sentence-transformers
(semantic) and scikit-learn TF-IDF + LSA. All produce a fixed-width,
L2-normalised numpy float32 vector so the pgvector table stays consistent.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

import numpy as np

from .config import EmbeddingConfig, ROOT, PROCESSED_DIR

# HuggingFace is often unreachable directly (e.g. from CN networks); the
# hf-mirror.com endpoint is a drop-in mirror. Overridable via HF_ENDPOINT.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

log = logging.getLogger(__name__)

STATE_META = PROCESSED_DIR / "embedding_meta.json"
STATE_TFIDF = PROCESSED_DIR / "tfidf_model.joblib"

_CJK_KW = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
_WORD_RE = re.compile(
    r"[a-zA-Z0-9]+(?:['’\-][a-zA-Z0-9]+)*|[\u3400-\u9fff\uf900-\ufaff]"
)


def _is_cjk(s: str) -> bool:
    return bool(_CJK_KW.fullmatch(s))


def _tokenize(text: str) -> list[str]:
    """Tokenizer for the TF-IDF backend that works for mixed CJK/latin text."""
    toks = _WORD_RE.findall(text.lower())
    out: list[str] = []
    for i, t in enumerate(toks):
        out.append(t)
        if _is_cjk(t) and i + 1 < len(toks) and _is_cjk(toks[i + 1]):
            out.append(t + toks[i + 1])
    return out


def _canonicalize(vecs: np.ndarray, dim: int) -> np.ndarray:
    """Ensure the vector matrix has exactly `dim` columns (pad/truncate)."""
    n = vecs.shape[0]
    if vecs.shape[1] == dim:
        return vecs.astype(np.float32)
    out = np.zeros((n, dim), dtype=np.float32)
    take = min(vecs.shape[1], dim)
    out[:, :take] = vecs[:, :take]
    return out


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


class Embedder:
    """Pluggable text->vector embedder.

    ``backend = "glm"`` calls the Zhipu AI embedding API (needs GLM_API_KEY).
    ``backend = "sentence-transformers"`` loads a HuggingFace model.
    ``backend = "tfidf"`` uses TF-IDF + LSA; call :meth:`fit` first on the bulk
    of documents so vocabulary is known. ``backend = "auto"`` prefers GLM when
    a key is configured, then tries the local model and falls back to TF-IDF.
    """

    _GLM_MAX_ATTEMPTS = 4

    def __init__(self, config: EmbeddingConfig):
        self.cfg = config
        self.backend = config.backend.lower()
        self.dim = config.dim
        self._model = None
        self._vectorizer = None
        self._svd = None
        self._requests = None  # module handle used by the glm backend

    # ------------------------------------------------------------------ load
    def load(self) -> "Embedder":
        if self.backend in {"glm", "zhipu", "zhipuai"}:
            self.backend = "glm"
            self._load_glm()
        elif self.backend in {"sentence-transformers", "st", "sbert", "semantic"}:
            self._load_sbert()
        elif self.backend in {"tfidf", "lsa", "bow"}:
            self.backend = "tfidf"
        else:  # auto: remote GLM first, then local model, then TF-IDF.
            if self.cfg.glm_api_key:
                try:
                    self._load_glm()
                    self.backend = "glm"
                    return self
                except Exception as exc:  # noqa: BLE001
                    log.warning("GLM embeddings unavailable (%s); trying local backends.", exc)
            try:
                self._load_sbert()
                self.backend = "sentence-transformers"
            except Exception as exc:  # noqa: BLE001
                log.warning("SentenceTransformers unavailable (%s); using TF-IDF.", exc)
                self.backend = "tfidf"
        return self

    def _load_glm(self) -> None:
        if not self.cfg.glm_api_key:
            raise RuntimeError(
                "GLM backend requires GLM_API_KEY (set the environment variable "
                "or put it in a git-ignored .env file; never commit the key)."
            )
        import requests

        self._requests = requests
        # Probe the API once so the schema's vector(N) column is created with
        # the dimension the service actually returns.
        try:
            vecs = self._glm_request(["dimension probe"])
        except Exception:
            # The requested `dimensions` value may be unsupported by the
            # model; retry with the model default and adopt whatever comes back.
            self.cfg.dim = 0
            vecs = self._glm_request(["dimension probe"])
        self.dim = int(vecs.shape[1])
        if self.dim != self.cfg.dim:
            log.info("GLM embedding dim is %d (config said %d); using %d.",
                     self.dim, self.cfg.dim, self.dim)
            self.cfg.dim = self.dim

    def _glm_request(self, texts: list[str]) -> np.ndarray:
        """One POST to the GLM embeddings endpoint; retries transient errors."""
        requests = self._requests
        headers = {
            "Authorization": f"Bearer {self.cfg.glm_api_key}",
            "Content-Type": "application/json",
        }
        payload: dict = {"model": self.cfg.glm_model, "input": texts}
        if self.cfg.dim > 0:
            payload["dimensions"] = self.cfg.dim
        last_exc: Exception | None = None
        for attempt in range(self._GLM_MAX_ATTEMPTS):
            try:
                r = requests.post(
                    self.cfg.glm_base_url,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.RequestException(
                        f"HTTP {r.status_code}: {r.text[:200]}"
                    )
                if r.status_code >= 400:
                    raise RuntimeError(
                        f"GLM embeddings HTTP {r.status_code}: {r.text[:300]}"
                    )
                data = r.json()["data"]
                data.sort(key=lambda d: d["index"])
                return np.asarray([d["embedding"] for d in data], dtype=np.float32)
            except (requests.RequestException, KeyError, ValueError) as exc:
                last_exc = exc
                if attempt < self._GLM_MAX_ATTEMPTS - 1:
                    time.sleep(1.5 * (attempt + 1))
        raise last_exc if last_exc else RuntimeError("GLM embeddings failed")

    def _load_sbert(self) -> None:
        from sentence_transformers import SentenceTransformer

        device = self.cfg.device or "cpu"
        try:
            self._model = SentenceTransformer(self.cfg.model_name, device=device)
        except Exception as exc:
            log.warning("Model %s failed on %s; retrying cpu. (%s)",
                        self.cfg.model_name, device, exc)
            self._model = SentenceTransformer(self.cfg.model_name, device="cpu")
        self.dim = int(self._model.get_sentence_embedding_dimension())
        if self.dim != self.cfg.dim:
            log.info("Model embedding dim is %d (config said %d); using %d.",
                     self.dim, self.cfg.dim, self.dim)
            self.cfg.dim = self.dim

    # ------------------------------------------------------------------ fit
    def fit(self, texts: list[str]) -> "Embedder":
        """Optional corpus pass required by the TF-IDF backend."""
        if self.backend != "tfidf" or not texts:
            return self
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            tokenizer=_tokenize,
            token_pattern=None,
            lowercase=True,
            sublinear_tf=True,
            max_features=self.cfg.tfidf_max_features,
            dtype=np.float32,
        )
        X = self._vectorizer.fit_transform(texts)
        n_comp = min(
            self.cfg.tfidf_lsa_components or self.cfg.dim,
            X.shape[0] - 1,
            X.shape[1] - 1,
        )
        n_comp = max(1, n_comp)
        self._svd = TruncatedSVD(n_components=n_comp, random_state=42).fit(X)
        self.dim = self.cfg.dim
        log.info("TF-IDF/LSA fit with %d components.", n_comp)
        return self

    # ---------------------------------------------------------------- embed
    def embed(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.backend == "glm":
            vecs = self._embed_glm(texts)
        elif self.backend == "sentence-transformers":
            vecs = self._embed_sbert(texts, is_query=is_query)
        else:
            vecs = self._embed_tfidf(texts)
        vecs = _canonicalize(vecs, self.dim)
        if self.cfg.normalize:
            vecs = _l2_normalize(vecs)
        return vecs

    def _embed_sbert(self, texts: list[str], *, is_query: bool) -> np.ndarray:
        to_encode = texts
        if is_query and self.cfg.query_instruction:
            to_encode = [self.cfg.query_instruction + t for t in texts]
        vecs = self._model.encode(
            to_encode,
            normalize_embeddings=self.cfg.normalize,
            batch_size=min(64, max(8, len(to_encode))),
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def _embed_glm(self, texts: list[str]) -> np.ndarray:
        # The GLM API is symmetric: no bge-style query instruction prefix.
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        bs = max(1, self.cfg.glm_batch_size)
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            vecs = self._glm_request(batch)
            if vecs.shape[0] != len(batch):
                raise RuntimeError(
                    f"GLM returned {vecs.shape[0]} vectors for {len(batch)} inputs"
                )
            if vecs.shape[1] != self.dim:
                raise RuntimeError(
                    f"GLM returned {vecs.shape[1]}-dim vectors, expected {self.dim}"
                )
            out[i : i + len(batch)] = vecs
        return out

    def _embed_tfidf(self, texts: list[str]) -> np.ndarray:
        if self._vectorizer is None or self._svd is None:
            raise RuntimeError("TF-IDF backend requires fit() before embed().")
        X = self._vectorizer.transform(texts)
        vecs = self._svd.transform(X)
        return vecs.astype(np.float32)

    # -------------------------------------------------------------- helpers
    def embed_query(self, text: str) -> np.ndarray:
        arr = self.embed([text], is_query=True)
        return arr[0]

    @property
    def is_ready(self) -> bool:
        return (
            self._model is not None
            or self._vectorizer is not None
            or (self.backend == "glm" and self._requests is not None)
        )

    # ----------------------------------------------------------- persistence
    def save_state(self) -> None:
        """Persist the active backend/dim and (for TF-IDF) the fitted model so a
        later `search` can rebuild the exact same vector space."""
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        model = self.cfg.glm_model if self.backend == "glm" else self.cfg.model_name
        STATE_META.write_text(
            json.dumps(
                {
                    "backend": self.backend,
                    "model_name": model,
                    "dim": self.dim,
                    "normalize": self.cfg.normalize,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if self.backend == "tfidf" and self._vectorizer is not None:
            import joblib

            joblib.dump(
                {"vectorizer": self._vectorizer, "svd": self._svd},
                STATE_TFIDF,
            )
        log.info("Persisted embedding state: backend=%s dim=%d", self.backend, self.dim)

    @classmethod
    def load_state(cls, config: EmbeddingConfig) -> "Embedder":
        """Rebuild an embedder matching the persisted state (for query-time)."""
        if STATE_META.exists():
            try:
                meta = json.loads(STATE_META.read_text(encoding="utf-8"))
                config.backend = meta.get("backend", config.backend)
                config.dim = int(meta.get("dim", config.dim))
                config.normalize = bool(meta.get("normalize", config.normalize))
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not read embedding meta (%s); using config.", exc)
        emb = cls(config).load()
        if emb.backend == "tfidf" and STATE_TFIDF.exists():
            import joblib

            try:
                m = joblib.load(STATE_TFIDF)
                emb._vectorizer = m["vectorizer"]
                emb._svd = m["svd"]
                emb.dim = config.dim
                log.info("Loaded persisted TF-IDF model (dim=%d).", emb.dim)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not load persisted TF-IDF model (%s).", exc)
        return emb
