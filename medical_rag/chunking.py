"""Text chunking, backed by LlamaIndex's SentenceSplitter.

The splitter is configured with a custom Chinese-aware tokenizer (CJK char ==
1 token, latin word == 1 token) and a Chinese sentence-boundary function, so
chunk sizes are measured in these tokens and tiktoken / nltk are never
invoked (offline-safe). The public interface — :class:`Chunk` and
:func:`chunk_document` — is unchanged, so the pipeline and DB layer are
unaffected.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from llama_index.core.node_parser import SentenceSplitter

from .config import ChunkConfig

# Chinese sentence boundaries (full-width + latin punctuation, newlines).
_SENT_BOUNDARY_RE = re.compile(r"(?<=[。！？!?\.;；])\s*|\n+", re.UNICODE)
# Token approximation: CJK characters are one token each, latin/digit runs one.
_TOKEN_RE = re.compile(
    r"[\u3400-\u9fff\uf900-\ufaff]|[a-zA-Z0-9]+(?:['’\-][a-zA-Z0-9]+)*|[^\s]"
)


@dataclass
class Chunk:
    index: int
    text: str
    char_length: int = field(init=False)
    token_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_length = len(self.text)
        self.token_count = count_tokens(self.text)

    def as_row(self) -> dict:
        return {
            "chunk_index": self.index,
            "chunk_text": self.text,
            "char_length": self.char_length,
            "token_count": self.token_count,
        }


def count_tokens(text: str) -> int:
    """Rough token count (CJK char == 1 token, latin word == 1 token)."""
    return len(_TOKEN_RE.findall(text))


def _tokenize(text: str) -> list[str]:
    """Tokenizer contract required by SentenceSplitter: return a sequence
    (it measures size via ``len(tokenizer(text))``)."""
    return _TOKEN_RE.findall(text)


def _split_sentences_zh(text: str) -> list[str]:
    """Chinese/latin sentence split for SentenceSplitter's primary level."""
    return [s for s in _SENT_BOUNDARY_RE.split(text) if s]


def chunk_document(text: str, config: ChunkConfig | None = None) -> list[Chunk]:
    cfg = config or ChunkConfig()
    size, overlap = cfg.size, cfg.overlap
    if overlap >= size:
        overlap = max(0, size // 4)

    splitter = SentenceSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        tokenizer=_tokenize,
        chunking_tokenizer_fn=_split_sentences_zh,
    )
    texts = [t.strip() for t in splitter.split_text(text) if t.strip()]

    chunks = [Chunk(index=i, text=t) for i, t in enumerate(texts)]
    # Drop degenerate short chunks and renumber.
    if cfg.min_chunk:
        chunks = [c for c in chunks if c.char_length >= cfg.min_chunk]
        for i, c in enumerate(chunks):
            c.index = i
    return chunks
