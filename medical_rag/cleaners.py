"""Text cleaning and normalization utilities.

The crawlers hand us raw HTML (or XML/JSON for API-backed sources). This module
turns that into clean, human-readable text and normalises it before chunking.
"""
from __future__ import annotations

import html as _html
import re
from typing import Iterable

from bs4 import BeautifulSoup, Tag

# Tags whose content is boilerplate / not part of the medical prose.
_STRIP_TAGS = {
    "script", "style", "noscript", "template", "svg", "iframe", "form",
    "nav", "header", "footer", "aside", "button", "input", "select",
    "option", "datalist", "dialog", "canvas", "video", "audio", "source",
    "picture-map", "figure", "figcaption", "menu", "ad", "ins",
}

# Approximate multi-space / newline runs.
_WS_RE = re.compile(r"[ \t\u00a0\u2007\u202f]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{2,}")
_CJK_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"
)
_BAD_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

_BLOCK_TAGS = {
    "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "br", "blockquote", "tr", "td", "th", "ul", "ol",
}


def soup_from_html(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _has_content(el: Tag) -> bool:
    """Rough check: an element holds real prose and not just whitespace."""
    txt = el.get_text(" ", strip=True)
    return len(txt) >= 40


def html_to_text(
    html: str,
    *,
    prefer_main: bool = True,
    min_paragraph: int = 24,
) -> str:
    """Convert an HTML document into clean plain text.

    Prefers <article>/<main> content when present and falls back to <body>.
    Removes boilerplate tags and keeps block structure so headings survive.
    """
    soup = soup_from_html(html)
    for t in soup.find_all(list(_STRIP_TAGS)):
        t.decompose()

    main_el = None
    if prefer_main:
        for sel in ("article", "main", "div[role=main]"):
            cand = soup.select_one(sel)
            if cand is not None and _has_content(cand if isinstance(cand, Tag) else cand):
                main_el = cand
                break
        # If we picked <main> but an <article> inside is richer, prefer it.
        if main_el is not None and main_el.find("article"):
            main_el = main_el.find("article")
    if main_el is None:
        body = soup.body or soup
        main_el = body

    parts: list[str] = []
    for el in main_el.find_all(True):
        if el.name in _BLOCK_TAGS:
            text = el.get_text(" ", strip=True)
            if text and len(text) >= min_paragraph:
                parts.append(text)
    if not parts:
        parts = [p for p in (s.get_text(" ", strip=True) for s in soup.find_all("p")) if p]

    return clean_text("\n\n".join(parts))


def clean_text(text: str) -> str:
    """Normalise a raw text blob for storage and chunking."""
    if not text:
        return ""
    text = _html.unescape(text)
    text = _BAD_CHARS.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = _WS_RE.sub(" ", line).strip()
        if line:
            lines.append(line)
    joined = "\n".join(lines)
    joined = _MULTI_NEWLINE_RE.sub("\n", joined).strip()
    return joined


def dedupe_lines(text: str) -> str:
    """Drop repeated consecutive identical lines (often from crawls)."""
    out: list[str] = []
    prev = None
    for line in text.split("\n"):
        if line != prev:
            out.append(line)
        prev = line
    return "\n".join(out)


def detect_lang(text: str) -> str:
    """Cheap language guess: 'zh' if a CJK ratio is high, else 'en'."""
    sample = text[:2000]
    if not sample:
        return "unknown"
    cjk = _CJK_RE.findall(sample)
    ratio = len(cjk) / max(1, len(re.sub(r"\s+", "", sample)))
    if ratio > 0.20:
        return "zh"
    return "en"


def strip_boilerplate(text: str, *, max_len: int | None = None) -> str:
    """Remove common cookie/consent boilerplate sentences (Chinese + English)."""
    patterns = [
        r"(?i)(cookie|privacy|terms of use|all rights reserved|©|©️).{0,120}",
        r"(?i)(本网站|版权所有|保留所有权利|使用条款|隐私政策|免责声明)[^\n]*",
        r"(?i)(微信|公众号|扫码|关注我们|订阅我们|下载应用)[^\n]*",
    ]
    for pat in patterns:
        text = re.sub(pat, " ", text)
    text = clean_text(text)
    return text[:max_len] if max_len else text


def iter_paragraphs(text: str) -> Iterable[str]:
    for p in re.split(r"\n\s*\n", text):
        p = p.strip()
        if p:
            yield p
