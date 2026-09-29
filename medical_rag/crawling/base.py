"""Crawler base classes and shared HTTP helpers.

Each site crawler yields :class:`RawDoc` objects (one per logical document,
e.g. one PubMed abstract, one MedlinePlus topic, one WHO fact sheet).  The
pipeline is responsible for cleaning, chunking and embedding; this layer only
fetches and shapes the raw content.
"""
from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

from ..config import CrawlConfig

log = logging.getLogger(__name__)


@dataclass
class RawDoc:
    source: str
    source_name: str
    source_url: str
    title: Optional[str]
    doc_metadata: dict[str, Any]
    raw: str
    content_kind: str = "html"          # "html" | "text"
    page_number: Optional[int] = None
    lang: Optional[str] = None


@dataclass
class FetchResult:
    url: str
    data: Any                          # str (html/text) or dict (json)
    content_type: str
    final_url: str
    status: int


class BaseCrawler(ABC):
    key: str = ""
    name: str = ""
    content_kind: str = "html"         # default for `raw`
    lang: Optional[str] = None
    respect_robots: bool = True        # API crawlers set False

    def __init__(self, cfg: CrawlConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": cfg.user_agent,
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
            }
        )
        self._last_request = 0.0
        self._robots: dict[str, Optional[RobotFileParser]] = {}

    # ------------------------------------------------------------------ HTTP
    def _sleep(self) -> None:
        # Small jitter around the configured delay to avoid burst requests.
        base_delay = getattr(self, "request_delay", None) or self.cfg.delay
        delay = base_delay * random.uniform(0.9, 1.2)
        elapsed = time.time() - self._last_request
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request = time.time()

    def _robots_allowed(self, url: str) -> bool:
        if not self.respect_robots or not self.cfg.respect_robots:
            return True
        parsed = urlparse(url)
        host = parsed.netloc
        if host not in self._robots:
            rp = RobotFileParser()
            try:
                rp.parse(self._fetch_robots(parsed))
            except Exception:  # noqa: BLE001
                rp = RobotFileParser()  # defensively allow if robots unreadable
            self._robots[host] = rp if rp.entries else None
        rp = self._robots.get(host)
        if rp is None:
            return True
        return rp.can_fetch(self.cfg.user_agent, url)

    def _fetch_robots(self, parsed) -> str:
        url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        try:
            r = self.session.get(url, timeout=self.cfg.timeout)
            return r.text if r.status_code == 200 else ""
        except requests.RequestException:
            return ""

    def _get(self, url: str, *, params: dict | None = None, headers: dict | None = None) -> requests.Response:
        self._sleep()
        if not self._robots_allowed(url):
            log.info("robots.txt forbids %s; skipping", url)
            # Raise so crawlers don't hammer; handled by caller as a skip.
            raise requests.exceptions.RequestException(f"robots disallow: {url}")
        last_exc: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = self.session.get(
                    url, params=params, headers=headers, timeout=self.cfg.timeout
                )
                if r.status_code in (429,) or r.status_code >= 500:
                    # Back off and retry on throttling / server errors.
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if r.status_code >= 400:
                    # Raise for browser-challenge/blocked pages so we skip them.
                    raise requests.exceptions.HTTPError(
                        f"HTTP {r.status_code} for {url}",
                        response=r,
                    )
                return r
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.cfg.max_retries:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise
        raise last_exc if last_exc else requests.exceptions.RequestException(url)

    def _get_html(self, url: str, **kw) -> str:
        return self._get(url, **kw).text

    def _get_json(self, url: str, **kw) -> Any:
        return self._get(url, **kw).json()

    def _get_text(self, url: str, **kw) -> str:
        return self._get(url, **kw).text

    # ------------------------------------------------------------- crawling
    @abstractmethod
    def crawl(self, max_pages: Optional[int] = None) -> Iterator[RawDoc]:
        """Yield RawDoc records. `max_pages` bounds the number of pages."""
        raise NotImplementedError

    def _limit(self, max_pages: Optional[int]) -> Optional[int]:
        return min(max_pages, self.cfg.max_pages_per_site) if max_pages else (self.cfg.max_pages_per_site or None)
