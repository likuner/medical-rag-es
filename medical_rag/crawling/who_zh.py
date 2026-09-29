"""WHO 中文实况报道爬虫。

与英文版 who.py 同构：抓取实况报道列表页，逐篇提取 ``#content`` 区域的
中文正文；robots.txt 未限制非名单爬虫，页面为服务端渲染。
"""
from __future__ import annotations

import logging
import re
from typing import Iterator, Optional

from bs4 import BeautifulSoup

from .base import BaseCrawler, RawDoc
from ..cleaners import clean_text
from ..config import CrawlConfig

log = logging.getLogger(__name__)

BASE = "https://www.who.int"
INDEX_URL = f"{BASE}/zh/news-room/fact-sheets"

# 列表页抓取失败时的兜底 slug（与英文版相同的疾病主题）。
FALLBACK_SLUGS = [
    "diabetes",
    "hypertension",
    "cardiovascular-diseases",
    "cancer",
    "asthma",
    "obesity",
    "tobacco",
    "hepatitis",
    "influenza",
    "tuberculosis",
    "dementia",
    "measles",
]


class WHOZhCrawler(BaseCrawler):
    key = "who_zh"
    name = "WHO 实况报道（中文）"
    content_kind = "html"
    lang = "zh"

    def crawl(self, max_pages: Optional[int] = None) -> Iterator[RawDoc]:
        cap = self._limit(max_pages) or 12
        slugs: list[str] = []
        try:
            html = self._get_html(INDEX_URL)
            slugs = re.findall(
                r'href="(/zh/news-room/fact-sheets/detail/[^"]+)"', html
            )
            slugs = [s.rsplit("/", 1)[-1] for s in slugs]
            slugs = list(dict.fromkeys(slugs))
        except Exception as exc:  # noqa: BLE001
            log.warning("WHO zh fact-sheets index failed (%s); using fallback slugs", exc)
        if not slugs:
            slugs = FALLBACK_SLUGS
        if len(slugs) > cap:
            # 在完整列表上等距采样，保证主题多样。
            step = len(slugs) / cap
            slugs = [slugs[int(i * step)] for i in range(cap)]

        for slug in slugs:
            if cap <= 0:
                return
            url = f"{BASE}/zh/news-room/fact-sheets/detail/{slug}"
            try:
                page_html = self._get_html(url)
            except Exception as exc:  # noqa: BLE001
                log.warning("skipped %s: %s", url, exc)
                continue
            title, body = self._extract(page_html)
            if not body or len(body) < 300:
                continue
            cap -= 1
            yield RawDoc(
                source=self.key,
                source_name=self.name,
                source_url=url,
                title=title,
                page_number=None,
                lang=self.lang,
                doc_metadata={"fact_sheet_slug": slug, "organization": "世界卫生组织"},
                raw=body,
                content_kind="text",
            )

    def _extract(self, html: str) -> tuple[Optional[str], str]:
        soup = BeautifulSoup(html, "lxml")
        title_el = soup.find("h1")
        title = title_el.get_text(strip=True) if title_el else None
        main = soup.select_one("#content") or soup.find(attrs={"role": "main"}) or soup.find("main")
        if main is None:
            return title, ""
        for junk in main.find_all(["script", "style", "noscript"]):
            junk.decompose()
        text = main.get_text("\n", strip=True)
        # 去掉顶部的中文面包屑与底部版权行。
        text = re.sub(r"^主页\s*/\s*新闻\s*/\s*实况报道\s*/\s*详情\s*/\s*[^\n]*", "", text)
        text = re.sub(r"^[^\n]*©[^\n]*\n?", "", text)
        return title, clean_text(text)
