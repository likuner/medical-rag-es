"""寻医问药疾病库（jib.xywy.com）爬虫。

从疾病库首页收集 ``il_sii_{id}.htm`` 疾病页链接，逐篇提取 ``.jib-rec``
区域的疾病概述。注意：站点编码为 GBK，取文本前必须显式设置编码。
"""
from __future__ import annotations

import logging
import re
from typing import Iterator, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from .base import BaseCrawler, RawDoc
from ..cleaners import clean_text
from ..config import CrawlConfig

log = logging.getLogger(__name__)

BASE = "https://jib.xywy.com"
INDEX_URL = f"{BASE}/"

# 首页抓取失败时的兜底疾病 id（百日咳/感冒/糖尿病等常见病）。
FALLBACK_IDS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]


class XywyCrawler(BaseCrawler):
    key = "xywy"
    name = "寻医问药疾病库"
    content_kind = "html"
    lang = "zh"

    def _get_html_gbk(self, url: str) -> str:
        """抓取 GBK 编码页面：requests 的 .text 依赖响应头字符集，这里显式指定。"""
        resp = self._get(url)
        resp.encoding = "gbk"
        return resp.text

    def crawl(self, max_pages: Optional[int] = None) -> Iterator[RawDoc]:
        cap = self._limit(max_pages) or 12
        ids: list[str] = []
        try:
            html = self._get_html_gbk(INDEX_URL)
            ids = re.findall(r'/(il_sii_(\d+)\.htm)', html)
            ids = [i[1] for i in ids]
            ids = list(dict.fromkeys(ids))
        except Exception as exc:  # noqa: BLE001
            log.warning("xywy index failed (%s); using fallback ids", exc)
        if not ids:
            ids = FALLBACK_IDS
        if len(ids) > cap:
            step = len(ids) / cap
            ids = [ids[int(i * step)] for i in range(cap)]

        for did in ids:
            if cap <= 0:
                return
            url = f"{BASE}/il_sii_{did}.htm"
            try:
                page_html, body = self._fetch_extract(url)
            except Exception as exc:  # noqa: BLE001
                log.warning("skipped %s: %s", url, exc)
                continue
            if not body or len(body) < 300:
                continue
            title = self._title(page_html)
            cap -= 1
            yield RawDoc(
                source=self.key,
                source_name=self.name,
                source_url=url,
                title=title,
                page_number=None,
                lang=self.lang,
                doc_metadata={"disease_id": did, "organization": "寻医问药网"},
                raw=body,
                content_kind="text",
            )

    def _fetch_extract(self, url: str, *, attempts: int = 2) -> tuple[str, str]:
        """抓取并提取正文；站点偶尔返回缺少 ``.jib-rec`` 的模板，
        按礼貌间隔重抓一次通常即可恢复。返回 (html, text)。"""
        html = ""
        for _ in range(max(1, attempts)):
            html = self._get_html_gbk(url)
            text = self._extract(html)
            if text:
                return html, text
        return html, ""

    def _title(self, html: str) -> Optional[str]:
        # h1 是问句式标题（"得了百日咳有什么症状…"），取 <title> 首段作病名。
        soup = BeautifulSoup(html, "lxml")
        t = soup.find("title")
        return (t.get_text(strip=True).split(",")[0].strip() or None) if t else None

    def _extract(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        # 正文容器是 class（.jib-rec），非 id。
        main = soup.select_one(".jib-rec") or soup.select_one(".main-sub") or soup.find("main")
        if main is None:
            return ""
        for junk in main.find_all(["script", "style", "noscript", "nav"]):
            junk.decompose()
        text = main.get_text("\n", strip=True)
        return clean_text(text)
