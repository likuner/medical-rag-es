"""A+医学百科（a-hospital.com）爬虫。

MediaWiki 架构的中文医学百科。从「疾病」枢纽页收集 ``/w/`` 文章链接，
逐篇提取 ``#bodyContent`` 正文。
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

BASE = "https://www.a-hospital.com"
# 「疾病」枢纽页（百分号编码）。
HUB_URL = f"{BASE}/w/%E7%96%BE%E7%97%85"

# 枢纽页抓取失败时的兜底词条（常见疾病/医学概念）。
FALLBACK_TITLES = [
    "糖尿病", "高血压", "冠心病", "肺炎", "肺结核", "肝炎", "胃癌",
    "白血病", "脑梗塞", "贫血", "肾病综合征", "甲状腺功能亢进",
    "类风湿性关节炎", "哮喘", "阑尾炎", "前列腺增生",
]

# 枢纽/导航页本身，不作为文档抓取。
HUB_TITLES = {"首页", "疾病", "药品", "症状", "检查", "手术", "中医"}


class AHospitalCrawler(BaseCrawler):
    key = "a_hospital"
    name = "A+医学百科"
    content_kind = "html"
    lang = "zh"

    def crawl(self, max_pages: Optional[int] = None) -> Iterator[RawDoc]:
        cap = self._limit(max_pages) or 12
        paths: list[str] = []
        try:
            html = self._get_html(HUB_URL)
            paths = re.findall(r'href="(/w/[^"#?]+)"', html)
            # 过滤命名空间页（含冒号）与编辑链接；枢纽页在提取阶段按标题剔除。
            paths = [
                p for p in paths
                if "%3A" not in p and ":" not in p and "action=" not in p
            ]
            paths = list(dict.fromkeys(paths))
        except Exception as exc:  # noqa: BLE001
            log.warning("A+医学百科 hub page failed (%s); using fallback titles", exc)
        if not paths:
            from urllib.parse import quote

            paths = [f"/w/{quote(t)}" for t in FALLBACK_TITLES]
        if len(paths) > cap:
            step = len(paths) / cap
            paths = [paths[int(i * step)] for i in range(cap)]

        for path in paths:
            if cap <= 0:
                return
            url = urljoin(BASE, path)
            try:
                page_html = self._get_html(url)
            except Exception as exc:  # noqa: BLE001
                log.warning("skipped %s: %s", url, exc)
                continue
            title, body = self._extract(page_html)
            if not body or len(body) < 400 or (title and title in HUB_TITLES):
                continue
            cap -= 1
            yield RawDoc(
                source=self.key,
                source_name=self.name,
                source_url=url,
                title=title,
                page_number=None,
                lang=self.lang,
                doc_metadata={
                    "article_title": title or path.rsplit("/", 1)[-1],
                    "site_type": "医学百科",
                },
                raw=body,
                content_kind="text",
            )

    def _extract(self, html: str) -> tuple[Optional[str], str]:
        soup = BeautifulSoup(html, "lxml")
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else None
        main = (
            soup.select_one("#bodyContent")
            or soup.select_one("#mw-content-text")
            or soup.select_one("#content")
        )
        if main is None:
            return title, ""
        for junk in main.find_all(["script", "style", "noscript", "nav", "footer"]):
            junk.decompose()
        # MediaWiki 的打印页脚、分类栏等噪声块。
        for sel in (".printfooter", "#catlinks", ".mw-jump-link"):
            for el in main.select(sel):
                el.decompose()
        text = main.get_text("\n", strip=True)
        return title, clean_text(text)
