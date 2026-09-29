"""39健康网疾病百科（jbk.39.net）爬虫。

从首页、按身体部位（/bw/）与 A–Z（/azb/）索引收集疾病主页链接
（形如 ``/{拼音码}/``），仿 NHS 模式将每个疾病的着陆页与四个子栏目页
（病理病因 / 症状体征 / 预防护理 / 就诊指南）合并为一篇文档。
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

BASE = "https://jbk.39.net"
INDEX_URLS = [f"{BASE}/", f"{BASE}/bw/", f"{BASE}/azb/"]

# 疾病主页下的子栏目：病理病因 / 症状体征 / 预防护理 / 就诊指南。
SUBSECTIONS = ["blby", "zztz", "yfhl", "jzzn"]

# 索引页上出现的频道/导航路径，不是疾病主页。
RESERVED_CODES = {
    "bw", "azb", "jiancha", "shoushu", "zicha", "zx", "jrsy", "yzjptc",
    "zc", "yw", "rxzs", "ask", "search", "news", "zt", "tag", "video",
    "club", "wiki", "jb", "m", "www", "hudongbaike", "taiji", "newarticle",
}


# 疾病主页上与知识无关的推荐/互动板块（按板块首行标题识别）。
NOISE_BOX_HEADERS = {"相关文章", "疾病用药", "医院医生", "用药经验", "疾病自测", "相关医学名词"}


class Jk39Crawler(BaseCrawler):
    key = "jk39"
    name = "39健康疾病百科"
    content_kind = "html"
    lang = "zh"

    def crawl(self, max_pages: Optional[int] = None) -> Iterator[RawDoc]:
        cap = self._limit(max_pages) or 10      # 疾病数
        codes: list[str] = []
        for idx in INDEX_URLS:
            try:
                html = self._get_html(idx)
            except Exception as exc:  # noqa: BLE001
                log.warning("index %s failed: %s", idx, exc)
                continue
            found = re.findall(r'href="(?:https://jbk\.39\.net)?/([a-z]{2,12})/"', html)
            codes += [c for c in found if c not in RESERVED_CODES]
        codes = list(dict.fromkeys(codes))
        if not codes:
            return
        if len(codes) > cap:
            step = len(codes) / cap
            codes = [codes[int(i * step)] for i in range(cap)]

        for code in codes:
            if cap <= 0:
                return
            landing_url = f"{BASE}/{code}/"
            try:
                landing_html, main_text = self._fetch_extract(landing_url)
            except Exception as exc:  # noqa: BLE001
                log.warning("skipped %s: %s", landing_url, exc)
                continue

            title = self._title(landing_html)
            sections: list[str] = []
            if main_text:
                sections.append(main_text)
            sub_pages: list[str] = []
            for sub in SUBSECTIONS:
                sub_url = urljoin(landing_url, f"{sub}/")
                try:
                    _, sub_text = self._fetch_extract(sub_url, attempts=1)
                except Exception as exc:  # noqa: BLE001
                    continue
                if sub_text and len(sub_text) > 120:
                    sections.append(sub_text)
                    sub_pages.append(sub_url)
            body = "\n\n".join(s for s in sections if s).strip()
            if not body or len(body) < 400:
                continue
            cap -= 1
            yield RawDoc(
                source=self.key,
                source_name=self.name,
                source_url=landing_url,
                title=title,
                page_number=None,
                lang=self.lang,
                doc_metadata={
                    "disease_code": code,
                    "sub_pages": sub_pages,
                    "organization": "39健康网",
                },
                raw=body,
                content_kind="text",
            )

    def _fetch_extract(self, url: str, *, attempts: int = 2) -> tuple[str, str]:
        """抓取并提取正文；站点偶尔返回缺少 ``.list_left`` 的模板，
        此时按礼貌间隔重抓一次通常即可恢复。返回 (html, text)。"""
        html = ""
        for _ in range(max(1, attempts)):
            html = self._get_html(url)
            text = self._extract_page(html)
            if text:
                return html, text
        return html, ""

    def _title(self, html: str) -> Optional[str]:
        soup = BeautifulSoup(html, "lxml")
        h1 = soup.find("h1")
        return clean_text(h1.get_text(" ", strip=True)) if h1 else None

    def _extract_page(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        # 站点模板里正文容器是 class（.list_left），非 id。
        main = soup.select_one(".list_left") or soup.find("main") or soup.find("article")
        if main is None:
            return ""
        for junk in main.find_all(["script", "style", "noscript"]):
            junk.decompose()
        for nav in main.find_all("nav"):
            nav.decompose()
        # 剔除医生推荐 / 用药经验等噪声板块。
        for box in main.select(".disease_box"):
            head = box.get_text("\n", strip=True).split("\n", 1)[0].strip()
            if head in NOISE_BOX_HEADERS:
                box.decompose()
        text = main.get_text("\n", strip=True)
        return clean_text(text)
