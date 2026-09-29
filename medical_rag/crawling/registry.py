"""Registry of available site crawlers (中文医学数据源)."""
from __future__ import annotations

import logging
from typing import Type

from ..config import CrawlConfig
from .a_hospital import AHospitalCrawler
from .base import BaseCrawler
from .jk39 import Jk39Crawler
from .msd_cn import MSDCnCrawler
from .who_zh import WHOZhCrawler
from .xywy import XywyCrawler

log = logging.getLogger(__name__)

CRAWLERS: dict[str, Type[BaseCrawler]] = {
    "who_zh": WHOZhCrawler,
    "msd_cn": MSDCnCrawler,
    "a_hospital": AHospitalCrawler,
    "jk39": Jk39Crawler,
    "xywy": XywyCrawler,
}

ALL_SITES = ["who_zh", "msd_cn", "a_hospital", "jk39", "xywy"]


def get_crawler(key: str, cfg: CrawlConfig) -> BaseCrawler:
    if key not in CRAWLERS:
        raise KeyError(f"Unknown site '{key}'. Available: {list(CRAWLERS)}")
    return CRAWLERS[key](cfg)


def get_crawlers(sites: list[str] | None, cfg: CrawlConfig) -> list[BaseCrawler]:
    keys = sites or ALL_SITES
    return [get_crawler(k, cfg) for k in keys]
