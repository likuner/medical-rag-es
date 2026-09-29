#!/usr/bin/env python3
"""CLI for the medical RAG pipeline.

Examples
--------
# Ingest all five Chinese sites (bounded crawls):
python run_pipeline.py ingest

# Ingest a single site, e.g. only WHO 中文:
python run_pipeline.py ingest --sites who_zh --max-pages 8

# Rebuild the DB from scratch:
python run_pipeline.py ingest --recreate

# Force the lightweight TF-IDF embedding backend:
python run_pipeline.py ingest --embed-backend tfidf

# Semantic search over what's stored:
python run_pipeline.py search --query "2型糖尿病的治疗" --k 5
"""
from __future__ import annotations

import argparse
import logging
import sys

from medical_rag.crawling.registry import ALL_SITES
from medical_rag.pipeline import run, search


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Crawl, clean, chunk, embed -> Elasticsearch")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="Run the crawl -> store pipeline")
    ing.add_argument("--sites", nargs="+", default=ALL_SITES,
                     help="site keys to crawl (default: all)")
    ing.add_argument("--max-pages", type=int, default=None,
                     help="max documents per site (default from config)")
    ing.add_argument("--recreate", action="store_true",
                     help="delete and recreate the ES indices before ingesting")
    ing.add_argument("--embed-backend",
                     choices=["glm", "sentence-transformers", "tfidf", "auto"],
                     default=None, help="override the embedding backend")

    q = sub.add_parser("search", help="Run a similarity search")
    q.add_argument("--query", required=True, help="the natural-language query")
    q.add_argument("--k", type=int, default=5, help="number of neighbours")
    q.add_argument("--sites", nargs="+", default=None, help="filter by source")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        if args.command == "ingest":
            result = run(
                sites=args.sites,
                max_pages=args.max_pages,
                recreate=args.recreate,
                embed_backend=args.embed_backend,
            )
            print(f"\nDone: {result}")
        else:  # search
            results = search(args.query, k=args.k, sites=args.sites)
            if not results:
                print("No results. Run `ingest` first.")
                return 1
            print(f"\nTop {len(results)} matches for: {args.query!r}\n")
            for i, r in enumerate(results, 1):
                print(f"{i}. [{r['source']}] sim={r['similarity']} "
                      f"{r['title']} (chunk {r['chunk_index']})")
                print(f"   url: {r['url']}")
                print(f"   {r['text'][:220]}...\n")
        return 0
    except Exception as exc:  # noqa: BLE001
        logging.exception("Pipeline failed")
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
