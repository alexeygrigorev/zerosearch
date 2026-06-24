#!/usr/bin/env python3
"""Benchmark zerosearch on the Simple English Wikipedia corpus.

The corpus preparation mirrors the benchmark in the sibling minsearch checkout:
extract article title/text/url documents from a Simple Wikipedia XML dump, then
sample search queries from article titles.
"""

from __future__ import annotations

import argparse
import bz2
import codecs
import json
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
import urllib.request
import xml.sax
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zerosearch import Index as CurrentIndex  # noqa: E402


# The benchmark shape mirrors ../minsearch/benchmark, but Wikimedia moved the
# dump path. The latest Simple Wiki pages-articles dump is a stable alias.
DEFAULT_DUMP_URL = (
    "https://dumps.wikimedia.org/simplewiki/latest/"
    "simplewiki-latest-pages-articles.xml.bz2"
)

DEFAULT_JSONL = ROOT / "benchmarks" / "data" / "simplewiki_1000.jsonl"
DEFAULT_RESULTS_DIR = ROOT / "benchmarks" / "results"


class StopParsing(Exception):
    """Raised internally when the requested document limit is reached."""


class WikipediaHandler(xml.sax.ContentHandler):
    """SAX handler that writes benchmark documents as JSONL."""

    skip_prefixes = {
        "User",
        "User talk",
        "Wikipedia",
        "File",
        "File talk",
        "Template",
        "Template talk",
        "Help",
        "Help talk",
        "Category",
        "Category talk",
        "Portal",
        "Talk",
        "MediaWiki",
    }

    def __init__(self, output_file, max_docs: int | None = None) -> None:
        super().__init__()
        self.output_file = output_file
        self.max_docs = max_docs
        self.in_page = False
        self.in_title = False
        self.in_text = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.current_page: dict[str, str] = {}
        self.doc_count = 0
        self.skip_count = 0

    def startElement(self, name: str, attrs) -> None:  # noqa: N802
        if name == "page":
            self.in_page = True
            self.current_page = {}
            self.title_parts = []
            self.text_parts = []
            return

        if not self.in_page:
            return
        if name == "title":
            self.in_title = True
            self.title_parts = []
        elif name == "text":
            self.in_text = True
            self.text_parts = []

    def endElement(self, name: str) -> None:  # noqa: N802
        if not self.in_page:
            return

        if name == "title":
            self.in_title = False
            self.current_page["title"] = "".join(self.title_parts)
        elif name == "text":
            self.in_text = False
            self.current_page["text"] = "".join(self.text_parts)
        elif name == "page":
            self.in_page = False
            self._emit_page()

    def characters(self, content: str) -> None:
        if self.in_title:
            self.title_parts.append(content)
        elif self.in_text:
            self.text_parts.append(content)

    def _emit_page(self) -> None:
        title = self.current_page.get("title", "")
        text = self.current_page.get("text", "").strip()
        if not title or len(text) <= 50:
            return

        if ":" in title and title.split(":", 1)[0] in self.skip_prefixes:
            self.skip_count += 1
            return

        doc = {
            "title": title,
            "text": text,
            "url": f"https://simple.wikipedia.org/wiki/{title.replace(' ', '_')}",
        }
        self.output_file.write(json.dumps(doc, ensure_ascii=False) + "\n")
        self.doc_count += 1
        if self.max_docs is not None and self.doc_count >= self.max_docs:
            raise StopParsing()


def ensure_jsonl(path: Path, dump_url: str, max_docs: int) -> None:
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Streaming {max_docs:,} Simple Wikipedia docs from {dump_url}")
    print(f"Writing {path}")

    parser = xml.sax.make_parser()
    with urllib.request.urlopen(dump_url, timeout=60) as response:
        decompressor = bz2.BZ2Decompressor()
        decoder = codecs.getincrementaldecoder("utf-8")()
        with path.open("w", encoding="utf-8") as output:
            handler = WikipediaHandler(output, max_docs=max_docs)
            parser.setContentHandler(handler)

            try:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    text = decoder.decode(decompressor.decompress(chunk))
                    if text:
                        parser.feed(text)
                tail = decoder.decode(b"", final=True)
                if tail:
                    parser.feed(tail)
                parser.close()
            except StopParsing:
                print(f"Reached {handler.doc_count:,} documents; stopping stream")


def load_docs(path: Path, limit: int | None) -> list[dict[str, Any]]:
    docs = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            docs.append(json.loads(line))
            if limit is not None and len(docs) >= limit:
                break
    return docs


def make_queries(docs: list[dict[str, Any]], count: int) -> list[str]:
    titles = [doc["title"] for doc in docs if doc.get("title")]
    rng = random.Random(42)
    if count <= len(titles):
        sampled = rng.sample(titles, count)
    else:
        sampled = [rng.choice(titles) for _ in range(count)]
    return [title.split("(", 1)[0].strip().lower() for title in sampled]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((pct / 100) * (len(ordered) - 1)))
    return ordered[index]


def index_footprint(index) -> dict[str, int]:
    arrays = [
        "_post_off",
        "_post_doc",
        "_post_field",
        "_post_tf",
        "_doc_freq",
        "_lengths",
    ]
    packed_array_bytes = sum(
        getattr(index, name).buffer_info()[1] * getattr(index, name).itemsize
        for name in arrays
        if hasattr(index, name)
    )
    return {
        "index_serialized_bytes": len(index.dumps()),
        "index_packed_array_bytes": packed_array_bytes,
        "vocab_terms": len(index._vocab),
        "postings": len(index._post_doc),
    }


def benchmark(docs: list[dict[str, Any]], queries: list[str]) -> dict[str, Any]:
    total_chars = sum(len(str(doc.get("text", ""))) for doc in docs)

    tracemalloc.start()
    build_start = time.perf_counter()
    index = INDEX_CLASS(text_fields=["text"]).fit(docs)
    build_seconds = time.perf_counter() - build_start
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if queries:
        index.search(queries[0], num_results=10)

    timings = []
    hit_counts = []
    for query in queries:
        start = time.perf_counter()
        results = index.search(query, num_results=10)
        timings.append(time.perf_counter() - start)
        hit_counts.append(len(results))

    avg = statistics.fmean(timings) if timings else 0.0
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "engine": INDEX_LABEL,
        "docs": len(docs),
        "queries": len(queries),
        "total_text_chars": total_chars,
        "build_seconds": build_seconds,
        "build_current_bytes_tracemalloc": current_bytes,
        "build_peak_bytes_tracemalloc": peak_bytes,
        **index_footprint(index),
        "search_avg_ms": avg * 1000,
        "search_median_ms": (statistics.median(timings) * 1000) if timings else 0.0,
        "search_p95_ms": percentile(timings, 95) * 1000,
        "search_min_ms": (min(timings) * 1000) if timings else 0.0,
        "search_max_ms": (max(timings) * 1000) if timings else 0.0,
        "qps": (1 / avg) if avg else 0.0,
        "avg_hits": statistics.fmean(hit_counts) if hit_counts else 0.0,
    }


def print_summary(results: dict[str, Any]) -> None:
    print("\nSimple Wikipedia benchmark")
    print(f"docs:              {results['docs']:,}")
    print(f"queries:           {results['queries']:,}")
    print(f"text chars:        {results['total_text_chars']:,}")
    print(f"build:             {results['build_seconds']:.3f} s")
    print(f"index current mem: {results['build_current_bytes_tracemalloc'] / 1024 / 1024:.1f} MB")
    print(f"build peak memory: {results['build_peak_bytes_tracemalloc'] / 1024 / 1024:.1f} MB")
    print(f"serialized index:  {results['index_serialized_bytes'] / 1024 / 1024:.1f} MB")
    print(f"packed arrays:     {results['index_packed_array_bytes'] / 1024 / 1024:.1f} MB")
    print(f"search avg:        {results['search_avg_ms']:.3f} ms")
    print(f"search median:     {results['search_median_ms']:.3f} ms")
    print(f"search p95:        {results['search_p95_ms']:.3f} ms")
    print(f"search qps:        {results['qps']:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--dump-url", default=DEFAULT_DUMP_URL)
    parser.add_argument("--sample-docs", type=int, default=1000)
    parser.add_argument("--num-docs", type=int, default=None)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--index-ref",
        default=None,
        help="Load Index from a git ref, e.g. HEAD. Defaults to the working tree.",
    )
    args = parser.parse_args()

    configure_index(args.index_ref)
    ensure_jsonl(args.input, args.dump_url, args.sample_docs)
    docs = load_docs(args.input, args.num_docs)
    queries = make_queries(docs, args.num_queries)
    results = benchmark(docs, queries)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result_path = args.output_dir / f"{label}.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    print_summary(results)
    print(f"results:           {result_path}")


def configure_index(index_ref: str | None) -> None:
    global INDEX_CLASS, INDEX_LABEL
    if index_ref is None:
        INDEX_CLASS = CurrentIndex
        INDEX_LABEL = "zerosearch"
        return

    source = subprocess.check_output(
        ["git", "show", f"{index_ref}:zerosearch/index.py"],
        cwd=ROOT,
        text=True,
    )
    namespace: dict[str, Any] = {}
    exec(compile(source, f"{index_ref}:zerosearch/index.py", "exec"), namespace)
    INDEX_CLASS = namespace["Index"]
    INDEX_LABEL = f"zerosearch@{index_ref}"


INDEX_CLASS = CurrentIndex
INDEX_LABEL = "zerosearch"


if __name__ == "__main__":
    main()
