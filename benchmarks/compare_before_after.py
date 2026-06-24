#!/usr/bin/env python3
"""Compare the working-tree Index against git HEAD on benchmark corpora."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zerosearch import Index as CurrentIndex  # noqa: E402

SIMPLEWIKI_JSONL = ROOT / "benchmarks" / "data" / "simplewiki_10000.jsonl"
FAQ_ASSISTANT = ROOT.parent / "faq-assistant"
FAQ_CORPUS = FAQ_ASSISTANT / "artifacts" / "search" / "search-corpus.json"
FAQ_RESULTS = FAQ_ASSISTANT / "evals" / "results"
OUT_DIR = ROOT / "benchmarks" / "results"


def load_head_index() -> type:
    source = subprocess.check_output(
        ["git", "show", "HEAD:zerosearch/index.py"],
        cwd=ROOT,
        text=True,
    )
    module = ModuleType("zerosearch_head_index")
    exec(compile(source, "HEAD:zerosearch/index.py", "exec"), module.__dict__)
    return module.Index


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def ranked_key(records: list[dict[str, Any]], id_field: str) -> list[str]:
    return [str(record[id_field]) for record in records]


def compare_records(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    *,
    id_field: str,
    score_abs_tol: float = 1e-12,
) -> tuple[bool, str | None]:
    before_ids = ranked_key(before, id_field)
    after_ids = ranked_key(after, id_field)
    if before_ids != after_ids:
        return False, f"ranked ids differ: before={before_ids} after={after_ids}"

    for old, new in zip(before, after):
        if not math.isclose(
            float(old["score"]),
            float(new["score"]),
            rel_tol=0.0,
            abs_tol=score_abs_tol,
        ):
            return False, (
                f"score differs for {old[id_field]}: "
                f"before={old['score']!r} after={new['score']!r}"
            )
    return True, None


def simplewiki_queries(docs: list[dict[str, Any]], count: int) -> list[str]:
    titles = [doc["title"] for doc in docs if doc.get("title")]
    rng = random.Random(42)
    if count <= len(titles):
        sampled = rng.sample(titles, count)
    else:
        sampled = [rng.choice(titles) for _ in range(count)]
    return [title.split("(", 1)[0].strip().lower() for title in sampled]


def compare_simplewiki(HeadIndex: type, path: Path, queries: int, docs_limit: int | None) -> dict:
    docs = read_jsonl(path, docs_limit)
    query_list = simplewiki_queries(docs, queries)
    unique_queries = list(dict.fromkeys(query_list))
    before = HeadIndex(text_fields=["text"]).fit(docs)
    after = CurrentIndex(text_fields=["text"]).fit(docs)

    mismatches = []
    for query in unique_queries:
        old_results = before.search(query, num_results=10)
        new_results = after.search(query, num_results=10)
        ok, detail = compare_records(old_results, new_results, id_field="url")
        if not ok:
            mismatches.append({"query": query, "detail": detail})

    return {
        "docs": len(docs),
        "queries": len(query_list),
        "unique_queries_checked": len(unique_queries),
        "mismatches": mismatches,
        "matched": not mismatches,
    }


def import_faq_config() -> tuple[list[str], list[str], dict[str, Any]]:
    src = FAQ_ASSISTANT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    spec = importlib.util.find_spec("faq_assistant.search_index")
    if spec is None:
        raise RuntimeError("could not import faq_assistant.search_index")

    from faq_assistant.generated_config import CONFIG
    from faq_assistant.search_index import KEYWORD_FIELDS, TEXT_FIELDS

    return TEXT_FIELDS, KEYWORD_FIELDS, CONFIG


def hit_rate(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(item in relevant for item in ranked_ids[:k]) else 0.0


def mrr(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    for index, item in enumerate(ranked_ids[:k], 1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def faq_search(index, query: str, course: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    retrieval = config["retrieval"]
    records = index.search(
        query=query,
        filter_dict={"course": [course, ""]},
        boost_dict=retrieval.get("boosts", {}),
        num_results=int(retrieval["default_limit"]),
    )
    min_score = float(retrieval.get("min_score", 0))
    return [record for record in records if float(record.get("score", 0)) >= min_score]


def compare_faq(HeadIndex: type) -> dict:
    text_fields, keyword_fields, config = import_faq_config()
    corpus = json.loads(FAQ_CORPUS.read_text(encoding="utf-8"))
    before = HeadIndex(text_fields=text_fields, keyword_fields=keyword_fields).fit(corpus)
    after = CurrentIndex(text_fields=text_fields, keyword_fields=keyword_fields).fit(corpus)

    variants = {}
    total_mismatches = []
    for path in sorted(FAQ_RESULTS.glob("zerosearch__*.json")):
        if path.name == "summary__zerosearch.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics_before = defaultdict(float)
        metrics_after = defaultdict(float)
        mismatches = []

        for row in data["rows"]:
            query = row["rewritten"]
            course = row["course"]
            old_results = faq_search(before, query, course, config)
            new_results = faq_search(after, query, course, config)
            ok, detail = compare_records(old_results, new_results, id_field="id")
            if not ok:
                mismatches.append({"query": row["query"], "rewritten": query, "detail": detail})

            relevant = set(row["relevant"])
            old_ranked = ranked_key(old_results, "id")
            new_ranked = ranked_key(new_results, "id")
            for k in (1, 3, 5):
                metrics_before[f"hit@{k}"] += hit_rate(old_ranked, relevant, k)
                metrics_before[f"mrr@{k}"] += mrr(old_ranked, relevant, k)
                metrics_after[f"hit@{k}"] += hit_rate(new_ranked, relevant, k)
                metrics_after[f"mrr@{k}"] += mrr(new_ranked, relevant, k)

        n = len(data["rows"])
        before_summary = {key: round(value / n, 4) for key, value in metrics_before.items()}
        after_summary = {key: round(value / n, 4) for key, value in metrics_after.items()}
        before_summary["n"] = n
        after_summary["n"] = n
        variant = path.stem.removeprefix("zerosearch__")
        variants[variant] = {
            "before": before_summary,
            "after": after_summary,
            "mismatches": mismatches,
            "matched": not mismatches and before_summary == after_summary,
        }
        total_mismatches.extend({"variant": variant, **item} for item in mismatches)

    return {
        "docs": len(corpus),
        "variants": variants,
        "matched": not total_mismatches,
        "mismatch_count": len(total_mismatches),
        "mismatches": total_mismatches[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simplewiki", type=Path, default=SIMPLEWIKI_JSONL)
    parser.add_argument("--simplewiki-docs", type=int, default=None)
    parser.add_argument("--simplewiki-queries", type=int, default=100)
    parser.add_argument("--skip-simplewiki", action="store_true")
    parser.add_argument("--skip-faq", action="store_true")
    parser.add_argument("--label", default="compare_before_after")
    args = parser.parse_args()

    HeadIndex = load_head_index()
    result: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "before": "git HEAD:zerosearch/index.py",
        "after": "working tree zerosearch.Index",
    }
    if not args.skip_simplewiki:
        result["simplewiki"] = compare_simplewiki(
            HeadIndex,
            args.simplewiki,
            args.simplewiki_queries,
            args.simplewiki_docs,
        )
    if not args.skip_faq:
        result["faq"] = compare_faq(HeadIndex)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.label}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"results: {out_path}")


if __name__ == "__main__":
    main()
