# Benchmarks

These benchmarks compare the working-tree `zerosearch.Index` against the
pre-optimization implementation from `git HEAD`.

## Simple Wikipedia

The Simple English Wikipedia runner mirrors the `minsearch` benchmark shape:
parse Simple Wiki article `title`/`text` documents, build an index over `text`,
and sample search queries from article titles with `random.seed(42)`. When
`--num-queries` is larger than the number of available article titles, the
runner samples titles with replacement. This makes long throughput runs such as
100,000 query executions possible on the cached 10,000-document sample.

The runner streams a bounded sample from:

`https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2`

On June 24, 2026, Wikimedia's `latest` alias pointed at the June 1, 2026
Simple Wiki dump. The old `minsearch` pinned URL for the February 2026 content
dump returned 404, so the local benchmark uses the current Wikimedia dump path.

Run:

```bash
uv run python benchmarks/simplewiki_benchmark.py \
  --input benchmarks/data/simplewiki_10000.jsonl \
  --sample-docs 10000 \
  --num-queries 100 \
  --label final_10000
```

Results on this machine:

| sample | version | build | peak memory | avg search | median | p95 | qps |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10,000 docs | before | 35.798 s | 338.4 MB | 0.954 ms | 0.162 ms | 2.660 ms | 1,047.9 |
| 10,000 docs | after | 29.762 s | 399.1 MB | 0.312 ms | 0.094 ms | 1.319 ms | 3,209.3 |
| 10,000 docs / 100k queries | before | 27.998 s | 364.9 MB | 0.665 ms | 0.134 ms | 2.993 ms | 1,504.0 |
| 10,000 docs / 100k queries | after | 29.868 s | 399.1 MB | 0.321 ms | 0.087 ms | 1.474 ms | 3,113.6 |
| 1,000 docs | before | 5.295 s | 51.2 MB | 0.116 ms | 0.055 ms | 0.442 ms | 8,606.8 |
| 1,000 docs | after | 3.980 s | 63.1 MB | 0.063 ms | 0.038 ms | 0.212 ms | 15,970.8 |

On the 10,000-document sample, average search latency improved by 3.1x and p95
latency improved by 2.0x. On the longer 100,000-query run, average latency
improved by 2.1x and p95 latency improved by 2.0x. The final implementation is
slower than the earlier normalized-weight experiment, but it preserves the
pre-optimization ranking and score behavior exactly.

The 100,000-query run was added after the first report to reduce timing noise
from very small query samples. It uses the same 10,000 cached documents and the
same title-derived query distribution, but samples with replacement. The longer
run confirms the 100-query result: average search latency stays around `0.32 ms`
and throughput is about `3.1k QPS` on this machine.

Raw result files:

- `benchmarks/results/baseline_10000.json`
- `benchmarks/results/final_10000.json`
- `benchmarks/results/baseline_10000_100k_queries.json`
- `benchmarks/results/final_10000_100k_queries.json`
- `benchmarks/results/baseline.json`
- `benchmarks/results/final.json`

## Correctness Check

Run:

```bash
uv run python benchmarks/compare_before_after.py \
  --simplewiki benchmarks/data/simplewiki_10000.jsonl \
  --simplewiki-queries 100 \
  --label compare_before_after
```

Result: `benchmarks/results/compare_before_after.json`

Checks performed:

- Simple Wiki: 10,000 documents, 100 title-derived queries. The before/after
  ranked result IDs and scores matched exactly within `1e-12` absolute score
  tolerance. Mismatches: 0.
- Simple Wiki 100k: 10,000 documents, 100,000 sampled query executions. The
  comparison runner deduplicates repeated query strings before executing
  searches, so it checked 9,845 unique queries and still covers the full sampled
  workload. Mismatches: 0.
- FAQ assistant checkpoint: production corpus with 3,319 records and all saved
  zerosearch eval variants from `../faq-assistant/evals/results`. This covers 8
  variants x 130 queries. Ranked IDs and scores matched for every query.
  Mismatches: 0.
- FAQ hit/MRR metrics recomputed from the saved rewritten queries were identical
  before vs after for every variant. Production variant stayed at `hit@5=0.5692`
  and `mrr@5=0.4017`.

## What Changed

The speedup is from query-path changes in `zerosearch/index.py`:

- document frequencies for unfiltered searches are precomputed at fit time;
- vocabulary lookup uses a direct term-to-id map;
- search selects top-k document IDs before copying result dictionaries.

The public API is unchanged. The scoring expression intentionally remains the
same as before (`tf / sqrt(field_length)` is still computed during scoring),
because precomputing normalized posting weights caused tiny floating-point
association differences that could reorder tied or near-tied results. The
serialized index format version was still bumped because the packed layout now
stores document frequencies. Existing `.zsx` files should be rebuilt; no
backward-compatibility shim is kept for v1 artifacts.

The benchmark runner was also changed to support long query-throughput runs:

- if `--num-queries <= len(titles)`, it samples distinct title queries;
- if `--num-queries > len(titles)`, it samples title queries with replacement.

The comparison runner also deduplicates repeated query strings before executing
searches. This makes the 100k correctness check fast without changing what it
proves: a repeated query can only produce the same mismatch each time.
