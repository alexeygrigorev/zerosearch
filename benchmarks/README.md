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

> **Build timing note.** `fit` is timed **without** `tracemalloc` running.
> `tracemalloc` traces every allocation and slows `fit` ~4x (e.g. 8.5 s → 34 s
> on the 10,000-doc sample), so build memory is measured in a separate, untimed
> `fit`. Earlier versions of this table timed `fit` under `tracemalloc` and so
> reported build times ~4x too high; the numbers below are the corrected,
> tracemalloc-free build times.

Results on this machine:

| sample | version | build | peak memory | avg search | median | p95 | qps |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10,000 docs | before | 8.277 s | 338.3 MB | 0.875 ms | 0.144 ms | 2.693 ms | 1,142.5 |
| 10,000 docs | after | 8.531 s | 371.2 MB | 0.345 ms | 0.116 ms | 1.467 ms | 2,896.9 |
| 10,000 docs / 100k queries | before | 8.340 s | 338.3 MB | 0.628 ms | 0.134 ms | 3.066 ms | 1,592.9 |
| 10,000 docs / 100k queries | after | 8.793 s | 371.2 MB | 0.337 ms | 0.090 ms | 1.578 ms | 2,963.9 |
| 1,000 docs | before | 1.095 s | 51.1 MB | 0.087 ms | 0.040 ms | 0.337 ms | 11,478.0 |
| 1,000 docs | after | 1.106 s | 59.3 MB | 0.063 ms | 0.037 ms | 0.224 ms | 15,791.4 |

The optimization is a **search-path** change, so build time is essentially
unchanged before vs after (the small differences are run-to-run noise). On the
10,000-document sample, average search latency improved by 2.5x and p95 latency
by 1.8x. On the longer 100,000-query run, average latency improved by 1.9x and
p95 by 1.9x. The implementation preserves the pre-optimization ranking and score
behavior exactly.

The 100,000-query run was added after the first report to reduce timing noise
from very small query samples. It uses the same 10,000 cached documents and the
same title-derived query distribution, but samples with replacement. The longer
run confirms the 100-query result: average search latency stays around `0.33 ms`
and throughput is about `3.1k QPS` on this machine.

## Memory Footprint

The result JSON files include four memory/footprint measures:

- `build_peak_bytes_tracemalloc`: peak Python allocations during `fit`.
- `build_current_bytes_tracemalloc`: Python allocations still live immediately
  after `fit`. This excludes the already-loaded input corpus, because
  `tracemalloc` starts right before the (separate, untimed) memory-measurement
  `fit`.
- `index_serialized_bytes`: size of `index.dumps()`, a practical shipped artifact
  size including documents, vocabulary, arrays, and keyword indexes.
- `index_packed_array_bytes`: bytes used by the packed posting/length arrays only.

| sample | version | live after build | build peak | serialized | packed arrays |
|---:|---|---:|---:|---:|---:|
| 10,000 docs | before | 55.6 MB | 338.3 MB | 99.9 MB | 31.3 MB |
| 10,000 docs | after | 82.3 MB | 371.2 MB | 101.4 MB | 32.8 MB |
| 1,000 docs | before | 9.8 MB | 51.1 MB | 14.4 MB | 4.2 MB |
| 1,000 docs | after | 16.4 MB | 59.3 MB | 14.7 MB | 4.6 MB |

The persistent packed artifact cost is modest: about `+1.5 MB` serialized and
`+1.5 MB` packed-array bytes on the 10,000-document sample, mostly from the new
per-term document-frequency array and term-to-id map. Build/live Python memory
is higher because the optimized index keeps extra lookup structures for faster
query execution.

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

The runner now times `fit` outside `tracemalloc` and measures build memory in a
separate `fit`, because tracing every allocation slowed the timed build ~4x and
made build times look far worse than they are. Search timing was always measured
outside `tracemalloc`, so only the build column changed.
