# zerosearch

A tiny, **zero-dependency** BM25-lite in-memory text search index — standard
library only, a single small module, and good enough to power retrieval for a
RAG pipeline. Designed to run anywhere Python runs, including constrained
environments like Cloudflare Python Workers (Pyodide) where pulling in
`scikit-learn`/`numpy` is not an option.

It is a spiritual cousin of [`minsearch`](https://github.com/alexeygrigorev/minsearch):
same `Index` / `.fit()` / `.search()` API, but reimplemented from scratch with no
third-party dependencies.

## Drop-in replacement

`zerosearch` mirrors the [`minsearch`](https://github.com/alexeygrigorev/minsearch)
API — `Index(text_fields, keyword_fields)`, `index.fit(docs)`, and
`index.search(query, filter_dict, boost_dict, num_results)` — so you can swap it in
without changing your call sites. It is used exactly this way in
[DataTalksClub/faq-assistant](https://github.com/DataTalksClub/faq-assistant), where it
is vendored as the retrieval engine.

Note on ranking vs `minsearch`: `zerosearch` uses BM25-lite scoring, not `minsearch`'s
TF-IDF + cosine similarity — different algorithms, so the rankings are **not**
bit-for-bit identical. **Retrieval quality is on par, though:** on the faq-assistant
benchmark `zerosearch` matches `minsearch`'s **recall** (it surfaces the same relevant
documents in the top results), it just orders them differently. It *is* 100% identical
to the in-repo BM25-lite engine it replaced.

## Install

```bash
pip install zerosearch
```

## Usage

```python
from zerosearch import Index

docs = [
    {"id": "1", "title": "Docker compose basics", "text": "how to start services", "course": "de"},
    {"id": "2", "title": "Kafka consumers", "text": "consumer groups explained", "course": "de"},
]

index = Index(
    text_fields=["title", "text"],
    keyword_fields=["id", "course"],
)
index.fit(docs)

results = index.search(
    "how do I start docker compose",
    filter_dict={"course": "de"},             # exact-match keyword filter
    boost_dict={"title": 3.0, "text": 1.0},   # per-field boosts
    num_results=5,
)
for result in results:
    print(result["score"], result["title"])
```

Each result is a shallow copy of the original document dict with an added
`"score"` key.

## How it works

* **Tokenizer** — lowercased word/number tokens; keeps `+ . # _ -` *inside* a
  token so `c++`, `node.js`, `f-string` survive (a token must start with a
  letter/digit). Drops 1-character tokens and a small English stop-word list
  (both overridable).
* **Inverted index** — built once in `fit()`. A query only scores documents that
  actually contain a query term, so search is fast even on large corpora.
* **Ranking** — BM25-lite: each query term contributes
  `boost * idf * (term_frequency / sqrt(field_length))` per field. IDF and
  document frequencies are computed over the filtered candidate set.

## Customizing

```python
index = Index(
    text_fields=["title", "text"],
    stop_words={"the", "a", "an"},          # replace the default stop words
    tokenizer=lambda s: s.lower().split(),  # or plug in your own tokenizer
)
index.fit(docs)
```

## License

WTFPL.
