"""Tests for the packed runtime layout and save/load serialization.

The ranking behavior is pinned in ``test_index.py``. Here we additionally check
the packed scorer against an *independent* brute-force oracle (so it is not just
self-consistent), and that an index survives a save/load round-trip unchanged.
"""

import math
import random
from collections import Counter

import pytest

from zerosearch import DEFAULT_STOP_WORDS, Index, tokenize


# --- independent reference -------------------------------------------------

def brute_force(docs, text_fields, keyword_fields, query, *, filter_dict=None,
                boost_dict=None, num_results=10, stop_words=DEFAULT_STOP_WORDS):
    """A slow, obvious reimplementation of the documented BM25-lite formula."""
    filter_dict = filter_dict or {}
    boost_dict = boost_dict or {}
    query_terms = tokenize(query, stop_words)
    if not query_terms:
        return []
    query_tf = Counter(query_terms)

    if filter_dict:
        candidates = None
        for field, value in filter_dict.items():
            matched = {i for i, d in enumerate(docs) if str(d.get(field, "")) == str(value)}
            candidates = matched if candidates is None else (candidates & matched)
        candidates = candidates or set()
        if not candidates:
            return []
    else:
        candidates = set(range(len(docs)))

    document_count = len(candidates)
    tokens_by_doc = {
        i: {f: tokenize(str(docs[i].get(f, "")), stop_words) for f in text_fields}
        for i in candidates
    }

    document_frequencies = {}
    for term in query_tf:
        df = sum(any(term in toks[f] for f in text_fields) for toks in tokens_by_doc.values())
        if df:
            document_frequencies[term] = df
    idf = {
        t: math.log(1 + (document_count - df + 0.5) / (df + 0.5))
        for t, df in document_frequencies.items()
    }

    scores = {}
    for i, toks in tokens_by_doc.items():
        score = 0.0
        for field in text_fields:
            field_tokens = toks[field]
            if not field_tokens:
                continue
            counts = Counter(field_tokens)
            norm = math.sqrt(len(field_tokens))
            boost = float(boost_dict.get(field, 1.0))
            for term in query_terms:  # raw list -> query-term-frequency weighting
                if term in idf and counts.get(term, 0):
                    score += boost * idf[term] * (counts[term] / norm)
        if score > 0:
            scores[i] = score

    ranked = sorted(scores, key=lambda i: (-scores[i], i))
    return [(docs[i].get("id"), round(scores[i], 9)) for i in ranked[:num_results]]


def keyed(results):
    return [(r.get("id"), round(r["score"], 9)) for r in results]


# --- a deterministic, larger-than-toy corpus -------------------------------

VOCAB = ("docker compose kafka consumer python pandas merge join spark airflow "
         "mlflow conda pip env error deadline homework capstone node.js c++ "
         "f-string postgres sql index query group network container").split()
COURSES = ["de", "mlops", "ml", ""]
TEXT_FIELDS = ["title", "text"]
KEYWORD_FIELDS = ["id", "course", "kind"]


def make_corpus(n=300, seed=7):
    rng = random.Random(seed)
    docs = []
    for i in range(n):
        title = " ".join(rng.choice(VOCAB) for _ in range(rng.randint(1, 4)))
        text = " ".join(rng.choice(VOCAB) for _ in range(rng.randint(3, 30)))
        docs.append({
            "id": f"d{i}",
            "title": title,
            "text": text,
            "course": rng.choice(COURSES),
            "kind": rng.choice(["faq", "lesson"]),
            "meta": {"n": i},  # nested, marshal-able; checks docs survive round-trip
        })
    return docs


QUERIES = [
    "docker compose", "kafka kafka consumer", "pandas merge join", "mlflow",
    "python pip conda env error", "node.js c++ f-string", "deadline homework homework",
    "spark airflow", "zzz totally unknown term", "",
]
FILTERS = [None, {"course": "de"}, {"course": "mlops", "kind": "faq"}, {"course": "nope"}]
BOOSTS = [None, {"title": 3.0}, {"title": 0.5, "text": 2.0}]


@pytest.fixture(scope="module")
def corpus():
    return make_corpus()


@pytest.fixture(scope="module")
def index(corpus):
    return Index(text_fields=TEXT_FIELDS, keyword_fields=KEYWORD_FIELDS).fit(corpus)


# --- parity against the brute-force oracle ---------------------------------

@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize("filter_dict", FILTERS)
def test_packed_search_matches_brute_force(index, corpus, query, filter_dict):
    for boost in BOOSTS:
        got = keyed(index.search(query, filter_dict=filter_dict, boost_dict=boost, num_results=10))
        want = brute_force(corpus, TEXT_FIELDS, KEYWORD_FIELDS, query,
                           filter_dict=filter_dict, boost_dict=boost, num_results=10)
        assert got == want, (query, filter_dict, boost)


# --- save / load round-trips -----------------------------------------------

def test_loads_round_trip_is_identical(index, corpus):
    restored = Index.loads(index.dumps())
    for query in QUERIES:
        for filter_dict in FILTERS:
            assert keyed(restored.search(query, filter_dict=filter_dict)) == \
                   keyed(index.search(query, filter_dict=filter_dict))


def test_save_load_file_round_trip(index, tmp_path):
    path = tmp_path / "index.zsx"
    index.save(path)
    assert path.stat().st_size > 0
    restored = Index.load(path)
    assert keyed(restored.search("docker compose", boost_dict={"title": 3.0})) == \
           keyed(index.search("docker compose", boost_dict={"title": 3.0}))


def test_round_trip_preserves_docs_and_fields(index):
    restored = Index.loads(index.dumps())
    assert restored.docs == index.docs
    assert restored.text_fields == index.text_fields
    assert restored.keyword_fields == index.keyword_fields
    # Nested values survive (docs are returned, so they must be intact).
    assert restored.docs[0]["meta"] == {"n": 0}


def test_round_trip_keyword_filters_still_work(index, corpus):
    restored = Index.loads(index.dumps())
    got = keyed(restored.search("kafka consumer", filter_dict={"course": "de", "kind": "faq"}))
    want = brute_force(corpus, TEXT_FIELDS, KEYWORD_FIELDS, "kafka consumer",
                       filter_dict={"course": "de", "kind": "faq"})
    assert got == want


def test_empty_index_round_trips():
    empty = Index(text_fields=["title"], keyword_fields=["course"])
    restored = Index.loads(empty.dumps())
    assert restored.search("docker") == []
    # ...and still works after fitting the restored object is irrelevant; fit a fresh one
    restored2 = Index.loads(Index(text_fields=["title"]).fit([{"id": "1", "title": "docker"}]).dumps())
    assert [r["id"] for r in restored2.search("docker")] == ["1"]


def test_custom_stopwords_survive_round_trip():
    index = Index(text_fields=["text"], stop_words={"docker"}).fit(
        [{"id": "1", "text": "docker kafka"}]
    )
    restored = Index.loads(index.dumps())
    assert restored.search("docker") == []          # still a stop word after reload
    assert [r["id"] for r in restored.search("kafka")] == ["1"]


def test_custom_tokenizer_must_be_resupplied_on_load():
    # A bare whitespace splitter keeps stop words; the default tokenizer drops
    # them. "the" therefore only matches under the custom tokenizer.
    splitter = lambda s: s.lower().split()  # noqa: E731
    index = Index(text_fields=["title"], tokenizer=splitter).fit(
        [{"id": "1", "title": "the answer"}]
    )
    assert [r["id"] for r in index.search("the")] == ["1"]
    # Reload without the tokenizer falls back to the default: "the" is a stop
    # word, so the query is empty and nothing matches.
    default_reload = Index.loads(index.dumps())
    assert default_reload.search("the") == []
    # Reload with the same tokenizer matches again.
    same_reload = Index.loads(index.dumps(), tokenizer=splitter)
    assert [r["id"] for r in same_reload.search("the")] == ["1"]


# --- guards ----------------------------------------------------------------

def test_loads_rejects_non_index_bytes():
    import marshal
    with pytest.raises(ValueError, match="not a zerosearch index"):
        Index.loads(marshal.dumps({"hello": "world"}))


def test_loads_rejects_unknown_format_version(index):
    import marshal
    state = marshal.loads(index.dumps())
    state["format"] = 999
    with pytest.raises(ValueError, match="unsupported zerosearch index format"):
        Index.loads(marshal.dumps(state))


def test_loads_rejects_incompatible_platform_itemsizes(index):
    import marshal
    state = marshal.loads(index.dumps())
    state["itemsizes"] = [99, 99, 99, 99, 99]
    with pytest.raises(ValueError, match="incompatible platform"):
        Index.loads(marshal.dumps(state))


def test_loads_rejects_mismatched_python_version(index):
    import marshal
    state = marshal.loads(index.dumps())
    state["python"] = "2.7"
    with pytest.raises(ValueError, match="built on Python 2.7"):
        Index.loads(marshal.dumps(state))


def test_too_many_text_fields_is_rejected():
    fields = [f"f{i}" for i in range(257)]
    with pytest.raises(ValueError, match="at most 256 text fields"):
        Index(text_fields=fields).fit([{"f0": "x"}])


def test_dumps_is_marshal_bytes(index):
    assert isinstance(index.dumps(), (bytes, bytearray))


def test_query_term_present_but_filtered_out_of_candidates():
    docs = [
        {"id": "1", "title": "docker", "course": "de"},
        {"id": "2", "title": "kafka", "course": "mlops"},
    ]
    index = Index(text_fields=["title"], keyword_fields=["course"]).fit(docs)
    # "docker" is in the vocab but only in course=de; filtering to mlops leaves
    # it with no candidate docs, while "kafka" still matches.
    results = index.search("docker kafka", filter_dict={"course": "mlops"})
    assert [r["id"] for r in results] == ["2"]
