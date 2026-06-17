import math

import pytest

from zerosearch import DEFAULT_STOP_WORDS, Index, tokenize


DOCS = [
    {"id": "1", "title": "Docker compose basics", "text": "how to start services with docker", "course": "de"},
    {"id": "2", "title": "Kafka consumers", "text": "consumer groups explained in kafka", "course": "de"},
    {"id": "3", "title": "Docker networking", "text": "containers talk over a docker network", "course": "mlops"},
    {"id": "4", "title": "Pandas joins", "text": "merge and join dataframes", "course": "ml"},
]


def make_index():
    return Index(text_fields=["title", "text"], keyword_fields=["id", "course"]).fit(DOCS)


def test_tokenize_keeps_technical_tokens():
    # Punctuation is kept *inside* a token (a token must start with [a-z0-9]),
    # so a leading dot in ".env" is dropped.
    assert tokenize("Node.js and C++ with f-strings") == ["node.js", "c++", "f-strings"]


def test_tokenize_drops_stopwords_and_single_chars():
    assert "the" not in tokenize("the a docker")
    assert tokenize("a I") == []  # all stop words / single char


def test_empty_query_returns_nothing():
    assert make_index().search("") == []
    assert make_index().search("   ") == []


def test_basic_ranking_finds_relevant_doc():
    results = make_index().search("docker compose", num_results=5)
    assert results
    assert results[0]["id"] == "1"
    assert all("score" in r for r in results)


def test_results_are_sorted_by_score_desc():
    results = make_index().search("docker", num_results=5)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_keyword_filter_restricts_candidates():
    results = make_index().search("docker", filter_dict={"course": "mlops"}, num_results=5)
    assert [r["id"] for r in results] == ["3"]


def test_filter_with_no_matches_returns_empty():
    assert make_index().search("docker", filter_dict={"course": "nonexistent"}) == []


def test_boost_changes_ranking():
    index = make_index()
    # "kafka" appears in both title and text of doc 2; boosting title should not
    # crash and should keep doc 2 on top.
    results = index.search("kafka", boost_dict={"title": 5.0, "text": 1.0})
    assert results[0]["id"] == "2"


def test_num_results_caps_output():
    assert len(make_index().search("docker", num_results=1)) == 1


def test_search_does_not_mutate_source_docs():
    index = make_index()
    index.search("docker")
    assert all("score" not in doc for doc in DOCS)


def test_unknown_term_returns_empty():
    assert make_index().search("zzzznonexistentterm") == []


def test_custom_stopwords():
    index = Index(text_fields=["text"], stop_words={"docker"}).fit(DOCS)
    # "docker" is now a stop word, so a docker-only query finds nothing.
    assert index.search("docker") == []


def test_custom_tokenizer():
    index = Index(text_fields=["title"], tokenizer=lambda s: s.lower().split()).fit(DOCS)
    assert index.search("kafka")[0]["id"] == "2"


def test_idf_is_positive_and_finite():
    index = make_index()
    results = index.search("docker")
    for r in results:
        assert math.isfinite(r["score"]) and r["score"] > 0


def test_default_stopwords_frozen():
    assert "the" in DEFAULT_STOP_WORDS
    with pytest.raises(AttributeError):
        DEFAULT_STOP_WORDS.add("x")  # frozenset has no add
