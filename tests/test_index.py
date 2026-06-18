import math

import pytest

import zerosearch
from zerosearch import DEFAULT_STOP_WORDS, TOKEN_RE, Index, tokenize


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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Python 3.11, C#, foo_bar, e-mail", ["python", "3.11", "c#", "foo_bar", "e-mail"]),
        (".env +leading #tag", ["env", "leading", "tag"]),
        ("HELLO hello HeLLo", ["hello", "hello", "hello"]),
    ],
)
def test_tokenize_normalizes_supported_token_shapes(text, expected):
    assert tokenize(text) == expected


def test_tokenize_drops_stopwords_and_single_chars():
    assert "the" not in tokenize("the a docker")
    assert tokenize("a I") == []  # all stop words / single char


def test_tokenize_accepts_any_stopword_iterable():
    assert tokenize("alpha beta gamma", stop_words=("alpha", "gamma")) == ["beta"]


def test_search_before_fit_returns_nothing():
    assert Index(text_fields=["title"]).search("docker") == []


def test_fit_returns_self():
    index = Index(text_fields=["title"])
    assert index.fit(DOCS) is index


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


def test_keyword_filters_are_intersected():
    docs = [
        {"id": "1", "title": "docker", "course": "de", "kind": "lesson"},
        {"id": "2", "title": "docker", "course": "de", "kind": "faq"},
        {"id": "3", "title": "docker", "course": "mlops", "kind": "faq"},
    ]
    index = Index(text_fields=["title"], keyword_fields=["course", "kind"]).fit(docs)

    results = index.search("docker", filter_dict={"course": "de", "kind": "faq"})

    assert [result["id"] for result in results] == ["2"]


def test_unknown_keyword_filter_field_returns_empty():
    assert make_index().search("docker", filter_dict={"missing": "value"}) == []


def test_keyword_filters_coerce_values_to_strings():
    docs = [{"id": 1, "title": "docker"}, {"id": 2, "title": "kafka"}]
    index = Index(text_fields=["title"], keyword_fields=["id"]).fit(docs)

    assert [result["id"] for result in index.search("docker", filter_dict={"id": 1})] == [1]


def test_filter_with_no_matches_returns_empty():
    assert make_index().search("docker", filter_dict={"course": "nonexistent"}) == []


def test_filter_candidates_without_query_terms_return_empty():
    assert make_index().search("docker", filter_dict={"course": "ml"}) == []


def test_boost_changes_ranking():
    docs = [
        {"id": "title", "title": "spark", "text": ""},
        {"id": "body", "title": "", "text": "spark spark spark spark"},
    ]
    index = Index(text_fields=["title", "text"]).fit(docs)

    assert index.search("spark")[0]["id"] == "body"
    assert index.search("spark", boost_dict={"title": 3.0})[0]["id"] == "title"


def test_num_results_caps_output():
    assert len(make_index().search("docker", num_results=1)) == 1


def test_num_results_zero_returns_empty():
    assert make_index().search("docker", num_results=0) == []


def test_search_does_not_mutate_source_docs():
    index = make_index()
    index.search("docker")
    assert all("score" not in doc for doc in DOCS)


def test_results_are_independent_shallow_copies():
    docs = [{"id": "1", "title": "docker", "metadata": {"course": "de"}}]
    result = Index(text_fields=["title"]).fit(docs).search("docker")[0]

    result["title"] = "changed"
    result["metadata"]["course"] = "mlops"

    assert docs[0]["title"] == "docker"
    assert docs[0]["metadata"]["course"] == "mlops"


def test_existing_score_field_is_replaced_in_result_only():
    docs = [{"id": "1", "title": "docker", "score": "original"}]

    result = Index(text_fields=["title"]).fit(docs).search("docker")[0]

    assert result["score"] != "original"
    assert isinstance(result["score"], float)
    assert docs[0]["score"] == "original"


def test_unknown_term_returns_empty():
    assert make_index().search("zzzznonexistentterm") == []


def test_missing_text_fields_are_treated_as_empty_text():
    docs = [{"id": "missing"}, {"id": "match", "title": "docker"}]
    index = Index(text_fields=["title"]).fit(docs)

    assert [result["id"] for result in index.search("docker")] == ["match"]


def test_non_string_text_fields_are_tokenized_as_strings():
    docs = [{"id": "year", "title": 2026}, {"id": "word", "title": "docker"}]
    index = Index(text_fields=["title"]).fit(docs)

    assert [result["id"] for result in index.search("2026")] == ["year"]


def test_refit_replaces_previous_index_state():
    index = make_index()

    index.fit([{"id": "new", "title": "postgres", "course": "db"}])

    assert index.search("docker") == []
    assert [result["id"] for result in index.search("postgres")] == ["new"]
    assert index.search("postgres", filter_dict={"course": "de"}) == []


def test_fit_copies_document_list_not_document_dicts():
    docs = [{"id": "1", "title": "docker"}]
    index = Index(text_fields=["title"]).fit(docs)

    docs.append({"id": "2", "title": "kafka"})

    assert index.search("kafka") == []
    assert index.search("docker")[0]["id"] == "1"


def test_tied_scores_keep_document_order():
    docs = [
        {"id": "1", "title": "docker"},
        {"id": "2", "title": "docker"},
        {"id": "3", "title": "docker"},
    ]
    index = Index(text_fields=["title"]).fit(docs)

    assert [result["id"] for result in index.search("docker", num_results=3)] == ["1", "2", "3"]


def test_repeated_query_terms_increase_score():
    index = Index(text_fields=["title"]).fit([{"id": "1", "title": "docker"}])

    single = index.search("docker")[0]["score"]
    repeated = index.search("docker docker")[0]["score"]

    assert math.isclose(repeated, single * 2)


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


def test_public_exports_are_available():
    assert set(zerosearch.__all__) == {"Index", "tokenize", "DEFAULT_STOP_WORDS", "TOKEN_RE", "__version__"}
    assert TOKEN_RE is zerosearch.TOKEN_RE
    assert isinstance(zerosearch.__version__, str)
