"""A tiny, zero-dependency BM25-lite search index.

The whole engine is standard-library only. Documents are plain dicts. Text
fields are tokenized once when the index is built and kept as an inverted index,
so a query only scores the documents that actually contain a query term.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Callable, Iterable

__all__ = ["Index", "tokenize", "DEFAULT_STOP_WORDS", "TOKEN_RE"]

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+.#-]*", re.IGNORECASE)

DEFAULT_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
        "how", "i", "in", "is", "it", "of", "on", "or", "the", "to", "with",
    }
)

Tokenizer = Callable[[str], list]


def tokenize(text: str, stop_words: Iterable[str] = DEFAULT_STOP_WORDS) -> list[str]:
    """Lowercase word/number tokens, dropping 1-char tokens and stop words.

    The token pattern keeps ``+ . # _ -`` inside a token so technical terms such
    as ``c++``, ``node.js`` and ``f-string`` survive intact (a token must start
    with a letter or digit, so a leading ``.`` in ``.env`` is dropped).
    """
    stops = stop_words if isinstance(stop_words, (set, frozenset)) else set(stop_words)
    tokens = (match.group(0).lower() for match in TOKEN_RE.finditer(text))
    return [token for token in tokens if len(token) > 1 and token not in stops]


class Index:
    """In-memory search over a fixed list of documents.

    Parameters
    ----------
    text_fields:
        Document fields that are tokenized and ranked.
    keyword_fields:
        Document fields used for exact-match filtering (not ranked).
    stop_words:
        Tokens to ignore. Defaults to :data:`DEFAULT_STOP_WORDS`.
    tokenizer:
        Optional ``str -> list[str]`` override. Defaults to :func:`tokenize`.

    Ranking is BM25-lite: each query term contributes
    ``boost * idf * (term_frequency / sqrt(field_length))`` per field, where the
    IDF and document frequencies are computed over the filtered candidate set.
    """

    def __init__(
        self,
        text_fields: list[str],
        keyword_fields: list[str] | None = None,
        *,
        stop_words: Iterable[str] = DEFAULT_STOP_WORDS,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.text_fields = list(text_fields)
        self.keyword_fields = list(keyword_fields or [])
        self._stop_words = frozenset(stop_words)
        self._tokenize: Tokenizer = tokenizer or (lambda text: tokenize(text, self._stop_words))
        self.docs: list[dict[str, Any]] = []
        self._field_counts: list[dict[str, Counter]] = []
        self._field_lengths: list[dict[str, int]] = []
        self._postings: dict[str, set[int]] = {}
        self._keyword_index: dict[str, dict[str, set[int]]] = {}

    def fit(self, docs: list[dict[str, Any]]) -> "Index":
        """Build the inverted index from ``docs``. Returns ``self``."""
        self.docs = list(docs)
        self._field_counts = []
        self._field_lengths = []
        self._postings = {}
        self._keyword_index = {field: {} for field in self.keyword_fields}

        for doc_id, doc in enumerate(self.docs):
            counts: dict[str, Counter] = {}
            lengths: dict[str, int] = {}
            doc_terms: set[str] = set()
            for field in self.text_fields:
                field_counts = Counter(self._tokenize(str(doc.get(field, ""))))
                counts[field] = field_counts
                lengths[field] = sum(field_counts.values())
                doc_terms.update(field_counts)
            self._field_counts.append(counts)
            self._field_lengths.append(lengths)
            for term in doc_terms:
                self._postings.setdefault(term, set()).add(doc_id)

            for field in self.keyword_fields:
                value = str(doc.get(field, ""))
                self._keyword_index[field].setdefault(value, set()).add(doc_id)

        return self

    def search(
        self,
        query: str,
        filter_dict: dict[str, str] | None = None,
        boost_dict: dict[str, float] | None = None,
        num_results: int = 10,
    ) -> list[dict[str, Any]]:
        """Return up to ``num_results`` docs (copies, with a ``"score"`` key)."""
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        filter_dict = filter_dict or {}
        boost_dict = boost_dict or {}

        candidates = self._candidate_ids(filter_dict)
        if candidates is not None and not candidates:
            return []

        document_count = len(self.docs) if candidates is None else len(candidates)
        term_postings: dict[str, set[int]] = {}
        document_frequencies: dict[str, int] = {}
        docs_to_score: set[int] = set()
        for term in set(query_terms):
            postings = self._postings.get(term)
            if not postings:
                continue
            matched = postings if candidates is None else (postings & candidates)
            if not matched:
                continue
            term_postings[term] = matched
            document_frequencies[term] = len(matched)
            docs_to_score |= matched

        if not docs_to_score:
            return []

        idf = {
            term: math.log(1 + (document_count - df + 0.5) / (df + 0.5))
            for term, df in document_frequencies.items()
        }

        scored = []
        for doc_id in sorted(docs_to_score):
            score = self._score(doc_id, query_terms, term_postings, idf, boost_dict)
            if score > 0:
                record = dict(self.docs[doc_id])
                record["score"] = score
                scored.append(record)

        scored.sort(key=lambda record: float(record["score"]), reverse=True)
        return scored[:num_results]

    def _candidate_ids(self, filter_dict: dict[str, str]) -> set[int] | None:
        """Intersect keyword indexes for each filter. ``None`` means "all docs"."""
        if not filter_dict:
            return None
        candidates: set[int] | None = None
        for field, value in filter_dict.items():
            matched = self._keyword_index.get(field, {}).get(str(value), set())
            candidates = set(matched) if candidates is None else (candidates & matched)
            if not candidates:
                return set()
        return candidates

    def _score(
        self,
        doc_id: int,
        query_terms: list[str],
        term_postings: dict[str, set[int]],
        idf: dict[str, float],
        boost_dict: dict[str, float],
    ) -> float:
        counts = self._field_counts[doc_id]
        lengths = self._field_lengths[doc_id]
        score = 0.0
        for field in self.text_fields:
            field_length = lengths.get(field, 0)
            if not field_length:
                continue
            field_counts = counts[field]
            boost = float(boost_dict.get(field, 1.0))
            norm = math.sqrt(field_length)
            for term in query_terms:
                if doc_id not in term_postings.get(term, ()):
                    continue
                term_frequency = field_counts.get(term, 0)
                if term_frequency == 0:
                    continue
                score += boost * idf[term] * (term_frequency / norm)
        return score
