"""A tiny, zero-dependency BM25-lite search index.

The whole engine is standard-library only. Documents are plain dicts. Text
fields are tokenized once when the index is built and kept as an inverted index,
so a query only scores the documents that actually contain a query term.

Building (:meth:`Index.fit`) uses ``Counter`` scaffolding, but the runtime state
is then compacted into flat ``array`` buffers (a CSR-style postings list). That
packed form is what :meth:`Index.search` reads and what :meth:`Index.save` /
:meth:`Index.load` round-trip, so a prebuilt index loads in milliseconds without
re-tokenizing the corpus.
"""

from __future__ import annotations

import marshal
import math
import re
import sys
from array import array
from bisect import bisect_left
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

# On-disk format. ``_FORMAT_VERSION`` is bumped whenever the packed layout
# changes so an incompatible artifact fails loudly instead of scoring wrong.
_MAGIC = "zerosearch"
_FORMAT_VERSION = 1

# Array typecodes for the packed postings. Doc ids and term frequencies use an
# unsigned 32-bit int; the field index uses a single byte (so at most 256 text
# fields, which ``fit`` enforces).
_DOC_TC = "I"
_TF_TC = "I"
_FIELD_TC = "B"
_OFFSET_TC = "I"
_LENGTH_TC = "I"
_MAX_TEXT_FIELDS = 256


def tokenize(text: str, stop_words: Iterable[str] = DEFAULT_STOP_WORDS) -> list[str]:
    """Lowercase word/number tokens, dropping 1-char tokens and stop words.

    The token pattern keeps ``+ . # _ -`` inside a token so technical terms such
    as ``c++``, ``node.js`` and ``f-string`` survive intact (a token must start
    with a letter or digit, so a leading ``.`` in ``.env`` is dropped).
    """
    stops = stop_words if isinstance(stop_words, (set, frozenset)) else set(stop_words)
    tokens = (match.group(0).lower() for match in TOKEN_RE.finditer(text))
    return [token for token in tokens if len(token) > 1 and token not in stops]


def _zeroed_array(typecode: str, length: int) -> array:
    """A zero-filled ``array`` of ``length`` items, portable across itemsizes."""
    buffer = array(typecode)
    buffer.frombytes(bytes(buffer.itemsize * length))
    return buffer


def _array_from_bytes(typecode: str, data: bytes) -> array:
    buffer = array(typecode)
    buffer.frombytes(data)
    return buffer


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
    IDF and document frequencies are computed over the filtered candidate set. A
    term that appears more than once in the query is weighted by its query-term
    frequency.
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

        # Packed runtime state (populated by ``fit`` or ``load``).
        self._n_fields = len(self.text_fields)
        self._vocab: list[str] = []  # sorted; a term's id is its position here
        self._post_off = array(_OFFSET_TC, [0])  # term id -> [start, end) into the postings
        self._post_doc = array(_DOC_TC)
        self._post_field = array(_FIELD_TC)
        self._post_tf = array(_TF_TC)
        self._lengths = array(_LENGTH_TC)  # flat doc_id * n_fields + field_index -> field length
        self._keyword_index: dict[str, dict[str, set[int]]] = {}

    # -- building -----------------------------------------------------------

    def fit(self, docs: list[dict[str, Any]]) -> "Index":
        """Build the inverted index from ``docs``. Returns ``self``."""
        if len(self.text_fields) > _MAX_TEXT_FIELDS:
            raise ValueError(f"at most {_MAX_TEXT_FIELDS} text fields are supported")

        self.docs = list(docs)
        n_docs = len(self.docs)
        n_fields = len(self.text_fields)
        self._n_fields = n_fields

        # Scaffolding: term -> list of (doc_id, field_index, term_frequency).
        postings: dict[str, list[tuple[int, int, int]]] = {}
        lengths = _zeroed_array(_LENGTH_TC, n_docs * n_fields)
        keyword_index: dict[str, dict[str, set[int]]] = {field: {} for field in self.keyword_fields}

        for doc_id, doc in enumerate(self.docs):
            base = doc_id * n_fields
            for field_index, field in enumerate(self.text_fields):
                counts = Counter(self._tokenize(str(doc.get(field, ""))))
                lengths[base + field_index] = sum(counts.values())
                for term, term_frequency in counts.items():
                    postings.setdefault(term, []).append((doc_id, field_index, term_frequency))
            for field in self.keyword_fields:
                keyword_index[field].setdefault(str(doc.get(field, "")), set()).add(doc_id)

        self._pack(postings, lengths, keyword_index)
        return self

    def _pack(
        self,
        postings: dict[str, list[tuple[int, int, int]]],
        lengths: array,
        keyword_index: dict[str, dict[str, set[int]]],
    ) -> None:
        """Compact the build scaffolding into the flat runtime arrays."""
        vocab = sorted(postings)
        post_off = array(_OFFSET_TC)
        post_doc = array(_DOC_TC)
        post_field = array(_FIELD_TC)
        post_tf = array(_TF_TC)

        offset = 0
        for term in vocab:
            post_off.append(offset)
            # Sorted by doc so tied scores fall back to document order downstream.
            for doc_id, field_index, term_frequency in sorted(postings[term]):
                post_doc.append(doc_id)
                post_field.append(field_index)
                post_tf.append(term_frequency)
                offset += 1
        post_off.append(offset)

        self._vocab = vocab
        self._post_off = post_off
        self._post_doc = post_doc
        self._post_field = post_field
        self._post_tf = post_tf
        self._lengths = lengths
        self._keyword_index = keyword_index

    # -- querying -----------------------------------------------------------

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

        query_term_frequencies = Counter(query_terms)
        filter_dict = filter_dict or {}
        boost_dict = boost_dict or {}

        candidates = self._candidate_ids(filter_dict)
        if candidates is not None and not candidates:
            return []

        document_count = len(self.docs) if candidates is None else len(candidates)

        # Locate each distinct query term's posting slice in the sorted vocab.
        located: list[tuple[str, int, int]] = []
        for term in query_term_frequencies:
            term_id = self._term_id(term)
            if term_id < 0:
                continue
            start, end = self._post_off[term_id], self._post_off[term_id + 1]
            if end > start:
                located.append((term, start, end))
        if not located:
            return []

        # Document frequency = distinct candidate docs containing the term.
        document_frequencies: dict[str, int] = {}
        for term, start, end in located:
            docs_seen: set[int] = set()
            for j in range(start, end):
                doc_id = self._post_doc[j]
                if candidates is None or doc_id in candidates:
                    docs_seen.add(doc_id)
            if docs_seen:
                document_frequencies[term] = len(docs_seen)
        if not document_frequencies:
            return []

        idf = {
            term: math.log(1 + (document_count - df + 0.5) / (df + 0.5))
            for term, df in document_frequencies.items()
        }

        scores = self._accumulate_scores(located, idf, query_term_frequencies, candidates, boost_dict)

        scored: list[dict[str, Any]] = []
        for doc_id in sorted(scores):
            if scores[doc_id] > 0:
                record = dict(self.docs[doc_id])
                record["score"] = scores[doc_id]
                scored.append(record)

        scored.sort(key=lambda record: float(record["score"]), reverse=True)
        return scored[:num_results]

    def _accumulate_scores(
        self,
        located: list[tuple[str, int, int]],
        idf: dict[str, float],
        query_term_frequencies: dict[str, int],
        candidates: set[int] | None,
        boost_dict: dict[str, float],
    ) -> dict[int, float]:
        scores: dict[int, float] = {}
        n_fields = self._n_fields
        text_fields = self.text_fields
        post_doc = self._post_doc
        post_field = self._post_field
        post_tf = self._post_tf
        lengths = self._lengths

        for term, start, end in located:
            if term not in idf:
                continue
            weight = idf[term] * query_term_frequencies[term]
            for j in range(start, end):
                doc_id = post_doc[j]
                if candidates is not None and doc_id not in candidates:
                    continue
                field_index = post_field[j]
                field_length = lengths[doc_id * n_fields + field_index]
                if not field_length:  # pragma: no cover - postings imply length>=tf>0; guards corrupt loads
                    continue
                boost = float(boost_dict.get(text_fields[field_index], 1.0))
                contribution = boost * weight * (post_tf[j] / math.sqrt(field_length))
                scores[doc_id] = scores.get(doc_id, 0.0) + contribution
        return scores

    def _term_id(self, term: str) -> int:
        """Binary-search the sorted vocabulary; ``-1`` if the term is unknown."""
        vocab = self._vocab
        index = bisect_left(vocab, term)
        if index < len(vocab) and vocab[index] == term:
            return index
        return -1

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

    # -- serialization ------------------------------------------------------

    def dumps(self) -> bytes:
        """Serialize the packed index to bytes (uses :mod:`marshal`).

        Documents must contain only ``marshal``-able values (the JSON-like types
        a search corpus normally holds: str, int, float, bool, None, list, dict).
        """
        state = {
            "magic": _MAGIC,
            "format": _FORMAT_VERSION,
            "python": _python_tag(),
            "itemsizes": _itemsizes(),
            "text_fields": self.text_fields,
            "keyword_fields": self.keyword_fields,
            "stop_words": sorted(self._stop_words),
            "n_fields": self._n_fields,
            "docs": self.docs,
            "vocab": self._vocab,
            "post_off": self._post_off.tobytes(),
            "post_doc": self._post_doc.tobytes(),
            "post_field": self._post_field.tobytes(),
            "post_tf": self._post_tf.tobytes(),
            "lengths": self._lengths.tobytes(),
            "keyword_index": {
                field: {value: array(_DOC_TC, sorted(ids)).tobytes() for value, ids in values.items()}
                for field, values in self._keyword_index.items()
            },
        }
        return marshal.dumps(state)

    def save(self, path) -> None:
        """Write the packed index to ``path``."""
        with open(path, "wb") as handle:
            handle.write(self.dumps())

    @classmethod
    def loads(cls, data: bytes, *, tokenizer: Tokenizer | None = None) -> "Index":
        """Reconstruct an index from :meth:`dumps` bytes.

        Pass ``tokenizer`` if the index was built with a custom tokenizer; query
        text must be tokenized the same way it was at build time. Indexes built
        with the default tokenizer (plus their stop words) restore automatically.
        """
        state = marshal.loads(data)
        if not isinstance(state, dict) or state.get("magic") != _MAGIC:
            raise ValueError("not a zerosearch index")
        if state.get("format") != _FORMAT_VERSION:
            raise ValueError(
                f"unsupported zerosearch index format {state.get('format')!r} "
                f"(this build expects {_FORMAT_VERSION})"
            )
        # marshal is not guaranteed portable across Python versions; fail clearly
        # (rather than with a cryptic "bad marshal data") if the index was built
        # on a different one. Rebuild from source to fix.
        if state.get("python") != _python_tag():
            raise ValueError(
                f"zerosearch index was built on Python {state.get('python')}, "
                f"but this is Python {_python_tag()}; rebuild the index"
            )
        if state.get("itemsizes") != _itemsizes():
            raise ValueError("zerosearch index was built on an incompatible platform")

        index = cls(
            text_fields=state["text_fields"],
            keyword_fields=state["keyword_fields"],
            stop_words=frozenset(state["stop_words"]),
            tokenizer=tokenizer,
        )
        index.docs = state["docs"]
        index._n_fields = state["n_fields"]
        index._vocab = state["vocab"]
        index._post_off = _array_from_bytes(_OFFSET_TC, state["post_off"])
        index._post_doc = _array_from_bytes(_DOC_TC, state["post_doc"])
        index._post_field = _array_from_bytes(_FIELD_TC, state["post_field"])
        index._post_tf = _array_from_bytes(_TF_TC, state["post_tf"])
        index._lengths = _array_from_bytes(_LENGTH_TC, state["lengths"])
        index._keyword_index = {
            field: {value: set(_array_from_bytes(_DOC_TC, blob)) for value, blob in values.items()}
            for field, values in state["keyword_index"].items()
        }
        return index

    @classmethod
    def load(cls, path, *, tokenizer: Tokenizer | None = None) -> "Index":
        """Load a packed index previously written with :meth:`save`."""
        with open(path, "rb") as handle:
            return cls.loads(handle.read(), tokenizer=tokenizer)


def _itemsizes() -> list[int]:
    """Platform array itemsizes, recorded so a cross-platform load fails loudly."""
    return [array(typecode).itemsize for typecode in (_OFFSET_TC, _DOC_TC, _TF_TC, _FIELD_TC, _LENGTH_TC)]


def _python_tag() -> str:
    """``"major.minor"`` of the running interpreter (marshal format granularity)."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"
