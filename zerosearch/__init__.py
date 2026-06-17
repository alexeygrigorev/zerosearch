"""zerosearch: a tiny, zero-dependency BM25-lite in-memory search index."""

from zerosearch.__version__ import __version__
from zerosearch.index import DEFAULT_STOP_WORDS, TOKEN_RE, Index, tokenize

__all__ = ["Index", "tokenize", "DEFAULT_STOP_WORDS", "TOKEN_RE", "__version__"]
