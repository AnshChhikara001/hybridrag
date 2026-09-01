"""Analysis for the sparse (BM25) index.

This function is the reason hybrid retrieval can beat dense retrieval on this corpus, so
it is worth more care than a one-line `text.split()`.

A default tokenizer splits on punctuation and shreds `response_model` into `response` and
`model`. Those two words appear in hundreds of chunks, so the exact-match signal that
justifies running a keyword index at all is destroyed before BM25 ever sees it. The corpus
has 797 unique inline-code identifiers; they are precisely what a developer types into a
search box and precisely what an embedding model is worst at distinguishing.

The strategy is to emit **both** the whole identifier and its parts. A query for
`response_model` matches the exact term strongly, while a query phrased as "response model"
still matches through the parts. One term is unambiguous and rare -- so BM25's IDF gives it
real weight -- while the parts keep recall for natural-language phrasing.

The same function analyses documents and queries. Using different analysis on the two sides
is the classic way to build a keyword index that silently never matches.
"""

from __future__ import annotations

import re

# A token starts alphanumeric and may carry internal separators, so `app.get`,
# `response_model` and `bge-small` survive as single terms. Trailing separators are
# trimmed afterwards: `end.` should be `end`, not `end.`.
_RUN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*")
_SEPARATOR = re.compile(r"[_.\-]+")
# Split camelCase and acronym boundaries: HTTPException -> HTTP, Exception.
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_HAS_LETTER = re.compile(r"[A-Za-z]")


def tokenize(text: str) -> list[str]:
    """Analyse `text` into BM25 terms, preserving code identifiers as whole terms."""
    terms: list[str] = []
    for match in _RUN.finditer(text):
        token = match.group().rstrip("_.-")
        if not token:
            continue
        terms.append(token.lower())

        # Parts are only useful when the token is a compound identifier. Splitting a
        # version number would turn `0.95` into the meaningless terms `0` and `95`, so
        # anything without a letter is left whole.
        if not _HAS_LETTER.search(token):
            continue
        parts = [p for chunk in _SEPARATOR.split(token) for p in _CAMEL.split(chunk) if p]
        if len(parts) > 1:
            terms.extend(part.lower() for part in parts)
    return terms
