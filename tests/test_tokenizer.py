"""Sparse-index analysis tests.

These assertions encode the project's central retrieval claim: that keyword search can
find code identifiers that dense retrieval blurs together. If the tokenizer stops
preserving identifiers, hybrid search quietly degenerates into a slower dense search, and
Phase 4 would measure that without ever explaining it.
"""

from __future__ import annotations

import pytest

from hybridrag.indexing import tokenize


class TestIdentifiersSurvive:
    """The whole justification for running a sparse index alongside a dense one."""

    def test_snake_case_identifier_is_kept_whole(self) -> None:
        assert "response_model" in tokenize("Use response_model to filter output.")

    def test_snake_case_identifier_also_yields_its_parts(self) -> None:
        """So a query phrased as "response model" still matches."""
        terms = tokenize("response_model")
        assert {"response_model", "response", "model"} <= set(terms)

    def test_dotted_call_is_kept_whole(self) -> None:
        terms = tokenize('@app.get("/items/")')
        assert {"app.get", "app", "get", "items"} <= set(terms)

    def test_camel_case_class_is_split_and_kept(self) -> None:
        terms = tokenize("raise HTTPException(status_code=404)")
        assert {"httpexception", "http", "exception"} <= set(terms)
        assert {"status_code", "status", "code"} <= set(terms)

    def test_hyphenated_identifier_survives(self) -> None:
        assert "bge-small" in tokenize("the bge-small model")


class TestNoise:
    def test_punctuation_is_dropped(self) -> None:
        assert tokenize("Hello, world!") == ["hello", "world"]

    def test_trailing_separators_are_trimmed(self) -> None:
        """`end.` and `end` must be the same term or a sentence-final word never matches."""
        assert tokenize("the end.") == tokenize("the end")

    def test_version_numbers_are_not_shredded(self) -> None:
        """Splitting 0.95 into 0 and 95 would add two meaningless high-frequency terms."""
        assert tokenize("cosine above 0.95") == ["cosine", "above", "0.95"]

    def test_empty_text_yields_no_terms(self) -> None:
        assert tokenize("") == []

    @pytest.mark.parametrize("text", ["---", "   ", "!!!", "..."])
    def test_symbol_only_text_yields_no_terms(self, text: str) -> None:
        assert tokenize(text) == []


class TestQueryAndDocumentAgree:
    def test_the_same_analysis_is_applied_to_both_sides(self) -> None:
        """Different analysis on each side is how a keyword index silently never matches."""
        document = tokenize("Declare it with the response_model parameter.")
        query = tokenize("response_model")
        assert set(query) <= set(document)

    def test_case_does_not_affect_matching(self) -> None:
        assert tokenize("RESPONSE_MODEL") == tokenize("response_model")
