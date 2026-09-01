"""Segmentation primitive tests.

Span conventions must agree across these functions. When they disagree by even one
character, chunks land one byte short of a boundary and code examples are silently cut --
a defect found on the real corpus, not in theory.
"""

from __future__ import annotations

from hybridrag.chunking.segmentation import find_code_fences, paragraph_spans, sentence_spans


class TestFindCodeFences:
    def test_span_covers_exactly_the_fence(self) -> None:
        """End convention must match paragraph_spans: exclude the trailing newline."""
        text = "Intro.\n\n```py\nx = 1\n```\n\nAfter."
        ((start, end),) = find_code_fences(text)
        assert text[start:end] == "```py\nx = 1\n```"

    def test_paragraph_break_after_a_fence_is_a_valid_split_point(self) -> None:
        """If the fence span swallows the newline, prose after it fuses to the code."""
        text = "```py\nx = 1\n```\n\nProse afterwards."
        spans = paragraph_spans(text, 0, len(text))
        assert len(spans) == 2
        assert text[spans[1][0] : spans[1][1]] == "Prose afterwards."

    def test_multiple_fences_pair_correctly(self) -> None:
        text = "```a\n1\n```\n\nmid\n\n```b\n2\n```"
        fences = find_code_fences(text)
        assert len(fences) == 2
        assert text[fences[0][0] : fences[0][1]] == "```a\n1\n```"
        assert text[fences[1][0] : fences[1][1]] == "```b\n2\n```"

    def test_unterminated_fence_runs_to_end(self) -> None:
        text = "```py\nx = 1\n"
        ((_start, end),) = find_code_fences(text)
        assert end == len(text)

    def test_no_boundary_is_reported_inside_a_fence(self) -> None:
        text = "```py\nfirst = 1\n\nsecond = 2\n```"
        assert len(paragraph_spans(text, 0, len(text))) == 1
        assert len(sentence_spans(text, 0, len(text))) == 1
