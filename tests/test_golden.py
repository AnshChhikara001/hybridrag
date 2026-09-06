"""Golden-set schema, span resolution and the relevance predicate.

Two behaviours carry most of the weight. Quote resolution must refuse anything ambiguous,
because a span pointing at the wrong text corrupts every retrieval metric while failing
nothing. And coverage must be measured over the *union* of retrieved chunks, or the
overlapping fixed-size strategy scores above 100% and wins the comparison on a bug.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hybridrag.evaluation.golden import (
    AnswerSpan,
    GoldenQuestion,
    GoldenSet,
    QuestionCategory,
    SpanNotFoundError,
    covered_ratio,
    is_hit,
)
from hybridrag.models import Chunk, ChunkingStrategy, Document, SourceFormat

TEXT = "Alpha declares the model. Beta validates it. Gamma filters the output."


def document(path: str = "guide.md", text: str = TEXT) -> Document:
    return Document(
        doc_id=Document.make_id(path),
        relative_path=path,
        source_format=SourceFormat.MARKDOWN,
        text=text,
        content_hash="hash",
    )


def chunk(start: int, end: int, path: str = "guide.md", text: str | None = None) -> Chunk:
    body = text if text is not None else TEXT[start:end]
    return Chunk(
        chunk_id=f"c{start}-{end}-{path}",
        doc_id=Document.make_id(path),
        relative_path=path,
        text=body,
        chunk_index=0,
        strategy=ChunkingStrategy.STRUCTURE,
        token_count=max(1, len(body.split())),
        char_count=end - start,
        start_char=start,
        end_char=end,
    )


def question(
    category: QuestionCategory = QuestionCategory.LOOKUP,
    spans: list[AnswerSpan] | None = None,
    **kwargs: object,
) -> GoldenQuestion:
    defaults: dict[str, object] = {
        "question_id": "q001",
        "category": category,
        "question": "What declares the model?",
        "answer": "Alpha does.",
        "spans": spans
        if spans is not None
        else [AnswerSpan(relative_path="guide.md", quote="Alpha declares the model.")],
    }
    defaults.update(kwargs)
    return GoldenQuestion.model_validate(defaults)


class TestSpanResolution:
    def test_a_quote_resolves_to_offsets_in_the_document(self) -> None:
        located = question().locate({"guide.md": document()})

        assert located[0].start_char == 0
        assert located[0].end_char == 25
        assert TEXT[located[0].start_char : located[0].end_char] == "Alpha declares the model."

    def test_a_paraphrased_quote_is_refused(self) -> None:
        """Generated candidates paraphrase constantly; that must fail, not be approximated."""
        q = question(spans=[AnswerSpan(relative_path="guide.md", quote="Alpha declares a model")])

        with pytest.raises(SpanNotFoundError, match="not found"):
            q.locate({"guide.md": document()})

    def test_an_ambiguous_quote_is_refused(self) -> None:
        """Two occurrences means the span is undecidable, so it cannot be ground truth."""
        text = "Set the flag. Set the flag."
        q = question(spans=[AnswerSpan(relative_path="guide.md", quote="Set the flag.")])

        with pytest.raises(SpanNotFoundError, match="occurs 2 times"):
            q.locate({"guide.md": document(text=text)})

    def test_a_missing_document_is_refused(self) -> None:
        with pytest.raises(SpanNotFoundError, match="no document"):
            question().locate({})


class TestCategoryShapes:
    def test_a_lookup_takes_exactly_one_span(self) -> None:
        with pytest.raises(ValueError, match="exactly one span"):
            question(
                spans=[
                    AnswerSpan(relative_path="a.md", quote="one"),
                    AnswerSpan(relative_path="b.md", quote="two"),
                ]
            )

    def test_multi_hop_must_cross_documents(self) -> None:
        """Two spans in one file is a lookup with a long answer, and tests nothing multi-hop."""
        with pytest.raises(ValueError, match="two or more documents"):
            question(
                category=QuestionCategory.MULTI_HOP,
                spans=[
                    AnswerSpan(relative_path="guide.md", quote="Alpha declares the model."),
                    AnswerSpan(relative_path="guide.md", quote="Beta validates it."),
                ],
            )

    def test_a_no_answer_question_must_have_no_spans(self) -> None:
        with pytest.raises(ValueError, match="no spans"):
            question(category=QuestionCategory.NO_ANSWER)

    def test_a_no_answer_question_is_valid_when_empty(self) -> None:
        q = question(category=QuestionCategory.NO_ANSWER, spans=[], answer="")

        assert q.spans == []

    def test_an_answerable_category_needs_a_span(self) -> None:
        with pytest.raises(ValueError, match="at least one span"):
            question(category=QuestionCategory.AMBIGUOUS, spans=[])


class TestCoverage:
    def test_a_span_inside_one_chunk_is_fully_covered(self) -> None:
        located = question().locate({"guide.md": document()})[0]

        assert covered_ratio(located, [chunk(0, 40)]) == pytest.approx(1.0)

    def test_a_span_split_across_two_chunks_is_covered_by_their_union(self) -> None:
        """Scoring per chunk would penalise a strategy for where its boundaries fall."""
        located = question().locate({"guide.md": document()})[0]

        assert covered_ratio(located, [chunk(0, 12), chunk(12, 30)]) == pytest.approx(1.0)
        assert covered_ratio(located, [chunk(0, 12)]) == pytest.approx(12 / 25)

    def test_overlapping_chunks_do_not_count_shared_text_twice(self) -> None:
        """The fixed-size strategy overlaps by 64 tokens; naive summing would exceed 1.0."""
        located = question().locate({"guide.md": document()})[0]

        assert covered_ratio(located, [chunk(0, 20), chunk(10, 30)]) == pytest.approx(1.0)

    def test_chunks_from_another_document_do_not_count(self) -> None:
        located = question().locate({"guide.md": document()})[0]

        assert covered_ratio(located, [chunk(0, 40, path="other.md")]) == 0.0

    def test_no_chunks_covers_nothing(self) -> None:
        located = question().locate({"guide.md": document()})[0]

        assert covered_ratio(located, []) == 0.0


class TestRelevanceRules:
    def test_multi_hop_requires_every_span(self) -> None:
        q = question(
            category=QuestionCategory.MULTI_HOP,
            spans=[
                AnswerSpan(relative_path="guide.md", quote="Alpha declares the model."),
                AnswerSpan(relative_path="other.md", quote="Gamma filters the output."),
            ],
        )
        docs = {"guide.md": document(), "other.md": document("other.md")}
        located = q.locate(docs)

        both = [chunk(0, 30), chunk(44, 70, path="other.md")]
        one = [chunk(0, 30)]

        assert is_hit(q, located, both)
        assert not is_hit(q, located, one)

    def test_ambiguous_accepts_any_span(self) -> None:
        q = question(
            category=QuestionCategory.AMBIGUOUS,
            spans=[
                AnswerSpan(relative_path="guide.md", quote="Alpha declares the model."),
                AnswerSpan(relative_path="guide.md", quote="Gamma filters the output."),
            ],
        )
        located = q.locate({"guide.md": document()})

        assert is_hit(q, located, [chunk(44, 70)])

    def test_a_partially_covered_span_misses_at_full_ratio(self) -> None:
        q = question()
        located = q.locate({"guide.md": document()})

        assert not is_hit(q, located, [chunk(0, 12)])
        assert is_hit(q, located, [chunk(0, 12)], min_ratio=0.4)

    def test_no_answer_questions_have_no_retrieval_ground_truth(self) -> None:
        q = question(category=QuestionCategory.NO_ANSWER, spans=[], answer="")

        with pytest.raises(ValueError, match="refusal rate"):
            is_hit(q, [], [chunk(0, 10)])


class TestPersistence:
    def test_a_set_round_trips_through_yaml(self, tmp_path: Path) -> None:
        original = GoldenSet(corpus_ref="0.115.6", questions=[question(verified=True)])
        path = tmp_path / "golden_set.yaml"
        original.save(path)

        assert GoldenSet.load(path) == original

    def test_quotes_stay_readable_in_the_file(self, tmp_path: Path) -> None:
        """A human hand-verifies all 50, so the file must not be a wall of escapes."""
        path = tmp_path / "golden_set.yaml"
        GoldenSet(corpus_ref="0.115.6", questions=[question()]).save(path)

        assert "Alpha declares the model." in path.read_text()

    def test_duplicate_ids_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate question_id"):
            GoldenSet(
                corpus_ref="0.115.6",
                questions=[question(), question()],
            )

    def test_a_set_from_another_format_version_refuses_to_load(self, tmp_path: Path) -> None:
        path = tmp_path / "golden_set.yaml"
        path.write_text("format_version: 99\ncorpus_ref: '0.115.6'\nquestions: []\n")

        with pytest.raises(ValueError, match="format version"):
            GoldenSet.load(path)

    def test_only_verified_questions_are_reportable(self) -> None:
        golden = GoldenSet(
            corpus_ref="0.115.6",
            questions=[
                question(question_id="q001", verified=True),
                question(question_id="q002", verified=False),
            ],
        )

        assert [q.question_id for q in golden.verified()] == ["q001"]
        assert len(golden.by_category(QuestionCategory.LOOKUP)) == 2


class TestWhitespaceTolerantMatching:
    """Models collapse blank lines when copying; the passage is still the right passage."""

    def test_a_quote_that_collapsed_a_paragraph_break_still_resolves(self) -> None:
        text = "Intro line.\n\nThe scopes represent permissions.\n\nTrailing note."
        q = question(
            spans=[
                AnswerSpan(
                    relative_path="guide.md",
                    quote="Intro line.\nThe scopes represent permissions.",
                )
            ]
        )

        located = q.locate({"guide.md": document(text=text)})[0]

        assert located.start_char == 0
        assert located.end_char == len("Intro line.\n\nThe scopes represent permissions.")
        # Rewritten to the document's own bytes, so the span stays exact.
        assert located.quote == "Intro line.\n\nThe scopes represent permissions."

    def test_the_resolved_span_indexes_the_real_document_text(self) -> None:
        text = "Alpha.\n\n\nBeta   gamma.\n\nDelta."
        q = question(spans=[AnswerSpan(relative_path="guide.md", quote="Beta gamma.")])

        located = q.locate({"guide.md": document(text=text)})[0]

        assert text[located.start_char : located.end_char] == "Beta   gamma."

    def test_an_exact_unique_match_wins_over_a_whitespace_variant(self) -> None:
        """Exact is the stronger signal, so it is taken before whitespace is relaxed."""
        text = "Set the flag.\n\nLater: Set  the   flag."
        q = question(spans=[AnswerSpan(relative_path="guide.md", quote="Set the flag.")])

        located = q.locate({"guide.md": document(text=text)})[0]

        assert located.start_char == 0

    def test_ambiguity_is_still_refused_when_whitespace_is_ignored(self) -> None:
        """Neither occurrence matches exactly, and collapsed they are indistinguishable."""
        text = "Set  the flag.\n\nAlso: Set the  flag."
        q = question(spans=[AnswerSpan(relative_path="guide.md", quote="Set the flag.")])

        with pytest.raises(SpanNotFoundError, match="occurs 2 times"):
            q.locate({"guide.md": document(text=text)})

    def test_a_paraphrase_is_still_refused(self) -> None:
        """Tolerating whitespace must not become tolerating different words."""
        q = question(spans=[AnswerSpan(relative_path="guide.md", quote="Alpha declares a model")])

        with pytest.raises(SpanNotFoundError, match="even ignoring whitespace"):
            q.locate({"guide.md": document()})
