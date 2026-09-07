"""Claim segmentation: what counts as one assertion in an answer.

Everything downstream counts these units -- citation coverage, the composite confidence
score, the unsupported-citation flag -- so a wrong split becomes a confidently wrong
percentage rather than a visible error.
"""

from __future__ import annotations

from hybridrag.generation.claims import Claim, split_claims, structural_coverage


class TestSentenceSplitting:
    def test_two_sentences_are_two_claims(self) -> None:
        claims = split_claims("Workers are set with --workers [2]. The default is one [3].")
        assert [claim.citations for claim in claims] == [(2,), (3,)]

    def test_a_dotted_identifier_does_not_end_a_claim(self) -> None:
        """`app.get` and `0.95` are why the shared splitter requires a capital next."""
        (claim,) = split_claims("Register it with app.get and version 0.95 applies [1].")
        assert claim.citations == (1,)

    def test_an_uncited_sentence_is_still_a_claim(self) -> None:
        claims = split_claims("Set the flag [1]. This part has no source.")
        assert [claim.is_cited for claim in claims] == [True, False]


class TestMarkdownStructure:
    def test_each_bullet_is_its_own_claim(self) -> None:
        """The chunking splitter breaks on blank lines only and would fuse the list."""
        answer = "- Use `--workers` to set workers [1]\n- Use `--host` to bind [2]\n"
        claims = split_claims(answer)
        assert [claim.citations for claim in claims] == [(1,), (2,)]
        assert claims[0].text.startswith("Use")

    def test_numbered_list_markers_are_stripped(self) -> None:
        (claim,) = split_claims("1. Install the dependencies first [1]")
        assert claim.text == "Install the dependencies first [1]"

    def test_headings_are_not_claims(self) -> None:
        claims = split_claims("## Running in Docker\n\nBuild the image first [1].")
        assert [claim.text for claim in claims] == ["Build the image first [1]."]

    def test_table_separator_rows_are_not_claims(self) -> None:
        answer = "| flag | meaning |\n|---|---|\n| --host | bind address [1] |\n"
        assert [claim.citations for claim in split_claims(answer)] == [(), (1,)]

    def test_horizontal_rule_is_not_a_claim(self) -> None:
        assert [claim.text for claim in split_claims("---\n\nOnly this asserts [1].")] == [
            "Only this asserts [1]."
        ]

    def test_empty_answer_has_no_claims(self) -> None:
        assert split_claims("") == []
        assert split_claims("\n\n   \n") == []


class TestCodeFences:
    def test_a_fence_attaches_to_the_claim_that_introduces_it(self) -> None:
        answer = "Create a Dockerfile [1]:\n\n```dockerfile\nFROM python:3.12\n```"
        (claim,) = split_claims(answer)
        assert claim.citations == (1,)
        assert "FROM python:3.12" in claim.text

    def test_brackets_inside_the_fence_are_not_citations(self) -> None:
        answer = "Run the server [1]:\n\n```console\n$ fastapi dev\n[2248755]\n```"
        (claim,) = split_claims(answer)
        assert claim.citations == (1,)

    def test_a_leading_fence_with_no_prose_is_dropped(self) -> None:
        """There is no assertion to check it against, so it is not a claim of its own."""
        claims = split_claims("```python\nx = 1\n```\n\nThat sets the value [1].")
        assert [claim.text for claim in claims] == ["That sets the value [1]."]

    def test_prose_after_a_fence_starts_a_new_claim(self) -> None:
        answer = "First [1]:\n\n```\ncode\n```\n\nThen build it [2]."
        assert [claim.citations for claim in split_claims(answer)] == [(1,), (2,)]


class TestStructuralCoverage:
    def test_coverage_counts_cited_claims(self) -> None:
        claims = split_claims("Cited [1]. Uncited sentence here.")
        assert structural_coverage(claims) == 0.5

    def test_coverage_of_no_claims_is_none(self) -> None:
        """A refusal has nothing to attribute, which is not the same as scoring zero."""
        assert structural_coverage([]) is None

    def test_fully_cited_answer_scores_one(self) -> None:
        assert structural_coverage(split_claims("All of it is cited [1][2].")) == 1.0


class TestIndexing:
    def test_claims_are_numbered_in_order(self) -> None:
        claims = split_claims("One [1]. Two [2]. Three [3].")
        assert [claim.index for claim in claims] == [0, 1, 2]
        assert all(isinstance(claim, Claim) for claim in claims)


class TestTrailingCitationBlocks:
    """Citations with no sentence of their own belong to the sentence before them.

    Found on real output, not imagined: this generator ends answers with
    `... for operations. [1] [2] [4] [5]`, which every assertion rule correctly rejects as
    a claim. An earlier version discarded it, so an answer with four resolved citations
    was reported as having none and its coverage read 0.000.
    """

    def test_a_trailing_block_attaches_to_the_last_claim(self) -> None:
        answer = (
            "OAuth2 scopes are strings separated by spaces. They declare permissions "
            "for operations. [1] [2] [4] [5]"
        )
        claims = split_claims(answer)
        assert [claim.citations for claim in claims] == [(), (1, 2, 4, 5)]

    def test_a_bare_citation_line_after_a_fence_is_not_lost(self) -> None:
        answer = "Build the image:\n\n```console\ndocker build -t myimage .\n```\n\n[4]\n"
        (claim,) = split_claims(answer)
        assert claim.citations == (4,)

    def test_a_citation_repeated_in_prose_and_trailer_is_counted_once(self) -> None:
        (claim,) = split_claims("The flag sets the port [1]. [1] [2]")
        assert claim.citations == (1, 2)

    def test_a_citation_block_before_any_claim_is_dropped(self) -> None:
        """There is no sentence for it to belong to, and inventing one would be a fiction."""
        claims = split_claims("[1] [2]\n\nThe real assertion lives here.")
        assert [claim.citations for claim in claims] == [()]

    def test_the_trailing_block_does_not_become_its_own_claim(self) -> None:
        claims = split_claims("One assertion here. [1] [2]")
        assert len(claims) == 1
