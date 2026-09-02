"""Loader tests.

Each markdown test corresponds to something measured in the real FastAPI corpus, not to a
hypothetical. The code-fence case in particular guards a confirmed defect source: 20 lines
across 5 real files are code comments that a naive parser reads as headings.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from pathlib import Path

import pytest

from hybridrag.loaders import CorpusLoader, UnsupportedFormatError
from hybridrag.models import SourceFormat


@pytest.fixture
def loader(fixture_corpus: Path) -> CorpusLoader:
    return CorpusLoader(fixture_corpus)


@pytest.fixture
def basics(loader: CorpusLoader, fixture_corpus: Path):  # type: ignore[no-untyped-def]
    return loader.load(fixture_corpus / "guide" / "basics.md")


class TestMarkdownStructure:
    def test_title_comes_from_h1(self, basics) -> None:  # type: ignore[no-untyped-def]
        assert basics.title == "Basics"

    def test_heading_anchor_is_stripped(self, basics) -> None:  # type: ignore[no-untyped-def]
        """`# Basics { #basics }` must not leak the anchor into the breadcrumb."""
        assert all("{" not in part for s in basics.sections for part in s.heading_path)

    def test_frontmatter_is_removed(self, basics) -> None:  # type: ignore[no-untyped-def]
        assert "Should be stripped" not in basics.text

    def test_breadcrumbs_nest(self, basics) -> None:  # type: ignore[no-untyped-def]
        paths = {s.heading_path for s in basics.sections}
        assert ("Basics", "Installation") in paths
        assert ("Basics", "Installation", "Notes") in paths

    def test_inline_html_is_stripped(self, basics) -> None:  # type: ignore[no-untyped-def]
        assert "<dfn" not in basics.text
        assert "widgets" in basics.text

    def test_admonition_markers_dropped_content_kept(self, basics) -> None:  # type: ignore[no-untyped-def]
        assert "///" not in basics.text
        assert "Use a virtual environment." in basics.text


class TestMarkdownCodeFences:
    """The confirmed parser trap."""

    def test_comment_in_fence_is_not_a_heading(self, basics) -> None:  # type: ignore[no-untyped-def]
        for section in basics.sections:
            assert "This looks like a heading" not in " > ".join(section.heading_path)

    def test_fenced_comment_survives_as_content(self, basics) -> None:  # type: ignore[no-untyped-def]
        assert "# This looks like a heading but is a code comment" in basics.text

    def test_code_indentation_is_preserved(self, basics) -> None:  # type: ignore[no-untyped-def]
        """Collapsing whitespace would corrupt every Python example in the corpus."""
        assert '        return {"deeply": "indented"}' in basics.text


class TestMarkdownIncludes:
    def test_include_is_expanded(self, basics) -> None:  # type: ignore[no-untyped-def]
        """Without expansion the corpus loses the identifiers sparse search needs."""
        assert "@app.get" in basics.text
        assert "async def read_items" in basics.text

    def test_unresolved_include_is_recorded_not_silent(
        self, loader: CorpusLoader, fixture_corpus: Path
    ) -> None:
        assert loader.missing_includes == []
        loader.load(fixture_corpus / "guide" / "basics.md")
        assert any("does_not_exist" in m for m in loader.missing_includes)

    def test_includes_can_be_disabled(self, fixture_corpus: Path) -> None:
        plain = CorpusLoader(fixture_corpus, expand_includes=False)
        doc = plain.load(fixture_corpus / "guide" / "basics.md")
        assert "@app.get" not in doc.text

    def test_includes_can_resolve_from_a_root_outside_the_corpus(self, tmp_path: Path) -> None:
        """FastAPI's layout: prose in `docs/en/docs`, examples in a sibling `docs_src`.

        Widening the corpus root to reach them would sweep six `requirements*.txt` files
        into the corpus and rewrite every relative path, so only resolution widens.
        """
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs_src").mkdir()
        (tmp_path / "docs_src" / "example.py").write_text("@app.get('/items')\n")
        page = tmp_path / "docs" / "page.md"
        page.write_text("# Page\n\n{* ../../docs_src/example.py *}\n")

        narrow = CorpusLoader(tmp_path / "docs")
        wide = CorpusLoader(tmp_path / "docs", include_root=tmp_path)

        assert "@app.get" not in narrow.load(page).text
        assert narrow.missing_includes
        assert "@app.get" in wide.load(page).text
        assert wide.missing_includes == []

    def test_an_include_cannot_escape_the_include_root_either(self, tmp_path: Path) -> None:
        """The traversal guard follows the widened root rather than being dropped by it."""
        corpus = tmp_path / "repo" / "docs"
        corpus.mkdir(parents=True)
        (tmp_path / "secret.py").write_text("SECRET = 'leaked'")
        page = corpus / "page.md"
        page.write_text("# P\n\n{* ../../../../secret.py *}\n")

        doc = CorpusLoader(corpus, include_root=tmp_path / "repo").load(page)

        assert "leaked" not in doc.text

    def test_include_cannot_escape_corpus_root(self, fixture_corpus: Path, tmp_path: Path) -> None:
        secret = tmp_path / "secret.py"
        secret.write_text("SECRET = 'leaked'")
        sneaky = fixture_corpus / "guide" / "_traversal.md"
        sneaky.write_text(f"# T\n\n{{* ../../../../{secret.name} *}}\n")
        try:
            doc = CorpusLoader(fixture_corpus).load(sneaky)
            assert "leaked" not in doc.text
        finally:
            sneaky.unlink()


class TestSectionIdentity:
    def test_repeated_headings_get_distinct_ids(self, basics) -> None:  # type: ignore[no-untyped-def]
        """Two `### Notes` under one parent must not merge into one ground-truth unit."""
        notes = [s for s in basics.sections if s.heading_path[-1:] == ("Notes",)]
        assert len(notes) == 2
        assert notes[0].section_id != notes[1].section_id

    def test_ids_are_unique_within_a_document(self, basics) -> None:  # type: ignore[no-untyped-def]
        ids = [s.section_id for s in basics.sections]
        assert len(ids) == len(set(ids))

    def test_reloading_produces_identical_ids(
        self, loader: CorpusLoader, fixture_corpus: Path
    ) -> None:
        path = fixture_corpus / "guide" / "basics.md"
        first = [s.section_id for s in loader.load(path).sections]
        second = [s.section_id for s in loader.load(path).sections]
        assert first == second


class TestTextLoader:
    def test_single_section_no_headings(self, loader: CorpusLoader, fixture_corpus: Path) -> None:
        doc = loader.load(fixture_corpus / "notes.txt")
        assert doc.source_format is SourceFormat.TEXT
        assert len(doc.sections) == 1
        assert doc.sections[0].heading_path == ()


class TestHtmlLoader:
    @pytest.fixture
    def page(self, loader: CorpusLoader, fixture_corpus: Path):  # type: ignore[no-untyped-def]
        return loader.load(fixture_corpus / "guide" / "page.html")

    def test_title_from_head(self, page) -> None:  # type: ignore[no-untyped-def]
        assert page.title == "HTML Guide"

    def test_headings_become_breadcrumbs(self, page) -> None:  # type: ignore[no-untyped-def]
        paths = {s.heading_path for s in page.sections}
        assert ("Guide", "Configuration") in paths
        assert ("Guide", "Deployment") in paths

    def test_nested_heading_is_found(self, page) -> None:  # type: ignore[no-untyped-def]
        """The h2 is inside a div; a children-only walk would miss it."""
        assert "root_path" in page.text

    def test_script_and_style_are_not_indexed(self, page) -> None:  # type: ignore[no-untyped-def]
        assert "should not be indexed" not in page.text
        assert "color:red" not in page.text


class TestPdfLoader:
    def test_pages_become_sections_with_numbers(
        self, make_pdf: Callable[[list[str]], Path], tmp_path: Path
    ) -> None:
        pdf = make_pdf(["Alpha page content", "Beta page content"])
        doc = CorpusLoader(tmp_path).load(pdf)
        assert doc.source_format is SourceFormat.PDF
        assert [s.page for s in doc.sections] == [1, 2]
        assert "Alpha" in doc.sections[0].text

    def test_pages_get_distinct_ids_despite_empty_headings(
        self, make_pdf: Callable[[list[str]], Path], tmp_path: Path
    ) -> None:
        """Every PDF page has an empty breadcrumb, so without the page discriminator all
        pages would hash to one section_id and collapse into a single ground-truth unit."""
        doc = CorpusLoader(tmp_path).load(make_pdf(["One", "Two", "Three"]))
        ids = {s.section_id for s in doc.sections}
        assert len(ids) == 3

    def test_corrupt_pdf_raises_rather_than_returning_empty(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.pdf"
        broken.write_bytes(b"not a pdf at all")
        from hybridrag.loaders import LoaderError

        with pytest.raises(LoaderError):
            CorpusLoader(tmp_path).load(broken)


class TestDispatch:
    def test_unsupported_extension_raises_with_guidance(
        self, loader: CorpusLoader, tmp_path: Path
    ) -> None:
        odd = tmp_path / "data.xlsx"
        odd.write_text("x")
        with pytest.raises(UnsupportedFormatError, match="supported:"):
            loader.loader_for(odd)

    def test_iteration_is_deterministic(self, fixture_corpus: Path) -> None:
        """Chunk IDs derive from position, so ingestion order must not vary by filesystem."""
        first = [d.relative_path for d in CorpusLoader(fixture_corpus).iter_documents()]
        second = [d.relative_path for d in CorpusLoader(fixture_corpus).iter_documents()]
        assert first == second == sorted(first)

    def test_all_four_formats_are_registered(self, loader: CorpusLoader) -> None:
        assert {".md", ".txt", ".html", ".pdf"} <= loader.supported_extensions


class TestSectionOffsets:
    """Offsets must address `Document.text` exactly.

    Phase 4 defines a retrieval hit as a chunk covering an answer span, and spans are
    expressed in these coordinates. If offsets drift from the text by even one character,
    every downstream metric is silently wrong, so the invariant is asserted directly.
    """

    def test_offsets_slice_back_to_section_text(self, basics) -> None:  # type: ignore[no-untyped-def]
        for section in basics.sections:
            assert basics.text[section.start_char : section.end_char] == section.text

    def test_offsets_hold_for_every_format(
        self, loader: CorpusLoader, fixture_corpus: Path
    ) -> None:
        for document in loader.iter_documents():
            for section in document.sections:
                assert document.text[section.start_char : section.end_char] == section.text, (
                    f"offset drift in {document.relative_path}"
                )

    def test_offsets_hold_for_pdf_pages(
        self, make_pdf: Callable[[list[str]], Path], tmp_path: Path
    ) -> None:
        doc = CorpusLoader(tmp_path).load(make_pdf(["First page", "Second page", "Third"]))
        for section in doc.sections:
            assert doc.text[section.start_char : section.end_char] == section.text

    def test_sections_are_ordered_and_non_overlapping(self, basics) -> None:  # type: ignore[no-untyped-def]
        spans = [(s.start_char, s.end_char) for s in basics.sections]
        assert spans == sorted(spans)
        for (_, prev_end), (next_start, _) in pairwise(spans):
            assert prev_end <= next_start
