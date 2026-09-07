"""Processed-document storage.

Two things matter. A stale store must be detectable, or a rebuild silently indexes
yesterday's corpus and every metric describes a document set that no longer exists. And the
format must not be pickle: a corpus cache is exactly the file that gets copied between
machines and loaded without being read first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hybridrag.loaders import DocumentStore
from hybridrag.models import Document, Section, SourceFormat


def document(path: str = "guide.md", text: str = "Alpha declares the model.") -> Document:
    return Document(
        doc_id=Document.make_id(path),
        relative_path=path,
        source_format=SourceFormat.MARKDOWN,
        title="Guide",
        text=text,
        sections=[
            Section(
                section_id=Section.make_id(path, ("Guide",)),
                relative_path=path,
                heading_path=("Guide",),
                level=1,
                text=text,
                start_char=0,
                end_char=len(text),
            )
        ],
        content_hash="hash",
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "raw"
    root.mkdir()
    (root / "guide.md").write_text("# Guide\nAlpha declares the model.", encoding="utf-8")
    return root


class TestRoundTrip:
    def test_documents_survive_with_their_sections(self, tmp_path: Path, corpus: Path) -> None:
        store = DocumentStore(tmp_path / "processed")
        store.save([document()], corpus_root=corpus)

        loaded = list(store.load())

        assert loaded == [document()]
        assert loaded[0].sections[0].heading_path == ("Guide",)

    def test_the_store_is_plain_json_lines(self, tmp_path: Path, corpus: Path) -> None:
        """Never pickle: loading one executes arbitrary code, and this file travels."""
        store = DocumentStore(tmp_path / "processed")
        store.save([document(), document("other.md")], corpus_root=corpus)

        lines = store.documents_path.read_text(encoding="utf-8").strip().splitlines()

        assert len(lines) == 2
        assert lines[0].startswith("{")

    def test_an_empty_store_reports_itself_missing(self, tmp_path: Path) -> None:
        assert not DocumentStore(tmp_path / "processed").exists()


class TestStaleness:
    def test_an_unchanged_corpus_is_not_stale(self, tmp_path: Path, corpus: Path) -> None:
        store = DocumentStore(tmp_path / "processed")
        store.save([document()], corpus_root=corpus)

        assert store.stale_against(corpus) == []

    def test_an_edited_source_is_detected(self, tmp_path: Path, corpus: Path) -> None:
        store = DocumentStore(tmp_path / "processed")
        store.save([document()], corpus_root=corpus)

        (corpus / "guide.md").write_text("# Guide\nBeta validates it.", encoding="utf-8")

        assert store.stale_against(corpus) == ["guide.md"]

    def test_a_deleted_source_is_detected(self, tmp_path: Path, corpus: Path) -> None:
        store = DocumentStore(tmp_path / "processed")
        store.save([document()], corpus_root=corpus)

        (corpus / "guide.md").unlink()

        assert store.stale_against(corpus) == ["guide.md"]

    def test_a_new_source_is_detected(self, tmp_path: Path, corpus: Path) -> None:
        """A document added since the last build is missing from the index entirely."""
        store = DocumentStore(tmp_path / "processed")
        store.save([document()], corpus_root=corpus)

        (corpus / "new.md").write_text("# New\nGamma filters the output.", encoding="utf-8")

        assert store.stale_against(corpus) == ["new.md"]

    def test_staleness_is_keyed_on_raw_bytes_not_parsed_text(
        self, tmp_path: Path, corpus: Path
    ) -> None:
        """Checking parsed text would need the parse this store exists to avoid."""
        store = DocumentStore(tmp_path / "processed")
        store.save([document()], corpus_root=corpus)

        recorded = store.manifest().sources["guide.md"]

        assert recorded != document().content_hash


class TestManifest:
    def test_the_manifest_records_what_produced_the_store(
        self, tmp_path: Path, corpus: Path
    ) -> None:
        store = DocumentStore(tmp_path / "processed")
        store.save([document()], corpus_root=corpus, include_root=corpus.parent)

        manifest = store.manifest()

        assert manifest.documents == 1
        assert manifest.corpus_root == str(corpus)
        assert manifest.include_root == str(corpus.parent)

    def test_a_store_from_another_format_version_refuses_to_load(self, tmp_path: Path) -> None:
        store = DocumentStore(tmp_path / "processed")
        store.path.mkdir(parents=True)
        store.manifest_path.write_text('{"format_version": 99}', encoding="utf-8")
        store.documents_path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match="format version"):
            store.manifest()
