"""Markdown loader, written against what this corpus actually contains.

Every transformation here answers something measured in the real FastAPI docs rather than
something anticipated in the abstract:

* 20 lines inside fenced code blocks look like ATX headings (`# comment` in Python
  examples). A naive `^#` scan invents sections that do not exist, so fence state is
  tracked and headings are only recognised outside fences.
* 440 `{* path *}` directives inject source files at site-build time. Left unexpanded, the
  corpus loses all 461 code examples -- and with them the identifiers (`response_model`,
  `@app.get`, `async def`) that sparse retrieval exists to catch.
* 403 `///` admonition markers, 4 files with YAML frontmatter, inline HTML such as
  `<dfn title="...">`, and headings carrying explicit anchors (`# Title { #anchor }`)
  are all syntax noise that would otherwise end up inside section text and heading paths.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import ClassVar

import structlog

from hybridrag.loaders.base import DocumentLoader, SectionBlock, assemble_document
from hybridrag.models import Document, SourceFormat

log = structlog.get_logger(__name__)

_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
_FENCE = re.compile(r"^\s*(?:`{3,}|~{3,})")
_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_HEADING_ANCHOR = re.compile(r"\s*\{\s*#[-\w]+\s*\}\s*$")
_INCLUDE = re.compile(r"^\{\*\s*(?P<path>\S+)(?P<options>[^*]*?)\*\}\s*$")
_ADMONITION = re.compile(r"^\s*///")
_HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")

_LANGUAGE_BY_SUFFIX = {".py": "python", ".sh": "console", ".json": "json", ".yaml": "yaml"}


class MarkdownLoader(DocumentLoader):
    """Parses markdown into heading-delimited sections."""

    extensions: ClassVar[frozenset[str]] = frozenset({".md", ".markdown"})
    source_format: ClassVar[SourceFormat] = SourceFormat.MARKDOWN

    def __init__(self, *, expand_includes: bool = True, include_root: Path | None = None) -> None:
        self.expand_includes = expand_includes
        # Include directives reach outside the documentation tree: FastAPI's examples live
        # in `docs_src/`, a sibling of `docs/`. Resolving them from the corpus root would
        # force the corpus root up to the repository, which pulls six `requirements*.txt`
        # files into a documentation corpus and rewrites every relative path -- and every
        # id derived from one. Separating the two roots keeps discovery narrow and
        # resolution wide.
        self.include_root = include_root
        self.missing_includes: list[str] = []

    def load(self, path: Path, corpus_root: Path) -> Document:
        relative_path = self.relative_path_of(path, corpus_root)
        raw = path.read_text(encoding="utf-8", errors="replace")
        body = _FRONTMATTER.sub("", raw)
        if self.expand_includes:
            body = self._expand_includes(body, self.include_root or corpus_root, relative_path)

        blocks, title = self._split_into_sections(body)
        return assemble_document(
            relative_path=relative_path,
            source_format=self.source_format,
            blocks=blocks,
            title=title,
        )

    def _expand_includes(self, body: str, corpus_root: Path, relative_path: str) -> str:
        """Replace `{* path *}` directives with the referenced file, as a fenced block."""
        out: list[str] = []
        for line in body.split("\n"):
            match = _INCLUDE.match(line.strip())
            if match is None:
                out.append(line)
                continue

            target = self._resolve_include(match.group("path"), corpus_root)
            if target is None or not target.is_file():
                # Loud, not fatal: one unresolvable example must not abort a 155-file
                # ingest, but it must never pass unnoticed either.
                self.missing_includes.append(match.group("path"))
                log.warning(
                    "include_unresolved", directive=match.group("path"), document=relative_path
                )
                continue

            language = _LANGUAGE_BY_SUFFIX.get(target.suffix, "")
            out.append(f"```{language}")
            out.append(target.read_text(encoding="utf-8", errors="replace").rstrip())
            out.append("```")
        return "\n".join(out)

    @staticmethod
    def _resolve_include(directive_path: str, include_root: Path) -> Path | None:
        """Resolve an include directive against the include root.

        MkDocs resolves these against its configured docs directory, not against the file
        containing them -- so the literal `../../` in the directive does not point where
        plain path arithmetic says it does (verified: it lands on a path that does not
        exist). The reliable anchor is the first root-relative component, so leading
        `..` segments are discarded and the remainder is resolved from the root.
        """
        parts = [p for p in PurePosixPath(directive_path).parts if p not in ("..", ".")]
        if not parts:
            return None
        resolved = (include_root / PurePosixPath(*parts)).resolve()
        # Never let a directive escape the include root (path-traversal guard).
        if not resolved.is_relative_to(include_root.resolve()):
            return None
        return resolved

    @staticmethod
    def _clean(line: str) -> str:
        return _HTML_TAG.sub("", line)

    def _split_into_sections(self, body: str) -> tuple[list[SectionBlock], str | None]:
        """Walk the document once, tracking fence state and the heading stack."""
        blocks: list[SectionBlock] = []
        title: str | None = None
        stack: list[str] = []
        heading_path: tuple[str, ...] = ()
        level = 0
        buffer: list[str] = []
        in_fence = False

        def flush() -> None:
            if buffer:
                blocks.append(
                    SectionBlock(heading_path=heading_path, level=level, text="\n".join(buffer))
                )
                buffer.clear()

        for line in body.split("\n"):
            if _FENCE.match(line):
                in_fence = not in_fence
                buffer.append(line)
                continue

            # Inside a fence every line is content, headings included.
            if in_fence:
                buffer.append(line)
                continue

            heading = _ATX_HEADING.match(line)
            if heading is None:
                if not _ADMONITION.match(line):
                    buffer.append(self._clean(line))
                continue

            flush()
            depth = len(heading.group(1))
            text = self._clean(_HEADING_ANCHOR.sub("", heading.group(2))).strip()
            if title is None and depth == 1:
                title = text

            # Pop to the parent depth, then push. Handles skipped levels (h2 -> h4)
            # without losing the breadcrumb.
            del stack[depth - 1 :]
            while len(stack) < depth - 1:
                stack.append("")
            stack.append(text)
            heading_path = tuple(part for part in stack if part)
            level = depth

        flush()
        return blocks, title
