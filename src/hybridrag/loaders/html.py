"""HTML loader.

Walks the document tree in source order, treating h1-h6 as section boundaries so HTML
produces the same heading breadcrumbs as markdown. Script, style and noscript subtrees are
removed first -- their contents are not prose, and indexing them would pollute both the
dense and the sparse index with code that no user will ever ask about.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from selectolax.parser import HTMLParser, Node

from hybridrag.loaders.base import DocumentLoader, SectionBlock, assemble_document
from hybridrag.models import Document, SourceFormat

_HEADING_TAGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_DROP_TAGS = ("script", "style", "noscript")


class HtmlLoader(DocumentLoader):
    extensions: ClassVar[frozenset[str]] = frozenset({".html", ".htm"})
    source_format: ClassVar[SourceFormat] = SourceFormat.HTML

    def load(self, path: Path, corpus_root: Path) -> Document:
        relative_path = self.relative_path_of(path, corpus_root)
        tree = HTMLParser(path.read_text(encoding="utf-8", errors="replace"))
        for tag in _DROP_TAGS:
            for node in tree.css(tag):
                node.decompose()

        title_node = tree.css_first("title")
        title = title_node.text().strip() if title_node is not None else None

        blocks: list[SectionBlock] = []
        stack: list[str] = []
        heading_path: tuple[str, ...] = ()
        level = 0
        buffer: list[str] = []

        def flush() -> None:
            if buffer:
                blocks.append(
                    SectionBlock(heading_path=heading_path, level=level, text="\n".join(buffer))
                )
                buffer.clear()

        def walk(node: Node) -> None:
            nonlocal heading_path, level
            for child in node.iter(include_text=True):
                tag = child.tag
                if tag in _HEADING_TAGS:
                    flush()
                    depth = _HEADING_TAGS[tag]
                    # Pop to the parent depth, then push -- mirrors the markdown loader so
                    # both formats yield identical breadcrumb shapes.
                    del stack[depth - 1 :]
                    while len(stack) < depth - 1:
                        stack.append("")
                    stack.append(child.text(deep=True).strip())
                    heading_path = tuple(part for part in stack if part)
                    level = depth
                elif tag == "-text":
                    content = child.text().strip()
                    if content:
                        buffer.append(content)
                else:
                    # Recurse into non-heading elements only; a heading's text is taken
                    # whole above, so descending into it would duplicate it into the body.
                    walk(child)

        root = tree.body or tree.root
        if root is not None:
            walk(root)
        flush()

        return assemble_document(
            relative_path=relative_path,
            source_format=self.source_format,
            blocks=blocks,
            title=title,
        )
