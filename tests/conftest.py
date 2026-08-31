"""Shared fixtures.

`make_pdf` builds a genuinely valid PDF with correct cross-reference offsets rather than
mocking `pypdf`. Mocking the parser would test only that our code calls a library; this
tests that a real PDF byte stream round-trips through extraction into pages we can cite.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"


def _build_pdf(pages: list[str]) -> bytes:
    """Assemble a minimal multi-page PDF, computing the xref table offsets."""
    count = len(pages)
    page_ids = [3 + i for i in range(count)]
    content_ids = [3 + count + i for i in range(count)]
    font_id = 3 + 2 * count

    objects: list[tuple[int, str]] = [
        (1, "<< /Type /Catalog /Pages 2 0 R >>"),
        (
            2,
            f"<< /Type /Pages /Kids [{' '.join(f'{i} 0 R' for i in page_ids)}] /Count {count} >>",
        ),
        (font_id, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]
    for page_id, content_id in zip(page_ids, content_ids, strict=True):
        objects.append(
            (
                page_id,
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>",
            )
        )
    for content_id, text in zip(content_ids, pages, strict=True):
        escaped = text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET"
        objects.append((content_id, f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"))

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number, body in sorted(objects):
        offsets[number] = len(out)
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")

    xref_offset = len(out)
    highest = max(offsets)
    out += f"xref\n0 {highest + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for number in range(1, highest + 1):
        out += f"{offsets[number]:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {highest + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(out)


@pytest.fixture
def make_pdf(tmp_path: Path) -> Callable[[list[str]], Path]:
    def _make(pages: list[str]) -> Path:
        target = tmp_path / "sample.pdf"
        target.write_bytes(_build_pdf(pages))
        return target

    return _make


@pytest.fixture
def fixture_corpus() -> Path:
    return FIXTURE_CORPUS
