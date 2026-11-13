"""Shared parts of the eval scripts: the Split option and the checks every run makes first.

Each check stops the run with one `error:` line and exit code 1, before any time or money
is spent. A traceback would hide the one thing the person needs: what to do next.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from hybridrag.evaluation.golden import GoldenSet, Split
from hybridrag.evaluation.report import DirtyTreeError, require_clean_tree

SplitOption = Annotated[
    Split, typer.Option(help="Golden set Split. Only held_out gives published numbers.")
]


def _stop(message: str) -> NoReturn:
    typer.echo(f"error: {message}")
    raise typer.Exit(code=1)


def load_split(path: Path, split: Split) -> GoldenSet:
    """The Golden set at `path`, holding only `split`'s questions."""
    try:
        return GoldenSet.load(path).for_split(split)
    except ValueError as error:
        _stop(str(error))


def exit_if_dirty() -> None:
    """Stop when uncommitted changes would make the report's commit line untrue."""
    try:
        require_clean_tree()
    except DirtyTreeError as error:
        _stop(str(error))


def require_results_split(path: Path, split: Split) -> None:
    """Stop unless the Tier 1 results at `path` were scored on the same Split.

    Scripts join these results to questions by ID. Results from another Split would join
    to nothing, and every question would read as "retrieval missed it" with no error.
    """
    if not path.is_file():
        _stop(
            f"{path} does not exist. Run scripts/evaluate_retrieval.py --split {split.value} first."
        )
    recorded = json.loads(path.read_text(encoding="utf-8")).get("provenance", {}).get("split")
    if recorded != split.value:
        _stop(
            f"{path} holds {recorded or 'unlabelled (v0)'} results, but this run uses "
            f"{split.value}. Run scripts/evaluate_retrieval.py --split {split.value} first."
        )
