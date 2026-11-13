"""The checks every eval script makes before it spends time or money.

Each one stops the run with one `error:` line, never a traceback, because the person
running the script must see what to do next.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from hybridrag.evaluation import GoldenQuestion, GoldenSet, QuestionCategory, Split
from hybridrag.evaluation.cli import load_split, require_results_split


def _golden(path: Path) -> Path:
    GoldenSet(
        corpus_ref="test",
        questions=[
            GoldenQuestion(
                question_id="none-001",
                split=Split.TUNING,
                category=QuestionCategory.NO_ANSWER,
                question="What is the support SLA?",
                answer="",
                verified=True,
            )
        ],
    ).save(path)
    return path


def _results(path: Path, split: str | None) -> Path:
    provenance = {"corpus_ref": "test"} if split is None else {"split": split}
    path.write_text(json.dumps({"provenance": provenance, "arms": []}), encoding="utf-8")
    return path


class TestLoadSplit:
    def test_a_split_with_questions_loads(self, tmp_path: Path) -> None:
        golden = load_split(_golden(tmp_path / "g.yaml"), Split.TUNING)

        assert [q.question_id for q in golden.verified()] == ["none-001"]

    def test_an_empty_split_stops_with_one_error_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(typer.Exit) as stopped:
            load_split(_golden(tmp_path / "g.yaml"), Split.HELD_OUT)

        assert stopped.value.exit_code == 1
        assert capsys.readouterr().out.startswith("error: no verified held_out questions")


class TestResultsSplit:
    """Tier 1 results from one Split must never be joined to questions from another."""

    def test_results_from_the_same_split_pass(self, tmp_path: Path) -> None:
        require_results_split(_results(tmp_path / "r.json", "held_out"), Split.HELD_OUT)

    def test_results_from_another_split_stop_the_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(typer.Exit):
            require_results_split(_results(tmp_path / "r.json", "tuning"), Split.HELD_OUT)

        assert "holds tuning results" in capsys.readouterr().out

    def test_results_with_no_recorded_split_stop_the_run(self, tmp_path: Path) -> None:
        """v0 results predate the Split, so nothing proves which questions they scored."""
        with pytest.raises(typer.Exit):
            require_results_split(_results(tmp_path / "r.json", None), Split.TUNING)

    def test_a_missing_results_file_stops_the_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(typer.Exit):
            require_results_split(tmp_path / "absent.json", Split.HELD_OUT)

        assert "--split held_out" in capsys.readouterr().out
