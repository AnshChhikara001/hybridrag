"""Reciprocal Rank Fusion arithmetic.

Tested as a pure function against hand-computed values. The properties that matter are the
ones that justify choosing RRF at all: agreement between retrievers outranks a single
retriever's first place, and native scores never touch the arithmetic.
"""

from __future__ import annotations

import pytest

from hybridrag.retrieval.fusion import reciprocal_rank_fusion


def ids(results: list) -> list[str]:  # type: ignore[type-arg]
    return [result.chunk_id for result in results]


def test_one_ranking_comes_back_in_the_same_order() -> None:
    """RRF over a single list is monotone in rank, so an ablation cannot reorder it."""
    ranked = [("a", 9.0), ("b", 4.0), ("c", 1.0)]

    assert ids(reciprocal_rank_fusion({"sparse": ranked})) == ["a", "b", "c"]


def test_agreement_outranks_a_lone_first_place() -> None:
    """The property hybrid retrieval is bought for.

    `y` is nobody's best result but both retrievers found it; `x` and `z` are each ranked
    first by one retriever and missed entirely by the other.
    """
    fused = reciprocal_rank_fusion(
        {"dense": [("x", 0.9), ("y", 0.8)], "sparse": [("y", 12.0), ("z", 3.0)]}
    )

    assert ids(fused)[0] == "y"
    assert fused[0].retrievers == ("dense", "sparse")


def test_scores_are_the_reciprocal_rank_sum() -> None:
    fused = reciprocal_rank_fusion(
        {"dense": [("x", 0.9), ("y", 0.8)], "sparse": [("y", 12.0)]}, rank_constant=60
    )
    by_id = {result.chunk_id: result for result in fused}

    assert by_id["y"].score == pytest.approx(1 / 62 + 1 / 61)
    assert by_id["x"].score == pytest.approx(1 / 61)


def test_native_scores_are_carried_through_untouched() -> None:
    """BM25 weights and cosine similarities are recorded, never summed."""
    fused = reciprocal_rank_fusion({"dense": [("x", 0.83)], "sparse": [("x", 22.4)]})

    assert fused[0].hits["dense"].score == 0.83
    assert fused[0].hits["sparse"].score == 22.4
    assert fused[0].score == pytest.approx(2 / 61)


def test_hits_record_each_retrievers_own_rank() -> None:
    fused = reciprocal_rank_fusion(
        {"dense": [("a", 0.9), ("x", 0.5)], "sparse": [("b", 8.0), ("c", 5.0), ("x", 2.0)]}
    )
    hits = next(result for result in fused if result.chunk_id == "x").hits

    assert hits["dense"].rank == 2
    assert hits["sparse"].rank == 3
    assert hits["dense"].contribution == pytest.approx(1 / 62)


def test_a_chunk_only_one_retriever_found_records_only_that_retriever() -> None:
    fused = reciprocal_rank_fusion({"dense": [("x", 0.9)], "sparse": [("y", 4.0)]})
    by_id = {result.chunk_id: result for result in fused}

    assert by_id["x"].retrievers == ("dense",)
    assert "sparse" not in by_id["x"].hits


def test_weighting_decides_which_retriever_wins_a_disagreement() -> None:
    """Perfectly opposed rankings: each retriever's favourite is the other's runner-up.

    Unweighted the two are an exact tie, so the weight alone picks the winner -- which is
    the real use for it: leaning on sparse for identifier queries and dense for prose.
    """
    rankings = {
        "dense": [("prose", 0.9), ("code", 0.4)],
        "sparse": [("code", 22.0), ("prose", 3.0)],
    }
    unweighted = reciprocal_rank_fusion(rankings)

    assert unweighted[0].score == pytest.approx(unweighted[1].score)
    assert ids(reciprocal_rank_fusion(rankings, weights={"dense": 2.0}))[0] == "prose"
    assert ids(reciprocal_rank_fusion(rankings, weights={"sparse": 2.0}))[0] == "code"


def test_weighting_a_retriever_up_also_lifts_what_it_ranked_second() -> None:
    """Weight applies to a retriever's whole list, not only to its exclusive results.

    `y` keeps first place even when dense is weighted 5x for `x`, because dense ranked
    `y` too -- so the boost lands on both. Worth pinning: the intuition that weighting a
    retriever up promotes its top result is wrong whenever the lists overlap.
    """
    rankings = {"dense": [("x", 0.9), ("y", 0.8)], "sparse": [("y", 12.0), ("z", 3.0)]}

    assert ids(reciprocal_rank_fusion(rankings, weights={"dense": 5.0}))[:2] == ["y", "x"]


def test_a_zero_weighted_retriever_stops_contributing() -> None:
    """Weighting a side to zero is an ablation, so its exclusive results should not appear."""
    fused = reciprocal_rank_fusion(
        {"dense": [("x", 0.9)], "sparse": [("y", 4.0)]}, weights={"sparse": 0.0}
    )

    assert ids(fused) == ["x"]


def test_a_lower_rank_constant_favours_a_single_top_result() -> None:
    """`rank_constant` is the knob between 'agreement wins' and 'first place wins'."""
    rankings = {
        "dense": [("x", 0.9), ("a", 0.5), ("b", 0.4), ("y", 0.3)],
        "sparse": [("c", 9.0), ("d", 8.0), ("e", 7.0), ("y", 6.0)],
    }

    def rank_of(chunk_id: str, rank_constant: int) -> int:
        return ids(reciprocal_rank_fusion(rankings, rank_constant=rank_constant)).index(chunk_id)

    # At 60, two fourth places beat one first place. At 1, they no longer do.
    assert rank_of("y", 60) < rank_of("x", 60)
    assert rank_of("x", 1) < rank_of("y", 1)


def test_ties_break_on_chunk_id_so_runs_are_reproducible() -> None:
    fused = reciprocal_rank_fusion({"dense": [("zebra", 1.0)], "sparse": [("apple", 1.0)]})

    assert ids(fused) == ["apple", "zebra"]


def test_limit_truncates_the_fused_ranking() -> None:
    fused = reciprocal_rank_fusion(
        {"dense": [("a", 3.0), ("b", 2.0), ("c", 1.0)]},
        limit=2,
    )

    assert ids(fused) == ["a", "b"]


def test_an_empty_ranking_contributes_nothing() -> None:
    fused = reciprocal_rank_fusion({"dense": [], "sparse": [("x", 4.0)]})

    assert ids(fused) == ["x"]


def test_no_rankings_at_all_fuses_to_nothing() -> None:
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"dense": [], "sparse": []}) == []


def test_a_misspelled_weight_is_an_error_not_a_silent_no_op() -> None:
    with pytest.raises(ValueError, match="not supplied"):
        reciprocal_rank_fusion({"dense": [("x", 1.0)]}, weights={"spasre": 2.0})


def test_negative_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion({"dense": [("x", 1.0)]}, weights={"dense": -1.0})


@pytest.mark.parametrize("rank_constant", [0, -5])
def test_a_non_positive_rank_constant_is_rejected(rank_constant: int) -> None:
    with pytest.raises(ValueError, match="rank_constant"):
        reciprocal_rank_fusion({"dense": [("x", 1.0)]}, rank_constant=rank_constant)


def test_a_non_positive_limit_is_rejected() -> None:
    with pytest.raises(ValueError, match="limit"):
        reciprocal_rank_fusion({"dense": [("x", 1.0)]}, limit=0)
