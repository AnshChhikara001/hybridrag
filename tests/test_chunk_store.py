"""Chunk store behaviour.

The order contract in `get_many` gets the most attention here. Everything else in this
class fails loudly when it breaks; a reordered result set does not -- it returns the right
chunks in the wrong ranking, which reads as working retrieval and measures as noise.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from hybridrag.chunk_store import ChunkStore, MissingChunkError
from hybridrag.models import Chunk, ChunkingStrategy

MakeChunk = Callable[..., Chunk]


@pytest.fixture
def store() -> ChunkStore:
    return ChunkStore.in_memory()


def test_a_stored_chunk_round_trips_unchanged(store: ChunkStore, make_chunk: MakeChunk) -> None:
    chunk = make_chunk(0)
    store.add([chunk])
    assert store.get(chunk.chunk_id) == chunk


def test_heading_path_survives_as_a_tuple(store: ChunkStore, make_chunk: MakeChunk) -> None:
    """JSON has no tuple type, so this would come back as a list without validation."""
    chunk = make_chunk(0, heading_path=("Advanced", "WebSockets"))
    store.add([chunk])

    restored = store.get(chunk.chunk_id)
    assert restored is not None
    assert restored.heading_path == ("Advanced", "WebSockets")
    assert isinstance(restored.heading_path, tuple)


def test_an_unknown_id_returns_none(store: ChunkStore) -> None:
    assert store.get("no-such-chunk") is None


def test_membership_and_length(store: ChunkStore, make_chunk: MakeChunk) -> None:
    chunks = [make_chunk(i) for i in range(3)]
    store.add(chunks)

    assert len(store) == 3
    assert chunks[1].chunk_id in store
    assert "no-such-chunk" not in store


def test_get_many_returns_chunks_in_the_order_requested(
    store: ChunkStore, make_chunk: MakeChunk
) -> None:
    chunks = [make_chunk(i, f"chunk number {i}") for i in range(5)]
    store.add(chunks)

    # Deliberately not insertion order: this is what a ranked result set looks like.
    ranking = [chunks[3].chunk_id, chunks[0].chunk_id, chunks[4].chunk_id]
    assert [chunk.chunk_id for chunk in store.get_many(ranking)] == ranking


def test_get_many_preserves_order_across_lookup_batches(
    store: ChunkStore, make_chunk: MakeChunk
) -> None:
    """More ids than one SQL statement takes, so the batching cannot reorder results."""
    chunks = [make_chunk(i, f"chunk number {i}") for i in range(1200)]
    store.add(chunks)

    ranking = [chunk.chunk_id for chunk in reversed(chunks)]
    assert [chunk.chunk_id for chunk in store.get_many(ranking)] == ranking


def test_get_many_repeats_a_duplicated_id_in_place(
    store: ChunkStore, make_chunk: MakeChunk
) -> None:
    chunks = [make_chunk(i) for i in range(2)]
    store.add(chunks)

    ranking = [chunks[0].chunk_id, chunks[1].chunk_id, chunks[0].chunk_id]
    assert [chunk.chunk_id for chunk in store.get_many(ranking)] == ranking


def test_get_many_of_nothing_is_empty(store: ChunkStore) -> None:
    assert store.get_many([]) == []


def test_get_many_raises_when_an_id_is_missing(store: ChunkStore, make_chunk: MakeChunk) -> None:
    chunk = make_chunk(0)
    store.add([chunk])

    with pytest.raises(MissingChunkError, match="different runs"):
        store.get_many([chunk.chunk_id, "orphaned-id"])


def test_adding_the_same_chunks_again_is_idempotent(
    store: ChunkStore, make_chunk: MakeChunk
) -> None:
    chunks = [make_chunk(i) for i in range(3)]
    store.add(chunks)
    store.add(chunks)

    assert len(store) == 3


def test_re_adding_an_id_replaces_its_content(store: ChunkStore, make_chunk: MakeChunk) -> None:
    original = make_chunk(0, "the original text")
    store.add([original])

    revised = original.model_copy(update={"text": "revised text", "char_count": 12, "end_char": 12})
    store.add([revised])

    stored = store.get(original.chunk_id)
    assert stored is not None
    assert stored.text == "revised text"
    assert len(store) == 1


def test_chunk_ids_can_be_filtered_by_strategy(store: ChunkStore, make_chunk: MakeChunk) -> None:
    store.add([make_chunk(i, strategy=ChunkingStrategy.STRUCTURE) for i in range(2)])
    store.add([make_chunk(i, strategy=ChunkingStrategy.FIXED) for i in range(3)])

    assert len(store.chunk_ids()) == 5
    assert len(store.chunk_ids(ChunkingStrategy.STRUCTURE)) == 2
    assert len(store.chunk_ids(ChunkingStrategy.FIXED)) == 3


def test_iter_chunks_streams_one_strategy_in_id_order(
    store: ChunkStore, make_chunk: MakeChunk
) -> None:
    store.add([make_chunk(i, strategy=ChunkingStrategy.SEMANTIC) for i in range(4)])
    store.add([make_chunk(i, strategy=ChunkingStrategy.FIXED) for i in range(2)])

    semantic = list(store.iter_chunks(ChunkingStrategy.SEMANTIC))
    assert len(semantic) == 4
    assert all(chunk.strategy is ChunkingStrategy.SEMANTIC for chunk in semantic)
    assert [chunk.chunk_id for chunk in semantic] == sorted(chunk.chunk_id for chunk in semantic)
    assert len(list(store.iter_chunks())) == 6


def test_deleting_a_strategy_leaves_the_others_alone(
    store: ChunkStore, make_chunk: MakeChunk
) -> None:
    store.add([make_chunk(i, strategy=ChunkingStrategy.FIXED) for i in range(3)])
    store.add([make_chunk(i, strategy=ChunkingStrategy.STRUCTURE) for i in range(2)])

    assert store.delete_strategy(ChunkingStrategy.FIXED) == 3
    assert store.chunk_ids() == store.chunk_ids(ChunkingStrategy.STRUCTURE)
    assert len(store) == 2


def test_deleting_a_document_leaves_other_documents_and_strategies_alone(
    store: ChunkStore, make_chunk: MakeChunk
) -> None:
    a_fixed = [
        make_chunk(i, strategy=ChunkingStrategy.FIXED, relative_path="a.md") for i in range(2)
    ]
    b_fixed = make_chunk(9, strategy=ChunkingStrategy.FIXED, relative_path="b.md")
    a_structure = make_chunk(0, strategy=ChunkingStrategy.STRUCTURE, relative_path="a.md")
    store.add([*a_fixed, b_fixed, a_structure])

    assert store.delete_document("a.md", ChunkingStrategy.FIXED) == 2
    assert store.chunk_ids(ChunkingStrategy.FIXED) == {b_fixed.chunk_id}
    assert store.chunk_ids(ChunkingStrategy.STRUCTURE) == {a_structure.chunk_id}
    assert len(store) == 2


def test_deleting_a_document_absent_from_this_strategy_is_a_no_op(
    store: ChunkStore, make_chunk: MakeChunk
) -> None:
    store.add([make_chunk(0, strategy=ChunkingStrategy.FIXED, relative_path="a.md")])
    assert store.delete_document("a.md", ChunkingStrategy.STRUCTURE) == 0
    assert len(store) == 1


def test_chunks_survive_closing_and_reopening_the_file(
    tmp_path: Path, make_chunk: MakeChunk
) -> None:
    path = tmp_path / "nested" / "chunks.sqlite"
    chunk = make_chunk(0, "persisted across processes")

    writer = ChunkStore(path)
    writer.add([chunk])
    writer.close()

    reader = ChunkStore(path)
    assert reader.get(chunk.chunk_id) == chunk
    reader.close()


def test_a_store_written_by_another_schema_version_refuses_to_open(tmp_path: Path) -> None:
    path = tmp_path / "chunks.sqlite"
    store = ChunkStore(path)
    store._db.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    store._db.commit()
    store.close()

    with pytest.raises(ValueError, match="schema version"):
        ChunkStore(path)
