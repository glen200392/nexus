"""
Tests for CheckpointStore — Durable execution state persistence.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from nexus.core.orchestrator.checkpoint import CheckpointStore, Checkpoint


@pytest.fixture
async def store(tmp_path):
    """Fresh CheckpointStore with temp database."""
    s = CheckpointStore(db_path=tmp_path / "test_checkpoints.db")
    await s.initialize()
    yield s
    await s.close()


class TestCheckpointStore:

    @pytest.mark.asyncio
    async def test_save_and_load(self, store):
        cp = Checkpoint(thread_id="t1", node_name="search", step=1, state_json='{"data": "test"}')
        cid = await store.save(cp)
        assert cid == cp.checkpoint_id

        loaded = await store.load("t1", step=1)
        assert loaded is not None
        assert loaded.thread_id == "t1"
        assert loaded.node_name == "search"
        assert loaded.state_json == '{"data": "test"}'

    @pytest.mark.asyncio
    async def test_load_latest(self, store):
        await store.save(Checkpoint(thread_id="t1", step=0, node_name="a"))
        await store.save(Checkpoint(thread_id="t1", step=1, node_name="b"))
        await store.save(Checkpoint(thread_id="t1", step=2, node_name="c"))

        latest = await store.load_latest("t1")
        assert latest is not None
        assert latest.step == 2
        assert latest.node_name == "c"

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, store):
        for i in range(5):
            await store.save(Checkpoint(thread_id="t1", step=i, node_name=f"node_{i}"))

        cps = await store.list_checkpoints("t1")
        assert len(cps) == 5
        assert [cp.step for cp in cps] == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_fork(self, store):
        await store.save(Checkpoint(
            thread_id="t1", step=2, node_name="analyze",
            state_json='{"data": "at step 2"}'
        ))

        new_thread = await store.fork("t1", step=2)
        assert new_thread != "t1"

        forked = await store.load_latest(new_thread)
        assert forked is not None
        assert forked.step == 2
        assert forked.state_json == '{"data": "at step 2"}'
        assert forked.metadata["forked_from"] == "t1"

    @pytest.mark.asyncio
    async def test_fork_nonexistent_raises(self, store):
        with pytest.raises(ValueError, match="No checkpoint found"):
            await store.fork("nonexistent", step=0)

    @pytest.mark.asyncio
    async def test_delete_thread(self, store):
        for i in range(3):
            await store.save(Checkpoint(thread_id="t1", step=i))

        count = await store.delete_thread("t1")
        assert count == 3

        cps = await store.list_checkpoints("t1")
        assert len(cps) == 0

    @pytest.mark.asyncio
    async def test_load_missing_returns_none(self, store):
        assert await store.load("nonexistent") is None

    @pytest.mark.asyncio
    async def test_cleanup(self, store):
        import time
        old = Checkpoint(thread_id="old", step=0, created_at=time.time() - 86400 * 60)
        recent = Checkpoint(thread_id="recent", step=0)

        await store.save(old)
        await store.save(recent)

        cleaned = await store.cleanup(older_than_days=30)
        assert cleaned == 1

        assert await store.load("recent") is not None
        assert await store.load("old") is None
