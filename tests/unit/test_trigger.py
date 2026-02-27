"""
Tests for Trigger Layer — Priority queue ordering.
Bug 11: Events should be dequeued in priority order.
"""
from __future__ import annotations

import asyncio
import pytest

from nexus.core.orchestrator.trigger import (
    TriggerManager, CLITrigger, TaskEvent,
    TriggerSource, TriggerPriority,
)


class TestPriorityQueue:
    """Bug 11: PriorityQueue should order events by priority."""

    @pytest.mark.asyncio
    async def test_critical_dequeued_first(self):
        """CRITICAL events should come before NORMAL events."""
        mgr = TriggerManager()
        cli = mgr.build_cli_trigger()

        # Submit in reverse priority order
        await cli.submit("low task", priority=TriggerPriority.LOW)
        await cli.submit("critical task", priority=TriggerPriority.CRITICAL)
        await cli.submit("normal task", priority=TriggerPriority.NORMAL)

        # Dequeue should be: CRITICAL → NORMAL → LOW
        e1 = await mgr.next_event()
        e2 = await mgr.next_event()
        e3 = await mgr.next_event()

        assert e1.raw_input == "critical task"
        assert e2.raw_input == "normal task"
        assert e3.raw_input == "low task"

    @pytest.mark.asyncio
    async def test_same_priority_fifo(self):
        """Events with same priority should be FIFO."""
        mgr = TriggerManager()
        cli = mgr.build_cli_trigger()

        await cli.submit("first", priority=TriggerPriority.NORMAL)
        await cli.submit("second", priority=TriggerPriority.NORMAL)

        e1 = await mgr.next_event()
        e2 = await mgr.next_event()

        assert e1.raw_input == "first"
        assert e2.raw_input == "second"

    @pytest.mark.asyncio
    async def test_all_priority_levels(self):
        """Test all 5 priority levels are correctly ordered."""
        mgr = TriggerManager()
        cli = mgr.build_cli_trigger()

        priorities = [
            TriggerPriority.IDLE,
            TriggerPriority.HIGH,
            TriggerPriority.CRITICAL,
            TriggerPriority.NORMAL,
            TriggerPriority.LOW,
        ]
        for p in priorities:
            await cli.submit(f"task_{p.name}", priority=p)

        events = []
        for _ in range(5):
            events.append(await mgr.next_event())

        # Verify order: CRITICAL(1) → HIGH(2) → NORMAL(3) → LOW(4) → IDLE(5)
        assert events[0].raw_input == "task_CRITICAL"
        assert events[1].raw_input == "task_HIGH"
        assert events[2].raw_input == "task_NORMAL"
        assert events[3].raw_input == "task_LOW"
        assert events[4].raw_input == "task_IDLE"


class TestCLITrigger:
    """Test CLITrigger submit, pause, resume."""

    @pytest.mark.asyncio
    async def test_submit_returns_event_id(self):
        mgr = TriggerManager()
        cli = mgr.build_cli_trigger()
        event_id = await cli.submit("test task")
        assert event_id is not None
        assert len(event_id) > 0

    @pytest.mark.asyncio
    async def test_pause_creates_critical_event(self):
        mgr = TriggerManager()
        cli = mgr.build_cli_trigger()
        await cli.pause("task-123")

        event = await mgr.next_event()
        assert "__PAUSE__" in event.raw_input
        assert event.payload["action"] == "pause"

    @pytest.mark.asyncio
    async def test_resume_creates_high_priority_event(self):
        mgr = TriggerManager()
        cli = mgr.build_cli_trigger()
        await cli.resume("task-123")

        event = await mgr.next_event()
        assert "__RESUME__" in event.raw_input
        assert event.resume_task_id == "task-123"
