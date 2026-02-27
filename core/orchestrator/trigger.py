"""
NEXUS Trigger Layer — Layer 1
Unified event ingestion from all trigger sources:
  - Scheduler (cron / interval)
  - File system hooks (watchdog)
  - Webhook receivers (FastAPI)
  - Manual CLI / chat
  - Git hooks, message queues
All triggers normalize into a TaskEvent and push to the perception queue.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger("nexus.trigger")


# ─── Trigger Sources ──────────────────────────────────────────────────────────

class TriggerSource(str, Enum):
    SCHEDULER  = "scheduler"    # APScheduler cron / interval
    FILESYSTEM = "filesystem"   # watchdog file system events
    WEBHOOK    = "webhook"      # HTTP POST from external service
    CLI        = "cli"          # Manual user command
    CHAT       = "chat"         # Message from clawd / clawdbot
    GIT_HOOK   = "git_hook"     # Post-commit, post-push hooks
    HEARTBEAT  = "heartbeat"    # Periodic health / proactive check
    MQ         = "message_queue"  # Redis pub/sub, MQTT


class TriggerPriority(int, Enum):
    CRITICAL = 1   # Immediate execution
    HIGH     = 2   # Next available slot
    NORMAL   = 3   # Standard queue
    LOW      = 4   # Background / batch
    IDLE     = 5   # Only when system is quiet


# ─── Task Event (Trigger → Perception) ───────────────────────────────────────

@dataclass
class TaskEvent:
    """
    Normalized event from any trigger source.
    This is the contract between Layer 1 and Layer 2.
    """
    event_id:    str = field(default_factory=lambda: str(uuid.uuid4()))
    source:      TriggerSource = TriggerSource.CLI
    priority:    TriggerPriority = TriggerPriority.NORMAL

    # Content
    raw_input:   str = ""          # Original text/command
    payload:     dict = field(default_factory=dict)  # Structured data from source
    attachments: list[str] = field(default_factory=list)  # File paths

    # Routing hints (from trigger config, not yet analyzed by Perception)
    hint_domain: Optional[str] = None
    hint_privacy: Optional[str] = None

    # Pause / resume
    resume_task_id: Optional[str] = None   # If set, resume a paused task
    checkpoint:     Optional[dict] = None

    # Metadata
    created_at:  float = field(default_factory=time.time)
    session_id:  str = ""
    user_id:     str = "local"


# ─── Base Trigger ─────────────────────────────────────────────────────────────

class BaseTrigger:
    """Abstract base for all trigger implementations."""
    source: TriggerSource = TriggerSource.CLI

    def __init__(self, event_queue: asyncio.Queue, name: str = ""):
        self.queue   = event_queue
        self.name    = name or self.source.value
        self._running = False
        self._logger  = logging.getLogger(f"nexus.trigger.{self.name}")

    async def start(self) -> None:
        self._running = True
        self._logger.info("Trigger '%s' started", self.name)
        await self._run()

    async def stop(self) -> None:
        self._running = False
        self._logger.info("Trigger '%s' stopped", self.name)

    async def _run(self) -> None:
        """Override in subclass."""
        pass

    async def emit(self, event: TaskEvent) -> None:
        """Push a TaskEvent to the perception queue."""
        await self.queue.put(event)
        self._logger.debug("Emitted event %s from %s", event.event_id[:8], self.source)


# ─── Scheduler Trigger ────────────────────────────────────────────────────────

class SchedulerTrigger(BaseTrigger):
    """
    Cron / interval-based trigger.
    Uses APScheduler for flexible schedule definitions.

    Schedule formats:
      interval: {"seconds": 30}
      cron:     {"hour": 9, "minute": 0, "day_of_week": "mon-fri"}
    """
    source = TriggerSource.SCHEDULER

    def __init__(self, event_queue: asyncio.Queue, jobs: list[dict]):
        super().__init__(event_queue, "scheduler")
        self.jobs = jobs  # [{id, schedule_type, schedule, task, priority}]
        self._scheduler = None

    async def _run(self) -> None:
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.cron import CronTrigger
            from apscheduler.triggers.interval import IntervalTrigger

            self._scheduler = AsyncIOScheduler()

            for job in self.jobs:
                stype = job.get("schedule_type", "interval")
                if stype == "interval":
                    trigger = IntervalTrigger(**job["schedule"])
                elif stype == "cron":
                    trigger = CronTrigger(**job["schedule"])
                else:
                    continue

                task_config = job["task"]
                priority    = TriggerPriority[job.get("priority", "NORMAL")]

                async def fire(cfg=task_config, prio=priority):
                    await self.emit(TaskEvent(
                        source=TriggerSource.SCHEDULER,
                        priority=prio,
                        raw_input=cfg.get("prompt", ""),
                        hint_domain=cfg.get("domain"),
                        payload=cfg,
                    ))

                self._scheduler.add_job(fire, trigger, id=job["id"])

            self._scheduler.start()
            self._logger.info("Scheduler started with %d jobs", len(self.jobs))

            while self._running:
                await asyncio.sleep(1)

        except ImportError:
            self._logger.warning("APScheduler not installed; scheduler trigger disabled")
        finally:
            if self._scheduler:
                self._scheduler.shutdown()


# ─── File System Trigger ──────────────────────────────────────────────────────

class FileSystemTrigger(BaseTrigger):
    """
    Watches directories for file changes and emits task events.
    Example use: auto-ingest new documents dropped into watched_dir.
    """
    source = TriggerSource.FILESYSTEM

    def __init__(self, event_queue: asyncio.Queue, watch_paths: list[dict]):
        super().__init__(event_queue, "filesystem")
        self.watch_paths = watch_paths  # [{path, patterns, recursive, domain, prompt_template}]

    async def _run(self) -> None:
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            loop = asyncio.get_event_loop()

            class Handler(FileSystemEventHandler):
                def __init__(self, trigger_self, config):
                    self._trigger = trigger_self
                    self._config  = config

                def on_created(self, event):
                    if event.is_directory:
                        return
                    path = event.src_path
                    patterns = self._config.get("patterns", ["*"])
                    if any(path.endswith(p.lstrip("*")) for p in patterns):
                        prompt = self._config.get(
                            "prompt_template",
                            "New file detected: {path}. Please process it."
                        ).format(path=path)
                        asyncio.run_coroutine_threadsafe(
                            self._trigger.emit(TaskEvent(
                                source=TriggerSource.FILESYSTEM,
                                raw_input=prompt,
                                payload={"file_path": path},
                                attachments=[path],
                                hint_domain=self._config.get("domain"),
                            )),
                            loop,
                        )

            observer = Observer()
            for wp in self.watch_paths:
                observer.schedule(
                    Handler(self, wp),
                    path=wp["path"],
                    recursive=wp.get("recursive", False),
                )
            observer.start()
            self._logger.info("Watching %d paths", len(self.watch_paths))

            while self._running:
                await asyncio.sleep(1)

            observer.stop()
            observer.join()

        except ImportError:
            self._logger.warning("watchdog not installed; filesystem trigger disabled")


# ─── CLI Trigger ──────────────────────────────────────────────────────────────

class CLITrigger(BaseTrigger):
    """
    Reads commands from stdin (interactive) or a command string.
    Used for manual task submission from terminal.
    """
    source = TriggerSource.CLI

    def __init__(self, event_queue: asyncio.Queue, session_id: str = ""):
        super().__init__(event_queue, "cli")
        self.session_id = session_id or str(uuid.uuid4())[:8]

    async def submit(
        self,
        prompt: str,
        priority: TriggerPriority = TriggerPriority.NORMAL,
        attachments: Optional[list[str]] = None,
        domain: Optional[str] = None,
    ) -> str:
        """Submit a manual task. Returns event_id."""
        event = TaskEvent(
            source=TriggerSource.CLI,
            priority=priority,
            raw_input=prompt,
            attachments=attachments or [],
            hint_domain=domain,
            session_id=self.session_id,
        )
        await self.emit(event)
        return event.event_id

    async def pause(self, task_id: str) -> str:
        """Request pause of a running task."""
        event = TaskEvent(
            source=TriggerSource.CLI,
            priority=TriggerPriority.CRITICAL,
            raw_input=f"__PAUSE__{task_id}",
            payload={"action": "pause", "task_id": task_id},
            session_id=self.session_id,
        )
        await self.emit(event)
        return event.event_id

    async def resume(self, task_id: str, checkpoint: Optional[dict] = None) -> str:
        """Resume a paused task, optionally from a checkpoint."""
        event = TaskEvent(
            source=TriggerSource.CLI,
            priority=TriggerPriority.HIGH,
            raw_input=f"__RESUME__{task_id}",
            payload={"action": "resume", "task_id": task_id},
            resume_task_id=task_id,
            checkpoint=checkpoint,
            session_id=self.session_id,
        )
        await self.emit(event)
        return event.event_id


# ─── Trigger Manager ─────────────────────────────────────────────────────────

class TriggerManager:
    """
    Manages all triggers and the shared event queue.
    Called by nexus.py at startup.
    """

    def __init__(self, queue_maxsize: int = 100):
        self.queue    = asyncio.Queue(maxsize=queue_maxsize)
        self.triggers: list[BaseTrigger] = []
        self._tasks:   list[asyncio.Task] = []

    def register(self, trigger: BaseTrigger) -> None:
        self.triggers.append(trigger)
        logger.info("Registered trigger: %s", trigger.name)

    async def start_all(self) -> None:
        for trigger in self.triggers:
            task = asyncio.create_task(trigger.start(), name=f"trigger:{trigger.name}")
            self._tasks.append(task)
        logger.info("All %d triggers started", len(self.triggers))

    async def stop_all(self) -> None:
        for trigger in self.triggers:
            await trigger.stop()
        for task in self._tasks:
            task.cancel()

    async def next_event(self) -> TaskEvent:
        """Get the next event from the priority queue."""
        return await self.queue.get()

    def build_cli_trigger(self) -> CLITrigger:
        t = CLITrigger(self.queue)
        self.register(t)
        return t
