"""Tests for :class:`IdleStopSubscriber`.

The subscriber owns the whole idle-stop policy: it counts in-flight work from
the ``on_message`` telemetry stream and stops the team itself when the count
stays at zero for ``delay`` seconds.

Two threads are in play. ``on_message``/``set_restoring`` arrive on caller
threads; the timeout callback fires on the ``Timer`` thread and offloads
``stop_team`` onto a third, daemon thread. Tests that exercise the countdown
therefore poll or wait on an ``Event`` rather than sleeping blindly.

Constructing the subscriber arms a real ``threading.Timer``, so every test here
either injects a delay short enough to fire on purpose or cancels the countdown
before asserting.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

import pytest
from akgentic.core.messages.message import Message, UserMessage
from akgentic.core.messages.orchestrator import ProcessedMessage, ReceivedMessage
from akgentic.core.orchestrator import EventSubscriber
from akgentic.core.utils.timer import Timer

from akgentic.team.subscriber import (
    DEFAULT_IDLE_STOP_DELAY_SECONDS,
    IDLE_STOP_DELAY_ENV_VAR,
    IdleStopSubscriber,
)


class _RecordingTeamManager:
    """Minimal ``TeamManager`` stub recording stop_team calls."""

    def __init__(self, raise_exc: Exception | None = None) -> None:
        self.calls: list[uuid.UUID] = []
        self._raise_exc = raise_exc

    def stop_team(self, team_id: uuid.UUID) -> None:
        self.calls.append(team_id)
        if self._raise_exc is not None:
            raise self._raise_exc


class _RecordingTimer(Timer):
    """``Timer`` double that records ``start``/``cancel`` without arming threads.

    ``task_started``/``task_completed`` are inherited unchanged, so the task
    counting under test is the real thing; only the two thread-touching methods
    they delegate to are replaced by recorders.
    """

    def __init__(self) -> None:
        super().__init__(delay=DEFAULT_IDLE_STOP_DELAY_SECONDS, timeout_callback=lambda: None)
        self.events: list[str] = []

    def start(self) -> None:
        self.events.append("start")

    def cancel(self) -> None:
        self.events.append("cancel")


def _wait_for(condition: Any, timeout: float = 2.0) -> None:
    """Poll ``condition`` (a callable) until true or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():  # type: ignore[operator]
            return
        time.sleep(0.01)
    pytest.fail("timed out waiting for condition")


def _instrumented(
    manager: _RecordingTeamManager | None = None,
) -> tuple[IdleStopSubscriber, uuid.UUID, _RecordingTimer]:
    """Build a subscriber whose countdown is a recorder, not a live timer.

    The real timer armed by ``__init__`` is cancelled first, so the poke tests
    observe start/cancel without leaking a thread or racing a real timeout.
    """
    team_id = uuid.uuid4()
    sub = IdleStopSubscriber(
        manager or _RecordingTeamManager(),  # type: ignore[arg-type]
        team_id,
        delay=DEFAULT_IDLE_STOP_DELAY_SECONDS,
    )
    sub._timer.cancel()
    timer = _RecordingTimer()
    sub._timer = timer
    return sub, team_id, timer


# --- AC1: the class and its Protocol surface ---------------------------------


def test_subclasses_event_subscriber_and_owns_a_core_timer() -> None:
    sub, _, _ = _instrumented()
    # EventSubscriber is a bare Protocol (not @runtime_checkable), so the
    # relationship is asserted through the MRO rather than isinstance.
    assert EventSubscriber in type(sub).__mro__
    assert isinstance(sub._timer, Timer)


def test_defines_the_three_protocol_hooks_and_no_stop_request() -> None:
    sub, _, _ = _instrumented()
    assert callable(sub.set_restoring)
    assert callable(sub.on_message)
    assert callable(sub.on_stop)
    assert not hasattr(sub, "on_stop_request")


# --- AC2: on_message drives the clock, and nothing else does -----------------


def test_received_message_starts_a_task_and_suppresses_the_countdown() -> None:
    sub, _, timer = _instrumented()

    sub.on_message(ReceivedMessage(message_id=uuid.uuid4()))

    assert timer.task_count == 1
    assert timer.events == ["cancel"]


def test_processed_message_completes_the_task_and_restarts_the_countdown() -> None:
    sub, _, timer = _instrumented()

    sub.on_message(ReceivedMessage(message_id=uuid.uuid4()))
    sub.on_message(ProcessedMessage(message_id=uuid.uuid4()))

    assert timer.task_count == 0
    assert timer.events == ["cancel", "start"]


@pytest.mark.parametrize(
    "msg",
    [
        UserMessage(content="hello"),
        Message(),
    ],
)
def test_other_message_types_leave_the_timer_untouched(msg: Message) -> None:
    sub, _, timer = _instrumented()

    sub.on_message(msg)

    assert timer.task_count == 0
    assert timer.events == []


# --- AC3: nested pairs restart only on the outermost completion --------------


def test_nested_pairs_restart_the_countdown_only_on_the_outermost_completion() -> None:
    sub, _, timer = _instrumented()

    sub.on_message(ReceivedMessage(message_id=uuid.uuid4()))
    assert timer.task_count == 1
    sub.on_message(ReceivedMessage(message_id=uuid.uuid4()))
    assert timer.task_count == 2

    sub.on_message(ProcessedMessage(message_id=uuid.uuid4()))
    assert timer.task_count == 1
    assert "start" not in timer.events, "countdown restarted while work was still in flight"

    sub.on_message(ProcessedMessage(message_id=uuid.uuid4()))
    assert timer.task_count == 0
    assert timer.events == ["cancel", "cancel", "start"]


# --- AC4: the restore guard --------------------------------------------------


def test_restore_burst_leaves_the_task_count_unmoved() -> None:
    sub, team_id, timer = _instrumented()

    sub.set_restoring(team_id, True)
    for _ in range(5):
        sub.on_message(ReceivedMessage(message_id=uuid.uuid4()))
        sub.on_message(ProcessedMessage(message_id=uuid.uuid4()))

    assert timer.task_count == 0
    assert timer.events == [], "a replayed burst started or cancelled the countdown"


def test_set_restoring_neither_starts_nor_cancels_the_countdown() -> None:
    sub, team_id, timer = _instrumented()

    sub.set_restoring(team_id, True)
    sub.set_restoring(team_id, False)

    assert timer.events == []


def test_counting_resumes_once_the_restore_ends() -> None:
    sub, team_id, timer = _instrumented()

    sub.set_restoring(team_id, True)
    sub.on_message(ReceivedMessage(message_id=uuid.uuid4()))
    assert timer.task_count == 0

    sub.set_restoring(team_id, False)
    sub.on_message(ReceivedMessage(message_id=uuid.uuid4()))
    assert timer.task_count == 1
    sub.on_message(ProcessedMessage(message_id=uuid.uuid4()))
    assert timer.task_count == 0
    assert timer.events == ["cancel", "start"]


# --- AC5: timeout offloads stop_team onto a daemon thread --------------------


class _CallbackThreadRecordingSubscriber(IdleStopSubscriber):
    """Records the thread the timeout callback itself runs on (the Timer thread)."""

    def __init__(self, team_manager: Any, team_id: uuid.UUID, delay: int) -> None:
        self.callback_thread_id: int | None = None
        super().__init__(team_manager, team_id, delay=delay)

    def _on_idle(self) -> None:
        self.callback_thread_id = threading.get_ident()
        super()._on_idle()


def test_timeout_calls_stop_team_once_on_a_separate_daemon_thread() -> None:
    seen: dict[str, int | bool] = {}
    done = threading.Event()
    calls: list[uuid.UUID] = []

    class _ThreadRecordingManager:
        def stop_team(self, team_id: uuid.UUID) -> None:
            calls.append(team_id)
            seen["thread_id"] = threading.get_ident()
            seen["is_daemon"] = threading.current_thread().daemon
            done.set()

    team_id = uuid.uuid4()
    sub = _CallbackThreadRecordingSubscriber(_ThreadRecordingManager(), team_id, delay=0)

    assert done.wait(timeout=2.0), "stop_team never ran"
    assert calls == [team_id]

    callback_thread_id = sub.callback_thread_id
    assert callback_thread_id is not None
    assert seen["thread_id"] != callback_thread_id, (
        "stop_team ran on the Timer callback thread — an inline call self-deadlocks "
        "against the orchestrator busy servicing the current message"
    )
    assert seen["is_daemon"] is True, "stop_team must run on a daemon thread"

    # The countdown is one-shot: no second stop arrives.
    time.sleep(0.1)
    assert calls == [team_id]


# --- AC6: exceptions never escape the timeout callback -----------------------


def test_timeout_swallows_already_stopped_value_error_and_logs_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _RecordingTeamManager(raise_exc=ValueError("already stopped"))
    team_id = uuid.uuid4()

    with caplog.at_level("DEBUG", logger="akgentic.team.subscriber"):
        sub = IdleStopSubscriber(manager, team_id, delay=0)  # type: ignore[arg-type]
        _wait_for(lambda: manager.calls == [team_id])
        _wait_for(
            lambda: any(
                "idempotent no-op" in record.getMessage() and record.levelname == "DEBUG"
                for record in caplog.records
            )
        )

    sub.on_stop(team_id)


def test_timeout_logs_an_unexpected_error_as_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = _RecordingTeamManager(raise_exc=RuntimeError("unexpected"))
    team_id = uuid.uuid4()

    with caplog.at_level("WARNING", logger="akgentic.team.subscriber"):
        sub = IdleStopSubscriber(manager, team_id, delay=0)  # type: ignore[arg-type]
        _wait_for(lambda: manager.calls == [team_id])
        _wait_for(
            lambda: any(
                "stop_team failed" in record.getMessage() and record.exc_info is not None
                for record in caplog.records
            )
        )

    sub.on_stop(team_id)


# --- AC7: per-team binding ---------------------------------------------------


def test_lifecycle_methods_reject_a_foreign_team_id() -> None:
    manager = _RecordingTeamManager()
    sub, _, _ = _instrumented(manager)
    wrong = uuid.uuid4()

    with pytest.raises(AssertionError):
        sub.set_restoring(wrong, True)
    with pytest.raises(AssertionError):
        sub.on_stop(wrong)

    time.sleep(0.05)
    assert manager.calls == []


# --- AC8: on_stop cancels the countdown --------------------------------------


def test_on_stop_cancels_the_countdown() -> None:
    manager = _RecordingTeamManager()
    team_id = uuid.uuid4()
    sub = IdleStopSubscriber(manager, team_id, delay=1)  # type: ignore[arg-type]

    sub.on_stop(team_id)

    # Wait well past the injected delay: the countdown must never fire.
    time.sleep(1.5)
    assert manager.calls == []


# --- AC9: the timer is armed when the subscriber is wired --------------------


def test_construction_arms_the_countdown_without_any_message() -> None:
    manager = _RecordingTeamManager()
    team_id = uuid.uuid4()

    IdleStopSubscriber(manager, team_id, delay=0)  # type: ignore[arg-type]

    # No on_message ever arrives — a team that never talks still idle-stops.
    _wait_for(lambda: manager.calls == [team_id])


# --- AC10: the delay resolves from the environment ---------------------------


def test_delay_reads_the_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(IDLE_STOP_DELAY_ENV_VAR, "42")
    sub = IdleStopSubscriber(_RecordingTeamManager(), uuid.uuid4())  # type: ignore[arg-type]
    try:
        assert sub._timer.delay == 42
    finally:
        sub._timer.cancel()


def test_delay_falls_back_to_the_module_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(IDLE_STOP_DELAY_ENV_VAR, raising=False)
    sub = IdleStopSubscriber(_RecordingTeamManager(), uuid.uuid4())  # type: ignore[arg-type]
    try:
        assert sub._timer.delay == DEFAULT_IDLE_STOP_DELAY_SECONDS
    finally:
        sub._timer.cancel()
