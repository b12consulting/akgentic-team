"""Per-team ``EventSubscriber`` implementations: event persistence and idle-stop."""

from __future__ import annotations

import logging
import os
import threading
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from akgentic.core.messages.message import Message
from akgentic.core.messages.orchestrator import (
    ProcessedMessage,
    ReceivedMessage,
    StateChangedMessage,
)
from akgentic.core.orchestrator import EventSubscriber
from akgentic.core.utils.timer import Timer
from akgentic.team.models import AgentStateSnapshot, PersistedEvent
from akgentic.team.ports import EventStore

if TYPE_CHECKING:
    from akgentic.team.manager import TeamManager

logger = logging.getLogger(__name__)

# Seconds of inactivity before an idle team is stopped, when the environment
# says nothing. Historical value: the implicit one-hour auto-stop the
# Orchestrator used to apply before the clock moved into this package.
DEFAULT_IDLE_STOP_DELAY_SECONDS = 3600

# The knob keeps the name it had while core owned the clock; only the read site
# moved here. Deployments already set it.
IDLE_STOP_DELAY_ENV_VAR = "ORCHESTRATOR_TIMEOUT_DELAY"


def resolve_idle_stop_delay() -> int:
    """Read the inactivity delay from the environment.

    Returns:
        The value of ``ORCHESTRATOR_TIMEOUT_DELAY`` in seconds, or
        :data:`DEFAULT_IDLE_STOP_DELAY_SECONDS` when the variable is unset.

    Raises:
        ValueError: The variable is set to something non-numeric. This mirrors
            the bare ``int()`` read it replaces — a malformed value is a
            deployment error and fails loudly at wiring time.
    """
    raw = os.environ.get(IDLE_STOP_DELAY_ENV_VAR)
    if raw is None:
        return DEFAULT_IDLE_STOP_DELAY_SECONDS
    return int(raw)


class PersistenceSubscriber(EventSubscriber):
    """Bridges EventSubscriber (akgentic-core) with EventStore (akgentic-team).

    Receives all messages from the orchestrator and persists:
    1. Every message as an append-only event (for replay on resume)
    2. Agent state snapshots on StateChangedMessage (for fast restore)

    LLM conversation history is NOT persisted separately -- it is
    reconstructed from the event log during restore.

    All types used here come from akgentic-core (Message, StateChangedMessage,
    BaseState) -- no imports from akgentic-llm or akgentic-agent needed.
    """

    def __init__(
        self,
        team_id: uuid.UUID,
        event_store: EventStore,
        initial_sequence: int = 0,
    ) -> None:
        """Initialize the persistence subscriber.

        Args:
            team_id: Unique identifier of the team whose events are persisted.
            event_store: Storage backend for events and agent state snapshots.
            initial_sequence: Starting sequence number. Use 0 (default) for new
                teams. For resumed teams, pass the max existing sequence so that
                new events continue monotonically without duplicating numbers.
        """
        self._team_id = team_id
        self._event_store = event_store
        self._sequence = initial_sequence
        self._restoring = False

    def on_message(self, msg: Message) -> None:
        """Persist a message as a PersistedEvent and optionally save agent state.

        If the restoring flag is True, skips all persistence to avoid duplicate
        writes during event replay.

        Args:
            msg: The message flowing through the orchestrator.
        """
        if self._restoring:
            return

        now = datetime.now(UTC)

        if isinstance(msg, StateChangedMessage) and msg.sender is not None:
            snapshot = AgentStateSnapshot(
                team_id=self._team_id,
                agent_id=str(msg.sender.agent_id),
                name=msg.sender.name,
                state=msg.state.serializable_copy(),
                updated_at=now,
            )
            self._event_store.save_agent_state(snapshot)
        else:
            self._sequence += 1
            event = PersistedEvent(
                team_id=self._team_id,
                sequence=self._sequence,
                event=msg,
                timestamp=now,
            )
            self._event_store.save_event(event)

    def set_restoring(self, team_id: uuid.UUID, restoring: bool) -> None:  # noqa: FBT001
        """Set the restoring flag to skip or resume persistence.

        Args:
            team_id: ``team_id`` of the orchestrator dispatching this lifecycle
                event. MUST match ``self._team_id`` — the per-team subscriber
                is bound at construction and never accepts traffic from a
                foreign orchestrator.
            restoring: If True, on_message will skip all persistence.
        """
        assert team_id == self._team_id
        self._restoring = restoring

    def on_stop(self, team_id: uuid.UUID) -> None:
        """No-op: required by EventSubscriber protocol.

        Args:
            team_id: ``team_id`` of the orchestrator dispatching this lifecycle
                event. MUST match ``self._team_id``.
        """
        assert team_id == self._team_id


class IdleStopSubscriber(EventSubscriber):
    """``EventSubscriber`` that detects an idle team *and* stops it.

    Counts in-flight work from the telemetry it already receives —
    ``ReceivedMessage`` starts a task, ``ProcessedMessage`` completes one —
    and drives a :class:`akgentic.core.utils.timer.Timer` with those two
    pokes. When the count stays at zero for ``delay`` seconds the countdown
    fires and the team is stopped through ``TeamManager.stop_team``. The
    countdown is armed at construction, so a team that never receives a
    single message still stops.

    Three threads meet here, which is what shapes the code:

    - ``on_message`` arrives on the orchestrator's actor thread. It must
      return immediately and touch neither the ``TeamManager`` nor the actor
      system.
    - The timeout callback fires on the ``Timer`` thread, which has no
      supervisor — an exception escaping it is silent, so everything that can
      raise lives inside ``_drain_to_stop_team``'s ``try``.
    - ``stop_team`` runs on a third, daemon thread. Calling it inline from the
      callback deadlocks: it reaches ``_teardown_team``, which issues a
      ``proxy_ask`` into the orchestrator and waits on the answer.

    ``set_restoring`` suppresses the counting, not the countdown. A resume
    replays a burst of ``ReceivedMessage``/``ProcessedMessage``; without the
    guard those drive the clock and a team can idle-stop moments after coming
    back, or mid-restore.

    Idempotent: if ``stop_team`` raises :class:`ValueError` because the team is
    already ``STOPPED`` or ``DELETED``, the error is swallowed and logged at
    DEBUG.
    """

    def __init__(
        self,
        team_manager: TeamManager,
        team_id: uuid.UUID,
        delay: int | None = None,
    ) -> None:
        """Bind to a team and arm the inactivity countdown.

        Args:
            team_manager: Owner of ``stop_team``, called when the team goes idle.
            team_id: Team this subscriber is bound to. Lifecycle callbacks
                carrying any other ``team_id`` are rejected.
            delay: Seconds of inactivity before the team is stopped. When
                ``None``, resolved from the environment via
                :func:`resolve_idle_stop_delay`.
        """
        self._team_manager = team_manager
        self._team_id = team_id
        self._restoring = False
        self._timer = Timer(
            resolve_idle_stop_delay() if delay is None else delay,
            timeout_callback=self._on_idle,
        )
        self._timer.start()

    def set_restoring(self, team_id: uuid.UUID, restoring: bool) -> None:  # noqa: FBT001
        """Toggle the replay guard.

        Only flips the flag: the countdown keeps running during a restore,
        exactly as it did when the orchestrator owned the clock. What the flag
        gates is the task counting in ``on_message``.

        Args:
            team_id: ``team_id`` of the orchestrator dispatching this lifecycle
                event. MUST match ``self._team_id``.
            restoring: ``True`` while replayed events are being delivered.
        """
        assert team_id == self._team_id
        self._restoring = restoring

    def on_message(self, msg: Message) -> None:
        """Drive the countdown from the two telemetry signals that bound work.

        Args:
            msg: The message flowing through the orchestrator. Anything other
                than ``ReceivedMessage``/``ProcessedMessage`` leaves the timer
                untouched.
        """
        if self._restoring:
            return
        if isinstance(msg, ReceivedMessage):
            self._timer.task_started()
        elif isinstance(msg, ProcessedMessage):
            self._timer.task_completed()

    def on_stop(self, team_id: uuid.UUID) -> None:
        """Cancel the countdown — the team is stopping, idle or not.

        Args:
            team_id: ``team_id`` of the orchestrator dispatching this lifecycle
                event. MUST match ``self._team_id``.
        """
        assert team_id == self._team_id
        self._timer.cancel()

    def _on_idle(self) -> None:
        """Timeout callback: hand the stop to a daemon thread and return.

        Runs on the ``Timer`` thread. Doing the stop here would deadlock the
        orchestrator, so this method does nothing but dispatch.
        """
        thread = threading.Thread(
            target=self._drain_to_stop_team,
            name=f"idle-stop-subscriber-{self._team_id}",
            daemon=True,
        )
        thread.start()

    def _drain_to_stop_team(self) -> None:
        """Daemon-thread body: call ``stop_team`` once, swallow idempotent errors."""
        try:
            self._team_manager.stop_team(self._team_id)
        except ValueError as exc:
            logger.debug(
                "IdleStopSubscriber idempotent no-op team_id=%s err=%s",
                self._team_id,
                exc,
            )
        except Exception:
            logger.warning(
                "IdleStopSubscriber.stop_team failed team_id=%s",
                self._team_id,
                exc_info=True,
            )
