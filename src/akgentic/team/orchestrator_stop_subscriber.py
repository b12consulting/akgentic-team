"""Timer-stop bridge subscriber for TeamManager.

Bridges the orchestrator's inactivity-timer ``on_stop`` hook (fired
from ``akgentic.core.orchestrator.Orchestrator.on_stop``) into
``TeamManager.stop_team`` so the full team-lifecycle teardown runs —
``Process.status=STOPPED`` is persisted, the team is deregistered
from the service registry, and runtime tracking is cleaned up.

Without this bridge, a team that stops via the core inactivity timer
leaves a ghost ``Process.status=RUNNING`` entry in the event store:
``Orchestrator.on_stop`` only notifies its subscribers, it never
reaches into the team-state store.

The subscriber is auto-attached by ``TeamManager.create_team`` and
``TeamManager.resume_team`` so every consumer of ``TeamManager``
gets the behaviour for free.

Boundary: depends only on ``akgentic-core`` (``EventSubscriber``
Protocol) and ``akgentic-team`` (``TeamManager``). The ``TeamManager``
import is guarded by :data:`typing.TYPE_CHECKING` to avoid a circular
import at runtime.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import TYPE_CHECKING

from akgentic.core.messages.message import Message

if TYPE_CHECKING:
    from akgentic.team.manager import TeamManager

logger = logging.getLogger(__name__)


class OrchestratorStopSubscriber:
    """``EventSubscriber`` that drains an orchestrator timer-stop into stop_team.

    When ``Orchestrator.on_stop`` fires (e.g. after the inactivity
    timer expires), this subscriber calls
    :meth:`TeamManager.stop_team` so the full shutdown path runs:
    ``Process.status=STOPPED`` is persisted and the team is
    deregistered from ``ServiceRegistry``. Without this bridge only
    the actor tree is torn down; the team-state record remains
    ``RUNNING`` indefinitely.

    Idempotent: if ``stop_team`` raises :class:`ValueError` because
    the team is already ``STOPPED`` or ``DELETED``, the error is
    swallowed and logged at DEBUG. A repeat ``on_stop`` after a
    completed teardown is therefore a safe no-op.
    """

    def __init__(self, team_manager: TeamManager, team_id: uuid.UUID) -> None:
        self._team_manager = team_manager
        self._team_id = team_id

    def set_restoring(self, restoring: bool) -> None:  # noqa: FBT001
        """No-op: timer-stop bridging is orthogonal to replay guarding."""
        del restoring

    def on_stop(self) -> None:
        """Drain the orchestrator stop into TeamManager.stop_team (async).

        The orchestrator calls this inside its own ``on_stop`` on the
        actor thread. Calling ``TeamManager.stop_team`` synchronously
        here would deadlock: ``stop_team`` → ``_teardown_team`` issues
        ``proxy_ask`` on the same orchestrator, which is already inside
        ``on_stop`` and cannot service the request. The work is therefore
        offloaded to a daemon thread so ``on_stop`` returns immediately
        and the actor thread can finish its own stop; by the time the
        daemon's ``stop_team`` reaches ``_teardown_team``, the
        orchestrator has already stopped and ``stop_team``'s subsequent
        state-store writes land cleanly.
        """
        thread = threading.Thread(
            target=self._drain_to_stop_team,
            name=f"orchestrator-stop-subscriber-{self._team_id}",
            daemon=True,
        )
        thread.start()

    def _drain_to_stop_team(self) -> None:
        """Daemon-thread body: call ``stop_team`` once, swallow idempotent errors."""
        try:
            self._team_manager.stop_team(self._team_id)
        except ValueError as exc:
            logger.debug(
                "OrchestratorStopSubscriber idempotent no-op team_id=%s err=%s",
                self._team_id,
                exc,
            )
        except Exception:
            logger.warning(
                "OrchestratorStopSubscriber.stop_team failed team_id=%s",
                self._team_id,
                exc_info=True,
            )

    def on_message(self, msg: Message) -> None:
        """No-op: this subscriber only reacts to orchestrator stop."""
        del msg


__all__ = ["OrchestratorStopSubscriber"]
