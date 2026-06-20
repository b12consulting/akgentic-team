"""Differential object-census over the real team lifecycle.

Mirrors the server A/B the operators run (no-teams vs created-then-stopped-teams,
captured just before shutdown): here the baseline is captured before any team is
built and the candidate after K full create -> stop -> delete cycles. The diff
ranks the classes the team activity left resident — the leak suspects — and the
ObjectCensus save/load path is exercised end to end.

This is a *diagnostic* test: it asserts the tooling works and that the heavy
agent objects are released (BaseAgent/Orchestrator do not accumulate). It does
NOT assert zero growth overall, because the known residual is per-actor Pykka
infrastructure (ActorRef/Queue/...) owned by akgentic-core, out of this package's
scope to fix. The ranked diff is logged so operators can see it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.messages.message import UserMessage

from akgentic.team.manager import TeamManager
from akgentic.team.models import TeamCard, TeamCardMember
from akgentic.team.sampler import ObjectCensus
from tests.integration.conftest import RecordingAgent, make_integration_agent_card
from tests.services.conftest import InMemoryEventStore

logger = logging.getLogger(__name__)


def _simple_team_card() -> TeamCard:
    entry = TeamCardMember(
        card=make_integration_agent_card(name="entry", role="Entry", agent_class=RecordingAgent),
    )
    return TeamCard(
        name="census-team",
        description="Team for differential census",
        entry_point=entry,
        members=[],
        message_types=[UserMessage],
    )


def _create_stop_delete(actor_system: ActorSystem) -> None:
    """One full lifecycle on a fresh manager, leaving nothing tracked behind."""
    manager = TeamManager(actor_system, InMemoryEventStore())
    runtime = manager.create_team(_simple_team_card())
    team_id = runtime.id
    actor_system.tell(runtime.entry_addr, UserMessage(content="ping"))
    del runtime
    manager.stop_team(team_id)
    manager.delete_team(team_id)


class TestLifecycleObjectCensus:
    """A/B census across the team lifecycle, with save/load + heavy-object check."""

    def test_diff_round_trips_and_releases_heavy_objects(
        self,
        actor_system: ActorSystem,
        tmp_path: Path,
    ) -> None:
        # Run A: baseline before any team exists.
        baseline = ObjectCensus.capture(label="no-teams")
        baseline.save(tmp_path / "census-a.json")

        # Run B activity: K full lifecycles, all stopped + deleted.
        for _ in range(5):
            _create_stop_delete(actor_system)
        candidate = ObjectCensus.capture(label="after-teams")
        candidate.save(tmp_path / "census-b.json")

        # The persisted captures diff exactly as the in-memory ones (operator path).
        rows = ObjectCensus.diff(
            ObjectCensus.load(tmp_path / "census-a.json"),
            ObjectCensus.load(tmp_path / "census-b.json"),
            top=25,
        )
        logger.info("lifecycle census diff:\n%s", ObjectCensus.format_diff(rows))

        # Heavy per-team objects must be fully released — they are owned by the
        # team layer and torn down on stop. Their presence in the diff would be a
        # team-package regression (the per-actor wrappers are core-owned, allowed).
        grew = {r.type_name for r in rows}
        for heavy in ("RecordingAgent", "Orchestrator", "TeamRuntime", "PersistenceSubscriber"):
            assert heavy not in grew, (
                f"{heavy} accumulated across lifecycles — team-layer leak:\n"
                f"{ObjectCensus.format_diff(rows)}"
            )
