"""Memory-release regression tests for the team lifecycle.

Drives the real ``TeamManager`` over repeated create → stop → delete cycles and
asserts that the live population of framework objects (orchestrator, agents,
state, subscribers, runtimes) returns to baseline — i.e. teardown actually
releases the team's RAM. ``TeamMemoryProbe`` names the holder when it does not.

The probe is constructed with ``tracked_names=frozenset()`` so it tracks only the
importable team-owned bases (``Akgent``/``BaseState``/``TeamRuntime``/subscribers),
not the broader name-matched set. In-process, asyncio's default-loop policy keeps
one benign thread-local reference to the last ``set_event_loop``'d loop — a bounded,
team-external residual that would otherwise produce a false positive here.
"""

from __future__ import annotations

import gc
from typing import Any

from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.messages.message import UserMessage

from akgentic.team.diagnostics import TeamMemoryProbe
from akgentic.team.manager import TeamManager
from akgentic.team.models import TeamCard, TeamCardMember
from tests.integration.conftest import (
    RecordingAgent,
    make_integration_agent_card,
    wait_for_agent_state,
)
from tests.services.conftest import InMemoryEventStore


def _simple_team_card(agent_class: type[Akgent[Any, Any]] = RecordingAgent) -> TeamCard:
    """Single entry-point agent team — the minimal lifecycle unit."""
    entry = TeamCardMember(
        card=make_integration_agent_card(name="entry", role="Entry", agent_class=agent_class),
    )
    return TeamCard(
        name="leak-probe-team",
        description="Minimal team for memory-release tests",
        entry_point=entry,
        members=[],
        message_types=[UserMessage],
    )


def _run_one_cycle(manager: TeamManager, actor_system: ActorSystem, probe: TeamMemoryProbe) -> Any:
    """Create a team, exercise it, snapshot at peak, then stop + delete.

    Returns the LeakReport. Crucially keeps NO strong handle to the runtime past
    ``mark_peak`` so the probe measures the framework's retention, not the test's.
    """
    probe.mark_baseline()
    runtime = manager.create_team(_simple_team_card())
    team_id = runtime.id

    actor_system.tell(runtime.entry_addr, UserMessage(content="ping"))
    wait_for_agent_state(
        runtime.entry_addr,
        lambda state: getattr(state, "counter", 0) >= 1,
        timeout=3.0,
    )

    probe.mark_peak()
    del runtime  # drop the only test-side strong ref before teardown

    manager.stop_team(team_id)
    manager.delete_team(team_id)
    return probe.report(label=f"team-{team_id}")


class TestTeamMemoryRelease:
    """Lifecycle teardown must release framework objects, cycle after cycle."""

    def test_single_cycle_releases_all_team_objects(
        self,
        actor_system: ActorSystem,
    ) -> None:
        """After stop + delete, no orchestrator/agent/state/subscriber survives."""
        manager = TeamManager(actor_system, InMemoryEventStore())
        probe = TeamMemoryProbe(tracked_names=frozenset())
        gc.collect()

        report = _run_one_cycle(manager, actor_system, probe)

        assert not report.leaked, report.format()

    def test_repeated_cycles_do_not_accumulate(
        self,
        actor_system: ActorSystem,
    ) -> None:
        """Live framework-object count stays flat across 5 create/stop/delete cycles.

        A genuine leak makes residual grow cycle on cycle; a clean teardown keeps
        it at zero. Auto-GC is disabled so the probe's own gc.collect() is the
        only collection — making the before/after-gc distinction meaningful.
        """
        manager = TeamManager(actor_system, InMemoryEventStore())
        probe = TeamMemoryProbe(tracked_names=frozenset())

        gc.collect()
        gc.disable()
        try:
            residuals: list[int] = []
            for _ in range(5):
                report = _run_one_cycle(manager, actor_system, probe)
                residuals.append(report.alive_after_gc)
        finally:
            gc.enable()

        assert all(r == 0 for r in residuals), (
            f"Team objects leaked across cycles (residual per cycle: {residuals}). "
            f"Last report:\n{report.format()}"
        )

    def test_probe_detects_a_deliberately_retained_runtime(
        self,
        actor_system: ActorSystem,
    ) -> None:
        """Guard the probe itself: a held runtime must be reported with its holder.

        If this fails, the probe is blind and the green lifecycle tests above are
        meaningless — so we prove a real leak is caught and the referrer named.
        """
        manager = TeamManager(actor_system, InMemoryEventStore())
        probe = TeamMemoryProbe(tracked_names=frozenset())
        gc.collect()

        probe.mark_baseline()
        runtime = manager.create_team(_simple_team_card())
        team_id = runtime.id
        probe.mark_peak()

        # Deliberately leak: stash the runtime in a container that outlives stop.
        leaked_holder = {"runtime": runtime}
        del runtime
        manager.stop_team(team_id)

        report = probe.report(label="deliberate-leak")
        try:
            assert report.leaked, "Probe failed to detect a retained runtime"
            runtime_leak = next(
                (rt for rt in report.retained if rt.type_name == "TeamRuntime"), None
            )
            assert runtime_leak is not None, report.format()
            assert any("dict" in ref.referrer_type for ref in runtime_leak.referrers), (
                f"Holder (the leaking dict) not identified:\n{report.format()}"
            )
        finally:
            leaked_holder.clear()
            manager.delete_team(team_id)
