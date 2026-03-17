"""Tests for TeamFactory.build — AC 1-10."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import patch

import pytest
from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.agent_card import AgentCard
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.messages.message import Message

from akgentic.team.factory import TeamFactory
from akgentic.team.models import TeamCard, TeamCardMember, TeamRuntime

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class StubAgent(Akgent[BaseConfig, BaseState]):
    """Minimal agent for factory tests."""

    pass


class FailingAgent(Akgent[BaseConfig, BaseState]):
    """Agent that raises during __init__ for rollback tests."""

    def __init__(self, **kwargs: Any) -> None:
        msg = "FailingAgent intentional error"
        raise RuntimeError(msg)


class StubSubscriber:
    """Minimal EventSubscriber for testing."""

    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.stopped: bool = False

    def on_stop(self) -> None:
        self.stopped = True

    def on_message(self, msg: Message) -> None:
        self.messages.append(msg)


def _make_card(
    name: str,
    role: str = "TestRole",
    agent_class: type[Akgent[Any, Any]] = StubAgent,
    routes_to: list[str] | None = None,
) -> AgentCard:
    return AgentCard(
        role=role,
        description=f"Test: {role}",
        skills=["testing"],
        agent_class=agent_class,
        config=BaseConfig(name=name, role=role),
        routes_to=routes_to or [],
    )


def _make_member(
    name: str,
    role: str = "TestRole",
    agent_class: type[Akgent[Any, Any]] = StubAgent,
    headcount: int = 1,
    members: list[TeamCardMember] | None = None,
    routes_to: list[str] | None = None,
) -> TeamCardMember:
    return TeamCardMember(
        card=_make_card(name, role, agent_class, routes_to),
        headcount=headcount,
        members=members or [],
    )


def _make_team_card(
    entry_point: TeamCardMember | None = None,
    members: list[TeamCardMember] | None = None,
    name: str = "test-team",
) -> TeamCard:
    ep = entry_point or _make_member("lead", "Lead")
    return TeamCard(
        name=name,
        description="Test team",
        entry_point=ep,
        members=members or [],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def actor_system() -> ActorSystem:  # type: ignore[misc]
    system = ActorSystem()
    yield system  # type: ignore[misc]
    system.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTeamFactoryBuild:
    """AC 1-8: TeamFactory.build creates a running team from a TeamCard."""

    # -- 3.1: Successful build with single agent -------------------------

    def test_build_single_agent(self, actor_system: ActorSystem) -> None:
        """AC 1,7: Build returns TeamRuntime with orchestrator, entry, addrs."""
        tc = _make_team_card()
        runtime = TeamFactory.build(tc, actor_system)

        assert isinstance(runtime, TeamRuntime)
        assert isinstance(runtime.id, uuid.UUID)
        assert runtime.orchestrator_addr is not None
        assert runtime.entry_addr is not None
        assert "lead" in runtime.addrs
        assert runtime.orchestrator_addr.is_alive()
        assert runtime.entry_addr.is_alive()

    # -- 3.2: Successful build with multiple agents (hierarchical) -------

    def test_build_hierarchical_team(self, actor_system: ActorSystem) -> None:
        """AC 2: All agents in TeamCard member tree are spawned."""
        worker = _make_member("worker", "Worker")
        supervisor = _make_member("supervisor", "Supervisor", members=[worker])
        tc = _make_team_card(members=[supervisor])

        runtime = TeamFactory.build(tc, actor_system)

        assert "lead" in runtime.addrs
        assert "supervisor" in runtime.addrs
        assert "worker" in runtime.addrs
        assert len(runtime.addrs) == 3
        for addr in runtime.addrs.values():
            assert addr.is_alive()

    # -- 3.3: Headcount > 1 spawns multiple instances --------------------

    def test_headcount_multiple_instances(self, actor_system: ActorSystem) -> None:
        """AC 3: Headcount > 1 spawns multiple actor instances."""
        multi = _make_member("worker", "Worker", headcount=3)
        tc = _make_team_card(members=[multi])

        runtime = TeamFactory.build(tc, actor_system)

        assert "worker_0" in runtime.addrs
        assert "worker_1" in runtime.addrs
        assert "worker_2" in runtime.addrs
        assert "worker" not in runtime.addrs
        # lead + 3 workers = 4
        assert len(runtime.addrs) == 4

    # -- 3.4: Subscriber registration -----------------------------------

    def test_subscriber_registration(self, actor_system: ActorSystem) -> None:
        """AC 5: Provided subscribers are registered with the orchestrator."""
        sub = StubSubscriber()
        tc = _make_team_card()

        runtime = TeamFactory.build(tc, actor_system, subscribers=[sub])

        # Verify subscriber is registered by stopping the orchestrator,
        # which calls on_stop on all subscribers.
        runtime.orchestrator_addr.stop()
        assert sub.stopped is True

    # -- 3.5: routes_to wiring ------------------------------------------

    def test_routes_to_registered(self, actor_system: ActorSystem) -> None:
        """AC 4,6: Agent profiles with routes_to are registered with orchestrator."""
        worker = _make_member("worker", "Worker")
        ep = _make_member("lead", "Lead", routes_to=["Worker"])
        tc = _make_team_card(entry_point=ep, members=[worker])

        runtime = TeamFactory.build(tc, actor_system)

        # Verify agent profiles are registered
        catalog = runtime.orchestrator_proxy.get_agent_catalog()
        roles = {c.role for c in catalog}
        assert "Lead" in roles
        assert "Worker" in roles

        # Verify routes_to is preserved
        lead_card = next(c for c in catalog if c.role == "Lead")
        assert "Worker" in lead_card.routes_to

    # -- 3.6: Partial failure rollback -----------------------------------

    def test_partial_failure_rollback(self, actor_system: ActorSystem) -> None:
        """AC 8: Partial failure tears down already-spawned actors."""
        # FailingAgent will raise on start -- this should trigger rollback
        failing = _make_member("failing", "Failing", agent_class=FailingAgent)
        tc = _make_team_card(members=[failing])

        with pytest.raises(Exception):
            TeamFactory.build(tc, actor_system)

    # -- 3.7: TeamRuntime.id equals orchestrator's team_id ---------------

    def test_runtime_id_is_team_id(self, actor_system: ActorSystem) -> None:
        """AC 7: TeamRuntime.id is the team_id assigned to all actors."""
        tc = _make_team_card()
        runtime = TeamFactory.build(tc, actor_system)

        # The orchestrator's team_id should match runtime.id
        assert runtime.orchestrator_addr.team_id == runtime.id
        assert runtime.entry_addr.team_id == runtime.id

    # -- 3.8: Agent profiles registered with orchestrator ----------------

    def test_agent_profiles_registered(self, actor_system: ActorSystem) -> None:
        """AC 6: orchestrator.get_agent_catalog() returns all agent cards."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])

        runtime = TeamFactory.build(tc, actor_system)

        catalog = runtime.orchestrator_proxy.get_agent_catalog()
        roles = {c.role for c in catalog}
        assert "Lead" in roles
        assert "Worker" in roles
        assert len(catalog) == 2

    # -- Additional edge-case tests --------------------------------------

    def test_build_with_no_subscribers(self, actor_system: ActorSystem) -> None:
        """Build works when subscribers is None."""
        tc = _make_team_card()
        runtime = TeamFactory.build(tc, actor_system, subscribers=None)
        assert runtime.orchestrator_addr.is_alive()

    def test_supervisor_addrs_populated(self, actor_system: ActorSystem) -> None:
        """Supervisor addresses are populated for members with subordinates."""
        worker = _make_member("worker", "Worker")
        ep = _make_member("lead", "Lead", members=[worker])
        tc = _make_team_card(entry_point=ep)

        runtime = TeamFactory.build(tc, actor_system)

        # lead has subordinates, so it's a supervisor
        assert "lead" in runtime.supervisor_addrs
        assert runtime.supervisor_addrs["lead"] == runtime.addrs["lead"]

    def test_entry_point_headcount_gt1_raises(self, actor_system: ActorSystem) -> None:
        """Entry point with headcount > 1 raises ValueError."""
        ep = _make_member("lead", "Lead", headcount=2)
        tc = _make_team_card(entry_point=ep)

        with pytest.raises(ValueError, match="must have headcount=1"):
            TeamFactory.build(tc, actor_system)

    def test_rollback_handles_stop_failure(self, actor_system: ActorSystem) -> None:
        """Rollback continues even if stopping an actor raises."""
        failing = _make_member("failing", "Failing", agent_class=FailingAgent)
        tc = _make_team_card(members=[failing])

        with patch.object(ActorAddress, "stop", side_effect=RuntimeError("stop failed")):
            with pytest.raises(RuntimeError, match="intentional error"):
                TeamFactory.build(tc, actor_system)
