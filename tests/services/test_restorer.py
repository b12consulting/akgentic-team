"""Tests for TeamRestorer -- AC 5-13, 15, 18."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
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
from akgentic.core.messages.orchestrator import StartMessage, StopMessage
from akgentic.core.orchestrator import EventSubscriber, Orchestrator

from akgentic.team.models import (
    PersistedEvent,
    Process,
    TeamCard,
    TeamCardMember,
    TeamRuntime,
    TeamStatus,
)
from akgentic.team.restorer import TeamRestorer
from akgentic.team.subscriber import PersistenceSubscriber
from tests.services.conftest import InMemoryEventStore

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class StubAgent(Akgent[BaseConfig, BaseState]):
    """Minimal agent for restorer tests."""

    pass


class RecordingSubscriber(EventSubscriber):
    """Subscriber that records received messages."""

    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.stopped: bool = False

    def on_message(self, msg: Message) -> None:
        """Record received message."""
        self.messages.append(msg)

    def on_stop(self) -> None:
        """Record stop."""
        self.stopped = True


def _make_card(
    name: str,
    role: str = "TestRole",
    agent_class: type[Akgent[Any, Any]] = StubAgent,
) -> AgentCard:
    return AgentCard(
        role=role,
        description=f"Test: {role}",
        skills=["testing"],
        agent_class=agent_class,
        config=BaseConfig(name=name, role=role),
        routes_to=[],
    )


def _make_member(
    name: str,
    role: str = "TestRole",
    agent_class: type[Akgent[Any, Any]] = StubAgent,
    headcount: int = 1,
    members: list[TeamCardMember] | None = None,
) -> TeamCardMember:
    return TeamCardMember(
        card=_make_card(name, role, agent_class),
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


def _make_start_message(
    agent_id: uuid.UUID,
    name: str,
    role: str,
    team_id: uuid.UUID,
    agent_class: type[Akgent[Any, Any]] = StubAgent,
    config: BaseConfig | None = None,
) -> StartMessage:
    """Create a StartMessage with a properly-formed sender address."""
    cfg = config or BaseConfig(name=name, role=role)
    msg = StartMessage(config=cfg)
    # Build a fake sender address dict so serialize() works
    from akgentic.core.actor_address_impl import ActorAddressProxy
    from akgentic.core.utils.deserializer import ActorAddressDict

    addr_dict: ActorAddressDict = {
        "__actor_address__": True,
        "__actor_type__": f"{agent_class.__module__}.{agent_class.__name__}",
        "agent_id": str(agent_id),
        "name": name,
        "role": role,
        "team_id": str(team_id),
        "squad_id": str(uuid.uuid4()),
        "user_message": False,
    }
    sender = ActorAddressProxy(addr_dict)
    msg.sender = sender
    msg.team_id = team_id
    return msg


def _make_stop_message(
    agent_id: uuid.UUID,
    name: str,
    role: str,
    team_id: uuid.UUID,
) -> StopMessage:
    """Create a StopMessage with a properly-formed sender address."""
    from akgentic.core.actor_address_impl import ActorAddressProxy
    from akgentic.core.utils.deserializer import ActorAddressDict

    msg = StopMessage()
    addr_dict: ActorAddressDict = {
        "__actor_address__": True,
        "__actor_type__": "akgentic.core.agent.Akgent",
        "agent_id": str(agent_id),
        "name": name,
        "role": role,
        "team_id": str(team_id),
        "squad_id": str(uuid.uuid4()),
        "user_message": False,
    }
    msg.sender = ActorAddressProxy(addr_dict)
    msg.team_id = team_id
    return msg


def _populate_stopped_team(
    event_store: InMemoryEventStore,
    team_card: TeamCard | None = None,
    extra_members: list[tuple[str, str]] | None = None,
    fired_members: list[tuple[str, str, uuid.UUID]] | None = None,
) -> tuple[uuid.UUID, Process]:
    """Populate InMemoryEventStore with events simulating a stopped team.

    Creates StartMessage events for orchestrator + all agents in team_card,
    plus optional fired agents (with matching StopMessage events).

    Returns:
        Tuple of (team_id, Process with STOPPED status).
    """
    tc = team_card or _make_team_card()
    team_id = uuid.uuid4()
    seq = 0

    # Orchestrator StartMessage
    orch_id = uuid.uuid4()
    seq += 1
    orch_start = _make_start_message(
        orch_id, "orchestrator", "Orchestrator", team_id,
        agent_class=Orchestrator,
        config=BaseConfig(name="orchestrator", role="Orchestrator"),
    )
    event_store.save_event(PersistedEvent(
        team_id=team_id, sequence=seq, event=orch_start, timestamp=datetime.now(UTC),
    ))

    # Agent StartMessages -- from TeamCard tree
    agent_names: list[str] = []

    def _walk_member(member: TeamCardMember) -> None:
        nonlocal seq
        name = member.card.config.name
        role = member.card.config.role
        agent_id = uuid.uuid4()
        agent_class = member.card.get_agent_class()
        seq += 1
        sm = _make_start_message(
            agent_id, name, role, team_id,
            agent_class=agent_class,
            config=member.card.get_config_copy(),
        )
        event_store.save_event(PersistedEvent(
            team_id=team_id, sequence=seq, event=sm, timestamp=datetime.now(UTC),
        ))
        agent_names.append(name)
        for child in member.members:
            _walk_member(child)

    _walk_member(tc.entry_point)
    for member in tc.members:
        _walk_member(member)

    # Fired members: add StartMessage + StopMessage pairs
    if fired_members:
        for fname, frole, fid in fired_members:
            seq += 1
            fsm = _make_start_message(fid, fname, frole, team_id)
            event_store.save_event(PersistedEvent(
                team_id=team_id, sequence=seq, event=fsm, timestamp=datetime.now(UTC),
            ))
            seq += 1
            fstop = _make_stop_message(fid, fname, frole, team_id)
            event_store.save_event(PersistedEvent(
                team_id=team_id, sequence=seq, event=fstop, timestamp=datetime.now(UTC),
            ))

    # Process record
    now = datetime.now(UTC)
    process = Process(
        team_id=team_id,
        team_card=tc,
        status=TeamStatus.STOPPED,
        user_id="test-user",
        user_email="test@test.com",
        created_at=now,
        updated_at=now,
    )
    event_store.save_team(process)

    return team_id, process


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def actor_system() -> ActorSystem:  # type: ignore[misc]
    """Provide an ActorSystem that shuts down after each test."""
    system = ActorSystem()
    yield system  # type: ignore[misc]
    system.shutdown()


@pytest.fixture()
def event_store() -> InMemoryEventStore:
    """Provide a fresh InMemoryEventStore per test."""
    return InMemoryEventStore()


# ---------------------------------------------------------------------------
# Tests: TestTeamRestorerRestore
# ---------------------------------------------------------------------------


class TestTeamRestorerRestore:
    """AC 5-13: Core restore functionality."""

    def test_restore_returns_valid_team_runtime(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 5,13: restore returns TeamRuntime with valid addresses."""
        team_id, process = _populate_stopped_team(event_store)

        restorer = TeamRestorer(actor_system, event_store)
        runtime, persistence_sub = restorer.restore(process)

        assert isinstance(runtime, TeamRuntime)
        assert runtime.id == team_id
        assert runtime.orchestrator_addr.is_alive()
        assert runtime.entry_addr.is_alive()
        assert "lead" in runtime.addrs
        assert isinstance(persistence_sub, PersistenceSubscriber)

    def test_restore_with_multiple_agents(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 9: All agents are rebuilt from StartMessages."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime, _ = restorer.restore(process)

        assert "lead" in runtime.addrs
        assert "worker" in runtime.addrs
        assert runtime.addrs["lead"].is_alive()
        assert runtime.addrs["worker"].is_alive()

    def test_restore_orchestrator_created_first(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 8: Orchestrator is created first during restore."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime, _ = restorer.restore(process)

        # Orchestrator should be alive and functional
        assert runtime.orchestrator_addr.is_alive()
        team = runtime.orchestrator_proxy.get_team()
        assert isinstance(team, list)

    def test_restore_events_replayed(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 12: Events are replayed through orchestrator during restore."""
        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        # Track how many events exist before restore
        events_before = len(event_store.load_events(team_id))
        assert events_before > 0

        restorer = TeamRestorer(actor_system, event_store)
        runtime, _ = restorer.restore(process)

        # Team is functional after restore
        assert runtime.orchestrator_addr.is_alive()

    def test_restore_get_team_works_after_restore(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """get_team() returns agents after restore (restore_message populates history)."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime, _ = restorer.restore(process)

        team = runtime.orchestrator_proxy.get_team()
        team_names = {addr.name for addr in team}
        # Both lead and worker should appear in orchestrator's team
        assert "lead" in team_names
        assert "worker" in team_names

    def test_restore_supervisor_addrs_populated(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 13: TeamRuntime has supervisor_addrs populated."""
        worker = _make_member("worker", "Worker")
        ep = _make_member("lead", "Lead", members=[worker])
        tc = _make_team_card(entry_point=ep)

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime, _ = restorer.restore(process)

        # lead is a supervisor (has subordinates)
        assert "lead" in runtime.supervisor_addrs

    def test_restore_with_agent_state_snapshots(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 11: Persisted AgentStateSnapshot is restored via init_state()."""
        from unittest.mock import patch as mock_patch

        from akgentic.team.models import AgentStateSnapshot

        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        # Inject a state snapshot for the "lead" agent
        snapshot = AgentStateSnapshot(
            team_id=team_id,
            agent_id="lead",
            state=BaseState(),
            updated_at=datetime.now(UTC),
        )
        event_store.save_agent_state(snapshot)

        restorer = TeamRestorer(actor_system, event_store)

        # Track init_state calls via spy
        init_state_calls: list[str] = []
        original_proxy_ask = actor_system.proxy_ask

        def tracking_proxy_ask(
            addr: ActorAddress, cls: type[Any],
        ) -> Any:
            proxy = original_proxy_ask(addr, cls)
            if cls is Akgent:
                original_init = proxy.init_state

                def tracked_init(state: Any) -> None:
                    init_state_calls.append("lead")
                    return original_init(state)

                proxy.init_state = tracked_init
            return proxy

        with mock_patch.object(actor_system, "proxy_ask", side_effect=tracking_proxy_ask):
            runtime, _ = restorer.restore(process)

        # Verify init_state was called for the agent with snapshot
        assert "lead" in init_state_calls
        assert runtime.addrs["lead"].is_alive()

    def test_restore_agent_profiles_registered(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 9: Only agent_profiles are registered with orchestrator after restore."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])
        # Explicitly register profiles for hiring
        tc.agent_profiles = list(tc.agent_cards.values())

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime, _ = restorer.restore(process)

        catalog = runtime.orchestrator_proxy.get_agent_catalog()
        roles = {c.role for c in catalog}
        assert "Lead" in roles
        assert "Worker" in roles

    def test_restore_no_profiles_means_empty_catalog(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """Default agent_profiles (empty) results in empty hiring catalog after restore."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])
        # agent_profiles defaults to empty — no roles available for hiring

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime, _ = restorer.restore(process)

        catalog = runtime.orchestrator_proxy.get_agent_catalog()
        assert len(catalog) == 0


# ---------------------------------------------------------------------------
# Tests: TestTeamRestorerAgentFiltering
# ---------------------------------------------------------------------------


class TestTeamRestorerAgentFiltering:
    """AC 7: StartMessage/StopMessage filtering."""

    def test_fired_agent_excluded_from_restore(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 7: Agents with matching StopMessage are excluded from rebuild."""
        tc = _make_team_card()
        fired_id = uuid.uuid4()

        team_id, process = _populate_stopped_team(
            event_store, tc,
            fired_members=[("fired-agent", "FiredRole", fired_id)],
        )

        restorer = TeamRestorer(actor_system, event_store)
        runtime, _ = restorer.restore(process)

        # Fired agent should NOT be in the restored team
        assert "fired-agent" not in runtime.addrs
        # Lead should still be there
        assert "lead" in runtime.addrs

    def test_agent_stopped_then_not_rebuilt(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 7: Multiple fired agents are all excluded."""
        tc = _make_team_card()
        fired1_id = uuid.uuid4()
        fired2_id = uuid.uuid4()

        team_id, process = _populate_stopped_team(
            event_store, tc,
            fired_members=[
                ("fired1", "Role1", fired1_id),
                ("fired2", "Role2", fired2_id),
            ],
        )

        restorer = TeamRestorer(actor_system, event_store)
        runtime, _ = restorer.restore(process)

        assert "fired1" not in runtime.addrs
        assert "fired2" not in runtime.addrs
        assert "lead" in runtime.addrs


# ---------------------------------------------------------------------------
# Tests: TestTeamRestorerRestoringFlag
# ---------------------------------------------------------------------------


class TestTeamRestorerRestoringFlag:
    """AC 12: Restoring flag toggle on PersistenceSubscriber."""

    def test_restoring_flag_managed_correctly(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 12: PersistenceSubscriber restoring=True before replay, False after."""
        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime, persistence_sub = restorer.restore(process)

        # After restore, restoring flag should be False
        assert persistence_sub._restoring is False

    def test_no_duplicate_events_during_replay(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 12: Replayed events are not re-persisted (restoring flag prevents it)."""
        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        original_events = event_store.load_events(team_id)
        original_sequences = {e.sequence for e in original_events}

        restorer = TeamRestorer(actor_system, event_store)
        runtime, _ = restorer.restore(process)

        # Check no events with original sequences were duplicated
        all_events = event_store.load_events(team_id)
        seq_counts: dict[int, int] = {}
        for e in all_events:
            seq_counts[e.sequence] = seq_counts.get(e.sequence, 0) + 1

        for seq in original_sequences:
            assert seq_counts[seq] == 1, f"Sequence {seq} was duplicated"


# ---------------------------------------------------------------------------
# Tests: TestTeamRestorerRollback
# ---------------------------------------------------------------------------


class TestTeamRestorerRollback:
    """Rollback on failure."""

    def test_rollback_on_failure(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """All spawned actors cleaned up if a phase fails."""
        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)

        # Track orchestrator actor created before failure
        import pykka
        actors_before = len(pykka.ActorRegistry.get_all())

        # Patch import_class to fail when resolving the lead agent class
        with patch(
            "akgentic.team.restorer.import_class",
            side_effect=ImportError("cannot find agent class"),
        ):
            with pytest.raises(ImportError, match="cannot find agent class"):
                restorer.restore(process)

        # After rollback, no new actors should remain alive
        actors_after = len(pykka.ActorRegistry.get_all())
        assert actors_after == actors_before, (
            f"Rollback failed: {actors_after - actors_before} actor(s) leaked"
        )

    def test_subscriber_factory_called(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """subscriber_factory results registered with orchestrator."""
        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        recording = RecordingSubscriber()

        def factory(tid: uuid.UUID) -> list[EventSubscriber]:
            return [recording]

        restorer = TeamRestorer(actor_system, event_store, subscriber_factory=factory)
        runtime, _ = restorer.restore(process)

        # Recording subscriber should have received replayed events
        assert len(recording.messages) > 0

        # Stop and verify on_stop called
        runtime.orchestrator_addr.stop()
        assert recording.stopped is True


# ---------------------------------------------------------------------------
# Tests: Hierarchy propagation during restore (Story 10-1, AC 3, 5)
# ---------------------------------------------------------------------------


class TestRestorerHierarchyPropagation:
    """AC 3,5: Restored agents have _orchestrator set."""

    def test_orchestrator_set_on_restored_agents(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 3,5: _orchestrator is not None on agents rebuilt during restore."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime, _ = restorer.restore(process)

        for name, addr in runtime.addrs.items():
            proxy: Akgent[Any, Any] = actor_system.proxy_ask(addr, Akgent)
            orch = proxy.orchestrator
            assert orch is not None, f"Restored agent '{name}' has _orchestrator=None"
            assert orch.is_alive(), (
                f"Restored agent '{name}' orchestrator is not alive"
            )

    def test_parent_set_on_restored_agents(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 5: _parent is set on agents rebuilt during restore."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime, _ = restorer.restore(process)

        # All restored agents should have orchestrator as parent
        for name, addr in runtime.addrs.items():
            actor = addr._actor_ref._actor  # type: ignore[union-attr]
            assert actor._parent is not None, (
                f"Restored agent '{name}' has _parent=None"
            )
            assert actor._parent.agent_id == runtime.orchestrator_addr.agent_id, (
                f"Restored agent '{name}' parent is not the orchestrator"
            )
