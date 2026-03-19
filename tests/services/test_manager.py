"""Tests for TeamManager — AC 1-13."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.agent_card import AgentCard
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.messages.message import Message
from akgentic.core.orchestrator import EventSubscriber

from akgentic.team.manager import TeamManager
from akgentic.team.models import (
    Process,
    TeamCard,
    TeamCardMember,
    TeamRuntime,
    TeamStatus,
)
from akgentic.team.ports import NullServiceRegistry
from tests.services.conftest import InMemoryEventStore

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class StubAgent(Akgent[BaseConfig, BaseState]):
    """Minimal agent for manager tests."""

    pass


class FailingAgent(Akgent[BaseConfig, BaseState]):
    """Agent that raises during __init__ for rollback tests."""

    def __init__(self, **kwargs: Any) -> None:
        msg = "FailingAgent intentional error"
        raise RuntimeError(msg)


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


@pytest.fixture()
def manager(
    actor_system: ActorSystem, event_store: InMemoryEventStore
) -> TeamManager:
    """Provide a TeamManager with default NullServiceRegistry."""
    return TeamManager(actor_system=actor_system, event_store=event_store)


# ---------------------------------------------------------------------------
# Tests: create_team
# ---------------------------------------------------------------------------


class TestTeamManagerCreate:
    """AC 1-8: TeamManager.create_team creates teams via TeamFactory."""

    def test_create_team_happy_path(
        self,
        manager: TeamManager,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 2,4,5,7: create_team returns TeamRuntime and persists RUNNING Process."""
        from datetime import UTC, datetime, timedelta

        before = datetime.now(UTC)
        tc = _make_team_card()
        runtime = manager.create_team(tc, user_id="test-user", user_email="u@test.com")
        after = datetime.now(UTC)

        assert isinstance(runtime, TeamRuntime)
        assert runtime.orchestrator_addr.is_alive()

        # Process persisted with RUNNING status
        process = event_store.load_team(runtime.id)
        assert process is not None
        assert process.status == TeamStatus.RUNNING
        assert process.user_id == "test-user"
        assert process.user_email == "u@test.com"
        assert process.team_card.name == "test-team"

        # Timestamps are set to reasonable values
        assert before - timedelta(seconds=1) <= process.created_at <= after + timedelta(seconds=1)
        assert process.created_at == process.updated_at

    def test_create_team_with_subscriber_factory(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 2,3: subscriber_factory results appended after PersistenceSubscriber."""
        recording = RecordingSubscriber()

        def factory(team_id: uuid.UUID) -> list[EventSubscriber]:
            return [recording]

        mgr = TeamManager(
            actor_system=actor_system,
            event_store=event_store,
            subscriber_factory=factory,
        )
        tc = _make_team_card()
        runtime = mgr.create_team(tc)

        # Verify recording subscriber is registered by stopping orchestrator
        runtime.orchestrator_addr.stop()
        assert recording.stopped is True

    def test_create_team_rollback_on_build_failure(
        self,
        manager: TeamManager,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 8: If build fails, no Process is persisted."""
        failing = _make_member("failing", "Failing", agent_class=FailingAgent)
        tc = _make_team_card(members=[failing])

        with pytest.raises(RuntimeError, match="intentional error"):
            manager.create_team(tc)

        # No Process should be in event store
        assert len(event_store.teams) == 0

    def test_create_team_uses_pre_generated_team_id(
        self,
        manager: TeamManager,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 4: TeamManager pre-generates team_id and passes to TeamFactory.build."""
        tc = _make_team_card()
        runtime = manager.create_team(tc)

        # The team_id in Process must match the runtime id
        process = event_store.load_team(runtime.id)
        assert process is not None
        assert process.team_id == runtime.id


# ---------------------------------------------------------------------------
# Tests: get_team
# ---------------------------------------------------------------------------


class TestTeamManagerGet:
    """AC 9: TeamManager.get_team retrieves Process metadata."""

    def test_get_team_found(
        self,
        manager: TeamManager,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 9: get_team returns Process when team exists."""
        tc = _make_team_card()
        runtime = manager.create_team(tc)

        result = manager.get_team(runtime.id)
        assert result is not None
        assert result.team_id == runtime.id
        assert result.status == TeamStatus.RUNNING

    def test_get_team_not_found(self, manager: TeamManager) -> None:
        """AC 9: get_team returns None when team does not exist."""
        result = manager.get_team(uuid.uuid4())
        assert result is None


# ---------------------------------------------------------------------------
# Tests: State machine enforcement
# ---------------------------------------------------------------------------


class TestTeamManagerStateMachine:
    """AC 10-11: State machine enforcement for delete_team."""

    def test_delete_running_team_raises(
        self,
        manager: TeamManager,
    ) -> None:
        """AC 10: delete_team on RUNNING team raises ValueError."""
        tc = _make_team_card()
        runtime = manager.create_team(tc)

        with pytest.raises(ValueError, match="currently running"):
            manager.delete_team(runtime.id)

    def test_delete_stopped_team_succeeds(
        self,
        manager: TeamManager,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 11: delete_team on STOPPED team purges data."""
        tc = _make_team_card()
        runtime = manager.create_team(tc)

        # Manually transition to STOPPED
        process = event_store.load_team(runtime.id)
        assert process is not None
        stopped_process = Process(
            team_id=process.team_id,
            team_card=process.team_card,
            status=TeamStatus.STOPPED,
            user_id=process.user_id,
            user_email=process.user_email,
            created_at=process.created_at,
            updated_at=process.updated_at,
        )
        event_store.save_team(stopped_process)

        manager.delete_team(runtime.id)

        # Data should be purged
        assert event_store.load_team(runtime.id) is None

    def test_delete_nonexistent_team_raises(
        self,
        manager: TeamManager,
    ) -> None:
        """AC 11: delete_team on non-existent team raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            manager.delete_team(uuid.uuid4())

    def test_delete_already_deleted_team_raises(
        self,
        manager: TeamManager,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 11: delete_team on DELETED team raises ValueError."""
        from datetime import UTC, datetime

        team_id = uuid.uuid4()
        process = Process(
            team_id=team_id,
            team_card=_make_team_card(),
            status=TeamStatus.DELETED,
            user_id="cli",
            user_email="",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        event_store.save_team(process)

        with pytest.raises(ValueError, match="already deleted"):
            manager.delete_team(team_id)


# ---------------------------------------------------------------------------
# Tests: ServiceRegistry integration
# ---------------------------------------------------------------------------


class TestTeamManagerServiceRegistry:
    """AC 6: ServiceRegistry.register_team called on create."""

    def test_register_team_called_on_create(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 6: register_team is called with instance_id and team_id."""
        mock_registry = MagicMock(spec=NullServiceRegistry)
        instance_id = uuid.uuid4()
        mgr = TeamManager(
            actor_system=actor_system,
            event_store=event_store,
            service_registry=mock_registry,
            instance_id=instance_id,
        )
        tc = _make_team_card()
        runtime = mgr.create_team(tc)

        mock_registry.register_team.assert_called_once_with(instance_id, runtime.id)

    def test_deregister_team_called_on_delete(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """deregister_team is called with instance_id and team_id on delete."""
        from datetime import UTC, datetime

        mock_registry = MagicMock(spec=NullServiceRegistry)
        instance_id = uuid.uuid4()
        mgr = TeamManager(
            actor_system=actor_system,
            event_store=event_store,
            service_registry=mock_registry,
            instance_id=instance_id,
        )
        team_id = uuid.uuid4()
        process = Process(
            team_id=team_id,
            team_card=_make_team_card(),
            status=TeamStatus.STOPPED,
            user_id="cli",
            user_email="",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        event_store.save_team(process)

        mgr.delete_team(team_id)

        mock_registry.deregister_team.assert_called_once_with(instance_id, team_id)


# ---------------------------------------------------------------------------
# Tests: resume_team
# ---------------------------------------------------------------------------


def _create_and_stop_team(
    manager: TeamManager,
    event_store: InMemoryEventStore,
    team_card: TeamCard | None = None,
) -> uuid.UUID:
    """Create a team, populate events, stop it, and set Process to STOPPED.

    Manually injects StartMessage events for each agent so that
    TeamRestorer can rebuild the team from the event log.

    Returns the team_id.
    """
    from datetime import UTC, datetime

    from akgentic.core.actor_address_impl import ActorAddressProxy
    from akgentic.core.messages.orchestrator import StartMessage
    from akgentic.core.orchestrator import Orchestrator
    from akgentic.core.utils.deserializer import ActorAddressDict
    from akgentic.team.models import PersistedEvent

    tc = team_card or _make_team_card()
    runtime = manager.create_team(tc, user_id="test-user")

    team_id = runtime.id
    seq = 0

    # Inject Orchestrator StartMessage
    seq += 1
    orch_addr_dict: ActorAddressDict = {
        "__actor_address__": True,
        "__actor_type__": f"{Orchestrator.__module__}.{Orchestrator.__name__}",
        "agent_id": str(runtime.orchestrator_addr.agent_id),
        "name": "orchestrator",
        "role": "Orchestrator",
        "team_id": str(team_id),
        "squad_id": str(uuid.uuid4()),
        "user_message": False,
    }
    orch_start = StartMessage(
        config=BaseConfig(name="orchestrator", role="Orchestrator"),
    )
    orch_start.sender = ActorAddressProxy(orch_addr_dict)
    orch_start.team_id = team_id
    event_store.save_event(PersistedEvent(
        team_id=team_id, sequence=seq, event=orch_start, timestamp=datetime.now(UTC),
    ))

    # Inject StartMessages for all agents in the TeamCard tree
    def _inject_member(member: TeamCardMember) -> None:
        nonlocal seq
        name = member.card.config.name
        role = member.card.config.role
        agent_class = member.card.get_agent_class()
        addr = runtime.addrs.get(name)
        agent_id = addr.agent_id if addr else uuid.uuid4()
        seq += 1
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
        sm = StartMessage(config=member.card.get_config_copy())
        sm.sender = ActorAddressProxy(addr_dict)
        sm.team_id = team_id
        event_store.save_event(PersistedEvent(
            team_id=team_id, sequence=seq, event=sm, timestamp=datetime.now(UTC),
        ))
        for child in member.members:
            _inject_member(child)

    _inject_member(tc.entry_point)
    for member in tc.members:
        _inject_member(member)

    # Stop all actors
    runtime.orchestrator_addr.stop()
    for addr in runtime.addrs.values():
        if addr.is_alive():
            addr.stop()

    # Transition Process to STOPPED
    process = event_store.load_team(team_id)
    assert process is not None
    stopped_process = Process(
        team_id=process.team_id,
        team_card=process.team_card,
        status=TeamStatus.STOPPED,
        user_id=process.user_id,
        user_email=process.user_email,
        created_at=process.created_at,
        updated_at=process.updated_at,
    )
    event_store.save_team(stopped_process)

    return team_id


class TestTeamManagerResume:
    """AC 1-4, 14, 16: resume_team tests."""

    def test_resume_happy_path(
        self,
        manager: TeamManager,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 1,14: resume_team returns TeamRuntime and updates Process to RUNNING."""
        team_id = _create_and_stop_team(manager, event_store)

        runtime = manager.resume_team(team_id)

        assert isinstance(runtime, TeamRuntime)
        assert runtime.orchestrator_addr.is_alive()

        # Process should be RUNNING
        process = event_store.load_team(team_id)
        assert process is not None
        assert process.status == TeamStatus.RUNNING

    def test_resume_running_raises(
        self,
        manager: TeamManager,
    ) -> None:
        """AC 2: resume_team on RUNNING team raises ValueError."""
        tc = _make_team_card()
        runtime = manager.create_team(tc)

        with pytest.raises(ValueError, match="currently running"):
            manager.resume_team(runtime.id)

    def test_resume_deleted_raises(
        self,
        manager: TeamManager,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 3: resume_team on DELETED team raises ValueError."""
        from datetime import UTC, datetime

        team_id = uuid.uuid4()
        process = Process(
            team_id=team_id,
            team_card=_make_team_card(),
            status=TeamStatus.DELETED,
            user_id="cli",
            user_email="",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        event_store.save_team(process)

        with pytest.raises(ValueError, match="has been deleted"):
            manager.resume_team(team_id)

    def test_resume_nonexistent_raises(
        self,
        manager: TeamManager,
    ) -> None:
        """AC 4: resume_team on non-existent team raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            manager.resume_team(uuid.uuid4())

    def test_resume_registers_with_service_registry(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 14: ServiceRegistry.register_team called on successful resume."""
        mock_registry = MagicMock(spec=NullServiceRegistry)
        instance_id = uuid.uuid4()
        mgr = TeamManager(
            actor_system=actor_system,
            event_store=event_store,
            service_registry=mock_registry,
            instance_id=instance_id,
        )

        team_id = _create_and_stop_team(mgr, event_store)

        # Reset mock to clear the create_team call
        mock_registry.register_team.reset_mock()

        runtime = mgr.resume_team(team_id)

        mock_registry.register_team.assert_called_once_with(instance_id, team_id)


# ---------------------------------------------------------------------------
# Tests: delete data purge verification
# ---------------------------------------------------------------------------


class TestTeamManagerDeleteDataPurge:
    """AC 1: delete_team purges all persisted data."""

    def test_delete_purges_all_persisted_data(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """Verify Process, events, and agent states are all purged on delete."""
        from datetime import UTC, datetime

        from akgentic.core.messages.message import UserMessage
        from akgentic.team.models import AgentStateSnapshot, PersistedEvent

        manager = TeamManager(actor_system=actor_system, event_store=event_store)
        tc = _make_team_card()
        runtime = manager.create_team(tc)
        team_id = runtime.id

        # Manually populate events and agent states to verify full purge
        event_store.save_event(PersistedEvent(
            team_id=team_id,
            sequence=1,
            event=UserMessage(content="test"),
            timestamp=datetime.now(UTC),
        ))
        event_store.save_agent_state(AgentStateSnapshot(
            team_id=team_id,
            agent_id="test-agent",
            state=BaseState(),
            updated_at=datetime.now(UTC),
        ))

        # Verify all three data types exist before delete
        assert event_store.load_team(team_id) is not None
        assert len(event_store.load_events(team_id)) > 0
        assert len(event_store.load_agent_states(team_id)) > 0

        # Manually set process to STOPPED (stop_team not yet implemented)
        process = event_store.load_team(team_id)
        assert process is not None
        stopped_process = Process(
            team_id=process.team_id,
            team_card=process.team_card,
            status=TeamStatus.STOPPED,
            user_id=process.user_id,
            user_email=process.user_email,
            created_at=process.created_at,
            updated_at=datetime.now(UTC),
        )
        event_store.save_team(stopped_process)

        # Delete and verify complete purge of all three data types
        manager.delete_team(team_id)
        assert event_store.load_team(team_id) is None
        assert event_store.load_events(team_id) == []
        assert event_store.load_agent_states(team_id) == []
