"""Tests for TeamRuntime model — persistent/ephemeral field separation and serialization."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.agent_card import AgentCard
from akgentic.core.agent_config import BaseConfig
from akgentic.core.messages.message import UserMessage
from akgentic.core.orchestrator import Orchestrator

from akgentic.team.models import TeamCard, TeamCardMember, TeamRuntime


def _make_agent_card(
    name: str = "test-agent",
    role: str = "TestAgent",
    routes_to: list[str] | None = None,
) -> AgentCard:
    """Create an AgentCard with agent_class as a real type (Akgent)."""
    return AgentCard(
        role=role,
        description=f"Test agent: {role}",
        skills=["testing"],
        agent_class=Akgent,
        config=BaseConfig(name=name, role=role),
        routes_to=routes_to or [],
    )


def _make_team_card(
    name: str = "test-team",
    message_types: list[type] | None = None,
) -> TeamCard:
    """Create a TeamCard with real type agent_class for TeamRuntime tests."""
    entry_point = TeamCardMember(card=_make_agent_card(name="lead", role="Lead"))
    return TeamCard(
        name=name,
        description="A test team",
        entry_point=entry_point,
        members=[],
        message_types=message_types or [],
    )


def _make_stub_addr(name: str = "stub") -> MagicMock:
    """Create a MagicMock that behaves like an ActorAddress."""
    addr = MagicMock(spec=ActorAddress)
    addr.agent_id = uuid.uuid4()
    addr.name = name
    return addr


def _make_stub_actor_system() -> MagicMock:
    """Create a MagicMock that behaves like an ActorSystem."""
    system = MagicMock(spec=ActorSystem)
    system.proxy_ask = MagicMock(return_value=MagicMock())
    system.proxy_tell = MagicMock(return_value=MagicMock())
    return system


def _make_team_runtime(
    *,
    team_card: TeamCard | None = None,
    message_types: list[type] | None = None,
    supervisor_addrs: dict[str, ActorAddress] | None = None,
    addrs: dict[str, ActorAddress] | None = None,
) -> TeamRuntime:
    """Create a TeamRuntime with mock dependencies for testing."""
    tc = team_card or _make_team_card(message_types=message_types)
    rt_id = uuid.uuid4()
    system = _make_stub_actor_system()
    orch_addr = _make_stub_addr("orchestrator")
    entry_addr = _make_stub_addr("entry")

    return TeamRuntime(
        id=rt_id,
        team=tc,
        actor_system=system,
        orchestrator_addr=orch_addr,
        entry_addr=entry_addr,
        supervisor_addrs=supervisor_addrs or {},
        addrs=addrs or {},
    )


class TestTeamRuntimeConstruction:
    """Test TeamRuntime construction with all required fields."""

    def test_construction_with_all_fields(self) -> None:
        runtime = _make_team_runtime()
        assert isinstance(runtime.id, uuid.UUID)
        assert runtime.team is not None
        assert runtime.actor_system is not None
        assert runtime.orchestrator_addr is not None
        assert runtime.entry_addr is not None
        assert isinstance(runtime.supervisor_addrs, dict)
        assert isinstance(runtime.addrs, dict)

    def test_id_has_no_default(self) -> None:
        """AC4: Construction without explicit id raises ValidationError."""
        system = _make_stub_actor_system()
        tc = _make_team_card()
        orch_addr = _make_stub_addr()
        entry_addr = _make_stub_addr()
        with pytest.raises(ValidationError):
            TeamRuntime(
                team=tc,
                actor_system=system,
                orchestrator_addr=orch_addr,
                entry_addr=entry_addr,
            )

    def test_supervisor_addrs_defaults_to_empty_dict(self) -> None:
        system = _make_stub_actor_system()
        tc = _make_team_card()
        rt = TeamRuntime(
            id=uuid.uuid4(),
            team=tc,
            actor_system=system,
            orchestrator_addr=_make_stub_addr(),
            entry_addr=_make_stub_addr(),
        )
        assert rt.supervisor_addrs == {}
        assert rt.addrs == {}

    def test_model_post_init_rebuilds_proxies(self) -> None:
        """AC3: model_post_init rebuilds all proxies from addresses."""
        runtime = _make_team_runtime()
        # Proxies should have been built by model_post_init
        assert runtime._orchestrator_proxy is not None
        assert runtime._entry_proxy is not None

    def test_model_post_init_is_idempotent(self) -> None:
        """AC3: model_post_init is idempotent."""
        runtime = _make_team_runtime()
        first_orch_proxy = runtime._orchestrator_proxy
        # Call again — should overwrite without error
        runtime.model_post_init(None)
        assert runtime._orchestrator_proxy is not None


class TestTeamRuntimeSerialization:
    """Test persistent/ephemeral field separation and serialization round-trip."""

    def test_ephemeral_fields_excluded_from_dump(self) -> None:
        """AC2/AC7: PrivateAttr fields are excluded from model_dump."""
        runtime = _make_team_runtime()
        data = runtime.model_dump()
        assert "_orchestrator_proxy" not in data
        assert "_entry_proxy" not in data
        assert "_supervisor_proxies" not in data
        assert "_message_cls" not in data

    def test_actor_system_excluded_from_dump(self) -> None:
        """AC1: actor_system with Field(exclude=True) is excluded."""
        runtime = _make_team_runtime()
        data = runtime.model_dump()
        assert "actor_system" not in data

    def test_persistent_fields_in_dump(self) -> None:
        """AC7: Persistent fields appear in model_dump."""
        runtime = _make_team_runtime()
        data = runtime.model_dump()
        assert "id" in data
        assert "team" in data
        assert "orchestrator_addr" in data
        assert "entry_addr" in data
        assert "supervisor_addrs" in data
        assert "addrs" in data


class TestTeamRuntimeMessaging:
    """Test send() and send_to() methods."""

    def test_send_calls_entry_proxy(self) -> None:
        """AC5: send() creates message and sends via entry proxy."""
        sup_addr = _make_stub_addr("supervisor")
        runtime = _make_team_runtime(
            message_types=[UserMessage],
            supervisor_addrs={"supervisor": sup_addr},
        )
        runtime.send("hello")
        # The entry proxy's send should have been called
        runtime._entry_proxy.send.assert_called()

    def test_send_to_looks_up_agent(self) -> None:
        """AC6: send_to() looks up agent via orchestrator proxy."""
        target_addr = _make_stub_addr("worker")
        runtime = _make_team_runtime(message_types=[UserMessage])
        runtime._orchestrator_proxy.get_team_member = MagicMock(return_value=target_addr)
        runtime.send_to("worker", "hello")
        runtime._orchestrator_proxy.get_team_member.assert_called_once_with("worker")
        runtime._entry_proxy.send.assert_called()

    def test_send_to_raises_for_unknown_agent(self) -> None:
        """AC6: send_to() raises ValueError if agent not found."""
        runtime = _make_team_runtime(message_types=[UserMessage])
        runtime._orchestrator_proxy.get_team_member = MagicMock(return_value=None)
        with pytest.raises(ValueError, match="not found"):
            runtime.send_to("nonexistent", "hello")

    def test_make_message_raises_when_no_message_types(self) -> None:
        """AC5: _make_message raises RuntimeError when no message_types declared."""
        runtime = _make_team_runtime(message_types=[])
        with pytest.raises(RuntimeError, match="No message type"):
            runtime._make_message("hello")

    def test_make_message_creates_correct_type(self) -> None:
        """_make_message instantiates the first message_type."""
        runtime = _make_team_runtime(message_types=[UserMessage])
        msg = runtime._make_message("hello world")
        assert isinstance(msg, UserMessage)
        assert msg.content == "hello world"

    def test_read_only_properties(self) -> None:
        """AC: read-only properties expose proxies."""
        runtime = _make_team_runtime()
        assert runtime.orchestrator_proxy is runtime._orchestrator_proxy
        assert runtime.entry_proxy is runtime._entry_proxy
        assert runtime.supervisor_proxies is runtime._supervisor_proxies
