"""Tests for TeamRuntime model — persistent/ephemeral field separation and serialization."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest
from akgentic.core.agent import Akgent
from akgentic.core.messages.message import UserMessage
from pydantic import ValidationError

from akgentic.team.models import TeamCard, TeamCardMember, TeamRuntime
from tests.models.conftest import (
    make_agent_card,
    make_stub_actor_system,
    make_stub_addr,
    make_team_card,
    make_team_runtime,
)


class TestTeamRuntimeConstruction:
    """AC 1-4: TeamRuntime construction with persistent and ephemeral fields."""

    def test_construction_with_all_fields(self) -> None:
        runtime = make_team_runtime()
        assert isinstance(runtime.id, uuid.UUID)
        assert runtime.team is not None
        assert runtime.actor_system is not None
        assert runtime.orchestrator_addr is not None
        assert runtime.entry_addr is not None
        assert isinstance(runtime.supervisor_addrs, dict)
        assert isinstance(runtime.addrs, dict)

    def test_id_has_no_default(self) -> None:
        """AC4: Construction without explicit id raises ValidationError."""
        system = make_stub_actor_system()
        tc = make_team_card(agent_class=Akgent)
        with pytest.raises(ValidationError):
            TeamRuntime(
                team=tc,
                actor_system=system,
                orchestrator_addr=make_stub_addr(),
                entry_addr=make_stub_addr(),
            )

    def test_supervisor_addrs_defaults_to_empty_dict(self) -> None:
        system = make_stub_actor_system()
        tc = make_team_card(agent_class=Akgent)
        rt = TeamRuntime(
            id=uuid.uuid4(),
            team=tc,
            actor_system=system,
            orchestrator_addr=make_stub_addr(),
            entry_addr=make_stub_addr(),
        )
        assert rt.supervisor_addrs == {}
        assert rt.addrs == {}

    def test_model_post_init_rebuilds_proxies(self) -> None:
        """AC3: model_post_init rebuilds all proxies from addresses."""
        runtime = make_team_runtime()
        assert runtime._orchestrator_proxy is not None
        assert runtime._entry_proxy is not None

    def test_model_post_init_is_idempotent(self) -> None:
        """AC3: model_post_init is idempotent — safe to call multiple times."""
        runtime = make_team_runtime()
        initial_call_count = runtime.actor_system.proxy_ask.call_count
        runtime.model_post_init(None)
        # proxy_ask was called again, confirming proxies were rebuilt
        assert runtime.actor_system.proxy_ask.call_count > initial_call_count
        assert runtime._orchestrator_proxy is not None

    def test_model_post_init_builds_supervisor_proxies(self) -> None:
        """AC3: model_post_init rebuilds supervisor proxies from addresses."""
        supervisor_card = make_agent_card(name="supervisor", role="Supervisor", agent_class=Akgent)
        worker_card = make_agent_card(name="worker", role="Worker", agent_class=Akgent)
        entry_member = TeamCardMember(
            card=make_agent_card(name="lead", role="Lead", agent_class=Akgent),
        )
        # Supervisor has a subordinate worker — makes it a supervisor
        supervisor_member = TeamCardMember(
            card=supervisor_card,
            members=[TeamCardMember(card=worker_card)],
        )
        tc = TeamCard(
            name="team-with-supervisors",
            description="A team with supervisor hierarchy",
            entry_point=entry_member,
            members=[supervisor_member],
            message_types=[UserMessage],
        )
        sup_addr = make_stub_addr("supervisor")
        runtime = make_team_runtime(
            team_card=tc,
            supervisor_addrs={"supervisor": sup_addr},
        )
        assert "supervisor" in runtime._supervisor_proxies
        runtime.actor_system.proxy_ask.assert_any_call(sup_addr, Akgent)


class TestTeamRuntimeSerialization:
    """AC 2, 7: Persistent/ephemeral field separation and serialization round-trip."""

    def test_ephemeral_fields_excluded_from_dump(self) -> None:
        runtime = make_team_runtime()
        data = runtime.model_dump()
        assert "_orchestrator_proxy" not in data
        assert "_entry_proxy" not in data
        assert "_supervisor_proxies" not in data
        assert "_message_cls" not in data

    def test_actor_system_excluded_from_dump(self) -> None:
        runtime = make_team_runtime()
        data = runtime.model_dump()
        assert "actor_system" not in data

    def test_persistent_fields_in_dump(self) -> None:
        runtime = make_team_runtime()
        data = runtime.model_dump()
        assert "id" in data
        assert "team" in data
        assert "orchestrator_addr" in data
        assert "entry_addr" in data
        assert "supervisor_addrs" in data
        assert "addrs" in data


class TestTeamRuntimeMessaging:
    """AC 5, 6: send() and send_to() messaging facade."""

    def test_send_broadcasts_to_entry_and_all_supervisors(self) -> None:
        """AC5: send() routes through entry proxy to entry agent and each supervisor."""
        sup_addr_1 = make_stub_addr("supervisor-1")
        sup_addr_2 = make_stub_addr("supervisor-2")
        runtime = make_team_runtime(
            message_types=[UserMessage],
            supervisor_addrs={"sup1": sup_addr_1, "sup2": sup_addr_2},
        )
        runtime.send("hello")
        # entry agent + 2 supervisors = 3 calls
        assert runtime._entry_proxy.send.call_count == 3
        call_addrs = {call.args[0] for call in runtime._entry_proxy.send.call_args_list}
        assert call_addrs == {runtime.entry_addr, sup_addr_1, sup_addr_2}
        # Verify all messages are UserMessage with correct content
        for call in runtime._entry_proxy.send.call_args_list:
            msg = call.args[1]
            assert isinstance(msg, UserMessage)
            assert msg.content == "hello"

    def test_send_with_empty_supervisors_sends_to_entry(self) -> None:
        """AC5: send() with no supervisors still sends to entry agent."""
        runtime = make_team_runtime(message_types=[UserMessage])
        runtime.send("hello")
        runtime._entry_proxy.send.assert_called_once()
        call_args = runtime._entry_proxy.send.call_args
        assert call_args.args[0] is runtime.entry_addr
        assert isinstance(call_args.args[1], UserMessage)
        assert call_args.args[1].content == "hello"

    def test_send_to_looks_up_agent(self) -> None:
        """AC6: send_to() looks up agent via orchestrator proxy."""
        target_addr = make_stub_addr("worker")
        runtime = make_team_runtime(message_types=[UserMessage])
        runtime._orchestrator_proxy.get_team_member = MagicMock(return_value=target_addr)
        runtime.send_to("worker", "hello")
        runtime._orchestrator_proxy.get_team_member.assert_called_once_with("worker")
        runtime._entry_proxy.send.assert_called_once()
        call_args = runtime._entry_proxy.send.call_args
        assert call_args.args[0] is target_addr
        assert isinstance(call_args.args[1], UserMessage)
        assert call_args.args[1].content == "hello"

    def test_send_to_raises_for_unknown_agent(self) -> None:
        """AC6: send_to() raises ValueError if agent not found."""
        runtime = make_team_runtime(message_types=[UserMessage])
        runtime._orchestrator_proxy.get_team_member = MagicMock(return_value=None)
        with pytest.raises(ValueError, match="not found"):
            runtime.send_to("nonexistent", "hello")

    def test_make_message_raises_when_no_message_types(self) -> None:
        runtime = make_team_runtime(message_types=[])
        with pytest.raises(RuntimeError, match="No message type"):
            runtime._make_message("hello")

    def test_make_message_creates_correct_type(self) -> None:
        runtime = make_team_runtime(message_types=[UserMessage])
        msg = runtime._make_message("hello world")
        assert isinstance(msg, UserMessage)
        assert msg.content == "hello world"

    def test_read_only_properties(self) -> None:
        runtime = make_team_runtime()
        assert runtime.orchestrator_proxy is runtime._orchestrator_proxy
        assert runtime.entry_proxy is runtime._entry_proxy
        assert runtime.supervisor_proxies is runtime._supervisor_proxies


class TestTeamRuntimeSendToResolution:
    """AC 2 (Story 12.3): send_to() resolves proxy addresses via addr_map."""

    def test_send_to_resolves_proxy_address(self) -> None:
        """If orchestrator returns a proxy, send_to() resolves it via addr_map."""
        from akgentic.core.actor_address_impl import ActorAddressProxy
        from akgentic.core.utils.deserializer import ActorAddressDict

        agent_id = uuid.uuid4()
        team_id = uuid.uuid4()

        # Create a proxy address (as would come from deserialized data)
        addr_dict: ActorAddressDict = {
            "__actor_address__": True,
            "__actor_type__": "akgentic.core.agent.Akgent",
            "agent_id": str(agent_id),
            "name": "worker",
            "role": "Worker",
            "team_id": str(team_id),
            "squad_id": str(uuid.uuid4()),
            "user_message": False,
        }
        proxy_addr = ActorAddressProxy(addr_dict)

        # Create a live address mock
        live_addr = make_stub_addr("worker")
        live_addr.agent_id = agent_id

        runtime = make_team_runtime(message_types=[UserMessage])
        runtime._orchestrator_proxy.get_team_member = MagicMock(return_value=proxy_addr)
        runtime._addr_map = {agent_id: live_addr}

        runtime.send_to("worker", "hello")

        # Verify send was called with the live address, not the proxy
        runtime._entry_proxy.send.assert_called_once()
        call_args = runtime._entry_proxy.send.call_args
        assert call_args.args[0] is live_addr
        assert isinstance(call_args.args[1], UserMessage)

    def test_send_to_raises_for_unmapped_proxy(self) -> None:
        """If proxy has no mapping in addr_map, raise ValueError."""
        from akgentic.core.actor_address_impl import ActorAddressProxy
        from akgentic.core.utils.deserializer import ActorAddressDict

        addr_dict: ActorAddressDict = {
            "__actor_address__": True,
            "__actor_type__": "akgentic.core.agent.Akgent",
            "agent_id": str(uuid.uuid4()),
            "name": "ghost",
            "role": "Ghost",
            "team_id": str(uuid.uuid4()),
            "squad_id": str(uuid.uuid4()),
            "user_message": False,
        }
        proxy_addr = ActorAddressProxy(addr_dict)

        runtime = make_team_runtime(message_types=[UserMessage])
        runtime._orchestrator_proxy.get_team_member = MagicMock(return_value=proxy_addr)
        runtime._addr_map = {}  # No mapping

        with pytest.raises(ValueError, match="stale proxy address"):
            runtime.send_to("ghost", "hello")
