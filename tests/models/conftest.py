"""Test fixtures for team domain model tests."""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent_card import AgentCard
from akgentic.core.agent_config import BaseConfig

from akgentic.team.models import TeamCard, TeamCardMember, TeamRuntime


def make_agent_card(
    name: str = "test-agent",
    role: str = "TestAgent",
    routes_to: list[str] | None = None,
    agent_class: str | type = "tests.fixtures.MockAgent",
) -> AgentCard:
    """Create a minimal AgentCard for testing.

    Args:
        name: Config name for the agent.
        role: Agent role.
        routes_to: List of agent names this agent routes to.
        agent_class: Agent class or FQCN string.

    Returns:
        An AgentCard with the specified configuration.
    """
    return AgentCard(
        role=role,
        description=f"Test agent: {role}",
        skills=["testing"],
        agent_class=agent_class,
        config=BaseConfig(name=name, role=role),
        routes_to=routes_to or [],
    )


def make_team_card(
    name: str = "test-team",
    description: str = "A test team",
    entry_point_name: str = "lead",
    entry_point_role: str = "Lead",
    member_names: list[str] | None = None,
    member_roles: list[str] | None = None,
    message_types: list[type] | None = None,
    agent_class: str | type = "tests.fixtures.MockAgent",
) -> TeamCard:
    """Create a minimal TeamCard for testing.

    Args:
        name: Team name.
        description: Team description.
        entry_point_name: Config name of the entry point agent.
        entry_point_role: Role of the entry point agent.
        member_names: Config names for additional members.
        member_roles: Roles for additional members.
        message_types: Message classes the team handles.
        agent_class: Agent class or FQCN string for all members.

    Returns:
        A TeamCard with the specified structure.
    """
    entry_point = TeamCardMember(
        card=make_agent_card(
            name=entry_point_name, role=entry_point_role, agent_class=agent_class
        ),
    )
    members: list[TeamCardMember] = []
    if member_names and member_roles:
        for mname, mrole in zip(member_names, member_roles, strict=True):
            members.append(
                TeamCardMember(
                    card=make_agent_card(name=mname, role=mrole, agent_class=agent_class)
                ),
            )
    return TeamCard(
        name=name,
        description=description,
        entry_point=entry_point,
        members=members,
        message_types=message_types or [],
    )


def make_stub_addr(name: str = "stub") -> MagicMock:
    """Create a MagicMock that behaves like an ActorAddress.

    Args:
        name: Name for the stub address.

    Returns:
        A MagicMock with ActorAddress spec.
    """
    addr = MagicMock(spec=ActorAddress)
    addr.agent_id = uuid.uuid4()
    addr.name = name
    return addr


def make_stub_actor_system() -> MagicMock:
    """Create a MagicMock that behaves like an ActorSystem.

    Returns:
        A MagicMock with ActorSystem spec and proxy methods.
    """
    system = MagicMock(spec=ActorSystem)
    system.proxy_ask = MagicMock(return_value=MagicMock())
    system.proxy_tell = MagicMock(return_value=MagicMock())
    return system


def make_team_runtime(
    *,
    team_card: TeamCard | None = None,
    message_types: list[type] | None = None,
    supervisor_addrs: dict[str, ActorAddress] | None = None,
    addrs: dict[str, ActorAddress] | None = None,
) -> TeamRuntime:
    """Create a TeamRuntime with mock dependencies for testing.

    Args:
        team_card: Optional pre-built TeamCard.
        message_types: Message types for auto-generated TeamCard.
        supervisor_addrs: Supervisor address mapping.
        addrs: All agent address mapping.

    Returns:
        A TeamRuntime with mock ActorSystem and addresses.
    """
    from akgentic.core.agent import Akgent

    tc = team_card or make_team_card(message_types=message_types, agent_class=Akgent)
    return TeamRuntime(
        id=uuid.uuid4(),
        team=tc,
        actor_system=make_stub_actor_system(),
        orchestrator_addr=make_stub_addr("orchestrator"),
        entry_addr=make_stub_addr("entry"),
        supervisor_addrs=supervisor_addrs or {},
        addrs=addrs or {},
    )
