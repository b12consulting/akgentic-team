"""Test fixtures for team domain model tests."""

from __future__ import annotations

from akgentic.core.agent_card import AgentCard
from akgentic.core.agent_config import BaseConfig

from akgentic.team.models import TeamCard, TeamCardMember


def make_agent_card(
    name: str = "test-agent",
    role: str = "TestAgent",
    routes_to: list[str] | None = None,
) -> AgentCard:
    """Create a minimal AgentCard for testing."""
    return AgentCard(
        role=role,
        description=f"Test agent: {role}",
        skills=["testing"],
        agent_class="tests.fixtures.MockAgent",
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
) -> TeamCard:
    """Create a minimal TeamCard for testing.

    Args:
        name: Team name.
        description: Team description.
        entry_point_name: Config name of the entry point agent.
        entry_point_role: Role of the entry point agent.
        member_names: Config names for additional members.
        member_roles: Roles for additional members.

    Returns:
        A TeamCard with the specified structure.
    """
    entry_point = TeamCardMember(
        card=make_agent_card(name=entry_point_name, role=entry_point_role),
    )
    members: list[TeamCardMember] = []
    if member_names and member_roles:
        for mname, mrole in zip(member_names, member_roles, strict=True):
            members.append(
                TeamCardMember(card=make_agent_card(name=mname, role=mrole)),
            )
    return TeamCard(
        name=name,
        description=description,
        entry_point=entry_point,
        members=members,
    )
