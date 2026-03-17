"""Domain models for team lifecycle management.

TeamCard, TeamCardMember, TeamRuntime, TeamStatus, Process, PersistedEvent,
AgentStateSnapshot.
"""

from __future__ import annotations

from pydantic import Field

from akgentic.core.agent_card import AgentCard
from akgentic.core.utils.serializer import SerializableBaseModel


class TeamCardMember(SerializableBaseModel):
    """A member slot in a team card, wrapping an AgentCard with multiplicity.

    TeamCardMember is self-referential: each member can contain subordinate
    members, forming a tree that describes hierarchical team structures.

    Attributes:
        card: The agent card describing this member's role and capabilities.
        headcount: Number of agent instances to create for this slot.
        members: Subordinate members managed by this member.
    """

    card: AgentCard = Field(description="Agent card describing this member's role and capabilities")
    headcount: int = Field(default=1, description="Number of agent instances for this slot")
    members: list[TeamCardMember] = Field(
        default_factory=list,
        description="Subordinate members managed by this member",
    )


class TeamCard(SerializableBaseModel):
    """Declarative definition of a team's structure, entry point, and routing.

    TeamCard describes the full hierarchy of agents in a team. The entry_point
    is the agent that receives external messages. Members form a tree that can
    be walked to discover all agent cards and supervisory relationships.

    Attributes:
        name: Unique name identifying this team definition.
        description: Human-readable summary of what the team does.
        entry_point: The member that serves as the team's external interface.
        members: Top-level members of the team (excluding the entry point).
        message_types: Message classes the team handles; first is the default.
    """

    name: str = Field(description="Unique name identifying this team definition")
    description: str = Field(description="Human-readable summary of what the team does")
    entry_point: TeamCardMember = Field(
        description="The member that serves as the team's external interface",
    )
    members: list[TeamCardMember] = Field(
        default_factory=list,
        description="Top-level members of the team excluding the entry point",
    )
    message_types: list[type] = Field(
        default_factory=list,
        description="Message classes the team handles; first is the default",
    )

    @property
    def agent_cards(self) -> dict[str, AgentCard]:
        """Return a flat index of all AgentCards in the member tree.

        Walks the entry_point and all members recursively, collecting every
        AgentCard keyed by its ``config.name``.

        Returns:
            Dictionary mapping config name to AgentCard for every member.
        """
        result: dict[str, AgentCard] = {}
        self._collect_cards(self.entry_point, result)
        for member in self.members:
            self._collect_cards(member, result)
        return result

    @property
    def supervisors(self) -> list[AgentCard]:
        """Return AgentCards for members that have subordinate members.

        A member is a supervisor if its ``members`` list is non-empty,
        meaning it manages at least one subordinate.

        Returns:
            List of AgentCards belonging to supervisory members.
        """
        result: list[AgentCard] = []
        self._collect_supervisors(self.entry_point, result)
        for member in self.members:
            self._collect_supervisors(member, result)
        return result

    @staticmethod
    def _collect_cards(
        member: TeamCardMember,
        result: dict[str, AgentCard],
    ) -> None:
        """Recursively collect AgentCards from a member subtree.

        Args:
            member: The member node to start from.
            result: Accumulator dict to populate with discovered cards.
        """
        result[member.card.config.name] = member.card
        for child in member.members:
            TeamCard._collect_cards(child, result)

    @staticmethod
    def _collect_supervisors(
        member: TeamCardMember,
        result: list[AgentCard],
    ) -> None:
        """Recursively collect supervisor AgentCards from a member subtree.

        Args:
            member: The member node to start from.
            result: Accumulator list to populate with supervisor cards.
        """
        if member.members:
            result.append(member.card)
        for child in member.members:
            TeamCard._collect_supervisors(child, result)
