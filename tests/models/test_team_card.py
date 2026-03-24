"""Tests for TeamCard and TeamCardMember models."""

from __future__ import annotations

import pytest
from akgentic.core.agent_card import AgentCard
from akgentic.core.messages import UserMessage

from akgentic.team.models import TeamCard, TeamCardMember

from .conftest import make_agent_card, make_team_card


class TestTeamCardMember:
    """Tests for the TeamCardMember model."""

    def test_construction_with_defaults(self) -> None:
        """TeamCardMember with only card uses default headcount=1 and empty members."""
        card = make_agent_card(name="worker", role="Worker")
        member = TeamCardMember(card=card)
        assert member.card.config.name == "worker"
        assert member.headcount == 1
        assert member.members == []

    def test_construction_with_explicit_headcount(self) -> None:
        """TeamCardMember accepts an explicit headcount value."""
        card = make_agent_card(name="pool-agent", role="PoolAgent")
        member = TeamCardMember(card=card, headcount=5)
        assert member.headcount == 5

    def test_nested_members(self) -> None:
        """TeamCardMember can contain nested members (self-referential)."""
        child = TeamCardMember(card=make_agent_card(name="child", role="Child"))
        parent = TeamCardMember(
            card=make_agent_card(name="parent", role="Parent"),
            members=[child],
        )
        assert len(parent.members) == 1
        assert parent.members[0].card.config.name == "child"


class TestTeamCard:
    """Tests for the TeamCard model."""

    def test_agent_cards_returns_flat_index(self) -> None:
        """agent_cards property returns dict keyed by config name."""
        team = make_team_card(
            entry_point_name="lead",
            entry_point_role="Lead",
            member_names=["dev", "qa"],
            member_roles=["Developer", "QA"],
        )
        cards = team.agent_cards
        assert isinstance(cards, dict)
        assert set(cards.keys()) == {"lead", "dev", "qa"}
        assert all(isinstance(v, AgentCard) for v in cards.values())

    def test_agent_cards_includes_nested_members(self) -> None:
        """agent_cards flattens nested member trees."""
        grandchild = TeamCardMember(
            card=make_agent_card(name="gc", role="GrandChild"),
        )
        child = TeamCardMember(
            card=make_agent_card(name="child", role="Child"),
            members=[grandchild],
        )
        entry = TeamCardMember(
            card=make_agent_card(name="root", role="Root"),
        )
        team = TeamCard(
            name="nested-team",
            description="A nested team",
            entry_point=entry,
            members=[child],
        )
        cards = team.agent_cards
        assert set(cards.keys()) == {"root", "child", "gc"}

    def test_supervisors_returns_members_with_subordinates(self) -> None:
        """supervisors property returns entry point and members with subordinates."""
        worker = TeamCardMember(card=make_agent_card(name="worker", role="Worker"))
        supervisor = TeamCardMember(
            card=make_agent_card(name="sup", role="Supervisor"),
            members=[worker],
        )
        entry = TeamCardMember(card=make_agent_card(name="entry", role="Entry"))
        team = TeamCard(
            name="sup-team",
            description="Team with supervisor",
            entry_point=entry,
            members=[supervisor],
        )
        sups = team.supervisors
        sup_names = {s.config.name for s in sups}
        assert "entry" in sup_names  # entry point always included
        assert "sup" in sup_names    # member with subordinates
        assert len(sups) == 2

    def test_supervisors_excludes_leaf_members_except_entry(self) -> None:
        """supervisors includes only the entry point when no member has subordinates."""
        team = make_team_card(
            member_names=["a", "b"],
            member_roles=["A", "B"],
        )
        # No member has subordinates; only entry point is a supervisor
        assert len(team.supervisors) == 1
        assert team.supervisors[0].config.name == "lead"

    def test_empty_members_list(self) -> None:
        """TeamCard with only entry_point and no members works correctly."""
        team = TeamCard(
            name="solo-team",
            description="Just the entry point",
            entry_point=TeamCardMember(
                card=make_agent_card(name="solo", role="Solo"),
            ),
            members=[],
        )
        assert set(team.agent_cards.keys()) == {"solo"}
        # Entry point is always a supervisor
        assert len(team.supervisors) == 1
        assert team.supervisors[0].config.name == "solo"

    def test_message_types_default_empty(self) -> None:
        """message_types defaults to an empty list."""
        team = make_team_card()
        assert team.message_types == []

    def test_agent_cards_raises_on_duplicate_config_name(self) -> None:
        """agent_cards raises ValueError when two members share config.name."""
        entry = TeamCardMember(card=make_agent_card(name="entry", role="Entry"))
        dupe1 = TeamCardMember(card=make_agent_card(name="dupe", role="Role1"))
        dupe2 = TeamCardMember(card=make_agent_card(name="dupe", role="Role2"))
        team = TeamCard(
            name="dupe-team",
            description="Duplicate names",
            entry_point=entry,
            members=[dupe1, dupe2],
        )
        with pytest.raises(ValueError, match="Duplicate config name 'dupe'"):
            team.agent_cards

    def test_supervisors_includes_entry_point_without_subordinates(self) -> None:
        """Entry point with no members list is still in supervisors."""
        entry = TeamCardMember(card=make_agent_card(name="proxy", role="Proxy"))
        worker = TeamCardMember(card=make_agent_card(name="worker", role="Worker"))
        team = TeamCard(
            name="flat-team",
            description="Flat team with leaf entry point",
            entry_point=entry,
            members=[worker],
        )
        sups = team.supervisors
        sup_names = [s.config.name for s in sups]
        assert "proxy" in sup_names

    def test_supervisors_includes_entry_point_with_subordinates(self) -> None:
        """Entry point with subordinates is not duplicated in supervisors."""
        child = TeamCardMember(card=make_agent_card(name="child", role="Child"))
        entry = TeamCardMember(
            card=make_agent_card(name="boss", role="Boss"),
            members=[child],
        )
        team = TeamCard(
            name="entry-sup-team",
            description="Entry point supervises",
            entry_point=entry,
            members=[],
        )
        sups = team.supervisors
        assert len(sups) == 1
        assert sups[0].config.name == "boss"

    def test_supervisors_includes_both_entry_and_member_supervisors(self) -> None:
        """Entry point + members with subordinates are all present."""
        worker = TeamCardMember(card=make_agent_card(name="worker", role="Worker"))
        mid_sup = TeamCardMember(
            card=make_agent_card(name="mid", role="MiddleSup"),
            members=[worker],
        )
        entry = TeamCardMember(card=make_agent_card(name="lead", role="Lead"))
        team = TeamCard(
            name="mixed-team",
            description="Entry point (leaf) + member supervisor",
            entry_point=entry,
            members=[mid_sup],
        )
        sups = team.supervisors
        sup_names = {s.config.name for s in sups}
        assert "lead" in sup_names
        assert "mid" in sup_names
        assert len(sups) == 2


class TestTeamCardSerialization:
    """Tests for serialization round-trip of TeamCard and TeamCardMember."""

    def test_team_card_member_round_trip(self) -> None:
        """TeamCardMember survives model_dump/model_validate round-trip."""
        card = make_agent_card(name="rt-agent", role="RoundTrip")
        member = TeamCardMember(card=card, headcount=3)
        dumped = member.model_dump()
        restored = TeamCardMember.model_validate(dumped)
        assert restored.card.config.name == "rt-agent"
        assert restored.headcount == 3
        assert restored.members == []

    def test_team_card_round_trip(self) -> None:
        """TeamCard survives model_dump/model_validate round-trip."""
        team = make_team_card(
            name="rt-team",
            member_names=["x", "y"],
            member_roles=["X", "Y"],
        )
        dumped = team.model_dump()
        restored = TeamCard.model_validate(dumped)
        assert restored.name == "rt-team"
        assert len(restored.agent_cards) == len(team.agent_cards)
        assert set(restored.agent_cards.keys()) == set(team.agent_cards.keys())

    def test_nested_tree_round_trip(self) -> None:
        """3+ level deep nested tree preserves data through serialization."""
        l3 = TeamCardMember(card=make_agent_card(name="l3", role="L3"))
        l2 = TeamCardMember(
            card=make_agent_card(name="l2", role="L2"),
            members=[l3],
        )
        l1 = TeamCardMember(
            card=make_agent_card(name="l1", role="L1"),
            members=[l2],
        )
        entry = TeamCardMember(card=make_agent_card(name="root", role="Root"))
        team = TeamCard(
            name="deep-team",
            description="Deep nesting",
            entry_point=entry,
            members=[l1],
        )
        dumped = team.model_dump()
        restored = TeamCard.model_validate(dumped)
        assert set(restored.agent_cards.keys()) == {"root", "l1", "l2", "l3"}
        # Verify supervisors preserved (entry point always included)
        sup_names = {s.config.name for s in restored.supervisors}
        assert sup_names == {"root", "l1", "l2"}

    def test_routes_to_preserved_through_serialization(self) -> None:
        """routes_to on AgentCards within the tree survive round-trip."""
        card_with_routes = make_agent_card(
            name="router",
            role="Router",
            routes_to=["TargetA", "TargetB"],
        )
        member = TeamCardMember(card=card_with_routes)
        entry = TeamCardMember(card=make_agent_card(name="entry", role="Entry"))
        team = TeamCard(
            name="routing-team",
            description="Team with routing",
            entry_point=entry,
            members=[member],
        )
        dumped = team.model_dump()
        restored = TeamCard.model_validate(dumped)
        router_card = restored.agent_cards["router"]
        assert router_card.routes_to == ["TargetA", "TargetB"]

    def test_single_member_team_round_trip(self) -> None:
        """Entry-point only team serializes and deserializes correctly."""
        team = TeamCard(
            name="singleton",
            description="Single agent team",
            entry_point=TeamCardMember(
                card=make_agent_card(name="only", role="Only"),
            ),
            members=[],
        )
        dumped = team.model_dump()
        restored = TeamCard.model_validate(dumped)
        assert restored.name == "singleton"
        assert set(restored.agent_cards.keys()) == {"only"}

    def test_message_types_round_trip(self) -> None:
        """message_types with actual type references survive round-trip."""
        team = TeamCard(
            name="typed-team",
            description="Team with message types",
            entry_point=TeamCardMember(
                card=make_agent_card(name="entry", role="Entry"),
            ),
            members=[],
            message_types=[UserMessage],
        )
        dumped = team.model_dump()
        restored = TeamCard.model_validate(dumped)
        assert len(restored.message_types) == 1
        assert restored.message_types[0] is UserMessage
