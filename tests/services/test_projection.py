"""Tests for the one TeamCard -> TeamProjection derivation and the card hash."""

from __future__ import annotations

from typing import Any

import pytest
from akgentic.core.agent import Akgent
from akgentic.core.agent_card import AgentCard
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.messages.message import UserMessage
from akgentic.core.utils.serializer import SerializableBaseModel

from akgentic.team.models import TeamCard, TeamCardMember
from akgentic.team.projection import (
    TeamProjection,
    derive_team_projection,
    hash_agent_card,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class StubAgent(Akgent[BaseConfig, BaseState]):
    """Minimal agent so cards resolve a real class."""


class SampleMetadata(SerializableBaseModel):
    """A metadata contract a card may declare."""

    tenant: str = ""


def _make_card(name: str, role: str, **overrides: Any) -> AgentCard:
    return AgentCard(
        description=f"Test: {role}",
        skills=["testing"],
        agent_class=StubAgent,
        config=BaseConfig(name=name, role=role),
        **overrides,
    )


def _make_member(
    name: str,
    role: str,
    headcount: int = 1,
    members: list[TeamCardMember] | None = None,
) -> TeamCardMember:
    return TeamCardMember(
        card=_make_card(name, role),
        headcount=headcount,
        members=members or [],
    )


def _three_level_card() -> TeamCard:
    """A card with an entry point, two first-layer members and one grandchild."""
    return TeamCard(
        name="deep-team",
        description="A team three levels deep",
        entry_point=_make_member("@Lead", "Lead"),
        members=[
            _make_member(
                "@Analyst",
                "Analyst",
                members=[_make_member("@Junior", "Junior")],
            ),
            _make_member("@Writer", "Writer"),
        ],
    )


# ---------------------------------------------------------------------------
# The card hash
# ---------------------------------------------------------------------------


class TestHashAgentCard:
    """AC 16-18: a stable content hash that ignores hireability."""

    def test_the_same_card_hashes_the_same_twice(self) -> None:
        card = _make_card("@Lead", "Lead")
        assert hash_agent_card(card) == hash_agent_card(card)

    def test_two_equal_cards_hash_the_same(self) -> None:
        assert hash_agent_card(_make_card("@Lead", "Lead")) == hash_agent_card(
            _make_card("@Lead", "Lead")
        )

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("skills", ["testing", "extra"]),
            ("description", "a different description"),
            ("metadata", {"tier": "gold"}),
        ],
    )
    def test_a_card_differing_in_a_hashed_field_hashes_differently(
        self, field: str, value: Any
    ) -> None:
        base = _make_card("@Lead", "Lead")
        other = base.model_copy(update={field: value})
        assert hash_agent_card(other) != hash_agent_card(base)

    def test_a_card_differing_in_config_hashes_differently(self) -> None:
        base = _make_card("@Lead", "Lead")
        other = _make_card("@Lead2", "Lead")
        assert hash_agent_card(other) != hash_agent_card(base)

    def test_can_be_hired_is_excluded_from_the_hash(self) -> None:
        """AC 17: hireability is a property of the team, not of the card.

        A card reached through ``agent_profiles`` and the same card reached
        through the member tree must address one stored blob, or 31-7's store
        would hold two copies differing in a boolean.

        Mutation-verified: including ``can_be_hired`` in the canonicalisation
        turns this red.
        """
        card = _make_card("@Lead", "Lead")
        hireable = card.model_copy(update={"can_be_hired": True})

        assert card.can_be_hired is False
        assert hireable.can_be_hired is True
        assert hash_agent_card(hireable) == hash_agent_card(card)


# ---------------------------------------------------------------------------
# The derivation
# ---------------------------------------------------------------------------


class TestDeriveTeamProjection:
    """AC 8-14, 19: the single derivation from a TeamCard."""

    def test_takes_only_the_team_card_and_returns_a_projection(self) -> None:
        projection = derive_team_projection(_three_level_card())
        assert isinstance(projection, TeamProjection)

    def test_carries_the_scalar_projection_fields(self) -> None:
        tc = _three_level_card()
        tc.message_types = [UserMessage]
        tc.metadata_type = SampleMetadata

        projection = derive_team_projection(tc)

        assert projection.team_name == "deep-team"
        assert projection.message_types == [UserMessage]
        assert projection.metadata_type is SampleMetadata

    def test_team_description_is_none_and_is_not_seeded_from_the_card(self) -> None:
        """AC 19: the card describes a blueprint; this describes one team."""
        tc = _three_level_card()
        assert tc.description
        assert derive_team_projection(tc).team_description is None

    def test_entry_point_ref_is_the_entry_point_agent(self) -> None:
        projection = derive_team_projection(_three_level_card())
        assert projection.entry_point.name == "@Lead"
        assert projection.entry_point.role == "Lead"

    def test_supervisors_are_the_first_layer_only_in_declaration_order(self) -> None:
        """AC 9: children of first-layer members are excluded, as on TeamCard."""
        projection = derive_team_projection(_three_level_card())
        assert [(r.name, r.role) for r in projection.supervisors] == [
            ("@Analyst", "Analyst"),
            ("@Writer", "Writer"),
        ]

    def test_agent_cards_cover_the_entry_point_and_the_whole_tree(self) -> None:
        """AC 9: the grandchild's role is in the catalog even though it is no supervisor."""
        projection = derive_team_projection(_three_level_card())
        assert [c.role for c in projection.agent_cards] == [
            "Lead",
            "Analyst",
            "Junior",
            "Writer",
        ]

    def test_agent_profiles_roles_are_included_and_marked_hireable(self) -> None:
        """AC 10: a role sourced only from the tree is not hireable."""
        tc = _three_level_card()
        tc.agent_profiles = [_make_card("@Specialist", "Specialist")]

        projection = derive_team_projection(tc)

        hireable = {c.role: c.can_be_hired for c in projection.agent_cards}
        assert hireable == {
            "Lead": False,
            "Analyst": False,
            "Junior": False,
            "Writer": False,
            "Specialist": True,
        }

    def test_a_role_in_both_the_tree_and_the_profiles_appears_once_and_is_hireable(
        self,
    ) -> None:
        """AC 11."""
        tc = _three_level_card()
        tc.agent_profiles = [_make_card("@AnotherWriter", "Writer")]

        projection = derive_team_projection(tc)

        writers = [c for c in projection.agent_cards if c.role == "Writer"]
        assert len(writers) == 1
        assert writers[0].can_be_hired is True
        assert len(projection.cards) == len(projection.agent_cards)

    def test_two_members_sharing_a_role_give_two_refs_and_one_card(self) -> None:
        """AC 12: both identities are addressable; the card behind them is one."""
        tc = TeamCard(
            name="twin-team",
            description="Two writers",
            entry_point=_make_member("@Lead", "Lead"),
            members=[_make_member("@WriterA", "Writer"), _make_member("@WriterB", "Writer")],
        )

        projection = derive_team_projection(tc)

        assert [r.name for r in projection.supervisors] == ["@WriterA", "@WriterB"]
        assert [r.role for r in projection.supervisors] == ["Writer", "Writer"]
        assert [c.role for c in projection.agent_cards] == ["Lead", "Writer"]
        assert len(projection.cards) == 2

    def test_headcount_three_contributes_three_indexed_refs(self) -> None:
        """AC 14: headcount is expanded here and appears nowhere in the projection."""
        tc = TeamCard(
            name="crew-team",
            description="A crew",
            entry_point=_make_member("@Lead", "Lead"),
            members=[_make_member("@Worker", "Worker", headcount=3)],
        )

        projection = derive_team_projection(tc)

        assert [r.name for r in projection.supervisors] == [
            "@Worker_0",
            "@Worker_1",
            "@Worker_2",
        ]
        assert {r.role for r in projection.supervisors} == {"Worker"}
        assert [c.role for c in projection.agent_cards] == ["Lead", "Worker"]
        assert "headcount" not in projection.model_dump()

    def test_headcount_one_contributes_the_bare_card_name(self) -> None:
        """AC 14."""
        projection = derive_team_projection(_three_level_card())
        assert [r.name for r in projection.supervisors] == ["@Analyst", "@Writer"]

    def test_the_cards_list_matches_agent_cards_in_order(self) -> None:
        """AC 8: the refs and the cards 31-7 will persist are parallel."""
        tc = _three_level_card()
        tc.agent_profiles = [_make_card("@Specialist", "Specialist")]

        projection = derive_team_projection(tc)

        assert [c.role for c in projection.cards] == [c.role for c in projection.agent_cards]
        for card, ref in zip(projection.cards, projection.agent_cards, strict=True):
            assert hash_agent_card(card) == ref.card_hash
            assert card.can_be_hired == ref.can_be_hired

    def test_the_input_card_is_not_mutated(self) -> None:
        """AC 13: the hireable flag is applied to a copy, never written back."""
        profile = _make_card("@Specialist", "Specialist")
        tree_card = _make_card("@Writer", "Writer")
        tc = TeamCard(
            name="pristine-team",
            description="Nothing here gets flagged",
            entry_point=_make_member("@Lead", "Lead"),
            members=[TeamCardMember(card=tree_card)],
            agent_profiles=[profile, _make_card("@AnotherWriter", "Writer")],
        )
        before = tc.model_dump()

        projection = derive_team_projection(tc)

        assert profile.can_be_hired is False
        assert tree_card.can_be_hired is False
        assert all(card.can_be_hired is False for card in tc.agent_cards.values())
        assert all(card.can_be_hired is False for card in tc.agent_profiles)
        assert tc.model_dump() == before
        # ...and the projection did flag its own copies.
        assert any(card.can_be_hired for card in projection.cards)

    def test_the_projection_round_trips(self) -> None:
        tc = _three_level_card()
        tc.message_types = [UserMessage]
        tc.metadata_type = SampleMetadata
        tc.agent_profiles = [_make_card("@Specialist", "Specialist")]

        projection = derive_team_projection(tc)
        restored = TeamProjection.model_validate(projection.model_dump())

        assert restored.message_types == [UserMessage]
        assert restored.metadata_type is SampleMetadata
        assert restored.agent_cards == projection.agent_cards
        assert restored.supervisors == projection.supervisors
