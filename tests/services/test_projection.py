"""Tests for the one TeamCard -> TeamProjection derivation and the card hash."""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from akgentic.core.agent import Akgent
from akgentic.core.agent_card import AgentCard
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.messages.message import UserMessage
from akgentic.core.utils.serializer import SerializableBaseModel

from akgentic.team.models import AgentCardRef, Process, TeamCard, TeamCardMember
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


def _writer_ref(projection: TeamProjection) -> AgentCardRef:
    """Return the ``Writer`` ref of *projection*."""
    return next(ref for ref in projection.agent_cards if ref.role == "Writer")


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

    def test_can_be_hired_is_a_declared_field_of_agent_card(self) -> None:
        """The exclusion needs a subject that can be seen to disappear.

        ``hash_agent_card`` drops the key with ``payload.pop(..., None)``, which
        tolerates its absence — and ``model_copy(update=...)`` on an undeclared
        field never reaches ``model_fields``, so it never reaches the payload
        either. If ``akgentic-core`` ever renames or drops ``can_be_hired``, the
        exclusion goes vacuous while the test below stays **green**: both cards
        would then be byte-identical and hash the same for the wrong reason.
        This is the guard that can see its own subject vanish.
        """
        assert "can_be_hired" in AgentCard.model_fields

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

    def test_the_profiles_card_is_the_one_that_survives_the_dedup(self) -> None:
        """AC 15: WHICH card wins, not how many are left.

        The two cards for ``Writer`` differ in ``description`` and ``skills``, so
        the surviving one is identifiable by content rather than by a count. A
        spec that only pins ``len(writers) == 1`` and the hireable flag stays
        green under either precedence — which is exactly how the tree-wins
        behaviour survived unnoticed.
        """
        tc = _three_level_card()
        profile = _make_card("@AnotherWriter", "Writer")
        profile.description = "Hired to write, not the one already writing"
        profile.skills = ["ghostwriting"]
        tc.agent_profiles = [profile]

        tree_card = tc.agent_cards["@Writer"]
        assert tree_card.description != profile.description
        assert tree_card.skills != profile.skills

        projection = derive_team_projection(tc)

        survivor = next(c for c in projection.cards if c.role == "Writer")
        assert survivor.description == profile.description
        assert survivor.skills == profile.skills

        ref = next(r for r in projection.agent_cards if r.role == "Writer")
        assert ref.card_hash == hash_agent_card(
            profile.model_copy(update={"can_be_hired": True})
        )
        assert ref.card_hash != hash_agent_card(tree_card)

    def test_the_overriding_profile_keeps_the_role_at_its_tree_position(self) -> None:
        """AC 14: only the card behind the role changes, never its position.

        Assignment on an existing dict key does not move it, so ``agent_cards``
        keeps its discovery order — entry point, tree, then profiles-only roles.
        """
        tc = _three_level_card()
        tc.agent_profiles = [
            _make_card("@AnotherWriter", "Writer"),
            _make_card("@Specialist", "Specialist"),
        ]

        roles = [r.role for r in derive_team_projection(tc).agent_cards]

        assert roles == ["Lead", "Analyst", "Junior", "Writer", "Specialist"]

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

    def test_one_role_reached_two_ways_gives_refs_differing_only_in_the_flag(
        self,
    ) -> None:
        """The property the shared store rests on.

        The same role reached through the member tree (not hireable) and
        through ``agent_profiles`` (hireable) must address ONE stored blob, so
        the two refs may differ in ``can_be_hired`` and in nothing else.
        """
        card = _make_card("@Writer", "Writer")
        tree_only = TeamCard(
            name="tree-team",
            description="Writer via the member tree",
            entry_point=_make_member("@Lead", "Lead"),
            members=[TeamCardMember(card=card)],
        )
        profile_only = TeamCard(
            name="profile-team",
            description="Writer via agent_profiles",
            entry_point=_make_member("@Lead", "Lead"),
            agent_profiles=[card],
        )

        tree_ref = _writer_ref(derive_team_projection(tree_only))
        profile_ref = _writer_ref(derive_team_projection(profile_only))

        assert tree_ref.card_hash == profile_ref.card_hash
        assert tree_ref.can_be_hired is False
        assert profile_ref.can_be_hired is True
        assert tree_ref.model_dump() == profile_ref.model_dump() | {"can_be_hired": False}

    def test_the_derivation_hashes_in_exactly_one_place(self) -> None:
        """AC 1: one canonicalisation, one call site.

        A second ``hash_agent_card`` call inside the derivation is how the same
        card starts being addressed by two digests — the fork the store cannot
        detect, because both keys resolve.
        """
        source = inspect.getsource(derive_team_projection)
        assert source.count("hash_agent_card(") == 1

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


# ---------------------------------------------------------------------------
# The ref shape (AC 4) — verified, not reintroduced
# ---------------------------------------------------------------------------


class TestAgentCardRefShape:
    """The ref stays three scalar fields, and no whole card reaches a Process."""

    def test_agent_card_ref_has_exactly_three_fields(self) -> None:
        assert set(AgentCardRef.model_fields) == {"role", "card_hash", "can_be_hired"}

    def test_can_be_hired_defaults_to_false(self) -> None:
        ref = AgentCardRef(role="Lead", card_hash="deadbeef")
        assert ref.can_be_hired is False

    def test_process_agent_cards_holds_refs_not_cards(self) -> None:
        """Embedding a card here is the duplication the whole epic removes."""
        assert Process.model_fields["agent_cards"].annotation == list[AgentCardRef]

    def test_no_process_field_holds_an_agent_card(self) -> None:
        """No path in this story puts a whole ``AgentCard`` on a ``Process``.

        ``team_card`` still carries the card graph — story 31-3 deletes it — so
        this asserts no *new* field does, which is what "Process gains no field
        here" means in practice.
        """
        card_fields = [
            name
            for name, field in Process.model_fields.items()
            if field.annotation in (AgentCard, list[AgentCard], AgentCard | None)
        ]
        assert card_fields == []
