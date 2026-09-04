"""The one derivation from a ``TeamCard`` to the structural projection stored on ``Process``.

``Process`` used to hold a second copy of the input ``TeamCard``. That copy
recorded what was *declared*, never what was *spawned*, and nothing stopped the
two from drifting. This module replaces it with a flat projection and — the part
that matters — with exactly **one** function that produces it.

Follows the shape ``akgentic.team.metadata.derive_metadata_indexes`` already
establishes in this package: one derived value, one derivation function, imported
by every write path. ``TeamManager.create_team`` calls it today and the migration
scripts of story 31-5 will import the same function. A second derivation site is
how a stored record starts disagreeing with what a fresh team would produce.

Implements ADR-26 §Decision 5 (FR3, FR3a, FR4, FR12).
"""

from __future__ import annotations

import hashlib
import json

from pydantic import Field

from akgentic.core.agent_card import AgentCard
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.team.models import (
    AgentCardRef,
    AgentRef,
    TeamCard,
    TeamCardMember,
    spawned_names,
)


def hash_agent_card(card: AgentCard) -> str:
    """Return a stable content hash for *card*.

    Two canonicalisation decisions, both load-bearing:

    * **Serialization mode is** ``mode="json"``. It renders every value —
      including the ``{"__type__": ...}`` markers a ``type`` field becomes — to
      JSON primitives, so the digest depends on the card's content rather than on
      whichever Python objects happened to be in memory.
    * **Field ordering is sorted by key** (``sort_keys=True``), with the compact
      separators. Pydantic emits fields in declaration order, so two equal cards
      built through different code paths would otherwise hash differently the day
      a field is reordered.

    ``can_be_hired`` is **excluded**. Hireability is a property of the team that
    names a role, not of the card's content: the same card reached through
    ``agent_profiles`` and through the member tree must address one stored blob,
    or the store would hold two copies of one card that differ in a boolean.

    The key is dropped from the dumped dict rather than passed as
    ``model_dump(exclude=...)``: ``SerializableBaseModel`` installs a plain
    ``@model_serializer`` that builds the payload itself and never sees
    ``exclude``, so the argument is silently ignored on any model in this
    framework.

    Args:
        card: The card to hash.

    Returns:
        The hex SHA-256 digest of the canonical form.
    """
    payload = card.model_dump(mode="json")
    payload.pop("can_be_hired", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class TeamProjection(SerializableBaseModel):
    """The result of deriving a projection from a ``TeamCard``.

    The first seven fields are exactly the projection fields of ``Process`` and
    are copied onto it verbatim by the write path.

    Attributes:
        team_name: The team's name.
        team_description: Always ``None``. A team instance's description is
            mutable and user-owned; the card's description belongs to the
            blueprint, and seeding from it would make an edited description snap
            back on the next derivation.
        entry_point: Ref to the agent that receives external messages.
        supervisors: One ref per first-layer member, in declaration order, with
            ``headcount`` already expanded.
        agent_cards: One ref per role, in discovery order: the entry point, then
            the member tree, then ``agent_profiles``.
        message_types: Message classes the team handles; first is the default.
        metadata_type: Model class for the team's business metadata, or ``None``.
        cards: The deduplicated cards ``agent_cards`` points at, in the same
            order, each already carrying its final ``can_be_hired`` value. This
            is the set story 31-7's card store will persist; **no consumer yet**.
    """

    team_name: str | None = Field(default=None, description="The team's name")
    team_description: str | None = Field(
        default=None, description="Always None; a team instance's description is user-owned"
    )
    entry_point: AgentRef = Field(description="Ref to the agent receiving external messages")
    supervisors: list[AgentRef] = Field(
        default_factory=list, description="One ref per first-layer member, headcount expanded"
    )
    agent_cards: list[AgentCardRef] = Field(
        default_factory=list, description="One ref per role, in discovery order"
    )
    message_types: list[type] = Field(
        default_factory=list, description="Message classes the team handles"
    )
    metadata_type: type[SerializableBaseModel] | None = Field(
        default=None, description="Model class for the team's business metadata, or None"
    )
    cards: list[AgentCard] = Field(
        default_factory=list,
        description="The deduped cards agent_cards points at; persisted by story 31-7",
    )


def _collect_cards_by_role(team_card: TeamCard) -> tuple[dict[str, AgentCard], set[str]]:
    """Collect one card per role, plus the set of roles that may be hired.

    Discovery order is entry point, then the member tree, then ``agent_profiles``
    — ``TeamCard.agent_cards`` already walks the first two in that order. The
    first card seen for a role wins; a role reachable from both the tree and
    ``agent_profiles`` therefore yields one entry, marked hireable.

    Note that ``AgentCard.role`` is a read-only property over ``config.role``,
    which is validated non-empty, while ``TeamCard.agent_cards`` is keyed by
    ``config.name``. The two keys are not interchangeable.

    Args:
        team_card: The card to walk.

    Returns:
        A ``(role -> card, hireable roles)`` pair, the mapping in discovery order.
    """
    by_role: dict[str, AgentCard] = {}
    for card in team_card.agent_cards.values():
        by_role.setdefault(card.role, card)

    hireable: set[str] = set()
    for card in team_card.agent_profiles:
        hireable.add(card.role)
        by_role.setdefault(card.role, card)

    return by_role, hireable


def _member_refs(member: TeamCardMember) -> list[AgentRef]:
    """Return one ``AgentRef`` per instance *member* is spawned as."""
    return [AgentRef(name=name, role=member.card.role) for name in spawned_names(member)]


def derive_team_projection(team_card: TeamCard) -> TeamProjection:
    """Derive the structural projection of *team_card* — the ONLY place it happens.

    The caller's ``TeamCard`` and every ``AgentCard`` reachable from it are left
    untouched: the hireable flag is applied to a ``model_copy`` and never written
    back, so a card handed in still reads ``can_be_hired=False`` afterwards.

    Args:
        team_card: The declarative definition to project.

    Returns:
        A ``TeamProjection`` carrying the seven ``Process`` fields and the
        deduplicated cards the refs point at.
    """
    by_role, hireable = _collect_cards_by_role(team_card)

    cards: list[AgentCard] = []
    refs: list[AgentCardRef] = []
    for role, card in by_role.items():
        can_be_hired = role in hireable
        # Copy rather than mutate: the caller's card is not ours to flag.
        resolved = card.model_copy(update={"can_be_hired": True}) if can_be_hired else card
        cards.append(resolved)
        refs.append(
            AgentCardRef(
                role=role,
                card_hash=hash_agent_card(resolved),
                can_be_hired=can_be_hired,
            )
        )

    supervisors: list[AgentRef] = []
    for member in team_card.members:
        supervisors.extend(_member_refs(member))

    return TeamProjection(
        team_name=team_card.name,
        team_description=None,
        entry_point=_member_refs(team_card.entry_point)[0],
        supervisors=supervisors,
        agent_cards=refs,
        message_types=list(team_card.message_types),
        metadata_type=team_card.metadata_type,
        cards=cards,
    )
