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
from typing import TYPE_CHECKING

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
from akgentic.team.ports import AgentCardNotFoundError

if TYPE_CHECKING:
    from akgentic.team.ports import EventStore

_HIREABLE_FIELD = "can_be_hired"
"""The one card field that is a team's policy rather than the card's content.

Named once because three things must agree about it: the hash excludes it, the
store normalises it away, and ``AgentCardRef`` carries it instead. A test pins
that it is a declared field of ``AgentCard`` — the exclusion is a ``pop`` that
tolerates the key's absence, so without that pin a rename in ``akgentic-core``
would make every guard here vacuously green.
"""


def hash_agent_card(card: AgentCard) -> str:
    """Return a stable content hash for *card*.

    Two canonicalisation decisions, both load-bearing:

    * **Serialization is whatever** ``SerializableBaseModel`` **produces**, asked
      for with ``mode="json"``. The mode is belt-and-braces only: the framework's
      ``@model_serializer`` takes no ``info`` argument, so it never sees the mode
      and builds the payload itself — every value, including the
      ``{"__type__": ...}`` marker a ``type`` field becomes, is already a JSON
      primitive by the time Pydantic sees it. The mode is kept so the digest
      still lands on primitives if that serializer is ever narrowed; nothing here
      may *depend* on it.
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
    payload.pop(_HIREABLE_FIELD, None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def storable_agent_card(card: AgentCard) -> AgentCard:
    """Return the form of *card* the content-addressed store holds.

    ``can_be_hired`` is normalised back to its declared default, so the stored
    bytes are a **pure function of the hash**. Excluding the flag from the hash
    is necessary but not sufficient: ``derive_team_projection`` hands over cards
    already carrying their final hireability, so two teams reaching one role with
    different policies would otherwise write the same key with different content
    and whichever wrote last would silently decide what every other team reads
    back — including a restore that hands the orchestrator a card whose own flag
    contradicts the authoritative one on its ``AgentCardRef``.

    The default is read off ``AgentCard.model_fields`` rather than written as a
    literal, so a card whose default ever changes normalises to the new one, and
    a card that loses the field raises here instead of normalising nothing.

    The caller's card is untouched — the flag is applied to a ``model_copy``,
    the same discipline ``derive_team_projection`` follows in the other
    direction.

    Args:
        card: The card to normalise.

    Returns:
        A copy with ``can_be_hired`` at its default. ``hash_agent_card`` of the
        result equals ``hash_agent_card`` of the input, by construction.
    """
    default = AgentCard.model_fields[_HIREABLE_FIELD].default
    return card.model_copy(update={_HIREABLE_FIELD: default})


def resolve_agent_cards(refs: list[AgentCardRef], store: EventStore) -> list[AgentCard]:
    """Resolve *refs* against the card store — the ONE place a hash becomes a card.

    One batch ``load_agent_cards`` call for the whole set, whatever its size:
    the restorer, and story 31-5's migration verification after it, go through
    here rather than reading per role. A per-role read returns the identical
    result, which is exactly why only a call-count assertion catches it coming
    back.

    Each returned card carries its ref's ``can_be_hired``, applied with
    ``model_copy(update=...)`` — never by mutating the card the store handed
    back, which a caching backend may share with the next caller.

    Args:
        refs: The ``Process``'s card refs, in order.
        store: The store to resolve against.

    Returns:
        One card per ref, in the same order.

    Raises:
        AgentCardNotFoundError: If any ref's hash is absent from the store,
            naming the first unresolved ref's role AND hash. A team that cannot
            describe one of its agents must refuse to restore (FR14).
    """
    resolved = store.load_agent_cards([ref.card_hash for ref in refs])
    cards: list[AgentCard] = []
    for ref in refs:
        stored = resolved.get(ref.card_hash)
        if stored is None:
            msg = f"No agent card stored for role '{ref.role}' at hash {ref.card_hash}"
            raise AgentCardNotFoundError(msg)
        cards.append(stored.model_copy(update={_HIREABLE_FIELD: ref.can_be_hired}))
    return cards


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
            order, each already carrying its final ``can_be_hired`` value.
            ``TeamManager.create_team`` hands this straight to
            ``EventStore.save_agent_cards``, before the ``Process`` that
            references it. The store normalises the hireable flag away — it
            belongs to the ``AgentCardRef``, not to the stored card — so the
            flag set here serves the *other* consumer, the orchestrator
            registration.
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
        description="The deduped cards agent_cards points at; persisted by save_agent_cards",
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
        # One ref, not a list: TeamFactory.build rejects an entry point whose
        # headcount is not 1 before any of this runs, so the expansion below can
        # only ever yield a single name. Indexing states that invariant rather
        # than quietly discarding names.
        entry_point=_member_refs(team_card.entry_point)[0],
        supervisors=supervisors,
        agent_cards=refs,
        message_types=list(team_card.message_types),
        metadata_type=team_card.metadata_type,
        cards=cards,
    )
