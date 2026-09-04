"""Shared test fixtures for akgentic-team tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from akgentic.team.models import TeamCard
from akgentic.team.projection import derive_team_projection

if TYPE_CHECKING:
    from akgentic.team.ports import EventStore


def projection_kwargs(team_card: TeamCard) -> dict[str, Any]:
    """Return the seven ``Process`` projection fields derived from *team_card*.

    Every test that constructs a ``Process`` by hand routes through here, so no
    fixture can hand-write a projection that the real write path would never
    produce — and none of them has to be revisited when the projection grows a
    field.
    """
    projection = derive_team_projection(team_card)
    return {
        "team_name": projection.team_name,
        "team_description": projection.team_description,
        "entry_point": projection.entry_point,
        "supervisors": projection.supervisors,
        "agent_cards": projection.agent_cards,
        "message_types": projection.message_types,
        "metadata_type": projection.metadata_type,
    }


def seed_agent_cards(store: EventStore, team_card: TeamCard) -> None:
    """Seed *store* with the cards a real ``create_team`` would have written.

    The counterpart of ``projection_kwargs``: that helper produces the
    ``AgentCardRef``s a hand-built ``Process`` carries, and this one puts the
    blobs those refs point at into the store — through the same derivation, so
    the hashes agree by construction rather than by a hand-written fixture that
    the real write path would never produce.

    Both halves are needed together. A ``Process`` seeded without its cards is a
    document whose hashes resolve against nothing, which restore now fails on
    loudly (``AgentCardNotFoundError``) instead of silently restoring a team the
    orchestrator cannot describe.
    """
    store.save_agent_cards(derive_team_projection(team_card).cards)
