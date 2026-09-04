"""Shared test fixtures for akgentic-team tests."""

from __future__ import annotations

from typing import Any

from akgentic.team.models import TeamCard
from akgentic.team.projection import derive_team_projection


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
