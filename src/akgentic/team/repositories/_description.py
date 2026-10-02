"""The three values ``update_team_description`` writes, built in ONE place.

Every backend's ``update_team_description`` writes exactly ``team_description``,
``description_origin`` and ``updated_at``, and every backend's ``load_team``
validates what ``save_team`` wrote through ``Process.model_dump()``. The values
are therefore picked out of a ``model_dump()`` of a ``Process`` carrying them —
never the enum member or a raw ``datetime`` handed straight to ``yaml.safe_dump``,
BSON or ``json.dumps``, and never a hand-formatted ``updated_at``. A backend
that formatted the stamp itself would write a document its own ``load_team``
could refuse, and three backends formatting it separately would drift apart.
One helper, three callers, one definition.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from akgentic.team.models import (
    AgentCardRef,
    AgentRef,
    DescriptionOrigin,
    Process,
    TeamStatus,
)

DESCRIPTION_UPDATE_KEYS: tuple[str, str, str] = (
    "team_description",
    "description_origin",
    "updated_at",
)
"""The keys ``update_team_description`` writes, and the only keys it writes."""

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_TEMPLATE_ROLE = "description-update-template"


def _template_process() -> Process:
    """A minimal valid ``Process`` to dump the three values through.

    Nothing about it reaches storage: only the three keys are picked out of its
    dump. It exists so the values pass through exactly the serializer
    ``save_team`` uses, which is the one thing a reader is guaranteed to accept.
    """
    return Process(
        team_id=uuid.UUID(int=0),
        status=TeamStatus.RUNNING,
        created_at=_EPOCH,
        updated_at=_EPOCH,
        entry_point=AgentRef(name=_TEMPLATE_ROLE, role=_TEMPLATE_ROLE),
        agent_cards=[AgentCardRef(role=_TEMPLATE_ROLE, card_hash="")],
    )


def description_update_fields(
    description: str | None,
    origin: DescriptionOrigin,
) -> dict[str, object]:
    """Return the three stored values for a description write, serialised.

    ``updated_at`` is ``datetime.now(UTC)`` at the call. The result holds exactly
    :data:`DESCRIPTION_UPDATE_KEYS`, each in the form ``Process.model_dump()``
    emits for that field, so the document any backend writes it into reads back
    through ``load_team`` unchanged.

    Args:
        description: The new description, stored verbatim, or ``None``.
        origin: Who is writing.

    Returns:
        A three-key mapping ready for ``$set``, ``||`` or a raw-dict update.
    """
    dump = (
        _template_process()
        .model_copy(
            update={
                "team_description": description,
                "description_origin": origin,
                "updated_at": datetime.now(UTC),
            }
        )
        .model_dump()
    )
    return {key: dump[key] for key in DESCRIPTION_UPDATE_KEYS}
