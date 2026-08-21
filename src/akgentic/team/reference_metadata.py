"""One worked ``TeamMetadata`` subclass — documentation as much as code.

The metadata contract in :mod:`akgentic.team.metadata` is read by clients, not
just written by deployments: a catalog projects a declared ``metadata_type``
into a per-field descriptor (key, description, index, mandatory) and a form
renders it. This module holds the single example that exercises every state
that projection can report, so the contract has something to point at.
"""

from __future__ import annotations

from pydantic import Field

from akgentic.team.metadata import TeamMetadata


class ReferenceTeamMetadata(TeamMetadata):
    """A reference example of a team metadata contract — executable documentation.

    This is **not** a business contract any deployment is expected to adopt, and
    **not** a base to inherit from: declare your own ``TeamMetadata`` subclass
    with your own fields. It exists so that a single catalog entry exercises all
    four states a client-side field descriptor can report, one field each:

    - ``tenant`` — indexed and mandatory (no default, not nullable, described);
    - ``case_ref`` — indexed and optional (nullable, and so carrying ``= None``);
    - ``tier`` — not indexed, optional through a non-``None`` default;
    - ``note`` — declares no description, so a client's fallback to the field
      name as a label has something to fall back on.

    Declaration order is the order a form renders the fields, so it is kept.
    """

    tenant: str = Field(
        json_schema_extra={"indexed": True},
        description="Slug of the tenant the team belongs to.",
    )
    case_ref: str | None = Field(
        default=None,
        json_schema_extra={"indexed": True},
        description="Reference of the case the team was opened for, if any.",
    )
    tier: str = Field(
        default="standard",
        description="Service tier the team runs under.",
    )
    note: str = ""
