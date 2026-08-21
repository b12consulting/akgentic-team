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

    - ``tenant`` — indexed and mandatory (no default, not nullable, described),
      and **patterned**;
    - ``case_ref`` — indexed and optional (nullable, and so carrying ``= None``),
      and **patterned**;
    - ``tier`` — not indexed, optional through a non-``None`` default, and
      unpatterned, so the ``pattern: None`` state is still demonstrated;
    - ``note`` — declares no description, so a client's fallback to the field
      name as a label has something to fall back on.

    Declaration order is the order a form renders the fields, so it is kept.

    **Two fields carry a pattern, and that is deliberate.** A pattern lands in
    two different places in a model's JSON Schema depending on the field's
    nullability: at the top level for ``tenant: str``, and nested inside
    ``anyOf`` for ``case_ref: str | None``. A catalog projecting these
    descriptors has to find both, so this model declares both — one patterned
    field would only ever exercise one encoding.

    Both patterns are written to be **ECMA-262-safe** — no named groups, no
    lookbehind, nothing whose meaning differs between Python's ``re`` and a
    browser's ``RegExp``. A client receives the pattern string verbatim and
    runs it, so a Python-only construct would validate one way on the server
    and another in the form.

    **A pattern never makes a field mandatory.** ``case_ref`` is patterned and
    still optional: the pattern constrains a value that is *present*, and says
    nothing about whether one has to be. Leaving it blank stays legal.
    """

    tenant: str = Field(
        pattern=r"^[a-z][a-z0-9-]{2,31}$",
        json_schema_extra={"indexed": True},
        description="Slug of the tenant the team belongs to.",
    )
    case_ref: str | None = Field(
        default=None,
        pattern=r"^[a-z]+-\d+$",
        json_schema_extra={"indexed": True},
        description="Reference of the case the team was opened for, if any.",
    )
    tier: str = Field(
        default="standard",
        description="Service tier the team runs under.",
    )
    note: str = ""
