"""Typed team metadata contract — indexable fields and their derived index.

A ``TeamMetadata`` subclass declares which of its fields are filterable by
marking them ``Field(json_schema_extra={"indexed": True})``. ``index_entries()``
flattens the marked fields into ``"key|value"`` strings that a backend stores as
a plain array of strings and filters on by **anchored prefix**.

``|`` is the separator, and the **value half is casefolded** — ``tenant="AzeFR"``
is stored as ``"tenant|azefr"``. A ``|`` inside a value is escaped as ``\\|`` so
that a value can never forge a second entry. Derivation and query construction
both go through :func:`make_index_entry`, so the fold and the escaping can never
diverge between the two sides.

Only scalars can be indexed (``str``, ``bool``, ``int``, ``UUID``, ``Enum``,
``date``, ``datetime``). The restriction is enforced when the subclass is
defined, not when a value is written. The model itself may hold nested models,
lists and dicts freely — they simply cannot be filtered on.

:class:`akgentic.team.reference_metadata.ReferenceTeamMetadata` is the worked
example: one subclass covering every state a client-side field descriptor can
report.

Implements ADR-24 §D4.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import date, datetime
from enum import Enum
from types import UnionType
from typing import Any, Union, get_args, get_origin

from akgentic.core.utils.serializer import SerializableBaseModel

INDEX_MARKER = "indexed"
"""``json_schema_extra`` key a field sets truthy to opt into indexing."""

INDEX_SEPARATOR = "|"
"""Separator between the key half and the value half of an index entry."""

_ESCAPED_SEPARATOR = "\\|"

_INDEXABLE_SCALARS: tuple[type, ...] = (bool, int, str, uuid.UUID, Enum, date, datetime)
"""Types permitted on an indexed field. ``float`` is excluded: float equality is
not a sound index key."""


def _render_scalar(value: Any) -> str:
    """Render an indexable scalar to its canonical index form.

    Dispatch order is load-bearing: ``Enum`` first (a ``StrEnum`` member is also
    a ``str``), ``bool`` before ``int`` (``bool`` subclasses ``int``), and
    ``datetime`` before ``date`` (``datetime`` subclasses ``date``).

    Args:
        value: The scalar to render.

    Returns:
        The canonical string form of *value*.
    """
    if isinstance(value, Enum):
        return _render_scalar(value.value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def make_index_entry(key: str, value: Any) -> str:
    """Build one ``"key|value"`` index entry, escaping ``|`` inside the value.

    This is the single place the entry format lives. Derivation calls it, and
    so must any caller building a query entry — the symmetry is what stops a
    value containing ``|`` from matching an entry it did not produce.

    The rendered value is **casefolded**; the key half is not. Folding on
    write is what buys case-insensitive matching without a case-insensitive
    query — a Mongo ``$options: "i"`` regex uses no index at all, so the seek
    the multikey index exists for would be gone (ADR-28 §D2). The fold happens
    here and nowhere else, so query construction and derivation cannot drift.

    Order is load-bearing: fold the rendered value, then escape ``|``, then
    prepend the key. Folding the *finished* entry would also fold the key —
    which passes today only because every declared key happens to be lowercase,
    and breaks the day a ``caseId`` field is declared.

    Args:
        key: Field name. Python identifiers cannot contain ``|``, so the key
            half is unforgeable and is not escaped.
        value: The scalar value to render.

    Returns:
        The entry string, e.g. ``"tenant|acme"``.
    """
    rendered = _render_scalar(value).casefold().replace(INDEX_SEPARATOR, _ESCAPED_SEPARATOR)
    return f"{key}{INDEX_SEPARATOR}{rendered}"


def make_index_prefix_groups(metadata: Mapping[str, list[str]] | None) -> list[list[str]]:
    """Render a ``list_teams`` metadata filter to its anchored match prefixes.

    The query-side counterpart of :func:`make_index_entry`, shared by all four
    ``EventStore`` implementations so the combination rule, the empty-term rule
    and the bare-``str`` rejection cannot drift per backend. Each prefix is
    matched against a stored entry with *stored startswith prefix* — never the
    reverse, which would let a term crafted to span two entries match.

    **The grouping is the combination rule.** One group per key: prefixes
    *within* a group are a disjunction, and the groups themselves are a
    conjunction. That is ordinary faceted search — same field ORs, different
    fields AND — and it is what repeating a query parameter means everywhere
    else. Terms on one key had to OR: under prefix matching two terms on the
    same key are either redundant (one is a prefix of the other) or jointly
    unsatisfiable, so a conjunction there would make the whole list dead weight.

    An **empty term contributes no constraint** for its key. ``make_index_entry(k, "")``
    yields ``"k|"``, whose prefix matches every entry for that key, so treating a
    blank as "no term" makes the store's answer independent of whether the caller
    sent one. Under a disjunction that rule is load-bearing rather than tidy: a
    blank surviving into a group would match *everything* for that key and
    silently widen the answer instead of narrowing it.

    **A key whose terms all render away yields no group at all** — never an empty
    one. This is structural, not a guard each backend has to remember: an empty
    group would become an empty ``$or``, which MongoDB rejects outright, and an
    empty conjunction, which matches zero documents. ADR-24 recorded the
    ``$all``-over-an-empty-array version of that hazard; the disjunction re-opens
    it one level further down.

    Args:
        metadata: Mapping of indexed field name to a list of prefix terms, or
            ``None`` for no metadata filter.

    Returns:
        One non-empty group of rendered prefixes per contributing key, in
        mapping order then term order — deterministic, so a backend can assert
        on the query it builds.

    Raises:
        TypeError: If a value is a bare ``str`` rather than a list of terms.
            ``str`` is itself a sequence of ``str``, so ``{"tenant": "acme"}``
            would otherwise filter on four one-character terms and return
            plausible wrong rows. A caller mypy does not cover must fail loudly.
    """
    if not metadata:
        return []
    groups: list[list[str]] = []
    for key, terms in metadata.items():
        if isinstance(terms, str | bytes):
            msg = (
                f"list_teams(metadata=...) takes a list of terms per key, but "
                f"{key!r} was given the bare {type(terms).__name__} {terms!r}. "
                f"Pass [{terms!r}] instead — a bare string would be read as one "
                f"term per character."
            )
            raise TypeError(msg)
        group = [make_index_entry(key, term) for term in terms if term]
        if group:
            groups.append(group)
    return groups


def _is_indexed(json_schema_extra: Any) -> bool:
    """Report whether a field's ``json_schema_extra`` marks it as indexed.

    ``json_schema_extra`` may legally be ``None``, a dict, or a callable that
    mutates the schema; only a dict carrying a truthy marker counts.
    """
    return isinstance(json_schema_extra, dict) and bool(json_schema_extra.get(INDEX_MARKER, False))


def _unwrap_optional(annotation: Any) -> Any:
    """Strip ``None`` from ``T | None`` / ``Optional[T]``, returning ``T``.

    Returns the annotation unchanged when it is not a union, and ``None`` when
    the union holds more than one non-``None`` member (not indexable).
    """
    if get_origin(annotation) not in (Union, UnionType):
        return annotation
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    return args[0] if len(args) == 1 else None


def _is_indexable(annotation: Any) -> bool:
    """Report whether *annotation* is a permitted indexed-field type."""
    inner = _unwrap_optional(annotation)
    if get_origin(inner) is not None or not isinstance(inner, type):
        return False
    return issubclass(inner, _INDEXABLE_SCALARS)


class TeamMetadata(SerializableBaseModel):
    """Base for a team's typed business metadata.

    Subclasses declare their own fields and mark the filterable ones with
    ``Field(json_schema_extra={"indexed": True})``. A subclass with no marked
    field is legal — it is simply not filterable.

    **A nullable field must carry ``= None``** — write ``owner: str | None =
    None``, never ``owner: str | None``. A client-facing field descriptor
    reports ``mandatory`` as *required and not nullable*, so a required-nullable
    field is reported non-mandatory and yet cannot be satisfied by leaving the
    input blank: the server answers 422 "field required" and the user has no way
    to clear it. Nothing enforces this at class-definition time, because a
    required-nullable field is a valid declaration for a client that is not a
    form; it is a rule for the declaring author to keep.

    See :class:`akgentic.team.reference_metadata.ReferenceTeamMetadata` for a
    worked declaration.
    """

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Reject indexed fields that are not scalars, at class-definition time.

        Pydantic calls this after the model is fully built, so ``model_fields``
        is populated — unlike ``__init_subclass__``, which would see nothing and
        let a bad declaration through until a write.

        Raises:
            TypeError: If a field marked indexed is annotated with anything but
                a permitted scalar (optionally ``| None``).
        """
        super().__pydantic_init_subclass__(**kwargs)
        for name, field in cls.model_fields.items():
            if not _is_indexed(field.json_schema_extra):
                continue
            if not _is_indexable(field.annotation):
                msg = (
                    f"{cls.__name__}.{name} is marked indexed but is annotated "
                    f"{field.annotation!r}. Indexed fields must be str, bool, int, UUID, "
                    f"Enum, date or datetime (optionally '| None')."
                )
                raise TypeError(msg)

    @classmethod
    def indexed_fields(cls) -> list[str]:
        """Return the names of the indexed fields, in declaration order.

        Fields inherited from an intermediate subclass are included, and come
        first — Pydantic collects base-class fields before the subclass's own.
        """
        return [
            name for name, field in cls.model_fields.items() if _is_indexed(field.json_schema_extra)
        ]

    def index_entries(self) -> list[str]:
        """Return one ``"key|value"`` entry per set indexed field.

        An indexed field holding ``None`` emits nothing — absent is not the
        empty string, which emits ``"key|"``.
        """
        entries: list[str] = []
        for name in type(self).indexed_fields():
            value = getattr(self, name)
            if value is None:
                continue
            entries.append(make_index_entry(name, value))
        return entries


def derive_metadata_indexes(metadata: SerializableBaseModel | None) -> list[str]:
    """Derive the flattened index for *metadata* — the ONLY place it is computed.

    Every write path calls this, so a team's metadata value and its index are
    never written independently. A second derivation site is how the index
    silently starts lying about what is stored.

    Args:
        metadata: The team's metadata value, or ``None``.

    Returns:
        ``metadata.index_entries()`` for a ``TeamMetadata``; ``[]`` for ``None``
        and for any other model — a card may legally declare a plain model as
        its ``metadata_type``, it simply carries no indexable contract.
    """
    if isinstance(metadata, TeamMetadata):
        return metadata.index_entries()
    return []
