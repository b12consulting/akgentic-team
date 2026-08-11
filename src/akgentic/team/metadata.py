"""Typed team metadata contract — indexable fields and their derived index.

A ``TeamMetadata`` subclass declares which of its fields are filterable by
marking them ``Field(json_schema_extra={"indexed": True})``. ``index_entries()``
flattens the marked fields into ``"key|value"`` strings that a backend stores as
a plain array of strings and filters on by equality.

``|`` is the separator. A ``|`` inside a value is escaped as ``\\|`` so that a
value can never forge a second entry. Derivation and query construction both go
through :func:`make_index_entry`, so the two sides can never diverge.

Only scalars can be indexed (``str``, ``bool``, ``int``, ``UUID``, ``Enum``,
``date``, ``datetime``). The restriction is enforced when the subclass is
defined, not when a value is written. The model itself may hold nested models,
lists and dicts freely — they simply cannot be filtered on.

Implements ADR-24 §D4.
"""

from __future__ import annotations

import uuid
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

    Args:
        key: Field name. Python identifiers cannot contain ``|``, so the key
            half is unforgeable and is not escaped.
        value: The scalar value to render.

    Returns:
        The entry string, e.g. ``"tenant|acme"``.
    """
    rendered = _render_scalar(value).replace(INDEX_SEPARATOR, _ESCAPED_SEPARATOR)
    return f"{key}{INDEX_SEPARATOR}{rendered}"


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
