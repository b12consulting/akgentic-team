"""Tests for the TeamMetadata indexing contract and the shared derivation helper."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum, IntEnum, StrEnum
from typing import Any, Optional

import pytest
from akgentic.core.utils.serializer import SerializableBaseModel
from pydantic import Field

from akgentic.team.metadata import (
    TeamMetadata,
    derive_metadata_indexes,
    make_index_entry,
)

from .conftest import AcmeTeamMetadata, PlainCardMetadata

_REF = uuid.UUID("11111111-2222-3333-4444-555555555555")
_OPENED_ON = date(2026, 8, 11)
_OPENED_AT = datetime(2026, 8, 11, 14, 30, 15, tzinfo=UTC)


class Channel(StrEnum):
    """String-valued enum — a member is simultaneously a str and an Enum."""

    EMAIL = "email"
    CHAT = "chat"


class Priority(IntEnum):
    """Int-valued enum — a member is simultaneously an int and an Enum."""

    LOW = 1
    HIGH = 3


class Region(Enum):
    """Plain enum whose value differs from its member name."""

    EU = "eu-west"


class Detail(SerializableBaseModel):
    """Nested model used to prove nesting is fine as long as it is unmarked."""

    note: str = ""


def _tweak_schema(schema: dict[str, Any]) -> None:
    """Callable form of json_schema_extra — must not be read as an index marker."""
    schema["title"] = "tweaked"


class ScalarMetadata(TeamMetadata):
    """One indexed field per permitted scalar type."""

    label: str = Field(default="acme", json_schema_extra={"indexed": True})
    active: bool = Field(default=True, json_schema_extra={"indexed": True})
    count: int = Field(default=7, json_schema_extra={"indexed": True})
    ref: uuid.UUID = Field(default=_REF, json_schema_extra={"indexed": True})
    channel: Channel = Field(default=Channel.EMAIL, json_schema_extra={"indexed": True})
    priority: Priority = Field(default=Priority.LOW, json_schema_extra={"indexed": True})
    region: Region = Field(default=Region.EU, json_schema_extra={"indexed": True})
    opened_on: date = Field(default=_OPENED_ON, json_schema_extra={"indexed": True})
    opened_at: datetime = Field(default=_OPENED_AT, json_schema_extra={"indexed": True})


class MarkerMetadata(TeamMetadata):
    """Every shape json_schema_extra can take, only one of which indexes."""

    marked: str = Field(default="", json_schema_extra={"indexed": True})
    bare: str = Field(default="")
    explicitly_off: str = Field(default="", json_schema_extra={"indexed": False})
    unrelated_key: str = Field(default="", json_schema_extra={"example": "x"})
    callable_extra: str = Field(default="", json_schema_extra=_tweak_schema)


class TenantMetadata(TeamMetadata):
    """Intermediate subclass contributing an inherited indexed field."""

    tenant: str = Field(default="acme", json_schema_extra={"indexed": True})


class ContosoMetadata(TenantMetadata):
    """Leaf subclass — inherits ``tenant`` and adds its own indexed field."""

    case_ref: str = Field(default="", json_schema_extra={"indexed": True})


class UnfilterableMetadata(TeamMetadata):
    """Legal metadata with no indexed field at all."""

    note: str = ""


class RichMetadata(TeamMetadata):
    """Unmarked fields of any type are accepted — only indexed ones are restricted."""

    tenant: str = Field(default="acme", json_schema_extra={"indexed": True})
    detail: Detail | None = None
    tags: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    ratio: float = 0.0


class OptionalScalarMetadata(TeamMetadata):
    """Optionality is orthogonal to indexability, in both spellings."""

    case_ref: str | None = Field(default=None, json_schema_extra={"indexed": True})
    channel: Optional[Channel] = Field(  # noqa: UP007, UP045
        default=None, json_schema_extra={"indexed": True}
    )


class TestIndexedFields:
    """indexed_fields() reads the json_schema_extra marker."""

    def test_returns_only_marked_fields_in_declaration_order(self) -> None:
        assert MarkerMetadata.indexed_fields() == ["marked"]

    def test_includes_fields_inherited_from_an_intermediate_subclass(self) -> None:
        assert ContosoMetadata.indexed_fields() == ["tenant", "case_ref"]

    def test_subclass_with_no_marked_field_is_legal(self) -> None:
        assert UnfilterableMetadata.indexed_fields() == []
        assert UnfilterableMetadata(note="hi").index_entries() == []

    def test_unmarked_fields_of_any_type_are_ignored(self) -> None:
        assert RichMetadata.indexed_fields() == ["tenant"]


class TestIndexEntries:
    """index_entries() derives one entry per set indexed field."""

    def test_one_entry_per_set_field_in_indexed_field_order(self) -> None:
        meta = ContosoMetadata(tenant="acme", case_ref="c-1")
        assert meta.index_entries() == ["tenant|acme", "case_ref|c-1"]

    def test_unmarked_fields_emit_nothing(self) -> None:
        meta = RichMetadata(tenant="acme", tags=["a"], ratio=1.5)
        assert meta.index_entries() == ["tenant|acme"]

    def test_unset_optional_field_emits_no_entry(self) -> None:
        assert OptionalScalarMetadata().index_entries() == []

    def test_empty_string_is_not_the_same_as_absent(self) -> None:
        assert OptionalScalarMetadata(case_ref="").index_entries() == ["case_ref|"]

    def test_pipe_in_a_value_is_escaped_and_yields_one_entry(self) -> None:
        entries = OptionalScalarMetadata(case_ref="a|b").index_entries()
        assert entries == ["case_ref|a\\|b"]
        assert len(entries) == 1

    def test_derivation_and_query_share_the_same_primitive(self) -> None:
        """A query entry built by hand must equal the derived one, pipes included."""
        meta = OptionalScalarMetadata(case_ref="a|b")
        assert meta.index_entries() == [make_index_entry("case_ref", "a|b")]


class TestScalarRendering:
    """Each permitted scalar renders to its canonical form."""

    def test_str_renders_verbatim(self) -> None:
        assert make_index_entry("k", "acme") == "k|acme"

    def test_bool_renders_lowercase_not_python_repr(self) -> None:
        assert make_index_entry("k", True) == "k|true"
        assert make_index_entry("k", False) == "k|false"

    def test_int_renders_as_decimal(self) -> None:
        assert make_index_entry("k", 7) == "k|7"

    def test_uuid_renders_canonical(self) -> None:
        assert make_index_entry("k", _REF) == f"k|{_REF}"

    def test_str_enum_renders_by_value(self) -> None:
        assert make_index_entry("k", Channel.EMAIL) == "k|email"

    def test_int_enum_renders_by_value(self) -> None:
        assert make_index_entry("k", Priority.HIGH) == "k|3"

    def test_enum_renders_value_never_member_name(self) -> None:
        assert make_index_entry("k", Region.EU) == "k|eu-west"

    def test_date_renders_isoformat(self) -> None:
        assert make_index_entry("k", _OPENED_ON) == "k|2026-08-11"

    def test_datetime_renders_isoformat_not_date_only(self) -> None:
        assert make_index_entry("k", _OPENED_AT) == f"k|{_OPENED_AT.isoformat()}"

    def test_unexpected_type_falls_back_to_str(self) -> None:
        """make_index_entry is public and takes query values too -- it never raises."""
        assert make_index_entry("k", Decimal("1.5")) == "k|1.5"

    def test_every_scalar_field_renders_in_one_pass(self) -> None:
        assert ScalarMetadata().index_entries() == [
            "label|acme",
            "active|true",
            "count|7",
            f"ref|{_REF}",
            "channel|email",
            "priority|1",
            "region|eu-west",
            "opened_on|2026-08-11",
            f"opened_at|{_OPENED_AT.isoformat()}",
        ]

    def test_rendering_is_stable_across_repeated_calls(self) -> None:
        meta = ScalarMetadata()
        assert meta.index_entries() == meta.index_entries()

    def test_rendering_survives_a_serialization_round_trip(self) -> None:
        meta = ScalarMetadata()
        restored = ScalarMetadata.model_validate(meta.model_dump())
        assert restored.index_entries() == meta.index_entries()


class TestScalarOnlyRestriction:
    """Marking a non-scalar raises when the class statement executes."""

    def test_nested_model_field_is_rejected(self) -> None:
        with pytest.raises(TypeError) as excinfo:

            class BadNested(TeamMetadata):
                detail: Detail = Field(json_schema_extra={"indexed": True})

        assert "detail" in str(excinfo.value)
        assert "Detail" in str(excinfo.value)

    def test_list_field_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="tags"):

            class BadList(TeamMetadata):
                tags: list[str] = Field(default_factory=list, json_schema_extra={"indexed": True})

    def test_dict_field_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="scores"):

            class BadDict(TeamMetadata):
                scores: dict[str, str] = Field(
                    default_factory=dict, json_schema_extra={"indexed": True}
                )

    def test_float_field_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="ratio"):

            class BadFloat(TeamMetadata):
                ratio: float = Field(default=0.0, json_schema_extra={"indexed": True})

    def test_multi_member_union_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="either"):

            class BadUnion(TeamMetadata):
                either: str | int = Field(default="", json_schema_extra={"indexed": True})

    def test_error_names_the_field_and_its_annotation(self) -> None:
        with pytest.raises(TypeError) as excinfo:

            class BadNamed(TeamMetadata):
                tags: list[str] = Field(default_factory=list, json_schema_extra={"indexed": True})

        message = str(excinfo.value)
        assert "BadNamed.tags" in message
        assert "list[str]" in message

    def test_optional_scalar_is_accepted(self) -> None:
        assert OptionalScalarMetadata.indexed_fields() == ["case_ref", "channel"]

    def test_optional_scalar_renders_when_set(self) -> None:
        meta = OptionalScalarMetadata(case_ref="c-1", channel=Channel.CHAT)
        assert meta.index_entries() == ["case_ref|c-1", "channel|chat"]

    def test_unmarked_non_scalar_fields_are_accepted(self) -> None:
        meta = RichMetadata(detail=Detail(note="n"), tags=["a"], scores={"s": 1.0}, ratio=2.5)
        assert meta.detail is not None
        assert meta.detail.note == "n"


class TestDeriveMetadataIndexes:
    """The single derivation helper both write paths call."""

    def test_returns_empty_for_none(self) -> None:
        assert derive_metadata_indexes(None) == []

    def test_returns_empty_for_a_non_team_metadata_model(self) -> None:
        assert derive_metadata_indexes(PlainCardMetadata(tenant="acme")) == []

    def test_returns_the_instance_entries_for_team_metadata(self) -> None:
        meta = AcmeTeamMetadata(tenant="acme", case_ref="c-1")
        assert derive_metadata_indexes(meta) == meta.index_entries()

    def test_returns_empty_for_metadata_with_no_indexed_field(self) -> None:
        assert derive_metadata_indexes(UnfilterableMetadata(note="hi")) == []
