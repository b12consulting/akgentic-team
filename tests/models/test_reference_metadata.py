"""Tests for the shipped reference TeamMetadata subclass.

These pin the base-class behaviours *against the worked example*: they are a
regression guard on the reference model, not new functionality. The model is
what a catalog entry will declare, so its four descriptor states are the
contract a form-driven client renders.
"""

from __future__ import annotations

from typing import get_args

import pytest
from akgentic.core.utils.deserializer import import_class
from pydantic import ValidationError
from pydantic.fields import FieldInfo

from akgentic.team.metadata import derive_metadata_indexes, make_index_entry
from akgentic.team.reference_metadata import ReferenceTeamMetadata


def _is_mandatory(field: FieldInfo) -> bool:
    """Compute ``mandatory`` the way a client-facing descriptor computes it.

    Required *and* not nullable. ``is_required()`` alone is not enough: it is
    ``True`` for ``owner: str | None`` with no default, which a form cannot
    satisfy by leaving the input blank. ``get_args`` returns ``()`` for a
    non-union annotation, so the ``type(None)`` check is safe for ``tenant: str``.
    """
    args = get_args(field.annotation)
    return field.is_required() and type(None) not in args


def _populated() -> ReferenceTeamMetadata:
    """A reference instance with every indexed field set."""
    return ReferenceTeamMetadata(tenant="acme", case_ref="case-1", tier="premium", note="hello")


def _sparse() -> ReferenceTeamMetadata:
    """A reference instance whose optional indexed field is unset."""
    return ReferenceTeamMetadata(tenant="contoso")


class TestFieldDescriptors:
    """The key and description halves of what a client-facing descriptor reports."""

    def test_the_four_fields_are_declared_in_the_order_a_form_renders_them(self) -> None:
        assert list(ReferenceTeamMetadata.model_fields) == ["tenant", "case_ref", "tier", "note"]

    def test_exactly_one_field_declares_no_description(self) -> None:
        undescribed = {
            name
            for name, field in ReferenceTeamMetadata.model_fields.items()
            if not field.description
        }
        assert undescribed == {"note"}


class TestIndexedFields:
    """The reference model declares exactly two indexed fields, in order."""

    def test_returns_the_two_marked_fields_in_declaration_order(self) -> None:
        assert ReferenceTeamMetadata.indexed_fields() == ["tenant", "case_ref"]


class TestIndexEntries:
    """index_entries() derives one entry per set indexed field."""

    def test_both_indexed_fields_set_yields_one_entry_each(self) -> None:
        assert _populated().index_entries() == [
            make_index_entry("tenant", "acme"),
            make_index_entry("case_ref", "case-1"),
        ]

    def test_unset_optional_indexed_field_yields_one_entry_and_no_trailing_pipe(self) -> None:
        entries = _sparse().index_entries()
        assert entries == [make_index_entry("tenant", "contoso")]
        assert len(entries) == 1
        assert not any(entry.endswith("|") for entry in entries)


class TestDeriveMetadataIndexes:
    """The shared derivation helper agrees with the model, populated or not."""

    def test_matches_index_entries_when_fully_populated(self) -> None:
        meta = _populated()
        assert derive_metadata_indexes(meta) == meta.index_entries()

    def test_matches_index_entries_when_the_optional_field_is_none(self) -> None:
        meta = _sparse()
        assert derive_metadata_indexes(meta) == meta.index_entries()


class TestMandatoryProjection:
    """Exactly one field is required and not nullable — the rule the model documents."""

    def test_only_tenant_is_mandatory(self) -> None:
        mandatory = {
            name
            for name, field in ReferenceTeamMetadata.model_fields.items()
            if _is_mandatory(field)
        }
        assert mandatory == {"tenant"}

    def test_the_nullable_field_carries_a_none_default(self) -> None:
        """case_ref must not be required — that is the trap the rule closes."""
        assert not ReferenceTeamMetadata.model_fields["case_ref"].is_required()
        assert _sparse().case_ref is None


class TestPublicResolution:
    """The dotted path a catalog __type__ tag will carry resolves to the class."""

    def test_import_class_resolves_the_package_level_path(self) -> None:
        assert import_class("akgentic.team.ReferenceTeamMetadata") is ReferenceTeamMetadata


class TestSerializationRoundTrip:
    """The model persists the way Process.metadata persists it."""

    def test_dump_carries_the_model_tag_and_validate_restores_the_value(self) -> None:
        meta = _populated()
        dumped = meta.model_dump()
        assert "__model__" in dumped
        assert ReferenceTeamMetadata.model_validate(dumped) == meta

    def test_the_emitted_model_tag_resolves_to_the_class(self) -> None:
        """The branch Process.metadata restore takes — it is typed SerializableBaseModel | None."""
        dumped = _populated().model_dump()
        assert import_class(dumped["__model__"]) is ReferenceTeamMetadata


class TestDeclaredPatterns:
    """Where each pattern lands in the JSON Schema — not merely that one was written.

    A catalog reads these descriptors out of ``model_json_schema()``, and a
    pattern sits in a different place depending on the field's nullability. A
    test asserting only that ``Field(pattern=...)`` was declared would pass even
    if the projection could never find the value, so each shape is pinned where
    it actually appears.
    """

    def test_the_required_field_carries_its_pattern_at_the_top_level(self) -> None:
        prop = ReferenceTeamMetadata.model_json_schema(by_alias=False)["properties"]["tenant"]
        assert prop["pattern"] == r"^[a-z][a-z0-9-]{2,31}$"

    def test_the_nullable_field_carries_its_pattern_nested_in_anyof(self) -> None:
        """The shape a top-level lookup drops -- and the one authors are told to write."""
        prop = ReferenceTeamMetadata.model_json_schema(by_alias=False)["properties"]["case_ref"]
        assert "pattern" not in prop
        assert [b["pattern"] for b in prop["anyOf"] if "pattern" in b] == [r"^[a-z]+-\d+$"]

    def test_the_unpatterned_fields_declare_none(self) -> None:
        props = ReferenceTeamMetadata.model_json_schema(by_alias=False)["properties"]
        for name in ("tier", "note"):
            assert "pattern" not in props[name]
            assert "anyOf" not in props[name]

    def test_both_patterns_accept_the_fixture_values(self) -> None:
        """The fixtures are the model's own worked example; a pattern must not invalidate them."""
        assert _populated().tenant == "acme"
        assert _populated().case_ref == "case-1"
        assert _sparse().tenant == "contoso"

    def test_a_malformed_value_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReferenceTeamMetadata(tenant="Acme Corp")

    def test_the_pattern_does_not_make_the_optional_field_mandatory(self) -> None:
        """A pattern constrains a value that is present; it never requires one."""
        assert _sparse().case_ref is None
