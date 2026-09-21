"""Validate public API exports from akgentic.team."""

from __future__ import annotations

import importlib

import pytest
from pydantic import ValidationError

import akgentic.team


def test_version_is_exported() -> None:
    """__version__ is exported and is a string."""
    assert hasattr(akgentic.team, "__version__")
    assert isinstance(akgentic.team.__version__, str)


def test_all_is_a_list() -> None:
    """__all__ is a list."""
    assert hasattr(akgentic.team, "__all__")
    assert isinstance(akgentic.team.__all__, list)


def test_all_entries_are_importable() -> None:
    """Every name in __all__ is importable from akgentic.team."""
    for name in akgentic.team.__all__:
        assert hasattr(akgentic.team, name), f"{name} listed in __all__ but not importable"


def test_version_in_all() -> None:
    """__version__ is listed in __all__."""
    assert "__version__" in akgentic.team.__all__


def test_metadata_contract_is_exported() -> None:
    """The metadata base, the reference model, the entry primitive and the helper are public."""
    for name in (
        "TeamMetadata",
        "ReferenceTeamMetadata",
        "make_index_entry",
        "derive_metadata_indexes",
    ):
        assert name in akgentic.team.__all__, f"{name} missing from __all__"
        assert hasattr(akgentic.team, name), f"{name} not importable from akgentic.team"


def test_the_query_side_helper_is_exported_alongside_the_entry_primitive() -> None:
    """Both halves of the index contract are public, or out-of-package stores drift.

    ``make_index_entry`` is public because query construction needs it;
    ``make_index_prefix_groups`` is the same argument one level up. It carries
    the combination rule, the empty-term rule and the bare-``str`` rejection,
    and every ``EventStore`` implementation outside this package — the infra
    tiers and the fakes each of them keeps — has to answer identically to the
    three in here. ``EventStore`` is not ``@runtime_checkable``, so nothing
    detects a hand-rolled reimplementation that gets one of those rules wrong;
    the only defence is that the shared helper is reachable.
    """
    assert "make_index_prefix_groups" in akgentic.team.__all__
    assert akgentic.team.make_index_prefix_groups({"tenant": ["AcM", ""]}) == [["tenant|acm"]]


def test_module_is_importable() -> None:
    """akgentic.team is importable as a module."""
    mod = importlib.import_module("akgentic.team")
    assert mod is not None


def test_the_card_store_surface_is_exported() -> None:
    """The error a consumer must catch, and the two functions it needs to raise it.

    ``AgentCardNotFoundError`` is the one an out-of-package caller has to be
    able to name — a restore that cannot resolve a card fails with it, and a
    consumer catching ``Exception`` instead is how a loud failure becomes a
    quiet one. ``resolve_agent_cards`` and ``storable_agent_card`` are exported
    beside it because every writer into, and reader out of, the store must use
    exactly these two — a second normalisation or a second resolution is how the
    store starts holding two blobs for one hash.
    """
    for name in ("AgentCardNotFoundError", "resolve_agent_cards", "storable_agent_card"):
        assert name in akgentic.team.__all__, f"{name} missing from __all__"
        assert hasattr(akgentic.team, name), f"{name} not importable from akgentic.team"


def test_the_card_store_enumeration_model_is_exported() -> None:
    """``AgentCardEntry`` is what ``list_agent_card_entries`` hands across the boundary.

    The consumer is out of package — ``akgentic-infra``'s reverse sweep — and it
    reads ``entry.first_seen_at``. A type it cannot name is a type it cannot
    annotate, which leaves the sweep passing a ``list[Any]`` around and losing
    the one distinction the model exists to make (a real ``None`` versus an age).
    """
    assert "AgentCardEntry" in akgentic.team.__all__
    assert hasattr(akgentic.team, "AgentCardEntry")


def test_the_enumeration_entry_never_defaults_its_age() -> None:
    """``first_seen_at`` is required, and ``None`` is an answer a caller STATES.

    A default would let a construction site omit the field and silently mean
    "unknown" when it simply forgot to ask — the one value that must never be
    produced by accident, since a consumer keys a deletion decision on it.
    """
    with pytest.raises(ValidationError):
        akgentic.team.AgentCardEntry(card_hash="a" * 64)  # type: ignore[call-arg]


def test_the_event_log_error_is_exported() -> None:
    """``EventLogUnreadableError`` must be nameable from outside the package.

    An out-of-package caller — the infra events endpoint above all — has to be
    able to tell "this log will not parse" from "your cursor is stale" and from
    every other failure. Without the export, the only way to catch it is
    ``except Exception``, which is how a loud failure becomes a quiet one again.
    """
    assert "EventLogUnreadableError" in akgentic.team.__all__
    assert hasattr(akgentic.team, "EventLogUnreadableError")
    # The type identity is half the contract: a caller that catches
    # EventNotFoundError must NOT also catch this one.
    assert not issubclass(
        akgentic.team.EventLogUnreadableError, akgentic.team.EventNotFoundError
    )
