"""Tests for port protocols: NullServiceRegistry conformance and behavior.

AC: 3, 5 — NullServiceRegistry satisfies ServiceRegistry via structural subtyping,
with all no-op implementations verified.
"""

from __future__ import annotations

import uuid

from akgentic.team.ports import NullServiceRegistry, ServiceRegistry


class TestNullServiceRegistry:
    """Verify NullServiceRegistry satisfies ServiceRegistry protocol (AC: 3, 5)."""

    def test_satisfies_service_registry_protocol(self) -> None:
        """NullServiceRegistry is recognized as a ServiceRegistry instance."""
        registry = NullServiceRegistry()
        assert isinstance(registry, ServiceRegistry)

    def test_find_team_returns_none(self) -> None:
        """find_team always returns None in single-process mode."""
        registry = NullServiceRegistry()
        assert registry.find_team(uuid.uuid4()) is None

    def test_get_active_instances_returns_empty_list(self) -> None:
        """get_active_instances always returns empty list in single-process mode."""
        registry = NullServiceRegistry()
        assert registry.get_active_instances() == []

    def test_register_instance_is_noop(self) -> None:
        """register_instance executes without error."""
        registry = NullServiceRegistry()
        result = registry.register_instance(uuid.uuid4())
        assert result is None

    def test_deregister_instance_is_noop(self) -> None:
        """deregister_instance executes without error."""
        registry = NullServiceRegistry()
        result = registry.deregister_instance(uuid.uuid4())
        assert result is None

    def test_register_team_is_noop(self) -> None:
        """register_team executes without error."""
        registry = NullServiceRegistry()
        result = registry.register_team(uuid.uuid4(), uuid.uuid4())
        assert result is None

    def test_deregister_team_is_noop(self) -> None:
        """deregister_team executes without error."""
        registry = NullServiceRegistry()
        result = registry.deregister_team(uuid.uuid4(), uuid.uuid4())
        assert result is None
