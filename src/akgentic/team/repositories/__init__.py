"""EventStore implementations: YAML (default) and MongoDB (optional)."""

from __future__ import annotations

from akgentic.team.repositories.yaml import YamlEventStore

__all__: list[str] = [
    "YamlEventStore",
]
