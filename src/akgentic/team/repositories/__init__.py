"""EventStore implementations: YAML (default) and MongoDB (optional); the MongoDB ResourceStore."""

from __future__ import annotations

from akgentic.team.repositories.yaml import YamlEventStore

__all__: list[str] = [
    "YamlEventStore",
]

try:
    from akgentic.team.repositories.mongo import MongoEventStore  # noqa: F401
    from akgentic.team.repositories.mongo_resource import MongoResourceStore  # noqa: F401

    __all__.append("MongoEventStore")
    __all__.append("MongoResourceStore")
except ImportError:
    pass
