"""Team lifecycle management — create, resume, stop, delete teams with event-sourced persistence.

Public API surface for the akgentic-team package. All public types are
re-exported here via explicit __all__.
"""

from __future__ import annotations

from akgentic.team.factory import TeamFactory
from akgentic.team.models import (
    AgentStateSnapshot,
    PersistedEvent,
    Process,
    TeamCard,
    TeamCardMember,
    TeamRuntime,
    TeamStatus,
)
from akgentic.team.ports import (
    EventStore,
    NullServiceRegistry,
    ServiceRegistry,
)

__version__ = "1.0.0-alpha.1"

__all__: list[str] = [
    "__version__",
    "AgentStateSnapshot",
    "EventStore",
    "NullServiceRegistry",
    "PersistedEvent",
    "Process",
    "ServiceRegistry",
    "TeamCard",
    "TeamCardMember",
    "TeamFactory",
    "TeamRuntime",
    "TeamStatus",
]
