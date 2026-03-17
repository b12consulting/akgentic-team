"""Team lifecycle management — create, resume, stop, delete teams with event-sourced persistence.

Public API surface for the akgentic-team package. All public types are
re-exported here via explicit __all__.
"""

from __future__ import annotations

from akgentic.team.models import TeamCard, TeamCardMember

__version__ = "1.0.0-alpha.1"

__all__: list[str] = [
    "__version__",
    "TeamCard",
    "TeamCardMember",
]
