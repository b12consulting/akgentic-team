# akgentic-team Examples

A progressive tutorial that walks through the akgentic-team package, from static team definitions to full lifecycle management with persistence and crash recovery.

Each example builds on concepts from previous ones. Start with example 01 and work through in order.

## Prerequisites

From the workspace root, install all packages:

```bash
uv sync --all-packages --all-extras
```

## Examples

| # | File | Summary | Status |
|---|------|---------|--------|
| 01 | `01_team_definition.py` | TeamCard & TeamCardMember hierarchies, agent_cards/supervisors inspection, Pydantic round-trip | Implemented |
| 02 | `02_team_factory.py` | TeamFactory.build(), TeamRuntime inspection, message sending, error paths, clean shutdown | Implemented |
| 03 | `03_team_manager.py` | TeamManager create/stop lifecycle, state machine transitions | Coming soon |
| 04 | `04_event_sourcing.py` | PersistenceSubscriber, YamlEventStore, event replay | Coming soon |
| 05 | `05_crash_recovery.py` | TeamRestorer, resume from persisted state | Coming soon |
| 06 | `06_mongodb_backend.py` | MongoEventStore with mongomock for testing | Coming soon |

Each `.py` file has a companion `.md` file with concepts, API patterns, and pitfalls.

## Running

From the workspace root:

```bash
uv run python packages/akgentic-team/examples/01_team_definition.py
uv run python packages/akgentic-team/examples/02_team_factory.py
```

Or from the package root (`packages/akgentic-team/`):

```bash
uv run python examples/01_team_definition.py
uv run python examples/02_team_factory.py
```

## Dependencies

- Examples 01-05 depend on `akgentic-core` and `akgentic-team` only.
- Example 06 additionally requires `mongomock`, available via the `[mongo]` extra.
