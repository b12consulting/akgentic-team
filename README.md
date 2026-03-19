# akgentic-team

[![CI](https://github.com/b12consulting/akgentic-team/actions/workflows/ci.yml/badge.svg)](https://github.com/b12consulting/akgentic-team/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/jltournay/708bb547b8679308d083be7beaf4448a/raw/coverage.json)](https://github.com/b12consulting/akgentic-team/actions/workflows/ci.yml)

Team lifecycle management for the [Akgentic](https://github.com/b12consulting/akgentic-quick-start)
multi-agent framework. Create, resume, stop, and delete multi-agent teams
with event-sourced persistence and crash recovery.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Team Definitions](#team-definitions)
- [Lifecycle Management](#lifecycle-management)
- [Persistence](#persistence)
- [CLI](#cli)
- [Examples](#examples)
- [Development](#development)
- [License](#license)

## Overview

`akgentic-team` provides the runtime lifecycle layer for Akgentic agent
teams. It sits between static team definitions and the running actor system,
providing:

- **Declarative team definitions** via `TeamCard` / `TeamCardMember` models
  with hierarchical member trees
- **One-call team building** via `TeamFactory` — from a `TeamCard` to live
  Pykka actors with routing wired
- **Full lifecycle management** via `TeamManager` — create, stop, resume,
  delete with state machine enforcement
- **Event-sourced persistence** via `PersistenceSubscriber` — every message
  is captured for crash recovery
- **Crash recovery** via `TeamRestorer` — 3-phase restore protocol rebuilds
  teams from persisted events
- **Two storage backends** (YAML files and MongoDB) behind a common
  `EventStore` protocol
- **CLI** (`ak-team`) for managing team instances from the command line

```
TeamCard ──▶ TeamFactory.build() ──▶ TeamRuntime (live actors)
                                          │
                              TeamManager  │  PersistenceSubscriber
                              ┌────────────┤  ┌──────────────────┐
                              │ create     │  │ save_event()     │
                              │ stop       │  │ save_agent_state │
                              │ resume ◀───┼──│ (event sourcing) │
                              │ delete     │  └──────────────────┘
                              └────────────┘
                                    │
                              EventStore Protocol
                              ┌─────┴──────┐
                              │            │
                         YamlEventStore  MongoEventStore
```

## Installation

### Workspace Installation (Recommended)

This package is designed for use within the Akgentic monorepo workspace:

```bash
git clone git@github.com:b12consulting/akgentic-quick-start.git
cd akgentic-quick-start
git submodule update --init --recursive

uv venv
source .venv/bin/activate
uv sync --all-packages --all-extras
```

All dependencies (`akgentic-core`) resolve automatically via workspace
configuration.

### Optional Extras

```bash
# CLI (Typer + Rich)
uv sync --extra cli

# MongoDB backend
uv sync --extra mongo

# Everything
uv sync --all-extras
```

## Quick Start

Create a team, send a message, stop it, and resume it:

```python
from pathlib import Path
from akgentic.core import ActorSystem, AgentCard, BaseConfig, BaseState
from akgentic.core.akgent import Akgent
from akgentic.team import (
    TeamCard, TeamCardMember, TeamManager, YamlEventStore,
)

# Define a simple agent
class EchoAgent(Akgent):
    def receiveMsg_UserMessage(self, msg):
        print(f"Echo: {msg.content}")

# Build a team definition
card = AgentCard(
    role="Echo", description="Echoes messages",
    skills=["echo"], agent_class=EchoAgent,
    config=BaseConfig(name="@Echo", role="Echo"),
)
team_card = TeamCard(
    name="echo-team", description="Simple echo team",
    entry_point=TeamCardMember(card=card),
    members=[TeamCardMember(card=card)],
)

# Create and manage a team
actor_system = ActorSystem()
event_store = YamlEventStore(Path("./data"))
manager = TeamManager(actor_system=actor_system, event_store=event_store)

runtime = manager.create_team(team_card)
runtime.send("Hello!")  # → Echo: Hello!

# Stop and resume
manager.stop_team(runtime.id)
resumed = manager.resume_team(runtime.id)  # full state restored
resumed.send("Back!")   # → Echo: Back!

# Clean up
manager.stop_team(resumed.id)
manager.delete_team(resumed.id)
```

## Architecture

The package follows a layered architecture with strict upward dependency
flow:

```
┌──────────────────────────────────────────────┐
│  Interfaces: CLI (ak-team), Python API       │
├──────────────────────────────────────────────┤
│  TeamManager (lifecycle facade)              │
│  TeamFactory / TeamRestorer                  │
│  PersistenceSubscriber                       │
├──────────────────────────────────────────────┤
│  Models: TeamCard, TeamRuntime, Process      │
│  Ports:  EventStore, ServiceRegistry         │
├──────────────────────────────────────────────┤
│  Repositories: YamlEventStore, MongoEventStore│
└──────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Role |
|---|---|
| **Models** | Pydantic models for team definitions, runtime state, and persistence |
| **Ports** | Protocol-based abstractions for storage and service discovery |
| **Services** | TeamFactory (build), TeamManager (lifecycle), TeamRestorer (recovery) |
| **Repositories** | EventStore implementations — YAML and MongoDB |
| **Interfaces** | CLI commands and direct Python imports |

### State Machine

Teams follow a strict lifecycle:

```
    create_team()        stop_team()         delete_team()
  ──────────────▶ RUNNING ──────────▶ STOPPED ──────────▶ DELETED
                    ▲                    │
                    └────────────────────┘
                       resume_team()
```

## Team Definitions

### TeamCard

Declarative team structure with hierarchical member trees:

```python
team_card = TeamCard(
    name="research-team",
    description="A research team with lead and workers",
    entry_point=TeamCardMember(card=lead_card),
    members=[
        TeamCardMember(card=lead_card, members=[
            TeamCardMember(card=researcher_card, headcount=3),
            TeamCardMember(card=reviewer_card),
        ]),
    ],
)

team_card.agent_cards    # flat index of all AgentCards by name
team_card.supervisors    # AgentCards with subordinates
```

### TeamRuntime

Live handle to a running team, returned by `create_team()` and
`resume_team()`:

```python
runtime.send("Hello!")                    # send to entry point
runtime.send_to("@Reviewer", message)    # directed messaging
runtime.id                                # team UUID
runtime.addrs                             # agent name → ActorAddress
```

## Persistence

### Event Sourcing

Every message flowing through the orchestrator is captured by
`PersistenceSubscriber` as an append-only event. Agent state snapshots
are saved on `StateChangedMessage`.

### Storage Backends

**YAML (default)** — zero infrastructure, per-team directory layout:

```
data/{team-uuid}/
  team.yaml              # Process metadata (overwrite)
  events.yaml            # All events (append-only)
  states/{agent-id}.yaml # Agent state snapshots (overwrite)
```

**MongoDB** — install the `[mongo]` extra:

```python
from akgentic.team import MongoEventStore
import pymongo

db = pymongo.MongoClient("mongodb://localhost:27017")["akgentic"]
event_store = MongoEventStore(db)
# Collections: teams, events, agent_states
```

### Crash Recovery

`TeamRestorer` executes a 3-phase protocol:
1. **Load** persisted events and agent state snapshots
2. **Rebuild** agents from the event log (Orchestrator first, then others)
3. **Replay** all events to reconstruct full state including LLM context

## CLI

The `ak-team` command is available when the `[cli]` extra is installed.
See the [CLI README](src/akgentic/team/cli/README.md) for full documentation.

```bash
# List all teams
ak-team list
ak-team list --status running

# Inspect a team
ak-team inspect <team-id>

# Create a team from a TeamCard YAML file (interactive — Ctrl+C to stop)
ak-team create team-card.yaml

# Resume a stopped team
ak-team resume <team-id>

# Delete a stopped team
ak-team delete <team-id>

# Use MongoDB backend
ak-team --backend mongodb --mongo-uri mongodb://localhost:27017 --mongo-db akgentic list
```

## Examples

Six progressive, self-contained examples in the [examples/](examples/)
directory. See the [Examples README](examples/README.md) for full
descriptions and learning path. Each includes a runnable `.py` script
and a companion `.md` explaining concepts and pitfalls.

```bash
uv run python examples/01_team_definition.py
```

| # | Script | Topic |
|---|---|---|
| 01 | `01_team_definition.py` | TeamCard & TeamCardMember hierarchies |
| 02 | `02_team_factory.py` | TeamFactory.build() & TeamRuntime |
| 03 | `03_team_manager_lifecycle.py` | Full lifecycle: create, stop, resume, delete |
| 04 | `04_event_sourcing.py` | PersistenceSubscriber & YamlEventStore |
| 05 | `05_crash_recovery.py` | TeamRestorer & crash recovery |
| 06 | `06_mongo_backend.py` | MongoEventStore & backend portability |

## Development

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
uv sync --all-extras
```

### Commands

```bash
# Run tests
uv run pytest packages/akgentic-team/tests/

# Run tests with coverage
uv run pytest packages/akgentic-team/tests/ --cov=akgentic.team --cov-fail-under=80

# Lint
uv run ruff check packages/akgentic-team/src/

# Format
uv run ruff format packages/akgentic-team/src/

# Type check
uv run mypy packages/akgentic-team/src/
```

### Project Structure

```
src/akgentic/team/
    __init__.py          # Public API (17 exports)
    models.py            # TeamCard, TeamRuntime, Process, TeamStatus, persistence models
    ports.py             # EventStore, ServiceRegistry protocols, NullServiceRegistry
    factory.py           # TeamFactory — static builder
    manager.py           # TeamManager — lifecycle facade
    restorer.py          # TeamRestorer — crash recovery
    subscriber.py        # PersistenceSubscriber — event sourcing bridge
    repositories/        # YamlEventStore, MongoEventStore
    cli/                 # ak-team CLI (Typer)
examples/                # 6 progressive examples with companion docs
tests/                   # 196 tests organized by domain
```

## License

See the repository root for license information.
