# akgentic-team

[![CI](https://github.com/b12consulting/akgentic-team/actions/workflows/ci.yml/badge.svg)](https://github.com/b12consulting/akgentic-team/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/gpiroux/e986fdd05c8c3d93e718782dc034e0c1/raw/coverage.json)](https://github.com/b12consulting/akgentic-team/actions/workflows/ci.yml)

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
- [Team Metadata](#team-metadata)
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
- **Three storage backends** (YAML files, MongoDB, and PostgreSQL via Nagra)
  behind a common `EventStore` protocol
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
                              ┌─────┬──────┬──────────────┐
                              │     │      │              │
                         YamlEventStore  MongoEventStore  NagraEventStore
                                                          (PostgreSQL via Nagra)
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

# PostgreSQL backend (Nagra)
uv sync --extra postgres

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
│  Repositories: YamlEventStore, MongoEventStore,│
│                NagraEventStore (PostgreSQL)    │
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

## Team Metadata

A team can carry typed business metadata — the tenant it belongs to, the case
it was opened for, the channel it came in on — and teams can be listed by it.
The schema is yours: you declare a model, mark the fields you want to filter
on, and hand the class to the `TeamCard`.

### Defining a metadata schema

Subclass `TeamMetadata` and mark each filterable field with
`Field(json_schema_extra={"indexed": True})`:

```python
from pydantic import Field

from akgentic.team import TeamMetadata


class SupportCaseMetadata(TeamMetadata):
    """Business context carried by each support-desk team instance."""

    tenant: str = Field(json_schema_extra={"indexed": True})
    case_ref: str = Field(json_schema_extra={"indexed": True})
    channel: str | None = Field(default=None, json_schema_extra={"indexed": True})
    notes: str = ""  # not marked — stored and returned, never filterable
```

Unmarked fields are ordinary model fields: they are persisted with the team and
come back on every read, they are simply not something you can filter on. A
subclass with no marked field at all is legal — it is just not filterable.

**Only indexed fields are restricted to scalars.** A marked field must be
annotated `str`, `bool`, `int`, `UUID`, `Enum`, `date` or `datetime` (optionally
`| None`); `float` is excluded, because float equality is not a sound index key.
The model *itself* may nest freely — sub-models, lists and dicts are all fine as
unmarked fields. Marking a non-scalar raises `TypeError` **when the class is
defined**, not when a write happens, so a bad declaration fails at import rather
than in production:

```python
class Broken(TeamMetadata):
    owner: Contact = Field(json_schema_extra={"indexed": True})
# TypeError: Broken.owner is marked indexed but is annotated ...
#            Indexed fields must be str, bool, int, UUID, Enum, date or datetime
```

### Declaring it on the TeamCard

`TeamCard.metadata_type` declares which model this team's metadata must be:

```python
team_card = TeamCard(
    name="support-desk",
    description="Handles inbound support cases",
    entry_point=TeamCardMember(card=triage_card),
    members=[TeamCardMember(card=agent_card)],
    metadata_type=SupportCaseMetadata,
)
```

`metadata_type` is a declared type field, not a generic parameter — there is no
`TeamCard[SupportCaseMetadata]`. A card that leaves it `None` **rejects**
metadata rather than ignoring it: supplying a value raises `ValueError`, so a
value can never be silently dropped.

### Setting and updating the value

Pass the value at creation, and replace it afterwards through
`TeamManager.update_team_metadata`:

```python
runtime = manager.create_team(
    team_card,
    metadata=SupportCaseMetadata(tenant="acme", case_ref="C-1234", channel="email"),
)

# Later — a complete replacement, returning the persisted Process
process = manager.update_team_metadata(
    runtime.id,
    SupportCaseMetadata(tenant="acme", case_ref="C-1234", channel="phone"),
)

manager.update_team_metadata(runtime.id, None)  # clears the metadata entirely
```

**Replace, never merge.** The value you pass is the complete document and must
validate on its own; a field that was set before and is absent now is gone from
both the stored value and its index. `None` clears the metadata.

The value is validated against the `metadata_type` the card declared *at
creation* — read back off the persisted team — so `metadata_type` cannot be
changed for a live team. A value that fails validation raises
`pydantic.ValidationError` and nothing is written.

### Filtering teams

`EventStore.list_teams` takes a plain `dict[str, str]` of key/value pairs:

```python
from akgentic.team import TeamStatus

teams = event_store.list_teams(
    user_id="alice",
    status=TeamStatus.RUNNING,
    metadata={"tenant": "acme", "channel": "email"},
)
```

Every key AND-combines with every other key, and the whole metadata filter
AND-combines with `user_id` and `status`. A filter left at `None` constrains
nothing; adding one can only narrow the result set, never widen it. An empty
dict is an empty conjunction and behaves exactly like `None`. Values are matched
against the *rendered* form of the stored field, so a typed field is filtered by
passing its rendered string (`"true"` for a `bool`, the ISO form for a `date`).

`user_id` scoping is applied server-side in every backend and is never weakened
by a metadata term — metadata is caller-supplied and non-secret, so narrowing by
it must not become a way to reach another owner's teams.

### The limits — read this before designing around metadata

> **Equality only.** A metadata filter matches a key to an exact value and
> nothing else. There are **no range queries, no prefix matching, no substring
> matching, and no sort-by-metadata**. `priority > 3` cannot be expressed
> through this mechanism at all; it would need a different index and a separate
> design decision. Filtering by creation time is *not* an example of this limit
> — `Process.created_at` is a first-class typed field, unrelated to metadata.

> **Filtering narrows, it does not paginate.** `list_teams` returns *every*
> matching team, and callers slice afterwards. A filter selecting 20 teams out
> of 5,000 still hydrates all 20 matches and still hands the caller all 20 —
> metadata filtering reduces the size of the result set, not the cost of a page.
> Store-side pagination push-down is a separate decision and does not ship here.

### How the index works

Indexed fields are flattened into a `Process.metadata_indexes` array of
`"key|value"` strings. For
`SupportCaseMetadata(tenant="acme", case_ref="C-1234", channel="email")` the
derived entries are:

```
["tenant|acme", "case_ref|C-1234", "channel|email"]
```

Three properties are worth knowing:

- **Derived on every write, never client-supplied.** `metadata_indexes` is
  recomputed from `metadata` each time the value is persisted, and the two are
  never written independently. Nothing you pass in can set it.
- **An unset optional indexed field emits no entry.** Absent is not the empty
  string — a `channel=None` contributes nothing, where `channel=""` would
  contribute `"channel|"`.
- **`|` is the separator and is escaped inside values.** `tenant="acme|contoso"`
  derives the single entry `tenant|acme\|contoso`, not two entries, so a value
  containing a pipe cannot forge a second index entry. Queries are built through
  the same helper as the derivation, so the two sides can never disagree.

### Backend support

| Backend | How the filter runs | Index |
|---|---|---|
| YAML | in-memory containment, applied to the raw parsed mapping before validation | none — the backend is here for parity, not throughput |
| MongoDB | pushed down into the same `find` filter as `user_id` and `status` | multikey `teams_metadata_indexes_idx`, provisioned by `ensure_indexes()` |
| PostgreSQL | interim in-memory filter after hydration — **push-down deferred** | none yet |

Correctness never depends on the index. A missing or un-created index makes a
query more expensive; it never changes which teams come back. That is what makes
the MongoDB opt-out safe: pass `auto_create_indexes=False` (or set
`MONGO_TEAM_AUTO_INDEX=0`) where the teams collection is too large to absorb a
foreground index build at boot, and provision out of band instead:

```bash
python -m akgentic.team.scripts.init_mongo
```

The PostgreSQL backend returns correct results today, but selects them in Python
after loading the rows — the database-side push-down is deferred. Do not
provision an index for a query that does not yet exist.

> **Not to be confused with `AgentCard.metadata`**, which is a free-form
> annotation bag on an individual agent. `Process.metadata` is the team's typed,
> filterable value described here; the two share a word and nothing else.

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

**PostgreSQL (Nagra)** — install the `[postgres]` extra:

```bash
uv sync --extra postgres
# or: uv add "akgentic-team[postgres]"
```

The PostgreSQL backend is built on [Nagra](https://pypi.org/project/nagra/)
and stores team state across three tables with promoted query keys plus a
`data JSONB` payload (the payload is authoritative — promoted columns are
indexes, not the source of truth):

| Table | Natural key | Purpose |
|---|---|---|
| `team_process_entries` | `id` | One row per team — `Process` snapshot |
| `event_entries` | `(team_id, sequence)` | Append-only event log |
| `agent_state_entries` | `(team_id, agent_id)` | Agent state snapshots |

Each public `NagraEventStore` method opens its own `Transaction`. The one
exception is `delete_team`, which spans a single transaction across the
three tables (ordered: `agent_state_entries` → `event_entries` →
`team_process_entries`) so cascade deletion is atomic. `save_event`
propagates the raw `psycopg`/Nagra `UniqueViolation` on duplicate
`(team_id, sequence)` — matching the Mongo backend's raw
`DuplicateKeyError` propagation.

**Environment variables.** The backend follows the V1 Akgentic conventions
so existing operator `.env` files work unchanged:

| Variable | Purpose |
|---|---|
| `POSTGRES_SERVER` | Database host |
| `POSTGRES_PORT` | Database port (typically `5432`) |
| `POSTGRES_USER` | Database user |
| `POSTGRES_PASSWORD` | Database password |
| `POSTGRES_DB` | Database name |
| `DB_CONN_STRING_PERSISTENCE` | Full libpq URL; what `NagraEventStore` receives as `conn_string` |

`DB_CONN_STRING_PERSISTENCE` is **shared verbatim with `akgentic-catalog`** —
both modules target the same database. Their tables are disjoint by design:
the catalog owns `template_entries`, `tool_entries`, `agent_entries`, and
`team_entries`; this package owns `team_process_entries` (renamed from
`team_entries` to prevent collision), `event_entries`, and
`agent_state_entries`. A single Postgres instance can serve both modules.

`NagraEventStore.__init__` takes `conn_string` directly as a positional
argument — env-var reading happens at the wiring layer (application
startup / infra code), **not** inside the event store. This keeps the
storage layer decoupled from process-level configuration.

**Schema initialisation.** Call `init_db(conn_string)` once per deployment
(at application startup or as a deploy hook). The call is idempotent —
it creates any missing tables and is safe to re-run. `NagraEventStore`'s
constructor does **not** call `init_db` implicitly.

```python
from akgentic.team.repositories.postgres import NagraEventStore, init_db

conn_string = "postgresql://akgentic:akgentic@localhost:5432/akgentic"

# One-time (idempotent) schema creation — run at deploy time.
init_db(conn_string)

# Construct the event store with the same conn_string.
event_store = NagraEventStore(conn_string)
```

Schema evolution is handled as a redeploy concern — the backend does not
adopt a migration framework. Drop-and-recreate semantics or manual
`ALTER TABLE` statements are the expected evolution path.

#### Database initialization (init container)

For Kubernetes / Nomad deployments, run the schema-creation hook as a
dedicated init container before the main team-runtime process starts:

```bash
python -m akgentic.team.scripts.init_db
```

The script reads `DB_CONN_STRING_PERSISTENCE` and exits with one of:

| Exit code | Meaning |
|---|---|
| `0` | Success — tables created or already present |
| `2` | `DB_CONN_STRING_PERSISTENCE` not set |
| `1` | Any other failure (nagra not installed, connection refused, `init_db` raised) |

Catalog and team can share a single init step — both modules expose the
same entry-point shape (`python -m akgentic.<module>.scripts.init_db`) and
read the same `DB_CONN_STRING_PERSISTENCE` env var.

**Kubernetes initContainer** snippet:

```yaml
spec:
  initContainers:
    - name: akgentic-team-init-db
      image: ghcr.io/b12consulting/akgentic-team:latest
      command: ["python", "-m", "akgentic.team.scripts.init_db"]
      env:
        - name: DB_CONN_STRING_PERSISTENCE
          valueFrom:
            secretKeyRef:
              name: akgentic-postgres
              key: conn-string
```

**Nomad prestart task** snippet:

```hcl
task "init-db" {
  driver = "docker"
  lifecycle {
    hook    = "prestart"
    sidecar = false
  }
  config {
    image   = "ghcr.io/b12consulting/akgentic-team:latest"
    command = "python"
    args    = ["-m", "akgentic.team.scripts.init_db"]
  }
  template {
    destination = "secrets/db.env"
    env         = true
    data        = <<EOF
DB_CONN_STRING_PERSISTENCE={{ with secret "kv/akgentic" }}{{ .Data.data.conn_string }}{{ end }}
EOF
  }
}
```

#### Out of scope: enterprise wiring

Wiring `NagraEventStore` into `akgentic-infra-enterprise`'s server +
worker bootstrap (opt-in via `AKGENTIC_EVENT_STORE = "postgres"` or
equivalent) is a **follow-up story tracked in `akgentic-infra-enterprise`**.
This package only ships the backend implementation, the `[postgres]`
extra, the deployment hook, and the documentation. The enterprise
deployment project owns the application-level switch.

### Crash Recovery

`TeamRestorer` executes a 3-phase protocol:
1. **Load** persisted events and agent state snapshots
2. **Rebuild** agents from the event log (Orchestrator first, then others)
3. **Replay** all events to reconstruct full state including LLM context

## CLI

The `ak-team` command is available when the `[cli]` extra is installed.
See the [CLI README](https://github.com/b12consulting/akgentic-team/blob/master/src/akgentic/team/cli/README.md) for full documentation.

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

Six progressive, self-contained examples in the [examples/](https://github.com/b12consulting/akgentic-team/tree/master/examples)
directory. See the [Examples README](https://github.com/b12consulting/akgentic-team/blob/master/examples/README.md) for full
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

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://github.com/b12consulting/akgentic-team/blob/master/LICENSE).

> **Dual licensing & CLA** — Akgentic is available under the AGPL-3.0 open-source license. A commercial license is also planned for organizations that require alternative terms. Contact [Yuma](https://www.weareyuma.com/en/contact) for more information. External contributions will be accepted once a Contributor License Agreement (CLA) is in place. Until then, please hold off on submitting pull requests.
