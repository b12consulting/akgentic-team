# akgentic-team

[![CI](https://github.com/b12consulting/akgentic-team/actions/workflows/ci.yml/badge.svg)](https://github.com/b12consulting/akgentic-team/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/gpiroux/e986fdd05c8c3d93e718782dc034e0c1/raw/coverage.json)](https://github.com/b12consulting/akgentic-team/actions/workflows/ci.yml)

Team lifecycle management for the [Akgentic](https://github.com/b12consulting/akgentic-framework)
multi-agent framework (open-source bundle). Create, resume, stop, and delete
multi-agent teams with event-sourced persistence and crash recovery.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Team Definitions](#team-definitions)
- [Messages](#messages)
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

- **Declarative team definitions** — `TeamCard` is the root object that
  parametrizes a whole team: its hierarchical `TeamCardMember` tree, entry
  point, message type, metadata schema and hireable profiles
- **One-call team building** via `TeamFactory` — from a `TeamCard` to live
  Pykka actors with routing wired
- **Full lifecycle management** via `TeamManager` — create, stop, resume,
  delete with state machine enforcement
- **Event-sourced persistence** via `PersistenceSubscriber` — every message is
  captured for crash recovery, with agent state diverted to a latest-per-agent
  snapshot rather than appended to the log
- **Idle-stop** via `IdleStopSubscriber` — a per-team inactivity countdown that
  stops the team itself once it has been idle long enough
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

Published on PyPI. Python 3.12 or newer.

```bash
uv add akgentic-team
# or
pip install akgentic-team
```

That is the whole install. `akgentic-core`, `pydantic` and `pyyaml` come with it
as ordinary dependencies — no workspace checkout, no submodules.

### Optional Extras

The base install gives you the lifecycle service and the YAML event store. Each
extra adds one optional surface:

| Extra      | Packages pulled in         | Enables                     |
|------------|----------------------------|-----------------------------|
| `cli`      | `typer`, `rich`            | `ak-team` console script    |
| `mongo`    | `pymongo`                  | `MongoEventStore`           |
| `postgres` | `nagra`, `psycopg[binary]` | `NagraEventStore`           |

```bash
uv add "akgentic-team[cli]"
uv add "akgentic-team[cli,postgres]"
```

`mongo` and `postgres` are alternative backends — pick the one you deploy. An
optional backend is imported lazily, so importing `akgentic.team` without
`pymongo` or `psycopg` installed is fine.

### As part of the framework bundle

`akgentic-framework` is the meta-distribution that pins every akgentic package
at versions built and tested together. Install `akgentic-team` through it when
you want the release-wide pin rather than a single package:

```bash
pip install "akgentic-framework[team]"   # this package + its closure, release-pinned
pip install "akgentic-framework[all]"    # the whole framework
```

### Working on the package itself

To develop `akgentic-team` rather than use it, clone the open-source bundle
[akgentic-framework](https://github.com/b12consulting/akgentic-framework), which
carries every package together as submodules:

```bash
git clone git@github.com:b12consulting/akgentic-framework.git
cd akgentic-framework
git submodule update --init
# uncomment the two "SOURCE MODE" blocks in pyproject.toml
uv sync
```

Source mode resolves `akgentic-*` to the local checkouts, editable.

## Quick Start

Create a team, send a message, stop it, and resume it:

```python
from pathlib import Path
from akgentic.core import ActorSystem, AgentCard, BaseConfig, BaseState
from akgentic.core.actor_address import ActorAddress
from akgentic.core.agent import Akgent
from akgentic.core.messages.message import UserMessage
from akgentic.team import (
    TeamCard, TeamCardMember, TeamManager, YamlEventStore,
)

# Define a simple agent
class EchoAgent(Akgent[BaseConfig, BaseState]):
    def receiveMsg_UserMessage(self, message: UserMessage, sender: ActorAddress) -> None:
        print(f"Echo: {message.content}")

# Build a team definition. The entry point is the team's external interface and
# is never a recipient of its own sends, so it needs a member to route to —
# every config.name in the tree must be unique.
# `role` is NOT a constructor argument — it is a read-only property reading
# config.role, so passing role= here is silently dropped. Set it on the config.
entry_card = AgentCard(
    description="Team entry point",
    skills=["routing"], agent_class=EchoAgent,
    config=BaseConfig(name="@Entry", role="Entry"),
)
echo_card = AgentCard(
    description="Echoes messages",
    skills=["echo"], agent_class=EchoAgent,
    config=BaseConfig(name="@Echo", role="Echo"),
)
team_card = TeamCard(
    name="echo-team", description="Simple echo team",
    entry_point=TeamCardMember(card=entry_card),
    members=[TeamCardMember(card=echo_card)],
    # Required for runtime.send(str): the first type is what a plain string is
    # wrapped in. Without it, send(str) raises RuntimeError — though send() also
    # accepts a pre-formed Message, which needs no declared type. See Messages.
    message_types=[UserMessage],
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

Two things to expect when you run this as a script:

- **Sends are asynchronous.** `send()` is fire-and-forget, so the echo prints
  once the actor gets around to the message, not on the line that sent it. Give
  it a moment before stopping the team if you want to see the output.
- **The process does not exit on its own** once the last line runs — the actor
  system's threads keep the interpreter alive. Long-running programs stop the
  actor system as part of their own shutdown; the `ak-team` CLI does this for
  you (see [CLI](#cli)).

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
| **Repositories** | EventStore implementations — YAML, MongoDB and PostgreSQL |
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

### Idle-stop

A running team does not stay up indefinitely. `TeamManager` gives every team its
own `IdleStopSubscriber` (in `akgentic.team.subscriber`), which owns the whole
idle-stop policy — the countdown included — and calls `stop_team` itself once the
team has been idle long enough. The `RUNNING → STOPPED` edge above is therefore
reached automatically as well as by an explicit `stop_team()`.

- **Constructed per team, on both the create and the resume path**, behind the
  `PersistenceSubscriber` and ahead of any shared subscribers you passed to
  `TeamManager`. You do not construct or register it.
- **Armed at construction, closed for good when a teardown begins.** A team that
  never receives a single message still stops. `on_stop_request` — raised by the
  orchestrator as the first statement of its own stop — closes the countdown
  terminally, because the mailbox drains behind that signal and a merely
  cancelled countdown is re-armed by the telemetry still arriving. `on_stop`
  keeps its plain cancel for the paths that never raise `on_stop_request`, such
  as a stop driven straight through Pykka. If the team fails to build,
  `TeamManager` cancels the countdown on the failure path rather than leaving it
  armed against a team that does not exist.
- **Driven by the event stream, not by a wall clock on the team.**
  `ReceivedMessage` starts a task and `ProcessedMessage` completes one; the
  countdown fires only while no task is in flight. Counting is suppressed during
  a restore, so replaying a stopped team's event log cannot drive the clock.
- **The stop runs on a daemon thread.** Stopping inline from the timeout callback
  would deadlock: the stop path issues a `proxy_ask` into the orchestrator and
  waits on the answer, which the orchestrator cannot service while that same
  call is in flight.
- **Idempotent.** If the team is already `STOPPED` or `DELETED` by the time the
  countdown fires, the resulting error is swallowed and logged at DEBUG.

The delay is read by this package's wiring from the `ORCHESTRATOR_TIMEOUT_DELAY`
environment variable, in seconds; it falls back to a module-level default when
the variable is unset, and a non-numeric value raises at wiring time rather than
failing silently later. The variable keeps the name it had while the countdown
lived in the orchestrator — only the site that reads it moved here, so existing
deployments need no change.

Because the countdown is armed as soon as a team is created, a short
`ORCHESTRATOR_TIMEOUT_DELAY` will stop the team in the
[Quick Start](#quick-start) before you get to `stop_team()`. That is the
mechanism working, not a fault.

## Team Definitions

### TeamCard — the root object

`TeamCard` is the one object that parametrizes a team. Everything the framework
needs in order to build the team and route into it is declared there: who is in
the team, who reports to whom, who faces the outside world, which message type
the team speaks, what business metadata it carries, and which extra profiles it
is allowed to hire while running.

It is a **pure Pydantic model** — no actor exists until the card is handed to
`TeamFactory.build()`, in practice through `TeamManager.create_team()`.

#### Creation — the card builds the team

```
create_team(team_card)
   │
   ├─▶ TeamFactory.build(team_card)
   │      1. Orchestrator actor
   │      2. entry_point, then the members tree — each agent is spawned
   │         through its parent and emits StartMessage(config, parent)
   │      3. agent_profiles ──▶ orchestrator role catalog
   │      4. welcome_message ──▶ WelcomeMessage on the event stream
   │      │
   │      └─▶ TeamRuntime          (live handle, returned to you)
   │
   │   every message above reaches the registered subscribers:
   │      PersistenceSubscriber ──▶ EventStore ─┬─ events
   │                                            └─ agent_states
   │
   └─▶ save_team(Process(team_id, team_card, RUNNING, metadata, …))
          │
          └─▶ team collection      (the card, stored for reference)
```

#### Restore — the card does *not* rebuild the team

The card kept on the `Process` is a reference: what this team was *asked* to be.
The roster that comes back is the one that was actually alive, and that is read
from the event store, not from the card.

```
resume_team(team_id)
   │
   ├─▶ load_team(team_id) ──▶ Process.team_card       (reference — see below)
   │
   └─▶ TeamRestorer.restore(process)
          1. load_events(team_id) + load_agent_states(team_id)
          2. roster = every StartMessage with no later StopMessage
                ├─ orchestrator StartMessage ──▶ Orchestrator actor
                └─ agent StartMessages ──▶ each agent respawned with its
                     persisted config, under its persisted parent, then its
                     agent_states snapshot applied
          3. replay the event log into the orchestrator (history and context)
          └─▶ TeamRuntime
```

| Rebuilt from | What it supplies |
|---|---|
| `StartMessage` / `StopMessage` in the event log | which agents are alive, and each one's name, role, config and parent |
| `agent_states` collection | each agent's persisted state |
| replay of the remaining events | the orchestrator's history and the agents' message context |
| `Process.team_card` | the team-level wiring only — which member is the entry point, which are the supervisors, which `agent_profiles` to re-register, which `metadata_type` to validate against |

The practical consequence: an agent hired at runtime comes back after a resume
even though no card ever mentioned it, and a member the card declares stays gone
if it was fired. Editing a card does not retro-fit a team already created from it.

Storing the card on the `Process` is nonetheless what makes it round-trip through
the team collection, which is why every field is a declared, serializable type —
and why the same card can be written as YAML and fed to the CLI (see [CLI](#cli)).

#### Fields

| Field | Type | Default | What it parametrizes |
|---|---|---|---|
| `name` | `str \| None` | `None` | Name of the *definition*. Descriptive only — a running team is identified by its `team_id` UUID, and two teams may share a card name. |
| `description` | `str \| None` | `None` | Human-readable summary of what the team does. |
| `entry_point` | `TeamCardMember` | *required* | The team's external interface — the agent every `send()` routes **through**. It is the sender, never a recipient of its own sends, so a card whose `members` is empty delivers nothing. |
| `members` | `list[TeamCardMember]` | `[]` | The team proper, as a tree. Its **first layer** are the supervisors — the agents the entry point actually delivers to. |
| `message_types` | `list[type]` | `[]` | The message classes the team speaks. **The first is the team default**: it is what a plain `str` handed to `send()` gets wrapped in. See [Messages](#messages). |
| `metadata_type` | `type[SerializableBaseModel] \| None` | `None` | The model class this team's business metadata must validate against, typically a `TeamMetadata` subclass. See [Team Metadata](#team-metadata). |
| `agent_profiles` | `list[AgentCard]` | `[]` | The profiles the orchestrator is given access to: cards registered in its catalog but **not** spawned. They are what an agent may hire at runtime, and what the `team_roles()` tool advertises in the system prompt. |
| `welcome_message` | `str \| None` | `None` | Static greeting published on the team's event stream when the team is first created (not on resume). `None` disables it. |

Two properties are derived from the tree rather than declared:

| Property | Returns |
|---|---|
| `agent_cards` | Flat `dict[name, AgentCard]` of every card in the tree, entry point included. **Raises `ValueError` on a duplicate `config.name`** — this is where name collisions surface. |
| `supervisors` | The `AgentCard`s of the **first layer of `members` only**. The entry point is excluded (it is the sender), and so is anything nested deeper. This is exactly the set `send()` fans out to. |

Two fields deserve a note beyond the table:

- **`agent_profiles` is the orchestrator's role catalog.** The factory registers
  it through `Orchestrator.register_agent_profiles()` at create *and* at restore,
  keyed by `card.role`. Two things read it back:

  - **`team_roles()`**, the Team tool's role-profiles prompt (exposed on the
    system-prompt and command channels). It renders the catalog into the agent's
    system prompt one line per profile — `role: description (Skills: …)` — which
    is how an LLM agent learns which roles exist for hiring, and why `description`
    and `skills` on those cards are prompt copy, not decoration.
  - **`hire_member(role)` / `hire_members(roles)`**, which look the card up by
    role and spawn it.

  Because the catalog is keyed by role, two profiles sharing a role silently
  overwrite each other. And keep live members out of the list: they are already in
  the room, so registering them lets the LLM hire a duplicate of an agent it can
  already talk to.
- **`metadata_type` is a declared field, not a type parameter.** A parameterised
  `TeamCard[X]` has no importable dotted path, which would break the very
  round-trip `Process.team_card` exists to perform.

### TeamCardMember — the member tree

`TeamCardMember` wraps an `AgentCard` with a multiplicity and its subordinates.
It is self-referential, so a team's hierarchy is just a tree of them:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `card` | `AgentCard` | *required* | The agent to spawn — its `agent_class`, `skills`, `description` and `config` (where `name` and `role` live). |
| `headcount` | `int` | `1` | How many instances of this member to spawn. |
| `members` | `list[TeamCardMember]` | `[]` | Subordinates, spawned **through this member**, so it owns their lifecycle. |

The rules the factory enforces, or silently relies on:

- **Every `config.name` in the tree must be unique.** Reusing one card in two
  slots is the common way to break this; `agent_cards` raises on it.
- **The entry point must have `headcount=1`** — `build()` raises `ValueError`
  otherwise.
- **`headcount > 1` renames the instances.** Three researchers named
  `@Researcher` become `@Researcher_0`, `@Researcher_1`, `@Researcher_2`; the
  bare name never exists at runtime, so `send_to("@Researcher")` will not find
  anyone. Children declared under such a member are spawned under the **last**
  instance.
- **Nesting is the spawn hierarchy, not a routing table.** A child is created
  through its parent's actor, so the parent owns it and takes it down with it.
  Who may talk to whom is decided by the agents at runtime, not by the tree.
- **`role` is not a constructor argument on `AgentCard`** — it is a read-only
  property over `config.role`, which must be non-empty. Passing `role=` to
  `AgentCard(...)` is silently dropped; set it on the config.
- **`routes_to` on an `AgentCard` is declared but not enforced** by this package:
  nothing in the team, core, agent or llm runtime consults it. Treat it as
  documentation until that changes.

#### A worked card

```python
team_card = TeamCard(
    name="research-team",
    description="A research team with a lead and workers",
    entry_point=TeamCardMember(card=entry_card),          # @Entry — faces the outside
    members=[
        TeamCardMember(card=lead_card, members=[          # @Lead — the only supervisor
            TeamCardMember(card=researcher_card, headcount=3),  # @Researcher_0..2
            TeamCardMember(card=reviewer_card),                 # @Reviewer
        ]),
    ],
    message_types=[UserMessage],                          # team default type
    agent_profiles=[translator_card],                     # hireable, not spawned
    welcome_message="Research team ready.",
)

team_card.agent_cards   # {"@Entry": ..., "@Lead": ..., "@Researcher": ..., "@Reviewer": ...}
team_card.supervisors   # [lead_card] — first layer of members only
```

`runtime.send("...")` here reaches `@Lead` and nobody else; `@Researcher_*` and
`@Reviewer` are `@Lead`'s to delegate to.

#### The same card as YAML

The card is serializable in both directions, so the CLI can create a team from a
file (`ak-team create card.yaml`). Class references travel as tagged dicts —
`agent_class` as a dotted string, types as `__type__`:

```yaml
name: research-team
description: A research team with a lead and workers
entry_point:
  card:
    agent_class: myapp.agents.EntryAgent
    description: Team entry point
    skills: [routing]
    config: {name: "@Entry", role: Entry}
members:
  - card:
      agent_class: myapp.agents.LeadAgent
      description: Leads the research
      skills: [planning]
      config: {name: "@Lead", role: Lead}
message_types:
  - __type__: akgentic.core.messages.message.UserMessage
```

### TeamRuntime

Live handle to a running team, returned by `create_team()` and `resume_team()`:

```python
runtime.id                                # team UUID
runtime.team                              # the TeamCard it was built from
runtime.addrs                             # agent name → ActorAddress (every member)
runtime.supervisor_addrs                  # the first-layer members send() fans out to

runtime.send("Hello!")                    # into the team, through the entry point
runtime.send_to("@Reviewer", "Hello!")    # directed to one member
runtime.send_from_to("@Lead", "@Reviewer", "Hello!")   # sender is @Lead, not @Entry
runtime.emitMessage(some_message)         # publish to the record, no agent involved
```

`TeamRuntime` is itself serializable: the addresses are persistent fields and the
actor proxies are rebuilt from them in `model_post_init`, which is what lets a
restored team be handed back as an ordinary runtime.

## Messages

### `message_types` — the type the team speaks

`TeamCard.message_types` declares the message classes a team handles, and **the
first entry is the team's default type**. It is used in exactly one place: when
`send()`, `send_to()` or `send_from_to()` is given a plain `str`, the string is
wrapped in `message_types[0]` before being routed.

```python
team_card = TeamCard(..., message_types=[UserMessage])

runtime.send("Hello!")            # → UserMessage(content="Hello!")
```

Consequences worth knowing before you pick one:

- **The class must accept `content=` as its only required argument** — the wrap
  is `message_types[0](content=content)`. `UserMessage` (core) and `AgentMessage`
  (akgentic-agent) both satisfy this.
- **A team with no `message_types` cannot be sent a `str`.** `send("...")` raises
  `RuntimeError: No message type declared for this team`. It can still be sent a
  `Message` — see below.
- **Entries beyond the first are declaration, not dispatch.** Nothing selects
  among them at runtime; they document what the team accepts.
- **The agent side must actually handle the type.** Routing is by handler name:
  an agent receives a `UserMessage` through `receiveMsg_UserMessage`. Declaring a
  type no agent has a handler for produces a silently ignored message, not an
  error.

### `send()` takes a `str` or a `Message`

All three processing verbs accept `str | Message`. A `str` is wrapped in the team
default as above; a `Message` is **passed through untouched**, so the caller — not
the card — picks the concrete type and fills its fields:

```python
from akgentic.core.messages.message import UserMessage

runtime.send("Hello!")                                  # team default type
runtime.send(UserMessage(content="Hello!"))             # caller's own instance
runtime.send(CaseUpdate(content="Hello!", severity="high"))  # caller's own type
```

This is how a product injects a richer, domain-specific message into a team
without the team card having to know that type — and it is the only way to send
into a team that declares no `message_types` at all.

Four things do **not** change when you pass a `Message`:

- **Routing is still the framework's.** `send()` still fans out to the
  supervisors, `send_to()` still targets the named member. A `recipient` you set
  on the message is not consulted for routing.
- **The sender is still the routing proxy** — the entry agent for `send()` and
  `send_to()`, the named sender for `send_from_to()`. A `sender` you set on the
  message is not authoritative.
- **The message is still processed by an agent.** It lands in a mailbox and the
  agent acts on it. To publish something *without* any agent touching it, use
  `emitMessage()`.
- **`send()` hands the same instance to every supervisor.** It is a passthrough,
  not a clone per recipient — do not mutate a message after sending it to a team
  with more than one supervisor.

### The four verbs

| Verb | Meaning | Reaches | Agent processing | Type chosen by |
|---|---|---|---|---|
| `send(content)` | converse with the team | every supervisor, sent by the entry agent | yes | team default, or the caller if a `Message` is passed |
| `send_to(name, content)` | converse with one member | that member, sent by the entry agent | yes | team default, or the caller |
| `send_from_to(sender, recipient, content)` | converse as a specific member | that recipient, sent by `sender` | yes | team default, or the caller |
| `emitMessage(message)` | publish a record into the team's event log | subscribers only — event store (durable) and live stream | **no** | the caller, always |

`emitMessage()` is the door for display or record messages — an ingestion
warning, a status banner — that must survive stop/resume and render live, but
that no agent should answer. It is fire-and-forget to the orchestrator, which
stamps `team_id` and fans it out to the subscribers with no routing and no
outbound dispatch.

All four are **asynchronous tells**: they return before anything has been
processed.

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

**A nullable field must carry `= None`.** Write `owner: str | None = None`, never
a bare `owner: str | None`. A client-facing field descriptor reports a field as
*mandatory* only when it is required **and** not nullable, so a bare
`owner: str | None` is advertised to a form as optional — while Pydantic still
counts it as required, because the key must be present. Leave that input blank
and the form sends no key at all, so the write answers 422 *field required*
naming a field it had just shown as optional — and a value once set can never be
cleared. The `= None` default is what makes the two halves agree, and it applies
to every nullable field, indexed or not.

Nothing catches a breach of this rule when the class is defined. Unlike the
scalar restriction below, a required-nullable field is a perfectly legal
declaration for any client that is not a form, so it is a rule for the declaring
author to keep rather than one the base class can enforce.

That is why `channel: str | None` above carries `default=None` — it is nullable,
so it must be defaultable too. `case_ref: str` needs nothing: being neither
nullable nor defaulted, it is reported mandatory, which is exactly what it is.

**Only indexed fields are restricted to scalars.** A marked field must be
annotated `str`, `bool`, `int`, `UUID`, `Enum`, `date` or `datetime` (optionally
`| None`); `float` is excluded, because float equality is not a sound index key.
The model *itself* may nest freely — sub-models, lists and dicts are all fine as
unmarked fields. Marking a non-scalar raises `TypeError` **when the class is
defined**, not when a write happens, so a bad declaration fails at import rather
than in production:

```python
class Broken(TeamMetadata):
    tags: list[str] = Field(default_factory=list, json_schema_extra={"indexed": True})
# TypeError: Broken.tags is marked indexed but is annotated list[str].
#            Indexed fields must be str, bool, int, UUID, Enum, date or datetime
#            (optionally '| None').
```

`ReferenceTeamMetadata` is the shipped, executable version of everything above:

```python
from akgentic.team import ReferenceTeamMetadata
```

It declares one field per state a client-side descriptor can report — indexed and
mandatory, indexed and nullable (carrying its `= None`), unindexed with a
non-`None` default, and one that declares no description at all. Read it as a
worked example, not as a contract to adopt or a base to inherit from: declare
your own `TeamMetadata` subclass, with your own fields.

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
  never written independently. No metadata API accepts it: neither `create_team`
  nor `update_team_metadata` takes an index argument, and both re-derive the
  array from the value they were given. If you build a service on top of this,
  keep it that way — a caller-supplied index is a lie the store cannot detect.
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
| PostgreSQL | pushed down as a `metadata_indexes @> …` containment term, in the same `WHERE` clause as `user_id` | GIN `team_process_metadata_indexes_idx` over the `TEXT[]` column, provisioned by `init_db` |

Correctness never depends on the index. A missing or un-created index makes a
query more expensive; it never changes which teams come back. That is what makes
the MongoDB opt-out safe: pass `auto_create_indexes=False` (or set
`MONGO_TEAM_AUTO_INDEX=0`) where the teams collection is too large to absorb a
foreground index build at boot, and provision out of band instead. The opt-out
covers the `teams` collection only — the `events` and `agent_states` indexes are
always created:

```bash
python -m akgentic.team.scripts.init_mongo
```

The PostgreSQL backend filters at the database. `Process.metadata_indexes` is
promoted to a `metadata_indexes TEXT[]` column on `team_process_entries`,
written by the same `save_team` statement that writes the JSON payload, and
`list_teams(metadata=…)` answers from a `metadata_indexes @> ARRAY[…]`
containment term AND-combined with the `user_id` scope. `init_db` provisions
the GIN index `team_process_metadata_indexes_idx` that serves it, and adds both
the column and the index to a database created before they existed — running

```bash
python -m akgentic.team.scripts.init_db
```

again is the whole upgrade. Rows written before the column existed carry `NULL`,
list normally, and match no metadata filter. The `status` filter is still applied
in Python after loading the rows on this backend; pushing it down needs its own
expression index and has not shipped.

> **Not to be confused with `AgentCard.metadata`**, which is a free-form
> annotation bag on an individual agent. `Process.metadata` is the team's typed,
> filterable value described here; the two share a word and nothing else.

## Persistence

### Event Sourcing

`PersistenceSubscriber` writes on two tracks, and every message takes exactly
one of them:

- **`StateChangedMessage` carrying a sender** is upserted as a latest-per-agent
  snapshot via `save_agent_state`, keyed by the agent's UUID and carrying its
  display name. It does **not** increment the sequence and is **never appended
  to the event log**.
- **Every other message** increments the sequence and is appended to the log via
  `save_event`.

The consequence is worth stating plainly: a snapshot write that is missed is a
**permanent** loss, not a late write. There is no log entry to replay it from,
so the agent's state is simply gone until that agent next changes state. A
missed `save_event` costs one event out of a replayable log; a missed
`save_agent_state` costs the state itself.

During a restore the subscriber returns immediately from `on_message` and writes
nothing at all — `TeamManager.resume_team` sets the guard around the replay so
the replayed log is not persisted a second time.

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
| `team_process_entries` | `id` | One row per team — `Process` snapshot. Also carries the promoted `metadata_indexes TEXT[]` column |
| `event_entries` | `(team_id, sequence)` | Append-only event log |
| `agent_state_entries` | `(team_id, agent_id)` | Agent state snapshots |

`init_db` creates two indexes on `team_process_entries`: the functional
expression index `team_process_user_id_idx` over `(data ->> 'user_id')`, and
the GIN index `team_process_metadata_indexes_idx` over `metadata_indexes`.
Both names are part of the contract — an operator inspecting a database
addresses them by name.

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
uv run pytest tests/

# Run tests with coverage
uv run pytest tests/ --cov=akgentic.team --cov-fail-under=80

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Type check
uv run mypy src/
```

All commands above are run from this repository's root. Inside the
`akgentic-framework` bundle they take a `packages/akgentic-team/` prefix, and
mypy then needs `--config-file packages/akgentic-team/pyproject.toml` — the
bundle-root config relaxes rules for other packages, so running mypy without it
proves less.

### Project Structure

```
src/akgentic/team/
    __init__.py          # Public API (__all__)
    models.py            # TeamCard, TeamRuntime, Process, TeamStatus, persistence models
    messages.py          # WelcomeMessage
    metadata.py          # TeamMetadata, make_index_entry, derive_metadata_indexes
    reference_metadata.py # ReferenceTeamMetadata — the worked metadata example
    ports.py             # EventStore, ServiceRegistry protocols, NullServiceRegistry
    factory.py           # TeamFactory — static builder
    manager.py           # TeamManager — lifecycle facade
    restorer.py          # TeamRestorer — crash recovery
    subscriber.py        # PersistenceSubscriber, IdleStopSubscriber
    repositories/        # YamlEventStore, MongoEventStore, postgres/NagraEventStore
    scripts/             # init_db / init_mongo deployment entry points
    cli/                 # ak-team CLI (Typer)
examples/                # 6 progressive examples with companion docs
tests/                   # organized by domain, mirroring src/
```

## License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://github.com/b12consulting/akgentic-team/blob/master/LICENSE).

> **Dual licensing & CLA** — Akgentic is available under the AGPL-3.0 open-source license. A commercial license is also planned for organizations that require alternative terms. Contact [Yuma](https://www.weareyuma.com/en/contact) for more information. External contributions will be accepted once a Contributor License Agreement (CLA) is in place. Until then, please hold off on submitting pull requests.
