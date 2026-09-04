"""Test fixtures for team domain model tests."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.agent_card import AgentCard
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.messages.message import Message, UserMessage
from akgentic.core.utils.serializer import SerializableBaseModel
from pydantic import Field

from akgentic.team.metadata import TeamMetadata, derive_metadata_indexes
from akgentic.team.models import (
    AgentStateSnapshot,
    PersistedEvent,
    Process,
    TeamCard,
    TeamCardMember,
    TeamRuntime,
    TeamStatus,
)
from tests.conftest import projection_kwargs


def make_agent_card(
    name: str = "test-agent",
    role: str = "TestAgent",
    agent_class: str | type = "tests.fixtures.MockAgent",
) -> AgentCard:
    """Create a minimal AgentCard for testing.

    Args:
        name: Config name for the agent.
        role: Agent role.
        agent_class: Agent class or FQCN string.

    Returns:
        An AgentCard with the specified configuration.
    """
    return AgentCard(
        role=role,
        description=f"Test agent: {role}",
        skills=["testing"],
        agent_class=agent_class,
        config=BaseConfig(name=name, role=role),
    )


def make_team_card(
    name: str = "test-team",
    description: str = "A test team",
    entry_point_name: str = "lead",
    entry_point_role: str = "Lead",
    member_names: list[str] | None = None,
    member_roles: list[str] | None = None,
    message_types: list[type] | None = None,
    agent_class: str | type = "tests.fixtures.MockAgent",
) -> TeamCard:
    """Create a minimal TeamCard for testing.

    Args:
        name: Team name.
        description: Team description.
        entry_point_name: Config name of the entry point agent.
        entry_point_role: Role of the entry point agent.
        member_names: Config names for additional members.
        member_roles: Roles for additional members.
        message_types: Message classes the team handles.
        agent_class: Agent class or FQCN string for all members.

    Returns:
        A TeamCard with the specified structure.
    """
    entry_point = TeamCardMember(
        card=make_agent_card(
            name=entry_point_name, role=entry_point_role, agent_class=agent_class
        ),
    )
    members: list[TeamCardMember] = []
    if member_names and member_roles:
        for mname, mrole in zip(member_names, member_roles, strict=True):
            members.append(
                TeamCardMember(
                    card=make_agent_card(name=mname, role=mrole, agent_class=agent_class)
                ),
            )
    return TeamCard(
        name=name,
        description=description,
        entry_point=entry_point,
        members=members,
        message_types=message_types or [],
    )


def make_stub_addr(
    name: str = "stub",
    role: str = "Stub",
    *,
    is_user_proxy: bool = False,
) -> MagicMock:
    """Create a MagicMock that behaves like an ActorAddress.

    ``role`` and ``is_user_proxy`` are set explicitly rather than left to the
    spec. A ``MagicMock(spec=ActorAddress)`` answers both with a truthy
    ``MagicMock``, which would flow into ``get_agent_profile`` and into the
    ``UserProxy`` check and make every assertion about them vacuously green.

    Args:
        name: Name for the stub address.
        role: Role for the stub address — the orchestrator catalog key.
        is_user_proxy: What the address reports about the actor's actual type.

    Returns:
        A MagicMock with ActorAddress spec.
    """
    addr = MagicMock(spec=ActorAddress)
    addr.agent_id = uuid.uuid4()
    addr.name = name
    addr.role = role
    addr.is_user_proxy = is_user_proxy
    return addr


def make_stub_actor_system(entry_agent_class: type | None = Akgent) -> MagicMock:
    """Create a MagicMock that behaves like an ActorSystem.

    The single ``proxy_ask`` return value doubles as the orchestrator proxy, so
    its ``get_agent_profile`` is what ``TeamRuntime.model_post_init`` reads to
    type the entry proxy.

    Args:
        entry_agent_class: The agent class the catalog reports for whatever role
            it is asked about. ``None`` makes the catalog answer with no entry,
            which is the construction-time ``ValueError`` path.

    Returns:
        A MagicMock with ActorSystem spec and proxy methods.
    """
    orchestrator = MagicMock()
    orchestrator.get_agent_profile = MagicMock(
        return_value=(
            None
            if entry_agent_class is None
            else make_agent_card(name="entry", role="Entry", agent_class=entry_agent_class)
        )
    )
    system = MagicMock(spec=ActorSystem)
    system.proxy_ask = MagicMock(return_value=orchestrator)
    system.proxy_tell = MagicMock(return_value=MagicMock())
    return system


def make_team_runtime(
    *,
    team_name: str | None = "test-team",
    message_types: list[type] | None = None,
    supervisor_addrs: dict[str, ActorAddress] | None = None,
    addrs: dict[str, ActorAddress] | None = None,
    actor_system: MagicMock | None = None,
    entry_addr: ActorAddress | None = None,
) -> TeamRuntime:
    """Create a TeamRuntime with mock dependencies for testing.

    Args:
        team_name: The team's name, as it appears in error messages.
        message_types: Message classes the team handles.
        supervisor_addrs: Supervisor address mapping.
        addrs: All agent address mapping.
        actor_system: Optional pre-configured stub actor system, for tests that
            need to steer the orchestrator catalog.
        entry_addr: Optional pre-built entry-point address.

    Returns:
        A TeamRuntime with mock ActorSystem and addresses.
    """
    return TeamRuntime(
        id=uuid.uuid4(),
        team_name=team_name,
        message_types=message_types or [],
        actor_system=actor_system or make_stub_actor_system(),
        orchestrator_addr=make_stub_addr("orchestrator", "Orchestrator"),
        entry_addr=entry_addr or make_stub_addr("entry", "Lead"),
        supervisor_addrs=supervisor_addrs or {},
        addrs=addrs or {},
    )


class SampleAgentState(BaseState):
    """Minimal BaseState subclass for testing polymorphic serialization."""

    task_count: int = 0


class AcmeTeamMetadata(TeamMetadata):
    """Sample deployment metadata: two indexed fields plus an unindexed one."""

    tenant: str = Field(default="acme", json_schema_extra={"indexed": True})
    case_ref: str | None = Field(default=None, json_schema_extra={"indexed": True})
    department: str = Field(default="")


class PlainCardMetadata(SerializableBaseModel):
    """A plain model a card may legally declare -- it carries no index contract."""

    tenant: str = ""


def make_process(
    team_id: uuid.UUID | None = None,
    team_card: TeamCard | None = None,
    status: TeamStatus = TeamStatus.RUNNING,
    user_id: str | None = None,
    catalog_namespace: str | None = None,
    metadata: SerializableBaseModel | None = None,
    metadata_indexes: list[str] | None = None,
) -> Process:
    """Create a Process with sensible defaults for testing.

    Args:
        team_id: Optional team identifier.
        team_card: Optional pre-built TeamCard.
        status: Lifecycle status.
        user_id: Optional owning user id. When ``None`` (default), the
            ``user_id`` kwarg is omitted from the ``Process(...)`` call so
            the underlying field falls back to its model default (``"cli"``
            per ``Process.user_id``). When a string, it is passed through.
        catalog_namespace: Optional catalog-namespace tag.
        metadata: Optional business metadata value.
        metadata_indexes: Optional pre-derived index entries. Passed through
            verbatim -- the helper never derives them, matching Process itself.

    Returns:
        A Process with the specified or default configuration.
    """
    now = datetime.now(UTC)
    tc = team_card or make_team_card()
    # Omit user_id when None so the field falls back to its model default.
    optional: dict[str, Any] = {} if user_id is None else {"user_id": user_id}
    return Process(
        team_id=team_id or uuid.uuid4(),
        team_card=tc,
        status=status,
        created_at=now,
        updated_at=now,
        catalog_namespace=catalog_namespace,
        metadata=metadata,
        metadata_indexes=metadata_indexes or [],
        **projection_kwargs(tc),
        **optional,
    )


def make_indexed_process(
    metadata: SerializableBaseModel | None,
    team_id: uuid.UUID | None = None,
    status: TeamStatus = TeamStatus.RUNNING,
    user_id: str | None = None,
) -> Process:
    """Create a Process whose index is derived exactly as a write path derives it.

    ``make_process`` passes ``metadata_indexes`` through verbatim and never
    derives it -- matching ``Process`` itself, which accepts the index rather
    than computing it. Filter tests need a team that looks the way the real
    create/update paths leave one, so this composes the two: the value and its
    index are seeded together, through the one derivation function.

    Args:
        metadata: The team's business metadata value, or ``None``.
        team_id: Optional team identifier.
        status: Lifecycle status.
        user_id: Optional owning user id, defaulting as in ``make_process``.

    Returns:
        A Process carrying ``metadata`` and its derived index entries.
    """
    return make_process(
        team_id=team_id,
        status=status,
        user_id=user_id,
        metadata=metadata,
        metadata_indexes=derive_metadata_indexes(metadata),
    )


def make_persisted_event(
    team_id: uuid.UUID | None = None,
    sequence: int = 0,
    event: Message | None = None,
) -> PersistedEvent:
    """Create a PersistedEvent with sensible defaults for testing.

    Args:
        team_id: Optional team identifier.
        sequence: Event sequence number.
        event: Optional Message subclass instance.

    Returns:
        A PersistedEvent with the specified or default configuration.
    """
    return PersistedEvent(
        team_id=team_id or uuid.uuid4(),
        sequence=sequence,
        event=event or UserMessage(content="test message"),
        timestamp=datetime.now(UTC),
    )


def make_agent_state_snapshot(
    team_id: uuid.UUID | None = None,
    agent_id: str = "test-agent",
    state: BaseState | None = None,
    name: str | None = None,
) -> AgentStateSnapshot:
    """Create an AgentStateSnapshot with sensible defaults for testing.

    Args:
        team_id: Optional team identifier.
        agent_id: Agent identifier (UUID string in current snapshots).
        state: Optional BaseState subclass instance.
        name: Optional agent display name (default None keeps existing callers
            working and mimics legacy snapshots).

    Returns:
        An AgentStateSnapshot with the specified or default configuration.
    """
    return AgentStateSnapshot(
        team_id=team_id or uuid.uuid4(),
        agent_id=agent_id,
        name=name,
        state=state or SampleAgentState(task_count=5),
        updated_at=datetime.now(UTC),
    )
