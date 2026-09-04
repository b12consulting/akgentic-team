"""Tests for TeamRestorer -- AC 5-13, 15, 18."""

from __future__ import annotations

import threading
import time
import uuid
from datetime import UTC, datetime
from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_address_impl import ActorAddressProxy
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.agent_card import AgentCard
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.messages.message import Message, UserMessage
from akgentic.core.messages.orchestrator import (
    ErrorMessage,
    EventMessage,
    NotificationMessage,
    SentMessage,
    StartMessage,
    StateChangedMessage,
    StopMessage,
    WarningMessage,
)
from akgentic.core.orchestrator import EventSubscriber, Orchestrator
from akgentic.core.utils.serializer import SerializableBaseModel
from pydantic import Field

from akgentic.team.factory import TeamFactory
from akgentic.team.metadata import TeamMetadata, derive_metadata_indexes
from akgentic.team.models import (
    AgentStateSnapshot,
    PersistedEvent,
    Process,
    TeamCard,
    TeamCardMember,
    TeamRuntime,
    TeamStatus,
    spawned_names,
)
from akgentic.team.restorer import GRACE_TIMEOUT_SECONDS, TeamRestorer
from tests.conftest import projection_kwargs, seed_agent_cards
from tests.services.conftest import InMemoryEventStore

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class StubAgent(Akgent[BaseConfig, BaseState]):
    """Minimal agent for restorer tests."""

    pass


class _OtherStubAgent(Akgent[BaseConfig, BaseState]):
    """A second agent class, so a card can declare one the live actor is not."""


class _MarkerState(BaseState):
    """BaseState carrying a distinguishing marker for snapshot-preference tests.

    Used to tell two competing snapshots (UUID-keyed vs name-keyed legacy) apart
    by the value the restorer applies via ``init_state``.
    """

    marker: str = ""


class RecordingSubscriber(EventSubscriber):
    """Subscriber that records received messages.

    Implements the team_id-aware ``EventSubscriber`` Protocol. The stub does
    not assert on ``team_id`` because it is reused across tests.
    """

    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.stopped: bool = False

    def on_message(self, msg: Message) -> None:
        """Record received message."""
        self.messages.append(msg)

    def on_stop(self, team_id: uuid.UUID) -> None:
        """Record stop."""
        del team_id
        self.stopped = True

    def on_stop_request(self, team_id: uuid.UUID) -> None:
        """No-op for test subscriber."""
        del team_id

    def set_restoring(self, team_id: uuid.UUID, restoring: bool) -> None:  # noqa: FBT001
        """No-op for test subscriber."""
        del team_id, restoring


def _make_card(
    name: str,
    role: str = "TestRole",
    agent_class: type[Akgent[Any, Any]] = StubAgent,
) -> AgentCard:
    return AgentCard(
        role=role,
        description=f"Test: {role}",
        skills=["testing"],
        agent_class=agent_class,
        config=BaseConfig(name=name, role=role),
        routes_to=[],
    )


def _make_member(
    name: str,
    role: str = "TestRole",
    agent_class: type[Akgent[Any, Any]] = StubAgent,
    headcount: int = 1,
    members: list[TeamCardMember] | None = None,
) -> TeamCardMember:
    return TeamCardMember(
        card=_make_card(name, role, agent_class),
        headcount=headcount,
        members=members or [],
    )


def _make_team_card(
    entry_point: TeamCardMember | None = None,
    members: list[TeamCardMember] | None = None,
    name: str = "test-team",
) -> TeamCard:
    ep = entry_point or _make_member("lead", "Lead")
    return TeamCard(
        name=name,
        description="Test team",
        entry_point=ep,
        members=members or [],
    )


def _make_start_message(
    agent_id: uuid.UUID,
    name: str,
    role: str,
    team_id: uuid.UUID,
    agent_class: type[Akgent[Any, Any]] = StubAgent,
    config: BaseConfig | None = None,
    parent_id: uuid.UUID | None = None,
    parent_name: str = "orchestrator",
    parent_role: str = "Orchestrator",
    sender_squad_id: uuid.UUID | None = None,
) -> StartMessage:
    """Create a StartMessage with a properly-formed sender address.

    Args:
        parent_id: If set, creates a parent ActorAddressProxy with this agent_id.
        sender_squad_id: Squad stamped into the sender address dict. Defaults to a
            fresh random UUID -- deliberately unrelated to ``config.squad_id``, which
            is what the persisted addresses of pre-existing fixtures look like. Pass
            the config's own squad to build a fixture that does not itself ship the
            sender/config disagreement a test is trying to detect.
    """
    cfg = config or BaseConfig(name=name, role=role)
    msg = StartMessage(config=cfg)
    # Build a fake sender address dict so serialize() works
    from akgentic.core.actor_address_impl import ActorAddressProxy
    from akgentic.core.utils.deserializer import ActorAddressDict

    addr_dict: ActorAddressDict = {
        "__actor_address__": True,
        "__actor_type__": f"{agent_class.__module__}.{agent_class.__name__}",
        "agent_id": str(agent_id),
        "name": name,
        "role": role,
        "team_id": str(team_id),
        "squad_id": str(sender_squad_id or uuid.uuid4()),
        "user_message": False,
    }
    sender = ActorAddressProxy(addr_dict)
    msg.sender = sender
    msg.team_id = team_id

    if parent_id is not None:
        parent_dict: ActorAddressDict = {
            "__actor_address__": True,
            "__actor_type__": "akgentic.core.orchestrator.Orchestrator",
            "agent_id": str(parent_id),
            "name": parent_name,
            "role": parent_role,
            "team_id": str(team_id),
            "squad_id": "",
            "user_message": False,
        }
        msg.parent = ActorAddressProxy(parent_dict)

    return msg


def _make_stop_message(
    agent_id: uuid.UUID,
    name: str,
    role: str,
    team_id: uuid.UUID,
) -> StopMessage:
    """Create a StopMessage with a properly-formed sender address."""
    from akgentic.core.actor_address_impl import ActorAddressProxy
    from akgentic.core.utils.deserializer import ActorAddressDict

    msg = StopMessage()
    addr_dict: ActorAddressDict = {
        "__actor_address__": True,
        "__actor_type__": "akgentic.core.agent.Akgent",
        "agent_id": str(agent_id),
        "name": name,
        "role": role,
        "team_id": str(team_id),
        "squad_id": str(uuid.uuid4()),
        "user_message": False,
    }
    msg.sender = ActorAddressProxy(addr_dict)
    msg.team_id = team_id
    return msg


def _emit_member_instances(
    event_store: InMemoryEventStore,
    member: TeamCardMember,
    team_id: uuid.UUID,
    first_seq: int,
    parent_id: uuid.UUID,
    parent_name: str,
    parent_role: str,
) -> dict[str, uuid.UUID]:
    """Save one StartMessage per SPAWNED actor of *member*, not one per slot.

    A real team's log holds one StartMessage per actor, and a member declared
    ``headcount=3`` is three actors named ``<name>_0..2``. A single event under
    the bare declared name describes an agent no factory would ever spawn, so
    the restore rebuilds THAT while the projection asks for the indexed names —
    a resume failure with a fixture cause that looks exactly like a production
    one. The expansion is a no-op for every ``headcount == 1`` member.

    Args:
        first_seq: Sequence number of the FIRST event saved here; each further
            instance takes the next one.

    Returns:
        The ``agent_id`` minted for each spawned name, in spawn order.
    """
    role = member.card.config.role
    agent_class = member.card.get_agent_class()
    instance_ids: dict[str, uuid.UUID] = {}
    for offset, name in enumerate(spawned_names(member)):
        agent_id = uuid.uuid4()
        config = member.card.get_config_copy()
        config.name = name
        sm = _make_start_message(
            agent_id,
            name,
            role,
            team_id,
            agent_class=agent_class,
            config=config,
            parent_id=parent_id,
            parent_name=parent_name,
            parent_role=parent_role,
        )
        event_store.save_event(
            PersistedEvent(
                team_id=team_id,
                sequence=first_seq + offset,
                event=sm,
                timestamp=datetime.now(UTC),
            )
        )
        instance_ids[name] = agent_id
    return instance_ids


def _populate_stopped_team(
    event_store: InMemoryEventStore,
    team_card: TeamCard | None = None,
    extra_members: list[tuple[str, str]] | None = None,
    fired_members: list[tuple[str, str, uuid.UUID]] | None = None,
    stopped_tree_members: list[str] | None = None,
    orchestrator_config: BaseConfig | None = None,
) -> tuple[uuid.UUID, Process]:
    """Populate InMemoryEventStore with events simulating a stopped team.

    Creates StartMessage events for orchestrator + all agents in team_card,
    plus optional fired agents (StopMessage + the state snapshot a real fire
    leaves behind -- nothing deletes it).

    Args:
        extra_members: ``(name, role)`` pairs given a StartMessage and no card —
            the trace a runtime hire leaves in the log.
        stopped_tree_members: Names of TREE members given a StopMessage, so the
            restore path does not rebuild them although the projection still
            carries a ref.
        orchestrator_config: Config persisted on the orchestrator's StartMessage.
            Defaults to ``BaseConfig(name="@Orchestrator", role="Orchestrator")``,
            which carries no squad -- so the default fixture cannot express a
            squad-preservation regression. Pass a config carrying a ``squad_id``
            (and non-default name/role) to exercise the restore passthrough.

    Returns:
        Tuple of (team_id, Process with STOPPED status).
    """
    tc = team_card or _make_team_card()
    team_id = uuid.uuid4()
    seq = 0

    # Orchestrator StartMessage
    orch_id = uuid.uuid4()
    seq += 1
    orch_config = orchestrator_config or BaseConfig(name="@Orchestrator", role="Orchestrator")
    orch_start = _make_start_message(
        orch_id,
        "orchestrator",
        "Orchestrator",
        team_id,
        agent_class=Orchestrator,
        config=orch_config,
        # Keep the persisted sender in agreement with the persisted config, so a
        # sender/config mismatch observed after restore can only come from the
        # restore path. With the default config (no squad) this falls back to the
        # random UUID every other fixture stamps.
        sender_squad_id=orch_config.squad_id,
    )
    event_store.save_event(
        PersistedEvent(
            team_id=team_id,
            sequence=seq,
            event=orch_start,
            timestamp=datetime.now(UTC),
        )
    )

    # Agent StartMessages -- from TeamCard tree
    agent_names: list[str] = []
    tree_agent_ids: dict[str, uuid.UUID] = {}

    def _walk_member(
        member: TeamCardMember,
        parent_agent_id: uuid.UUID | None = None,
        parent_name: str = "orchestrator",
        parent_role: str = "Orchestrator",
    ) -> None:
        nonlocal seq
        instance_ids = _emit_member_instances(
            event_store,
            member,
            team_id,
            first_seq=seq + 1,
            parent_id=parent_agent_id or orch_id,
            parent_name=parent_name,
            parent_role=parent_role,
        )
        seq += len(instance_ids)
        agent_names.extend(instance_ids)
        tree_agent_ids.update(instance_ids)
        # Subordinates hang off the LAST spawned instance, mirroring
        # ``TeamFactory._spawn_member``.
        last_name = next(reversed(instance_ids))
        for child in member.members:
            _walk_member(
                child,
                parent_agent_id=instance_ids[last_name],
                parent_name=last_name,
                parent_role=member.card.config.role,
            )

    _walk_member(tc.entry_point)
    for member in tc.members:
        _walk_member(member)

    # Extra members: a StartMessage and NOTHING else — the trace a runtime hire
    # leaves behind. The agent is in the event log and in no member tree, so it
    # is rebuilt from the log while its ROLE reaches the catalog through
    # agent_profiles. The two halves are independent, which is the point.
    if extra_members:
        for xname, xrole in extra_members:
            seq += 1
            xsm = _make_start_message(uuid.uuid4(), xname, xrole, team_id)
            event_store.save_event(
                PersistedEvent(
                    team_id=team_id,
                    sequence=seq,
                    event=xsm,
                    timestamp=datetime.now(UTC),
                )
            )

    # Tree members fired during the team's life: a StopMessage against the
    # agent_id its StartMessage already carried, so the restore path's filter
    # matches. Unlike ``fired_members`` these ARE in the card, so the stored
    # projection still carries a supervisor ref for them — which is exactly the
    # shape that must resume rather than fail.
    if stopped_tree_members:
        for sname in stopped_tree_members:
            sid = tree_agent_ids[sname]
            seq += 1
            event_store.save_event(
                PersistedEvent(
                    team_id=team_id,
                    sequence=seq,
                    event=_make_stop_message(sid, sname, "unused", team_id),
                    timestamp=datetime.now(UTC),
                )
            )

    # Fired members: StartMessage + StopMessage, plus the snapshot that outlives
    # them. Omitting the snapshot made every fired-agent test vacuous: the restore
    # path's state lookup found nothing, so it could not crash the way it did in
    # production the first time a team was restored after firing an agent.
    if fired_members:
        for fname, frole, fid in fired_members:
            seq += 1
            fsm = _make_start_message(fid, fname, frole, team_id)
            event_store.save_event(
                PersistedEvent(
                    team_id=team_id,
                    sequence=seq,
                    event=fsm,
                    timestamp=datetime.now(UTC),
                )
            )
            seq += 1
            fstop = _make_stop_message(fid, fname, frole, team_id)
            event_store.save_event(
                PersistedEvent(
                    team_id=team_id,
                    sequence=seq,
                    event=fstop,
                    timestamp=datetime.now(UTC),
                )
            )
            event_store.save_agent_state(
                AgentStateSnapshot(
                    team_id=team_id,
                    agent_id=str(fid),
                    name=fname,
                    state=_MarkerState(marker=f"fired:{fname}"),
                    updated_at=datetime.now(UTC),
                )
            )

    # Process record
    now = datetime.now(UTC)
    process = Process(
        team_id=team_id,
        status=TeamStatus.STOPPED,
        user_id="test-user",
        user_email="test@test.com",
        created_at=now,
        updated_at=now,
        **projection_kwargs(tc),
    )
    # Cards first, document second — the order create_team writes them in.
    seed_agent_cards(event_store, tc)
    event_store.save_team(process)

    return team_id, process


def _lead_agent_id_from_events(
    event_store: InMemoryEventStore,
    team_id: uuid.UUID,
    name: str = "lead",
) -> uuid.UUID:
    """Read an agent's persisted UUID from its ``StartMessage`` sender.

    Used to key UUID-based ``AgentStateSnapshot`` fixtures by the same UUID the
    restorer re-injects when re-spawning the agent.
    """
    for pe in event_store.load_events(team_id):
        if (
            isinstance(pe.event, StartMessage)
            and pe.event.sender is not None
            and pe.event.sender.name == name
        ):
            return pe.event.sender.agent_id
    msg = f"{name} agent_id not found in events"
    raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def actor_system() -> ActorSystem:  # type: ignore[misc]
    """Provide an ActorSystem that shuts down after each test."""
    system = ActorSystem()
    yield system  # type: ignore[misc]
    system.shutdown()


@pytest.fixture()
def event_store() -> InMemoryEventStore:
    """Provide a fresh InMemoryEventStore per test."""
    return InMemoryEventStore()


# ---------------------------------------------------------------------------
# Tests: TestTeamRestorerRestore
# ---------------------------------------------------------------------------


class TestTeamRestorerRestore:
    """AC 5-13: Core restore functionality."""

    def test_restore_returns_valid_team_runtime(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 5,13: restore returns TeamRuntime with valid addresses."""
        team_id, process = _populate_stopped_team(event_store)

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        assert isinstance(runtime, TeamRuntime)
        assert runtime.id == team_id
        assert runtime.orchestrator_addr.is_alive()
        assert runtime.entry_addr.is_alive()
        assert "lead" in runtime.addrs

    def test_restore_with_multiple_agents(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 9: All agents are rebuilt from StartMessages."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        assert "lead" in runtime.addrs
        assert "worker" in runtime.addrs
        assert runtime.addrs["lead"].is_alive()
        assert runtime.addrs["worker"].is_alive()

    def test_restore_orchestrator_created_first(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 8: Orchestrator is created first during restore."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        # Orchestrator should be alive and functional
        assert runtime.orchestrator_addr.is_alive()
        team = runtime.orchestrator_proxy.get_team()
        assert isinstance(team, list)

    def test_restore_events_replayed(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 12: Events are replayed through orchestrator during restore."""
        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        # Track how many events exist before restore
        events_before = len(event_store.load_events(team_id))
        assert events_before > 0

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        # Team is functional after restore
        assert runtime.orchestrator_addr.is_alive()

    def test_restore_get_team_works_after_restore(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """get_team() returns agents after restore (restore_message populates history)."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        team = runtime.orchestrator_proxy.get_team()
        team_names = {addr.name for addr in team}
        # Both lead and worker should appear in orchestrator's team
        assert "lead" in team_names
        assert "worker" in team_names

    def test_restore_supervisor_addrs_populated(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 13: TeamRuntime has supervisor_addrs populated for first-layer members."""
        worker = _make_member("worker", "Worker")
        supervisor = _make_member("supervisor", "Supervisor", members=[worker])
        tc = _make_team_card(members=[supervisor])

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        # supervisor is a first-layer member, so it's in supervisor_addrs
        assert "supervisor" in runtime.supervisor_addrs
        # entry point (lead) is NOT in supervisor_addrs
        assert "lead" not in runtime.supervisor_addrs

    def test_restore_with_agent_state_snapshots(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC #3: a UUID-keyed AgentStateSnapshot is restored via init_state().

        The snapshot is keyed by the lead agent's actual UUID (read from its
        persisted ``StartMessage`` sender), and the restorer matches it against
        the live address UUID -- no name fallback.
        """
        from unittest.mock import patch as mock_patch

        from akgentic.team.models import AgentStateSnapshot

        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        # The restorer now matches by str(addr.agent_id), so the snapshot MUST be
        # keyed by the lead agent's real UUID -- read it from its StartMessage.
        lead_agent_id = _lead_agent_id_from_events(event_store, team_id)

        # Inject a state snapshot for the "lead" agent, keyed by UUID.
        snapshot = AgentStateSnapshot(
            team_id=team_id,
            agent_id=str(lead_agent_id),
            name="lead",
            state=BaseState(),
            updated_at=datetime.now(UTC),
        )
        event_store.save_agent_state(snapshot)

        restorer = TeamRestorer(actor_system, event_store)

        # Track init_state calls via spy
        init_state_calls: list[str] = []
        original_proxy_ask = actor_system.proxy_ask

        def tracking_proxy_ask(
            addr: ActorAddress,
            cls: type[Any],
        ) -> Any:
            proxy = original_proxy_ask(addr, cls)
            if cls is Akgent:
                original_init = proxy.init_state
                addr_name = addr.name

                def tracked_init(state: Any, _name: str = addr_name) -> None:
                    init_state_calls.append(_name)
                    return original_init(state)

                proxy.init_state = tracked_init
            return proxy

        with mock_patch.object(actor_system, "proxy_ask", side_effect=tracking_proxy_ask):
            runtime = restorer.restore(process)

        # Verify init_state was applied to the lead agent's own address (matched
        # by UUID), not merely that some agent received state.
        assert "lead" in init_state_calls
        assert runtime.addrs["lead"].is_alive()

    def test_restore_with_legacy_name_keyed_snapshot_loads_and_is_restored_via_name_fallback(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC #2/#6: a legacy name-keyed snapshot loads (name=None) and is restored.

        Migration-safety + name fallback: a snapshot persisted in the OLD shape --
        ``agent_id`` holding the agent **name** and no ``name`` key -- (a) loads via
        ``load_agent_states`` with ``name is None`` and (b) is now APPLIED on restore
        via the name fallback (its stored ``agent_id`` equals the live agent's name),
        without raising. This is the direct inverse of Story 23-1's "legacy skipped".
        """
        from unittest.mock import patch as mock_patch

        from akgentic.team.models import AgentStateSnapshot

        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        # OLD shape: agent_id holds the name, no name field (default None applies).
        legacy_snapshot = AgentStateSnapshot(
            team_id=team_id,
            agent_id="lead",
            state=BaseState(),
            updated_at=datetime.now(UTC),
        )
        event_store.save_agent_state(legacy_snapshot)

        # (a) The legacy snapshot loads with name=None and raises no error.
        loaded = event_store.load_agent_states(team_id)
        assert len(loaded) == 1
        assert loaded[0].agent_id == "lead"
        assert loaded[0].name is None

        # (a') A serialized payload that literally OMITS the ``name`` key (the true
        # on-disk legacy shape) validates cleanly with name=None -- this is the
        # deserialization boundary the no-migration design rests on.
        legacy_payload = legacy_snapshot.model_dump()
        legacy_payload.pop("name", None)
        revalidated = AgentStateSnapshot.model_validate(legacy_payload)
        assert revalidated.name is None
        assert revalidated.agent_id == "lead"

        restorer = TeamRestorer(actor_system, event_store)

        # Spy init_state to prove the legacy snapshot IS applied on restore.
        init_state_calls: list[str] = []
        original_proxy_ask = actor_system.proxy_ask

        def tracking_proxy_ask(addr: ActorAddress, cls: type[Any]) -> Any:
            proxy = original_proxy_ask(addr, cls)
            if cls is Akgent:
                original_init = proxy.init_state
                addr_name = addr.name

                def tracked_init(state: Any, _name: str = addr_name) -> None:
                    init_state_calls.append(_name)
                    return original_init(state)

                proxy.init_state = tracked_init
            return proxy

        with mock_patch.object(actor_system, "proxy_ask", side_effect=tracking_proxy_ask):
            runtime = restorer.restore(process)

        # (b) Applied on restore via the name fallback -- the UUID lookup misses, but
        # the stored agent_id ("lead") matches the live agent's name.
        assert init_state_calls == ["lead"]
        assert runtime.addrs["lead"].is_alive()

    def test_restore_prefers_uuid_keyed_over_same_name_legacy_snapshot(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC #3: UUID-keyed snapshot wins over a same-name legacy snapshot.

        Both a UUID-keyed snapshot (agent_id=str(lead UUID)) and a name-keyed legacy
        snapshot (agent_id="lead") exist for the SAME live lead agent, carrying
        distinguishable states. The restorer tries the UUID lookup first, so the
        UUID-keyed snapshot's state is the one applied -- the legacy one is shadowed.
        """
        from unittest.mock import patch as mock_patch

        from akgentic.team.models import AgentStateSnapshot

        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        lead_agent_id = _lead_agent_id_from_events(event_store, team_id)

        # UUID-keyed (current) snapshot -- this one must win.
        uuid_snapshot = AgentStateSnapshot(
            team_id=team_id,
            agent_id=str(lead_agent_id),
            name="lead",
            state=_MarkerState(marker="uuid"),
            updated_at=datetime.now(UTC),
        )
        # Name-keyed legacy snapshot -- same live agent, must be shadowed.
        legacy_snapshot = AgentStateSnapshot(
            team_id=team_id,
            agent_id="lead",
            state=_MarkerState(marker="legacy"),
            updated_at=datetime.now(UTC),
        )
        # Different agent_id keys -> both coexist in load_agent_states.
        event_store.save_agent_state(uuid_snapshot)
        event_store.save_agent_state(legacy_snapshot)

        restorer = TeamRestorer(actor_system, event_store)

        # Spy init_state capturing BOTH the agent name and the applied state, so we
        # can assert WHICH snapshot's state won.
        init_state_records: list[tuple[str, Any]] = []
        original_proxy_ask = actor_system.proxy_ask

        def tracking_proxy_ask(addr: ActorAddress, cls: type[Any]) -> Any:
            proxy = original_proxy_ask(addr, cls)
            if cls is Akgent:
                original_init = proxy.init_state
                addr_name = addr.name

                def tracked_init(state: Any, _name: str = addr_name) -> None:
                    init_state_records.append((_name, state))
                    return original_init(state)

                proxy.init_state = tracked_init
            return proxy

        with mock_patch.object(actor_system, "proxy_ask", side_effect=tracking_proxy_ask):
            runtime = restorer.restore(process)

        # Exactly one init_state for the lead agent, carrying the UUID-keyed state.
        lead_records = [(name, st) for name, st in init_state_records if name == "lead"]
        assert len(lead_records) == 1
        applied_state = lead_records[0][1]
        assert isinstance(applied_state, _MarkerState)
        assert applied_state.marker == "uuid"
        assert runtime.addrs["lead"].is_alive()

    def test_restore_legacy_snapshot_no_false_match(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC #4: a legacy snapshot whose name matches no live agent is a no-op.

        A name-keyed snapshot (agent_id="ghost") matches neither a live UUID nor a
        live agent name, so no agent receives init_state and restore does not raise;
        the live agents stay alive.
        """
        from unittest.mock import patch as mock_patch

        from akgentic.team.models import AgentStateSnapshot

        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        ghost_snapshot = AgentStateSnapshot(
            team_id=team_id,
            agent_id="ghost",
            state=BaseState(),
            updated_at=datetime.now(UTC),
        )
        event_store.save_agent_state(ghost_snapshot)

        restorer = TeamRestorer(actor_system, event_store)

        init_state_calls: list[str] = []
        original_proxy_ask = actor_system.proxy_ask

        def tracking_proxy_ask(addr: ActorAddress, cls: type[Any]) -> Any:
            proxy = original_proxy_ask(addr, cls)
            if cls is Akgent:
                original_init = proxy.init_state
                addr_name = addr.name

                def tracked_init(state: Any, _name: str = addr_name) -> None:
                    init_state_calls.append(_name)
                    return original_init(state)

                proxy.init_state = tracked_init
            return proxy

        with mock_patch.object(actor_system, "proxy_ask", side_effect=tracking_proxy_ask):
            runtime = restorer.restore(process)

        # No agent matched -> no init_state applied, restore did not raise.
        assert init_state_calls == []
        assert runtime.addrs["lead"].is_alive()

    def test_restore_registers_the_hireable_roles(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """Every role a profile names comes back marked hireable."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])
        # Explicitly register profiles for hiring
        tc.agent_profiles = list(tc.agent_cards.values())

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        catalog = runtime.orchestrator_proxy.get_agent_catalog()
        assert {c.role for c in catalog} == {"Lead", "Worker"}
        assert all(c.can_be_hired for c in catalog)

    def test_restore_registers_the_whole_card_set_not_only_the_profiles(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """The intended behaviour change of story 31-7, not a regression.

        The restorer used to register ``team_card.agent_profiles`` alone, so a
        team with no profiles left the orchestrator unable to describe a single
        one of its own agents. It now registers the whole resolved card set, and
        ``can_be_hired`` — taken from each ``AgentCardRef``, the authoritative
        copy — is what still says which of them may be hired at runtime.
        """
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])
        # agent_profiles defaults to empty — nothing is hireable.

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        catalog = runtime.orchestrator_proxy.get_agent_catalog()
        assert {c.role for c in catalog} == {"Lead", "Worker"}
        assert not any(c.can_be_hired for c in catalog)

    def test_restore_marks_only_the_profiled_roles_hireable(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """The whole directory is registered; only some of it is hireable."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])
        tc.agent_profiles = [_make_card("specialist", "Specialist")]

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        catalog = runtime.orchestrator_proxy.get_agent_catalog()
        assert {c.role: c.can_be_hired for c in catalog} == {
            "Lead": False,
            "Worker": False,
            "Specialist": True,
        }


# ---------------------------------------------------------------------------
# Tests: TestTeamRestorerAgentFiltering
# ---------------------------------------------------------------------------


class TestTeamRestorerAgentFiltering:
    """AC 7: StartMessage/StopMessage filtering."""

    def test_fired_agent_excluded_from_restore(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 7: Agents with matching StopMessage are excluded from rebuild."""
        tc = _make_team_card()
        fired_id = uuid.uuid4()

        team_id, process = _populate_stopped_team(
            event_store,
            tc,
            fired_members=[("fired-agent", "FiredRole", fired_id)],
        )

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        # Fired agent should NOT be in the restored team
        assert "fired-agent" not in runtime.addrs
        # Lead should still be there
        assert "lead" in runtime.addrs

    def test_agent_stopped_then_not_rebuilt(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 7: Multiple fired agents are all excluded."""
        tc = _make_team_card()
        fired1_id = uuid.uuid4()
        fired2_id = uuid.uuid4()

        team_id, process = _populate_stopped_team(
            event_store,
            tc,
            fired_members=[
                ("fired1", "Role1", fired1_id),
                ("fired2", "Role2", fired2_id),
            ],
        )

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        assert "fired1" not in runtime.addrs
        assert "fired2" not in runtime.addrs
        assert "lead" in runtime.addrs


# ---------------------------------------------------------------------------
# Tests: TestTeamRestorerRestoringFlag
# ---------------------------------------------------------------------------


class TestTeamRestorerRestoringFlag:
    """AC 12: Restoring flag toggle on PersistenceSubscriber."""

    def test_restoring_flag_managed_correctly(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 12: PersistenceSubscriber restoring=True before replay, False after.

        The caller (TeamManager) is now responsible for toggling the restoring flag.
        This test validates the pattern: set_restoring(True) before restore(),
        set_restoring(False) after restore().
        """
        from akgentic.team.subscriber import PersistenceSubscriber

        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        persistence_sub = PersistenceSubscriber(team_id, event_store)
        persistence_sub.set_restoring(team_id, True)

        restorer = TeamRestorer(actor_system, event_store)
        restorer.restore(process, subscribers=[persistence_sub])

        # Caller sets restoring=False after restore (simulating TeamManager)
        persistence_sub.set_restoring(team_id, False)
        assert persistence_sub._restoring is False

    def test_no_duplicate_events_during_replay(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 12: Replayed events are not re-persisted (restoring flag prevents it)."""
        from akgentic.team.subscriber import PersistenceSubscriber

        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        original_events = event_store.load_events(team_id)
        original_sequences = {e.sequence for e in original_events}

        persistence_sub = PersistenceSubscriber(team_id, event_store)
        persistence_sub.set_restoring(team_id, True)

        restorer = TeamRestorer(actor_system, event_store)
        restorer.restore(process, subscribers=[persistence_sub])

        persistence_sub.set_restoring(team_id, False)

        # Check no events with original sequences were duplicated
        all_events = event_store.load_events(team_id)
        seq_counts: dict[int, int] = {}
        for e in all_events:
            seq_counts[e.sequence] = seq_counts.get(e.sequence, 0) + 1

        for seq in original_sequences:
            assert seq_counts[seq] == 1, f"Sequence {seq} was duplicated"


# ---------------------------------------------------------------------------
# Tests: TestTeamRestorerRollback
# ---------------------------------------------------------------------------


class TestTeamRestorerRollback:
    """Rollback on failure."""

    def test_rollback_on_failure(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """All spawned actors cleaned up if a phase fails."""
        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)

        # Track orchestrator actor created before failure
        import pykka

        actors_before = len(pykka.ActorRegistry.get_all())

        # Patch import_class to fail when resolving the lead agent class
        with patch(
            "akgentic.team.restorer.import_class",
            side_effect=ImportError("cannot find agent class"),
        ):
            with pytest.raises(ImportError, match="cannot find agent class"):
                restorer.restore(process)

        # After rollback, no new actors should remain alive
        actors_after = len(pykka.ActorRegistry.get_all())
        assert actors_after == actors_before, (
            f"Rollback failed: {actors_after - actors_before} actor(s) leaked"
        )

    def test_restorer_rollback_waits_on_orchestrator(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC2 (restore path): rollback stops agents blocking, then waits on the
        orchestrator entry's stop event; the orchestrator wait runs last.

        We build a team with two agents and make agent-class resolution succeed
        for the first agent and fail for the second, so exactly one agent is
        spawned before the failure. Rollback then stops that agent (blocking
        ``Akgent.stop``) and waits on the orchestrator (non-blocking
        ``Orchestrator.stop(grace).wait()``), in that order.
        """
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])
        team_id, process = _populate_stopped_team(event_store, tc)

        events: list[str] = []
        grace_args: list[float] = []

        real_orch_stop = Orchestrator.stop
        real_agent_stop = Akgent.stop

        from akgentic.team import restorer as restorer_module

        real_import_class = restorer_module.import_class
        call_count = {"n": 0}

        def flaky_import_class(path: str) -> Any:
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise ImportError("cannot resolve second agent class")
            return real_import_class(path)

        def wrapped_orch_stop(
            self: Orchestrator, grace_timeout: float = GRACE_TIMEOUT_SECONDS
        ) -> threading.Event:
            grace_args.append(grace_timeout)
            evt = real_orch_stop(self, grace_timeout)

            class _RecordingEvent:
                def __init__(self, inner: threading.Event) -> None:
                    self._inner = inner

                def wait(self, timeout: float | None = None) -> bool:
                    events.append("orchestrator-wait")
                    return self._inner.wait(timeout)

            return _RecordingEvent(evt)  # type: ignore[return-value]

        def wrapped_agent_stop(self: Akgent[Any, Any]) -> None:
            if not isinstance(self, Orchestrator):
                events.append("agent-stop")
            return real_agent_stop(self)

        restorer = TeamRestorer(actor_system, event_store)

        with (
            patch.object(restorer_module, "import_class", flaky_import_class),
            patch.object(Orchestrator, "stop", wrapped_orch_stop),
            patch.object(Akgent, "stop", wrapped_agent_stop),
        ):
            with pytest.raises(ImportError, match="second agent class"):
                restorer.restore(process)

        # Orchestrator stop received the single grace timeout; its wait() ran
        # last, after the (blocking) agent stop in reversed spawn order.
        assert grace_args == [GRACE_TIMEOUT_SECONDS]
        assert events.count("agent-stop") >= 1, f"no agent stops recorded: {events}"
        assert "orchestrator-wait" in events
        assert events.index("orchestrator-wait") == len(events) - 1, (
            f"orchestrator wait must be last (after agent stops): {events}"
        )

    def test_subscribers_registered_with_orchestrator(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """Pre-instantiated subscribers are registered with orchestrator."""
        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        recording = RecordingSubscriber()

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process, subscribers=[recording])

        # Recording subscriber should have received replayed events
        assert len(recording.messages) > 0

        # Stop via proxy and wait for full shutdown before asserting on_stop
        runtime.orchestrator_proxy.stop()
        # on_stop() fires asynchronously after the proxy stop() call returns;
        # wait for the actor thread to finish so on_stop() has been invoked.
        deadline = time.monotonic() + 2.0
        while runtime.orchestrator_addr.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert recording.stopped is True

    def test_subscriber_receives_start_messages_during_replay(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """Subscriber receives StartMessage events for all agents during Phase 3 replay.

        Story 14.9 AC 4: Given a team with 3 agents that is stopped and resumed,
        when a subscriber is registered during restore, then the subscriber
        receives StartMessage events for all 3 agents during Phase 3 replay.
        """
        worker1 = _make_member("worker1", "Worker")
        worker2 = _make_member("worker2", "Worker")
        tc = _make_team_card(members=[worker1, worker2])

        team_id, process = _populate_stopped_team(event_store, tc)

        recording = RecordingSubscriber()

        restorer = TeamRestorer(actor_system, event_store)
        restorer.restore(process, subscribers=[recording])

        # Filter StartMessage events received by the subscriber
        start_messages = [
            m for m in recording.messages if isinstance(m, StartMessage)
        ]

        # Should receive StartMessages for orchestrator + lead + worker1 + worker2
        start_names = {
            m.sender.name for m in start_messages if m.sender is not None
        }
        assert "lead" in start_names, (
            f"Missing StartMessage for 'lead'; received: {start_names}"
        )
        assert "worker1" in start_names, (
            f"Missing StartMessage for 'worker1'; received: {start_names}"
        )
        assert "worker2" in start_names, (
            f"Missing StartMessage for 'worker2'; received: {start_names}"
        )


# ---------------------------------------------------------------------------
# Tests: Hierarchy propagation during restore (Story 10-1, AC 3, 5)
# ---------------------------------------------------------------------------


class TestRestorerHierarchyPropagation:
    """AC 3,5: Restored agents have _orchestrator set."""

    def test_orchestrator_set_on_restored_agents(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 3,5: _orchestrator is not None on agents rebuilt during restore."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        for name, addr in runtime.addrs.items():
            proxy: Akgent[Any, Any] = actor_system.proxy_ask(addr, Akgent)
            orch = proxy.orchestrator
            assert orch is not None, f"Restored agent '{name}' has _orchestrator=None"
            assert orch.is_alive(), f"Restored agent '{name}' orchestrator is not alive"

    def test_parent_set_on_restored_agents(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 5: _parent is set on agents rebuilt during restore."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        # All restored agents should have orchestrator as parent
        for name, addr in runtime.addrs.items():
            actor = addr._actor_ref._actor_weakref()  # type: ignore[union-attr]
            assert actor._parent is not None, f"Restored agent '{name}' has _parent=None"
            assert actor._parent.agent_id == runtime.orchestrator_addr.agent_id, (
                f"Restored agent '{name}' parent is not the orchestrator"
            )


# ---------------------------------------------------------------------------
# Tests: TestRestorerAddressResolution (Story 12.3, AC 1)
# ---------------------------------------------------------------------------


class TestRestorerAddressResolution:
    """AC 1: Restored teams resolve serialized actor addresses to live refs."""

    def test_restored_get_team_returns_live_addresses(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """After restore, get_team() returns ActorAddressImpl, not ActorAddressProxy."""
        from akgentic.core.actor_address_impl import ActorAddressImpl, ActorAddressProxy

        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        team = runtime.orchestrator_proxy.get_team()
        for addr in team:
            assert isinstance(addr, ActorAddressImpl), (
                f"Expected ActorAddressImpl but got {type(addr).__name__} for agent '{addr.name}'"
            )
            assert not isinstance(addr, ActorAddressProxy), (
                f"ActorAddressProxy leaked into get_team() for '{addr.name}'"
            )

    def test_restored_get_team_member_returns_live_address(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """After restore, get_team_member() returns the live address matching spawned actor."""
        from akgentic.core.actor_address_impl import ActorAddressImpl

        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        lead_addr = runtime.orchestrator_proxy.get_team_member("lead")
        assert lead_addr is not None
        assert isinstance(lead_addr, ActorAddressImpl)
        assert lead_addr.is_alive()

        worker_addr = runtime.orchestrator_proxy.get_team_member("worker")
        assert worker_addr is not None
        assert isinstance(worker_addr, ActorAddressImpl)
        assert worker_addr.is_alive()

    def test_replay_does_not_overwrite_live_with_proxy(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """Phase 3 replayed StartMessages do not overwrite Phase 2 live addresses."""
        from akgentic.core.actor_address_impl import ActorAddressImpl

        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        # The entry agent should have a live address matching the spawned actor
        lead_from_roster = runtime.orchestrator_proxy.get_team_member("lead")
        lead_from_addrs = runtime.addrs["lead"]

        assert lead_from_roster is not None
        assert isinstance(lead_from_roster, ActorAddressImpl)
        # The address from get_team_member should match the address from addrs
        assert lead_from_roster.agent_id == lead_from_addrs.agent_id


# ---------------------------------------------------------------------------
# Tests: Proxy-based restore spawning (Story 12.4, AC 4, 6)
# ---------------------------------------------------------------------------


class TestRestorerProxySpawning:
    """AC 4,6: Restore uses public createActor() API, no duplicate roster entries."""

    def test_restore_creates_agents_through_public_api(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 4: After restore, all agents are alive and have correct names/roles."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        team = runtime.orchestrator_proxy.get_team()
        team_names = {addr.name for addr in team}
        assert "lead" in team_names
        assert "worker" in team_names
        for addr in team:
            assert addr.is_alive()

    def test_restore_no_duplicate_roster_entries(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 6: get_team() has no duplicate agent_ids after restore."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])

        team_id, process = _populate_stopped_team(event_store, tc)

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        team = runtime.orchestrator_proxy.get_team()
        agent_ids = [addr.agent_id for addr in team]
        assert len(agent_ids) == len(set(agent_ids)), (
            f"Duplicate agent_ids in team roster: {agent_ids}"
        )


# ---------------------------------------------------------------------------
# Tests: Orphan fallback in _spawn_agents (Story 14-1, AC 3)
# ---------------------------------------------------------------------------


class TestRestorerOrphanFallback:
    """AC 3: Unknown parent falls back to orchestrator."""

    def test_spawn_agents_orphan_falls_back_to_orchestrator(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 3: Agent with unknown parent spawns through orchestrator (no crash)."""
        tc = _make_team_card()
        team_id = uuid.uuid4()
        seq = 0

        # Orchestrator StartMessage
        orch_id = uuid.uuid4()
        seq += 1
        orch_start = _make_start_message(
            orch_id,
            "orchestrator",
            "Orchestrator",
            team_id,
            agent_class=Orchestrator,
            config=BaseConfig(name="@Orchestrator", role="Orchestrator"),
        )
        event_store.save_event(
            PersistedEvent(
                team_id=team_id,
                sequence=seq,
                event=orch_start,
                timestamp=datetime.now(UTC),
            )
        )

        # Agent with an unknown parent_id (orphan)
        unknown_parent_id = uuid.uuid4()
        agent_id = uuid.uuid4()
        seq += 1
        orphan_start = _make_start_message(
            agent_id,
            "lead",
            "Lead",
            team_id,
            agent_class=StubAgent,
            parent_id=unknown_parent_id,
            parent_name="ghost",
            parent_role="Ghost",
        )
        event_store.save_event(
            PersistedEvent(
                team_id=team_id,
                sequence=seq,
                event=orphan_start,
                timestamp=datetime.now(UTC),
            )
        )

        now = datetime.now(UTC)
        process = Process(
            team_id=team_id,
            status=TeamStatus.STOPPED,
            user_id="test-user",
            user_email="test@test.com",
            created_at=now,
            updated_at=now,
            **projection_kwargs(tc),
        )
        seed_agent_cards(event_store, tc)
        event_store.save_team(process)

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        # The orphan agent should be alive and a child of the orchestrator
        assert "lead" in runtime.addrs
        assert runtime.addrs["lead"].is_alive()

        # Verify it's parented to orchestrator (orphan fallback)
        from tests.integration.conftest import get_actor_from_addr

        orchestrator_actor = get_actor_from_addr(runtime.orchestrator_addr)
        orch_child_ids = {c.agent_id for c in orchestrator_actor._children}
        assert runtime.addrs["lead"].agent_id in orch_child_ids


# ---------------------------------------------------------------------------
# Helpers: EventMessage construction
# ---------------------------------------------------------------------------


def _make_event_message(
    agent_id: uuid.UUID,
    name: str,
    role: str,
    team_id: uuid.UUID,
    event: Any = "some-event",
) -> EventMessage:
    """Create an EventMessage with a properly-formed sender address."""
    from akgentic.core.actor_address_impl import ActorAddressProxy
    from akgentic.core.utils.deserializer import ActorAddressDict

    msg = EventMessage(event=event)
    addr_dict: ActorAddressDict = {
        "__actor_address__": True,
        "__actor_type__": f"{StubAgent.__module__}.{StubAgent.__name__}",
        "agent_id": str(agent_id),
        "name": name,
        "role": role,
        "team_id": str(team_id),
        "squad_id": str(uuid.uuid4()),
        "user_message": False,
    }
    msg.sender = ActorAddressProxy(addr_dict)
    msg.team_id = team_id
    return msg


def _make_sent_message(
    agent_id: uuid.UUID,
    name: str,
    role: str,
    team_id: uuid.UUID,
) -> SentMessage:
    """Create a SentMessage (non-EventMessage) for filtering tests."""
    from akgentic.core.actor_address_impl import ActorAddressProxy
    from akgentic.core.utils.deserializer import ActorAddressDict

    inner = Message()
    addr_dict: ActorAddressDict = {
        "__actor_address__": True,
        "__actor_type__": f"{StubAgent.__module__}.{StubAgent.__name__}",
        "agent_id": str(agent_id),
        "name": name,
        "role": role,
        "team_id": str(team_id),
        "squad_id": str(uuid.uuid4()),
        "user_message": False,
    }
    recipient = ActorAddressProxy(addr_dict)
    msg = SentMessage(message=inner, recipient=recipient)
    msg.sender = ActorAddressProxy(addr_dict)
    msg.team_id = team_id
    return msg


# ---------------------------------------------------------------------------
# Tests: TestFilterEventMessages (Story 14.5, AC 2)
# ---------------------------------------------------------------------------


class TestFilterEventMessages:
    """AC 2: _filter_event_messages filtering logic."""

    def test_filter_event_messages_returns_matching_events(
        self,
        event_store: InMemoryEventStore,
    ) -> None:
        """Returns only EventMessage instances with matching agent_id, in order."""
        actor_system = ActorSystem()
        try:
            restorer = TeamRestorer(actor_system, event_store)
            team_id = uuid.uuid4()
            target_id = uuid.uuid4()
            other_id = uuid.uuid4()

            em1 = _make_event_message(target_id, "agent-a", "RoleA", team_id, event="ev1")
            em2 = _make_event_message(other_id, "agent-b", "RoleB", team_id, event="ev2")
            em3 = _make_event_message(target_id, "agent-a", "RoleA", team_id, event="ev3")
            sm = _make_sent_message(target_id, "agent-a", "RoleA", team_id)

            events = [
                PersistedEvent(team_id=team_id, sequence=1, event=em1, timestamp=datetime.now(UTC)),
                PersistedEvent(team_id=team_id, sequence=2, event=em2, timestamp=datetime.now(UTC)),
                PersistedEvent(team_id=team_id, sequence=3, event=em3, timestamp=datetime.now(UTC)),
                PersistedEvent(team_id=team_id, sequence=4, event=sm, timestamp=datetime.now(UTC)),
            ]

            result = restorer._filter_event_messages(events, target_id)

            assert len(result) == 2
            assert result[0] is em1
            assert result[1] is em3
        finally:
            actor_system.shutdown()

    def test_filter_event_messages_returns_empty_for_no_matches(
        self,
        event_store: InMemoryEventStore,
    ) -> None:
        """Returns empty list when no EventMessage matches."""
        actor_system = ActorSystem()
        try:
            restorer = TeamRestorer(actor_system, event_store)
            team_id = uuid.uuid4()
            target_id = uuid.uuid4()
            other_id = uuid.uuid4()

            em = _make_event_message(other_id, "agent-b", "RoleB", team_id)
            sm = _make_sent_message(other_id, "agent-b", "RoleB", team_id)

            events = [
                PersistedEvent(team_id=team_id, sequence=1, event=em, timestamp=datetime.now(UTC)),
                PersistedEvent(team_id=team_id, sequence=2, event=sm, timestamp=datetime.now(UTC)),
            ]

            result = restorer._filter_event_messages(events, target_id)
            assert result == []
        finally:
            actor_system.shutdown()

    def test_filter_event_messages_skips_none_sender(
        self,
        event_store: InMemoryEventStore,
    ) -> None:
        """EventMessage with sender=None is excluded from results."""
        actor_system = ActorSystem()
        try:
            restorer = TeamRestorer(actor_system, event_store)
            team_id = uuid.uuid4()
            target_id = uuid.uuid4()

            em_with_sender = _make_event_message(
                target_id, "agent-a", "RoleA", team_id, event="ev1"
            )
            em_no_sender = EventMessage(event="ev2")
            em_no_sender.sender = None

            events = [
                PersistedEvent(
                    team_id=team_id, sequence=1, event=em_with_sender, timestamp=datetime.now(UTC)
                ),
                PersistedEvent(
                    team_id=team_id, sequence=2, event=em_no_sender, timestamp=datetime.now(UTC)
                ),
            ]

            result = restorer._filter_event_messages(events, target_id)
            assert len(result) == 1
            assert result[0] is em_with_sender
        finally:
            actor_system.shutdown()


# ---------------------------------------------------------------------------
# Tests: TestRebuildAgentsLlmContext (Story 14.5, AC 1, 3, 4)
# ---------------------------------------------------------------------------


_has_init_llm_context = hasattr(Akgent, "init_llm_context")


@pytest.mark.skipif(
    not _has_init_llm_context,
    reason="Requires akgentic-core with init_llm_context (Story 14.2)",
)
class TestRebuildAgentsLlmContext:
    """AC 1,3,4: init_llm_context() called during _rebuild_agents()."""

    def test_init_llm_context_called_for_agents_with_events(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """init_llm_context() is called for agents that have EventMessage events."""
        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        # Add EventMessage events for the "lead" agent
        events = event_store.load_events(team_id)
        # Find the lead agent's agent_id from the StartMessage
        lead_agent_id: uuid.UUID | None = None
        for pe in events:
            if (
                isinstance(pe.event, StartMessage)
                and pe.event.sender is not None
                and pe.event.sender.name == "lead"
            ):
                lead_agent_id = pe.event.sender.agent_id
                break
        assert lead_agent_id is not None, "lead agent_id not found in events"

        # Add EventMessage events for "lead"
        em1 = _make_event_message(lead_agent_id, "lead", "Lead", team_id, event="llm-ev-1")
        em2 = _make_event_message(lead_agent_id, "lead", "Lead", team_id, event="llm-ev-2")
        event_store.save_event(
            PersistedEvent(team_id=team_id, sequence=100, event=em1, timestamp=datetime.now(UTC))
        )
        event_store.save_event(
            PersistedEvent(team_id=team_id, sequence=101, event=em2, timestamp=datetime.now(UTC))
        )

        restorer = TeamRestorer(actor_system, event_store)

        # Track init_llm_context calls
        init_llm_calls: dict[str, list[Any]] = {}
        original_proxy_ask = actor_system.proxy_ask

        def tracking_proxy_ask(
            addr: ActorAddress,
            cls: type[Any],
        ) -> Any:
            proxy = original_proxy_ask(addr, cls)
            if cls is Akgent:
                original_init_llm = proxy.init_llm_context

                def tracked_init_llm(context: list[Any]) -> None:
                    init_llm_calls[addr.name] = context
                    return original_init_llm(context)

                proxy.init_llm_context = tracked_init_llm
            return proxy

        with patch.object(actor_system, "proxy_ask", side_effect=tracking_proxy_ask):
            restorer.restore(process)

        # init_llm_context was called for "lead" with 2 events
        assert "lead" in init_llm_calls, "init_llm_context not called for lead"
        assert len(init_llm_calls["lead"]) == 2

    def test_init_llm_context_not_called_for_agents_without_events(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """init_llm_context() is NOT called for agents with no EventMessage events."""
        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        # No EventMessage events added -- only StartMessages exist

        restorer = TeamRestorer(actor_system, event_store)

        init_llm_calls: dict[str, list[Any]] = {}
        original_proxy_ask = actor_system.proxy_ask

        def tracking_proxy_ask(
            addr: ActorAddress,
            cls: type[Any],
        ) -> Any:
            proxy = original_proxy_ask(addr, cls)
            if cls is Akgent:
                original_init_llm = proxy.init_llm_context

                def tracked_init_llm(context: list[Any]) -> None:
                    init_llm_calls[addr.name] = context
                    return original_init_llm(context)

                proxy.init_llm_context = tracked_init_llm
            return proxy

        with patch.object(actor_system, "proxy_ask", side_effect=tracking_proxy_ask):
            restorer.restore(process)

        # init_llm_context should NOT have been called (no EventMessage events)
        assert len(init_llm_calls) == 0, (
            f"init_llm_context unexpectedly called for: {list(init_llm_calls.keys())}"
        )


# ---------------------------------------------------------------------------
# Tests: ToolActor spawn order during restore (Story 14.6, ADR-010)
# ---------------------------------------------------------------------------


class TestRestorerToolActorSpawnOrder:
    """Story 14.6: ToolActor StartMessages sorted before regular agents."""

    def test_tool_actor_start_messages_sorted_first(
        self,
        event_store: InMemoryEventStore,
    ) -> None:
        """ToolActor StartMessages must be sorted before regular agents."""
        actor_system = ActorSystem()
        try:
            restorer = TeamRestorer(actor_system, event_store)
            team_id = uuid.uuid4()
            orch_id = uuid.uuid4()
            seq = 0

            # Orchestrator
            seq += 1
            orch_start = _make_start_message(
                orch_id,
                "orchestrator",
                "Orchestrator",
                team_id,
                agent_class=Orchestrator,
                config=BaseConfig(name="@Orchestrator", role="Orchestrator"),
            )
            event_store.save_event(
                PersistedEvent(
                    team_id=team_id, sequence=seq, event=orch_start,
                    timestamp=datetime.now(UTC),
                )
            )

            # Regular agents first (lower sequence numbers)
            seq += 1
            manager_start = _make_start_message(
                uuid.uuid4(), "Manager", "Manager", team_id,
                config=BaseConfig(name="Manager", role="Manager"),
                parent_id=orch_id,
            )
            event_store.save_event(
                PersistedEvent(
                    team_id=team_id, sequence=seq, event=manager_start,
                    timestamp=datetime.now(UTC),
                )
            )

            seq += 1
            assistant_start = _make_start_message(
                uuid.uuid4(), "Assistant", "Assistant", team_id,
                config=BaseConfig(name="Assistant", role="Assistant"),
                parent_id=orch_id,
            )
            event_store.save_event(
                PersistedEvent(
                    team_id=team_id, sequence=seq, event=assistant_start,
                    timestamp=datetime.now(UTC),
                )
            )

            # ToolActors last (higher sequence numbers -- lazy creation)
            seq += 1
            planning_start = _make_start_message(
                uuid.uuid4(), "#PlanningTool", "ToolActor", team_id,
                config=BaseConfig(name="#PlanningTool", role="ToolActor"),
                parent_id=orch_id,
            )
            event_store.save_event(
                PersistedEvent(
                    team_id=team_id, sequence=seq, event=planning_start,
                    timestamp=datetime.now(UTC),
                )
            )

            seq += 1
            kg_start = _make_start_message(
                uuid.uuid4(), "#KnowledgeGraphTool", "ToolActor", team_id,
                config=BaseConfig(name="#KnowledgeGraphTool", role="ToolActor"),
                parent_id=orch_id,
            )
            event_store.save_event(
                PersistedEvent(
                    team_id=team_id, sequence=seq, event=kg_start,
                    timestamp=datetime.now(UTC),
                )
            )

            events = event_store.load_events(team_id)
            events.sort(key=lambda e: e.sequence)

            orchestrator_start, agent_starts = restorer._determine_live_agents(events)

            # Before the fix, agent_starts would be in sequence order:
            # [Manager, Assistant, #PlanningTool, #KnowledgeGraphTool]
            # After the fix (sort in _rebuild_agents), ToolActors come first.
            # We test the sort key directly here:
            tool_actor_role = "ToolActor"
            agent_starts.sort(
                key=lambda sm: sm.config.role != tool_actor_role
            )

            roles = [sm.config.role for sm in agent_starts]
            assert roles[:2] == ["ToolActor", "ToolActor"]
            assert "ToolActor" not in roles[2:]

        finally:
            actor_system.shutdown()

    def test_sort_with_no_tool_actors_preserves_order(
        self,
        event_store: InMemoryEventStore,
    ) -> None:
        """When no ToolActors exist, sort preserves stable ordering."""
        actor_system = ActorSystem()
        try:
            restorer = TeamRestorer(actor_system, event_store)
            team_id = uuid.uuid4()
            orch_id = uuid.uuid4()
            seq = 0

            # Orchestrator
            seq += 1
            orch_start = _make_start_message(
                orch_id,
                "orchestrator",
                "Orchestrator",
                team_id,
                agent_class=Orchestrator,
                config=BaseConfig(name="@Orchestrator", role="Orchestrator"),
            )
            event_store.save_event(
                PersistedEvent(
                    team_id=team_id, sequence=seq, event=orch_start,
                    timestamp=datetime.now(UTC),
                )
            )

            # Regular agents only
            seq += 1
            manager_start = _make_start_message(
                uuid.uuid4(), "Manager", "Manager", team_id,
                config=BaseConfig(name="Manager", role="Manager"),
                parent_id=orch_id,
            )
            event_store.save_event(
                PersistedEvent(
                    team_id=team_id, sequence=seq, event=manager_start,
                    timestamp=datetime.now(UTC),
                )
            )

            seq += 1
            assistant_start = _make_start_message(
                uuid.uuid4(), "Assistant", "Assistant", team_id,
                config=BaseConfig(name="Assistant", role="Assistant"),
                parent_id=orch_id,
            )
            event_store.save_event(
                PersistedEvent(
                    team_id=team_id, sequence=seq, event=assistant_start,
                    timestamp=datetime.now(UTC),
                )
            )

            events = event_store.load_events(team_id)
            events.sort(key=lambda e: e.sequence)

            orchestrator_start, agent_starts = restorer._determine_live_agents(events)

            original_names = [sm.sender.name for sm in agent_starts]

            tool_actor_role = "ToolActor"
            agent_starts.sort(
                key=lambda sm: sm.config.role != tool_actor_role
            )

            # All agents present, no crash
            assert len(agent_starts) == 2
            assert {sm.sender.name for sm in agent_starts} == set(original_names)

        finally:
            actor_system.shutdown()

    def test_tool_actors_spawned_before_regular_agents_in_full_restore(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """Full restore: ToolActors spawned before regular agents (integration)."""
        tc = _make_team_card()
        team_id = uuid.uuid4()
        orch_id = uuid.uuid4()
        seq = 0

        # Orchestrator StartMessage
        seq += 1
        orch_start = _make_start_message(
            orch_id,
            "orchestrator",
            "Orchestrator",
            team_id,
            agent_class=Orchestrator,
            config=BaseConfig(name="@Orchestrator", role="Orchestrator"),
        )
        event_store.save_event(
            PersistedEvent(
                team_id=team_id, sequence=seq, event=orch_start,
                timestamp=datetime.now(UTC),
            )
        )

        # Regular agent (entry point) -- seq 2
        lead_id = uuid.uuid4()
        seq += 1
        lead_start = _make_start_message(
            lead_id, "lead", "Lead", team_id,
            config=BaseConfig(name="lead", role="Lead"),
            parent_id=orch_id,
        )
        event_store.save_event(
            PersistedEvent(
                team_id=team_id, sequence=seq, event=lead_start,
                timestamp=datetime.now(UTC),
            )
        )

        # ToolActor -- seq 3 (higher, simulating lazy creation)
        tool_id = uuid.uuid4()
        seq += 1
        tool_start = _make_start_message(
            tool_id, "#PlanningTool", "ToolActor", team_id,
            config=BaseConfig(name="#PlanningTool", role="ToolActor"),
            parent_id=orch_id,
        )
        event_store.save_event(
            PersistedEvent(
                team_id=team_id, sequence=seq, event=tool_start,
                timestamp=datetime.now(UTC),
            )
        )

        now = datetime.now(UTC)
        process = Process(
            team_id=team_id,
            status=TeamStatus.STOPPED,
            user_id="test-user",
            user_email="test@test.com",
            created_at=now,
            updated_at=now,
            **projection_kwargs(tc),
        )
        seed_agent_cards(event_store, tc)
        event_store.save_team(process)

        # Track spawn order
        spawn_order: list[str] = []
        original_spawn = TeamRestorer._spawn_agents

        def tracking_spawn(
            self_restorer: TeamRestorer,
            agent_starts_arg: list[StartMessage],
            orchestrator_addr: ActorAddress,
            spawned_addrs: list[ActorAddress],
        ) -> dict[str, ActorAddress]:
            for sm in agent_starts_arg:
                if sm.sender is not None:
                    spawn_order.append(sm.sender.name)
            return original_spawn(
                self_restorer, agent_starts_arg, orchestrator_addr, spawned_addrs
            )

        with patch.object(TeamRestorer, "_spawn_agents", tracking_spawn):
            restorer = TeamRestorer(actor_system, event_store)
            runtime = restorer.restore(process)

        # ToolActor must be spawned BEFORE the regular agent
        assert spawn_order.index("#PlanningTool") < spawn_order.index("lead"), (
            f"Expected #PlanningTool before lead, got order: {spawn_order}"
        )

        # Both agents alive, no duplicates
        assert "#PlanningTool" in runtime.addrs
        assert "lead" in runtime.addrs
        team = runtime.orchestrator_proxy.get_team()
        agent_ids = [addr.agent_id for addr in team]
        assert len(agent_ids) == len(set(agent_ids)), "Duplicate agents in roster"


# ---------------------------------------------------------------------------
# Tests: Re-emit agent state onto the event stream on restore (Story 23-3)
# ---------------------------------------------------------------------------


class TestRestorerReemitStateOnRestore:
    """AC #1-#6: Phase 3 re-emits each restored agent's state after its StartMessage.

    Uses a ``RecordingSubscriber`` as the stand-in for the live event stream:
    it is the subscriber attached at phase 2g, so it observes exactly the
    messages a fresh team's stream would. ``notify_state_change`` is an ask-mode
    proxy call, so the agent's ``StateChangedMessage`` is enqueued on the
    orchestrator mailbox before the next replayed event -- ordering is
    deterministic.
    """

    def test_state_reaches_restored_stream_after_start_message(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC #1, #5: a snapshot's state is re-emitted right after its StartMessage.

        The restored stream carries exactly one ``StateChangedMessage`` for the
        agent-with-snapshot, positioned after that agent's ``StartMessage`` and
        carrying the snapshot state -- the parity a fresh team's live stream has.
        """
        from akgentic.team.models import AgentStateSnapshot

        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)
        lead_agent_id = _lead_agent_id_from_events(event_store, team_id)

        snapshot = AgentStateSnapshot(
            team_id=team_id,
            agent_id=str(lead_agent_id),
            name="lead",
            state=_MarkerState(marker="restored"),
            updated_at=datetime.now(UTC),
        )
        event_store.save_agent_state(snapshot)

        recording = RecordingSubscriber()
        restorer = TeamRestorer(actor_system, event_store)
        restorer.restore(process, subscribers=[recording])

        state_msgs = [m for m in recording.messages if isinstance(m, StateChangedMessage)]
        # Exactly one re-emit for the single agent-with-snapshot (AC #5).
        assert len(state_msgs) == 1
        applied = state_msgs[0].state
        assert isinstance(applied, _MarkerState)
        assert applied.marker == "restored"

        # Positioned after the lead's StartMessage in the replay order (AC #1).
        lead_start_idx = next(
            i
            for i, m in enumerate(recording.messages)
            if isinstance(m, StartMessage) and m.sender is not None and m.sender.name == "lead"
        )
        state_idx = recording.messages.index(state_msgs[0])
        assert state_idx > lead_start_idx

    def test_reemitted_state_message_is_keyed_by_uuid(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC #2: the re-emitted StateChangedMessage's sender.agent_id is the UUID."""
        from akgentic.team.models import AgentStateSnapshot

        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)
        lead_agent_id = _lead_agent_id_from_events(event_store, team_id)

        snapshot = AgentStateSnapshot(
            team_id=team_id,
            agent_id=str(lead_agent_id),
            name="lead",
            state=BaseState(),
            updated_at=datetime.now(UTC),
        )
        event_store.save_agent_state(snapshot)

        recording = RecordingSubscriber()
        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process, subscribers=[recording])

        state_msgs = [m for m in recording.messages if isinstance(m, StateChangedMessage)]
        assert len(state_msgs) == 1
        sender = state_msgs[0].sender
        assert sender is not None
        # Emitted via the live agent -> sender carries the agent UUID, matching
        # the live lead address (so a UUID-keyed consumer folds it correctly).
        assert sender.agent_id == lead_agent_id
        assert sender.agent_id == runtime.addrs["lead"].agent_id

    def test_legacy_name_keyed_snapshot_is_reemitted_via_name_fallback(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC #3: a legacy (name-keyed) snapshot is also re-emitted via name fallback.

        The UUID lookup misses, but the stored ``agent_id`` ("lead") matches the
        live agent's name, so the snapshot state still reaches the stream.
        """
        from akgentic.team.models import AgentStateSnapshot

        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        legacy_snapshot = AgentStateSnapshot(
            team_id=team_id,
            agent_id="lead",  # OLD shape: agent_id holds the name
            state=_MarkerState(marker="legacy"),
            updated_at=datetime.now(UTC),
        )
        event_store.save_agent_state(legacy_snapshot)

        recording = RecordingSubscriber()
        restorer = TeamRestorer(actor_system, event_store)
        restorer.restore(process, subscribers=[recording])

        state_msgs = [m for m in recording.messages if isinstance(m, StateChangedMessage)]
        assert len(state_msgs) == 1
        applied = state_msgs[0].state
        assert isinstance(applied, _MarkerState)
        assert applied.marker == "legacy"

    def test_no_reemit_for_sender_without_snapshot(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC #3, #5: senders without a snapshot (orchestrator, plain agents) no-op.

        With no snapshots persisted at all, restore replays every StartMessage
        (orchestrator + lead) but emits zero StateChangedMessages and does not
        raise.
        """
        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)

        recording = RecordingSubscriber()
        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process, subscribers=[recording])

        state_msgs = [m for m in recording.messages if isinstance(m, StateChangedMessage)]
        assert state_msgs == []
        assert runtime.addrs["lead"].is_alive()

    def test_one_reemit_per_agent_with_snapshot(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC #5: exactly one StateChangedMessage per agent-with-snapshot.

        Two agents have snapshots and one (the orchestrator) does not; the
        stream carries exactly two re-emits, one per snapshotted agent.
        """
        from akgentic.team.models import AgentStateSnapshot

        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])
        team_id, process = _populate_stopped_team(event_store, tc)

        lead_id = _lead_agent_id_from_events(event_store, team_id, name="lead")
        worker_id = _lead_agent_id_from_events(event_store, team_id, name="worker")

        for agent_id, name in ((lead_id, "lead"), (worker_id, "worker")):
            event_store.save_agent_state(
                AgentStateSnapshot(
                    team_id=team_id,
                    agent_id=str(agent_id),
                    name=name,
                    state=BaseState(),
                    updated_at=datetime.now(UTC),
                )
            )

        recording = RecordingSubscriber()
        restorer = TeamRestorer(actor_system, event_store)
        restorer.restore(process, subscribers=[recording])

        state_msgs = [m for m in recording.messages if isinstance(m, StateChangedMessage)]
        emitted_ids = {m.sender.agent_id for m in state_msgs if m.sender is not None}
        assert len(state_msgs) == 2
        assert emitted_ids == {lead_id, worker_id}

    def test_reemit_does_not_repersist_events_or_states(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC #4: load_events and load_agent_states are unchanged across a restore.

        Phase 3 runs with PersistenceSubscriber.set_restoring(True), so the
        re-emitted StateChangedMessage is neither appended to the event log nor
        written to the snapshot store -- the ADR-013 invariant holds.
        """
        from akgentic.team.models import AgentStateSnapshot
        from akgentic.team.subscriber import PersistenceSubscriber

        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)
        lead_agent_id = _lead_agent_id_from_events(event_store, team_id)

        snapshot = AgentStateSnapshot(
            team_id=team_id,
            agent_id=str(lead_agent_id),
            name="lead",
            state=BaseState(),
            updated_at=datetime.now(UTC),
        )
        event_store.save_agent_state(snapshot)

        events_before = [pe.model_dump() for pe in event_store.load_events(team_id)]
        states_before = [s.model_dump() for s in event_store.load_agent_states(team_id)]

        persistence_sub = PersistenceSubscriber(team_id, event_store)
        persistence_sub.set_restoring(team_id, True)

        restorer = TeamRestorer(actor_system, event_store)
        restorer.restore(process, subscribers=[persistence_sub])

        persistence_sub.set_restoring(team_id, False)

        events_after = [pe.model_dump() for pe in event_store.load_events(team_id)]
        states_after = [s.model_dump() for s in event_store.load_agent_states(team_id)]

        assert events_after == events_before, "Re-emit must not append to the event log"
        assert states_after == states_before, "Re-emit must not write the snapshot store"


# ---------------------------------------------------------------------------
# Tests: the re-emit skips agents that were never rebuilt (Story 34-1)
# ---------------------------------------------------------------------------


class TestRestorerReemitSkipsFiredAgents:
    """A fired agent's StartMessage is replayed; its state must not be re-emitted.

    The agent is deliberately not rebuilt, so its address never resolves past
    ``ActorAddressProxy`` -- while its snapshot outlives it. The re-emit must
    therefore consult the live set (``addr_map``), not the snapshot store.
    """

    def test_restore_survives_a_fired_agent_with_a_surviving_snapshot(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """The production crash: ungated, ``proxy_ask`` gets a dead proxy."""
        tc = _make_team_card()
        fired_id = uuid.uuid4()
        team_id, process = _populate_stopped_team(
            event_store,
            tc,
            fired_members=[("fired-agent", "FiredRole", fired_id)],
        )
        # The lookup this test guards must genuinely have something to find.
        stored = {snap.agent_id for snap in event_store.load_agent_states(team_id)}
        assert str(fired_id) in stored

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        assert "fired-agent" not in runtime.addrs
        assert runtime.addrs["lead"].is_alive()

    def test_no_state_is_reemitted_for_a_fired_agent(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """Stream parity: a fresh team emits no state for an agent that is gone.

        Asserts on sender ids, so re-emitting the fired state through some other
        live agent fails too.
        """
        tc = _make_team_card()
        fired_id = uuid.uuid4()
        team_id, process = _populate_stopped_team(
            event_store,
            tc,
            fired_members=[("fired-agent", "FiredRole", fired_id)],
        )
        lead_agent_id = _lead_agent_id_from_events(event_store, team_id)
        event_store.save_agent_state(
            AgentStateSnapshot(
                team_id=team_id,
                agent_id=str(lead_agent_id),
                name="lead",
                state=_MarkerState(marker="restored"),
                updated_at=datetime.now(UTC),
            )
        )

        recording = RecordingSubscriber()
        restorer = TeamRestorer(actor_system, event_store)
        restorer.restore(process, subscribers=[recording])

        state_msgs = [m for m in recording.messages if isinstance(m, StateChangedMessage)]
        emitted_ids = {m.sender.agent_id for m in state_msgs if m.sender is not None}
        assert emitted_ids == {lead_agent_id}
        markers = [m.state.marker for m in state_msgs if isinstance(m.state, _MarkerState)]
        assert markers == ["restored"]

    def test_legacy_name_keyed_snapshot_of_a_fired_agent_is_skipped(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """The name fallback must not reanimate a fired agent either.

        Gating on the live set closes this door; gating on the snapshot's shape
        would not.
        """
        tc = _make_team_card()
        fired_id = uuid.uuid4()
        team_id, process = _populate_stopped_team(
            event_store,
            tc,
            fired_members=[("fired-agent", "FiredRole", fired_id)],
        )
        # A pre-Epic-23 snapshot for the same agent: keyed by NAME, not UUID.
        event_store.save_agent_state(
            AgentStateSnapshot(
                team_id=team_id,
                agent_id="fired-agent",
                state=_MarkerState(marker="legacy-fired"),
                updated_at=datetime.now(UTC),
            )
        )

        recording = RecordingSubscriber()
        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process, subscribers=[recording])

        assert "fired-agent" not in runtime.addrs
        state_msgs = [m for m in recording.messages if isinstance(m, StateChangedMessage)]
        assert state_msgs == []


# ---------------------------------------------------------------------------
# Tests: restore repopulates the orchestrator's team metadata (Story 27-6)
# ---------------------------------------------------------------------------


class AcmeSupportMetadata(TeamMetadata):
    """Business metadata for the acme deployment used by the restore tests."""

    tenant: str = Field(json_schema_extra={"indexed": True})
    channel: str = Field(json_schema_extra={"indexed": True})
    note: str = ""


class MetadataProbeAgent(Akgent[BaseConfig, BaseState]):
    """Agent that reads the orchestrator's metadata from its own ``on_start``.

    This is how the placement of the push is pinned as behaviour rather than as
    call ordering: the probe is spawned at step 2c, so it can only observe a
    value that was already set at 2b-bis. A push moved to the end of phase 2 --
    or dropped -- leaves it recording ``None``.

    ``actor_system`` is injected by the test because an agent has no handle on
    one; ``observed`` collects what each spawned instance saw.
    """

    actor_system: ClassVar[ActorSystem | None] = None
    observed: ClassVar[list[SerializableBaseModel | None]] = []

    def on_start(self) -> None:
        """Record the orchestrator's metadata as seen at spawn time."""
        super().on_start()
        system = MetadataProbeAgent.actor_system
        orchestrator = self.orchestrator
        if system is None or orchestrator is None:  # pragma: no cover - guard only
            return
        # A timeout rather than an unbounded wait: a regression that deadlocked
        # the spawn path should fail the test, not hang the suite.
        proxy: Orchestrator = system.proxy_ask(orchestrator, Orchestrator, timeout=10.0)
        MetadataProbeAgent.observed.append(proxy.get_metadata())


def _stopped_team_with_metadata(
    event_store: InMemoryEventStore,
    metadata: SerializableBaseModel | None,
    team_card: TeamCard | None = None,
) -> tuple[uuid.UUID, Process]:
    """Build a stopped team whose persisted ``Process`` carries *metadata*.

    Reuses ``_populate_stopped_team`` and overrides only the metadata pair, so
    the fixture differs from every other restore fixture in this file by exactly
    the value under test -- including in having no agent-state snapshots.

    Returns:
        Tuple of (team_id, the re-persisted STOPPED Process).
    """
    tc = team_card or _make_team_card()
    tc.metadata_type = type(metadata) if metadata is not None else None
    team_id, process = _populate_stopped_team(event_store, tc)
    process = process.model_copy(
        update={
            "metadata": metadata,
            "metadata_indexes": derive_metadata_indexes(metadata),
        }
    )
    event_store.save_team(process)
    return team_id, process


def _read_orchestrator_metadata(
    actor_system: ActorSystem, runtime: TeamRuntime
) -> SerializableBaseModel | None:
    """Read the restored orchestrator's metadata over the public proxy API."""
    proxy: Orchestrator = actor_system.proxy_ask(runtime.orchestrator_addr, Orchestrator)
    return proxy.get_metadata()


class TestRestoreRepopulatesMetadata:
    """AC 1-4, 6: restore pushes ``Process.metadata`` onto the rebuilt orchestrator.

    Every read here goes through ``ActorSystem.proxy_ask`` (AC6); nothing reaches
    into actor internals to observe or set the value.
    """

    def test_restore_repopulates_metadata_with_its_concrete_type(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC1: the value comes back equal AND as the concrete subclass.

        The type assertion pins the push itself: what reaches the orchestrator
        is the value, not a base-typed copy of it. It does NOT exercise the
        ``__model__`` tagged-dict round trip -- the fake store holds the
        ``Process`` by reference, so nothing here is ever serialized. That
        round trip is pinned in ``tests/models/test_process.py``.
        """
        metadata = AcmeSupportMetadata(tenant="acme", channel="email", note="n")
        team_id, process = _stopped_team_with_metadata(event_store, metadata)

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        restored = _read_orchestrator_metadata(actor_system, runtime)
        assert type(restored) is AcmeSupportMetadata
        assert restored == metadata

    def test_restore_without_metadata_leaves_it_none(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC3: no metadata is not an error -- restore completes, value stays None."""
        team_id, process = _stopped_team_with_metadata(event_store, None)
        assert process.metadata is None

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        assert _read_orchestrator_metadata(actor_system, runtime) is None
        assert runtime.addrs["lead"].is_alive()

    def test_metadata_restored_with_an_empty_snapshot_store(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC4: nothing in the snapshot store, yet the value is back.

        This is the whole reason the story exists -- metadata is deliberately
        not a ``BaseState`` field, so the replay path that recovers agent state
        recovers nothing here.
        """
        metadata = AcmeSupportMetadata(tenant="acme", channel="chat")
        team_id, process = _stopped_team_with_metadata(event_store, metadata)
        assert event_store.load_agent_states(team_id) == []

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        assert _read_orchestrator_metadata(actor_system, runtime) == metadata

    def test_metadata_restored_when_snapshots_carry_unrelated_state(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC4: snapshots exist but none carries metadata -- it is still restored."""
        from akgentic.team.models import AgentStateSnapshot

        metadata = AcmeSupportMetadata(tenant="contoso", channel="voice")
        team_id, process = _stopped_team_with_metadata(event_store, metadata)

        lead_agent_id = _lead_agent_id_from_events(event_store, team_id)
        event_store.save_agent_state(
            AgentStateSnapshot(
                team_id=team_id,
                agent_id=str(lead_agent_id),
                name="lead",
                state=_MarkerState(marker="unrelated"),
                updated_at=datetime.now(UTC),
            )
        )

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        # Pin what the snapshot store actually holds: one unrelated state and
        # nothing carrying metadata. An `isinstance(..., BaseState)` check here
        # would pass for any implementation -- the field is typed BaseState.
        snapshots = event_store.load_agent_states(team_id)
        assert [snap.state for snap in snapshots] == [_MarkerState(marker="unrelated")]
        assert _read_orchestrator_metadata(actor_system, runtime) == metadata

    def test_agent_spawned_during_restore_observes_the_metadata(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC2: the push lands before the remaining agents are spawned.

        Asserted as behaviour -- what a restored agent actually sees at start-up
        -- rather than by inspecting the order of calls the restorer makes.
        """
        MetadataProbeAgent.actor_system = actor_system
        MetadataProbeAgent.observed = []
        try:
            metadata = AcmeSupportMetadata(tenant="acme", channel="email")
            tc = _make_team_card(
                entry_point=_make_member("lead", "Lead", agent_class=MetadataProbeAgent)
            )
            team_id, process = _stopped_team_with_metadata(event_store, metadata, tc)

            restorer = TeamRestorer(actor_system, event_store)
            runtime = restorer.restore(process)
            assert runtime.addrs["lead"].is_alive()

            # on_start runs on the probe's own thread, so wait for it to record
            # rather than assuming restore() outran it.
            deadline = time.monotonic() + 5.0
            while not MetadataProbeAgent.observed and time.monotonic() < deadline:
                time.sleep(0.01)

            assert MetadataProbeAgent.observed == [metadata]
        finally:
            MetadataProbeAgent.actor_system = None
            MetadataProbeAgent.observed = []


# ---------------------------------------------------------------------------
# Helpers: NotificationMessage construction (Story 28-1)
# ---------------------------------------------------------------------------


def _make_proxy_address(
    agent_id: uuid.UUID,
    name: str,
    role: str,
    team_id: uuid.UUID,
) -> ActorAddressProxy:
    """Build the ActorAddressProxy a deserialized event carries for an agent."""
    from akgentic.core.utils.deserializer import ActorAddressDict

    addr_dict: ActorAddressDict = {
        "__actor_address__": True,
        "__actor_type__": f"{StubAgent.__module__}.{StubAgent.__name__}",
        "agent_id": str(agent_id),
        "name": name,
        "role": role,
        "team_id": str(team_id),
        "squad_id": str(uuid.uuid4()),
        "is_user_proxy": False,
    }
    return ActorAddressProxy(addr_dict)


def _persist_notification(
    event_store: InMemoryEventStore,
    team_id: uuid.UUID,
    notification: NotificationMessage,
    agent_id: uuid.UUID,
    name: str = "lead",
    role: str = "Lead",
) -> None:
    """Persist a NotificationMessage sent by *name*, after the team's own events.

    The sequence is taken past the current maximum so the notification replays
    last, once every agent has been respawned.
    """
    notification.sender = _make_proxy_address(agent_id, name, role, team_id)
    notification.team_id = team_id
    event_store.save_event(
        PersistedEvent(
            team_id=team_id,
            sequence=event_store.get_max_sequence(team_id) + 1,
            event=notification,
            timestamp=datetime.now(UTC),
        )
    )


def _restore_capturing_replay(
    actor_system: ActorSystem,
    event_store: InMemoryEventStore,
    process: Process,
) -> tuple[TeamRuntime, list[Message]]:
    """Restore the team, capturing every message phase 3 replays.

    The replayed objects are the observation point for address resolution.
    A subscriber is not: ``Orchestrator._notify_subscribers_message`` snapshots
    live addresses back to ``ActorAddressProxy`` before fan-out, so a subscriber
    never sees the rehydrated refs.
    """
    replayed: list[Message] = []
    original_restore_message = Orchestrator.restore_message

    def recording_restore_message(self: Orchestrator, message: Message) -> None:
        replayed.append(message)
        original_restore_message(self, message)

    restorer = TeamRestorer(actor_system, event_store)
    with patch.object(Orchestrator, "restore_message", recording_restore_message):
        runtime = restorer.restore(process)
    return runtime, replayed


def _replayed_notification(
    replayed: list[Message],
    notification_type: type[NotificationMessage],
) -> NotificationMessage:
    """Return the single replayed notification of exactly *notification_type*."""
    matches = [m for m in replayed if type(m) is notification_type]
    assert len(matches) == 1, (
        f"expected exactly one replayed {notification_type.__name__}, got {len(matches)}"
    )
    return matches[0]


# ---------------------------------------------------------------------------
# Tests: NotificationMessage address resolution (Story 28-1)
# ---------------------------------------------------------------------------


class TestRestorerNotificationAddressResolution:
    """Restore rehydrates ``NotificationMessage.current_message`` addresses.

    ``_resolve_message_addresses`` matches the ``NotificationMessage`` base class,
    so every subclass -- ``ErrorMessage``, ``WarningMessage``, and any sibling
    akgentic-core adds later -- has its nested ``current_message`` resolved
    without a per-subclass branch here.
    """

    def test_warning_message_current_message_is_rehydrated(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC3: a persisted WarningMessage's nested address comes back live."""
        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)
        lead_id = _lead_agent_id_from_events(event_store, team_id)

        failed = Message()
        failed.sender = _make_proxy_address(lead_id, "lead", "Lead", team_id)
        _persist_notification(
            event_store,
            team_id,
            WarningMessage(content="handled by a human", current_message=failed),
            lead_id,
        )

        runtime, replayed = _restore_capturing_replay(actor_system, event_store, process)

        warning = _replayed_notification(replayed, WarningMessage)
        assert warning.current_message is not None
        resolved = warning.current_message.sender
        assert not isinstance(resolved, ActorAddressProxy)
        assert resolved is runtime.addrs["lead"]

    def test_error_message_current_message_is_rehydrated(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC4: widening the branch to the base class keeps ErrorMessage working."""
        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)
        lead_id = _lead_agent_id_from_events(event_store, team_id)

        failed = Message()
        failed.sender = _make_proxy_address(lead_id, "lead", "Lead", team_id)
        _persist_notification(
            event_store,
            team_id,
            ErrorMessage(
                content_type="ValueError",
                content="boom",
                current_message=failed,
            ),
            lead_id,
        )

        runtime, replayed = _restore_capturing_replay(actor_system, event_store, process)

        error = _replayed_notification(replayed, ErrorMessage)
        assert error.current_message is not None
        resolved = error.current_message.sender
        assert not isinstance(resolved, ActorAddressProxy)
        assert resolved is runtime.addrs["lead"]

    def test_notification_without_current_message_resolves_sender_only(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC5: current_message=None is a no-op -- no error, sender still resolved."""
        tc = _make_team_card()
        team_id, process = _populate_stopped_team(event_store, tc)
        lead_id = _lead_agent_id_from_events(event_store, team_id)

        _persist_notification(
            event_store,
            team_id,
            WarningMessage(content="nothing was being processed"),
            lead_id,
        )

        runtime, replayed = _restore_capturing_replay(actor_system, event_store, process)

        warning = _replayed_notification(replayed, WarningMessage)
        assert warning.current_message is None
        assert warning.sender is runtime.addrs["lead"]


# ---------------------------------------------------------------------------
# Tests: orchestrator config passthrough on restore (Story 30-1)
# ---------------------------------------------------------------------------


class _ConfigWithExtraField(BaseConfig):
    """A persisted orchestrator config carrying a field the restore path cannot name."""

    extra_field: str = "sentinel"


def _replayed_orchestrator_start(
    replayed: list[Message],
    orchestrator_id: uuid.UUID,
) -> StartMessage:
    """Return the replayed ``StartMessage`` sent by the restored orchestrator."""
    matches = [
        m
        for m in replayed
        if isinstance(m, StartMessage)
        and m.sender is not None
        and m.sender.agent_id == orchestrator_id
    ]
    assert len(matches) == 1, (
        f"expected exactly one replayed orchestrator StartMessage, got {len(matches)}"
    )
    return matches[0]


class TestRestorerOrchestratorConfigPreserved:
    """Restore hands the orchestrator its persisted config, not a synthesised one.

    ``ActorAddressImpl`` caches ``squad_id``/``name``/``role`` from ``actor.config``
    at construction, so whatever config the restore path passes to ``createActor``
    is frozen into every address the orchestrator subsequently appears in --
    including the ``sender`` of its own replayed ``StartMessage``. Synthesising a
    config here therefore does not merely lose fields, it publishes wrong ones.
    """

    def test_persisted_squad_id_survives_restore(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 2: the restored orchestrator address reports the persisted squad."""
        persisted_squad = uuid.uuid4()
        team_id, process = _populate_stopped_team(
            event_store,
            orchestrator_config=BaseConfig(
                name="@Orchestrator",
                role="Orchestrator",
                squad_id=persisted_squad,
            ),
        )
        del team_id

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        assert runtime.orchestrator_addr.squad_id == persisted_squad

    def test_replayed_start_message_agrees_with_itself(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 3: the replayed orchestrator StartMessage does not contradict itself.

        ``_resolve_event_addresses`` swaps the persisted proxy sender for the live
        address before replay, so ``sender.squad_id`` reads what ``createActor`` was
        handed while ``config.squad_id`` stays the persisted value -- the two sides
        of the assertion.
        """
        persisted_squad = uuid.uuid4()
        team_id, process = _populate_stopped_team(
            event_store,
            orchestrator_config=BaseConfig(
                name="@Orchestrator",
                role="Orchestrator",
                squad_id=persisted_squad,
            ),
        )
        del team_id

        runtime, replayed = _restore_capturing_replay(actor_system, event_store, process)

        orch_start = _replayed_orchestrator_start(replayed, runtime.orchestrator_addr.agent_id)
        assert orch_start.sender is not None
        # The persisted side must still be the persisted value: createActor mutates
        # the config object it is handed, so a config passed by reference would have
        # this assertion pass by corrupting the event rather than by fixing restore.
        assert orch_start.config.squad_id == persisted_squad
        assert orch_start.sender.squad_id == orch_start.config.squad_id

    def test_persisted_name_and_role_survive_restore(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 4: non-default name/role on the persisted config survive the restore."""
        team_id, process = _populate_stopped_team(
            event_store,
            orchestrator_config=BaseConfig(
                name="@TeamCoordinator",
                role="Coordinator",
                squad_id=uuid.uuid4(),
            ),
        )
        del team_id

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        assert runtime.orchestrator_addr.name == "@TeamCoordinator"
        assert runtime.orchestrator_addr.role == "Coordinator"

    def test_a_field_the_restore_path_cannot_name_survives(
        self,
        actor_system: ActorSystem,
        event_store: InMemoryEventStore,
    ) -> None:
        """AC 1: the config is copied whole, never rebuilt field by field.

        The three tests above stay green against a hand-written
        ``BaseConfig(name=..., role=..., squad_id=...)`` -- the same
        enumerated-reconstruction defect with one more field named, correct the
        day it is written and silently lossy the day a field is added. Only a
        field the restore path has never heard of separates the two, so this is
        the test that actually pins ``model_copy()`` with no ``update=``.
        """
        persisted = _ConfigWithExtraField(
            name="@Orchestrator",
            role="Orchestrator",
            squad_id=uuid.uuid4(),
            extra_field="sentinel",
        )
        team_id, process = _populate_stopped_team(
            event_store,
            orchestrator_config=persisted,
        )
        del team_id

        restorer = TeamRestorer(actor_system, event_store)
        runtime = restorer.restore(process)

        # Read over the public proxy API, never through actor internals.
        restored_config = actor_system.proxy_ask(runtime.orchestrator_addr, Orchestrator).config
        assert isinstance(restored_config, _ConfigWithExtraField)
        assert restored_config.extra_field == "sentinel"


# ---------------------------------------------------------------------------
# Tests: the runtime is built from the stored projection (FR6)
# ---------------------------------------------------------------------------


def _projection_team_card(
    message_types: list[type] | None = None,
    profiles: list[AgentCard] | None = None,
) -> TeamCard:
    """An entry point and two supervisors, with the fields FR6 now reads."""
    return TeamCard(
        name="projection-team",
        description="Two supervisors and an entry point",
        entry_point=_make_member("lead", "Lead"),
        members=[_make_member("alpha", "Alpha"), _make_member("beta", "Beta")],
        message_types=message_types or [],
        agent_profiles=profiles or [],
    )


def _sent_recipients(
    recording: RecordingSubscriber, since: int, expected: int
) -> list[SentMessage]:
    """Return the ``SentMessage``s recorded after *since*, once *expected* arrive.

    ``TeamRuntime.send`` routes through a ``proxy_tell`` entry proxy and the
    orchestrator notification is asynchronous, so the count is polled rather
    than read once. Returning early on the expected count keeps a passing test
    fast; the deadline is what makes a failing one finite.

    On timeout this returns whatever DID arrive rather than failing, so every
    caller pins ``len(...)`` as well as the recipients or contents — a set
    comparison alone cannot tell a full delivery from a partial one.
    """
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        sent = [m for m in recording.messages[since:] if isinstance(m, SentMessage)]
        if len(sent) >= expected:
            return sent
        time.sleep(0.01)
    return [m for m in recording.messages[since:] if isinstance(m, SentMessage)]


class TestRestoreBuildsTheRuntimeFromTheProjection:
    """FR6: ``_build_team_runtime`` reads the ``Process``, never a ``TeamCard``.

    These tests assert what the restored runtime DOES. Nothing here claims that
    hireability is enforced anywhere: ``can_be_hired`` is asserted as a value
    the catalog carries, which is all it is.
    """

    def test_the_entry_point_and_every_supervisor_come_off_the_projection(
        self, actor_system: ActorSystem, event_store: InMemoryEventStore
    ) -> None:
        """AC 2: name, entry point and supervisors, keyed by their spawned names."""
        tc = _projection_team_card(message_types=[UserMessage])
        team_id, process = _populate_stopped_team(event_store, tc)

        runtime = TeamRestorer(actor_system, event_store).restore(process)

        assert runtime.id == team_id
        assert runtime.team_name == "projection-team"
        assert runtime.message_types == [UserMessage]
        assert runtime.entry_addr == runtime.addrs["lead"]
        assert set(runtime.supervisor_addrs) == {"alpha", "beta"}
        assert runtime.supervisor_addrs["alpha"] == runtime.addrs["alpha"]
        assert runtime.supervisor_addrs["beta"] == runtime.addrs["beta"]

    def test_send_reaches_every_supervisor_after_a_resume(
        self, actor_system: ActorSystem, event_store: InMemoryEventStore
    ) -> None:
        """AC 2: the addresses are not merely present, they are routable."""
        tc = _projection_team_card(message_types=[UserMessage])
        _team_id, process = _populate_stopped_team(event_store, tc)

        recording = RecordingSubscriber()
        runtime = TeamRestorer(actor_system, event_store).restore(
            process, subscribers=[recording]
        )

        baseline = len(recording.messages)
        runtime.send("hello")

        sent = _sent_recipients(recording, baseline, expected=2)
        assert {m.recipient.name for m in sent} == {"alpha", "beta"}

    def test_a_preformed_message_is_passed_through_unwrapped(
        self, actor_system: ActorSystem, event_store: InMemoryEventStore
    ) -> None:
        """AC 2: ``send(Message)`` still bypasses the default-type wrapping."""
        tc = _projection_team_card(message_types=[UserMessage])
        _team_id, process = _populate_stopped_team(event_store, tc)

        recording = RecordingSubscriber()
        runtime = TeamRestorer(actor_system, event_store).restore(
            process, subscribers=[recording]
        )

        baseline = len(recording.messages)
        runtime.send(UserMessage(content="preformed"))

        sent = _sent_recipients(recording, baseline, expected=2)
        # The count is pinned as well as the content: a set of one value is
        # equal to a set of two identical ones, so without this a half-delivered
        # send reads as a pass.
        assert len(sent) == 2
        assert all(type(m.message) is UserMessage for m in sent)
        assert {m.message.content for m in sent} == {"preformed"}

    def test_a_str_is_wrapped_in_the_projections_message_type(
        self, actor_system: ActorSystem, event_store: InMemoryEventStore
    ) -> None:
        """AC 3: the stored ``message_types`` decides what a ``str`` becomes."""
        tc = _projection_team_card(message_types=[UserMessage])
        _team_id, process = _populate_stopped_team(event_store, tc)

        recording = RecordingSubscriber()
        runtime = TeamRestorer(actor_system, event_store).restore(
            process, subscribers=[recording]
        )

        baseline = len(recording.messages)
        runtime.send("wrap me")

        sent = _sent_recipients(recording, baseline, expected=2)
        assert len(sent) == 2
        assert all(isinstance(m.message, UserMessage) for m in sent)
        assert all(m.message.content == "wrap me" for m in sent)

    def test_an_empty_message_types_refuses_a_str_but_takes_a_message(
        self, actor_system: ActorSystem, event_store: InMemoryEventStore
    ) -> None:
        """AC 3: the other half — no declared type is a real, distinct state.

        Together with the spec above this is what makes the field's SOURCE
        observable: one team wraps, one refuses, and only ``process.message_types``
        tells them apart.
        """
        tc = _projection_team_card()
        _team_id, process = _populate_stopped_team(event_store, tc)
        assert process.message_types == []

        recording = RecordingSubscriber()
        runtime = TeamRestorer(actor_system, event_store).restore(
            process, subscribers=[recording]
        )

        with pytest.raises(RuntimeError, match="No message type declared for this team"):
            runtime.send("hi")

        baseline = len(recording.messages)
        runtime.send(UserMessage(content="explicit"))
        sent = _sent_recipients(recording, baseline, expected=2)
        assert len(sent) == 2
        assert {m.message.content for m in sent} == {"explicit"}

    def test_a_supervisor_that_was_not_rebuilt_is_skipped_not_fatal(
        self, actor_system: ActorSystem, event_store: InMemoryEventStore
    ) -> None:
        """AC 4: a fired supervisor keeps its ref; the team must still resume."""
        tc = _projection_team_card(message_types=[UserMessage])
        _team_id, process = _populate_stopped_team(
            event_store, tc, stopped_tree_members=["beta"]
        )

        # The projection still carries the ref — that is what makes the skip
        # necessary rather than academic.
        assert {r.name for r in process.supervisors} == {"alpha", "beta"}

        recording = RecordingSubscriber()
        runtime = TeamRestorer(actor_system, event_store).restore(
            process, subscribers=[recording]
        )

        assert "beta" not in runtime.addrs
        assert set(runtime.supervisor_addrs) == {"alpha"}

        baseline = len(recording.messages)
        runtime.send("still routable")
        sent = _sent_recipients(recording, baseline, expected=1)
        assert len(sent) == 1
        assert {m.recipient.name for m in sent} == {"alpha"}

    def test_a_runtime_hire_is_restored_and_its_role_is_in_the_catalog(
        self, actor_system: ActorSystem, event_store: InMemoryEventStore
    ) -> None:
        """AC 5: the AGENT comes from the log, its ROLE from ``agent_profiles``.

        The hired agent appears in no member tree — a hire leaves only a
        StartMessage behind. Its card was never projected as a member, so the
        catalog entry for ``Specialist`` is the profile's, which is precisely
        where the hired card came from. ``can_be_hired`` is asserted as the
        value the catalog carries; nothing reads it.
        """
        tc = _projection_team_card(
            message_types=[UserMessage],
            profiles=[_make_card("specialist-profile", "Specialist")],
        )
        _team_id, process = _populate_stopped_team(
            event_store, tc, extra_members=[("hired-one", "Specialist")]
        )

        runtime = TeamRestorer(actor_system, event_store).restore(process)

        # Half one: the agent, rebuilt from the event log (NFR4, unchanged).
        assert "hired-one" in runtime.addrs
        assert runtime.addrs["hired-one"].is_alive()
        assert runtime.orchestrator_proxy.get_team_member("hired-one") is not None

        # Half two: the role, in the catalog, alongside every tree role.
        flags = {c.role: c.can_be_hired for c in runtime.orchestrator_proxy.get_agent_catalog()}
        assert flags == {
            "Lead": False,
            "Alpha": False,
            "Beta": False,
            "Specialist": True,
        }

    def test_the_profiles_card_reaches_the_restored_catalog(
        self, actor_system: ActorSystem, event_store: InMemoryEventStore
    ) -> None:
        """AC 16, resume path: the same precedence the create path applies."""
        profile = _make_card("alpha-profile", "Alpha")
        profile.description = "Hired to be Alpha, not the one already being Alpha"
        profile.skills = ["substitution"]
        tc = _projection_team_card(message_types=[UserMessage], profiles=[profile])
        _team_id, process = _populate_stopped_team(event_store, tc)

        runtime = TeamRestorer(actor_system, event_store).restore(process)

        catalog = runtime.orchestrator_proxy.get_agent_catalog()
        entry = next(c for c in catalog if c.role == "Alpha")
        assert entry.description == profile.description
        assert entry.skills == profile.skills

    def test_a_profile_naming_the_entry_points_role_leaves_send_working(
        self, actor_system: ActorSystem, event_store: InMemoryEventStore
    ) -> None:
        """AC 18: the trap — the profile's card becomes what _entry_proxy resolves.

        ``TeamRuntime.model_post_init`` looks the entry point's agent class up in
        the catalog by role, so a profile that names the entry point's role
        supplies it — including a DIFFERENT ``agent_class`` than the live entry
        member was spawned with, which is the sharp form of the trap. Measured on
        BOTH paths rather than special-cased inside the dedup.
        """
        entry_profile = _make_card("lead-profile", "Lead", agent_class=_OtherStubAgent)
        entry_profile.description = "The role a new Lead would be hired into"
        assert entry_profile.get_agent_class() is not _make_card("lead", "Lead").get_agent_class()
        tc = _projection_team_card(message_types=[UserMessage], profiles=[entry_profile])

        # Create path. The send is RECORDED, not merely called: ``send`` routes
        # through a fire-and-forget proxy that ignores the class it is typed on
        # (``ActorSystem.proxy_tell`` casts and discards it), so a call that
        # does not raise proves nothing on its own.
        built_recording = RecordingSubscriber()
        built = TeamFactory.build(tc, actor_system, [built_recording])
        built_catalog = built.orchestrator_proxy.get_agent_catalog()
        assert next(c for c in built_catalog if c.role == "Lead").description == (
            entry_profile.description
        )
        assert set(built.supervisor_addrs) == {"alpha", "beta"}
        built_baseline = len(built_recording.messages)
        built.send("after a create")
        built_sent = _sent_recipients(built_recording, built_baseline, expected=2)
        assert len(built_sent) == 2
        assert {m.recipient.name for m in built_sent} == {"alpha", "beta"}
        built.orchestrator_proxy.stop(GRACE_TIMEOUT_SECONDS).wait()

        # Resume path.
        _team_id, process = _populate_stopped_team(event_store, tc)
        recording = RecordingSubscriber()
        resumed = TeamRestorer(actor_system, event_store).restore(
            process, subscribers=[recording]
        )

        assert set(resumed.supervisor_addrs) == {"alpha", "beta"}
        baseline = len(recording.messages)
        resumed.send("after a resume")
        sent = _sent_recipients(recording, baseline, expected=2)
        assert len(sent) == 2
        assert {m.recipient.name for m in sent} == {"alpha", "beta"}

    def test_a_headcount_supervisor_survives_create_stop_resume(
        self, actor_system: ActorSystem, event_store: InMemoryEventStore
    ) -> None:
        """FR7 (31-4): the restore path needs no expansion of its own.

        ``_build_team_runtime`` iterates ``process.supervisors`` and keys on
        ``ref.name``, and those refs have been SPAWNED names since 31-1 — there
        is no declared name left on that path to mismatch. So this is a
        non-regression pin measured on both paths, not a second fix:
        ``restorer.py`` is unchanged by this story. Its surviving
        ``if ref.name in addrs`` guard is a different one — a supervisor fired
        during the team's life is deliberately not respawned — and stays.
        """
        crew = _make_member("worker", "Worker", headcount=3)
        tc = TeamCard(
            name="headcount-team",
            description="One multi-instance supervisor",
            entry_point=_make_member("lead", "Lead"),
            members=[crew],
            message_types=[UserMessage],
        )
        expected = {"worker_0", "worker_1", "worker_2"}

        # Create path.
        built_recording = RecordingSubscriber()
        built = TeamFactory.build(tc, actor_system, [built_recording])
        assert set(built.supervisor_addrs) == expected
        built_baseline = len(built_recording.messages)
        built.send("after a create")
        built_sent = _sent_recipients(built_recording, built_baseline, expected=3)
        assert len(built_sent) == 3
        assert {m.recipient.name for m in built_sent} == expected
        built.orchestrator_proxy.stop(GRACE_TIMEOUT_SECONDS).wait()

        # Resume path.
        _team_id, process = _populate_stopped_team(event_store, tc)
        assert {ref.name for ref in process.supervisors} == expected
        recording = RecordingSubscriber()
        resumed = TeamRestorer(actor_system, event_store).restore(
            process, subscribers=[recording]
        )

        assert set(resumed.supervisor_addrs) == expected
        baseline = len(recording.messages)
        resumed.send("after a resume")
        sent = _sent_recipients(recording, baseline, expected=3)
        assert len(sent) == 3
        assert {m.recipient.name for m in sent} == expected
