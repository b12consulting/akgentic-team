"""Tests for the identity and the metadata an agent can see AT ITS SPAWN.

Two invariants, both of which have to hold on the create path and the resume
path alike, because an agent resolves its tools while it is being spawned:

* the owning user's identity reaches every agent in the tree, including one
  hired later through a live member;
* the team's business metadata is already readable on the orchestrator by the
  time the first member spawns.

Every assertion here reads what the agent recorded **inside its own**
``on_start``. Asserting on ``build``'s arguments, or on the orchestrator after
``build`` returns, passes on the pre-story code and proves nothing: the old
create path did set the metadata, just too late for anyone to use it.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import pytest
from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.agent_card import AgentCard
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.utils.serializer import SerializableBaseModel
from pydantic import Field

from akgentic.team.factory import TeamFactory
from akgentic.team.manager import TeamManager
from akgentic.team.metadata import TeamMetadata
from akgentic.team.models import TeamCard, TeamCardMember, TeamRuntime
from tests.services.conftest import InMemoryEventStore

USER_ID = "alice"
USER_EMAIL = "alice@example.com"
SPAWN_RECORD_TIMEOUT = 5.0


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class AcmeMetadata(TeamMetadata):
    """Business metadata for the acme deployment — one indexed field."""

    tenant: str = Field(json_schema_extra={"indexed": True})


class IdentityAgent(Akgent[BaseConfig, BaseState]):
    """Records what it can see of its identity and of the team's metadata,
    at the instant it is spawned.

    The class-level defaults are what make the recording observable: pykka
    introspects the actor's attributes when a proxy is built, so an attribute
    that exists only after a successful ``on_start`` cannot be read back to
    show that ``on_start`` never ran. ``spawn_recorded`` separates "not yet
    recorded" from "recorded as None", which is exactly the distinction the
    metadata assertion turns on.
    """

    spawn_recorded: bool = False
    # ``Any`` with a reason: ``Akgent.__init__`` still declares
    # ``user_id: uuid.UUID | None`` while a ``str`` is what actually flows
    # through ``createActor``. Either concrete annotation would be a lie about
    # one end of that disagreement; ``str | None`` becomes correct once
    # akgentic-core's widening lands, and this line is where to change it.
    seen_user_id: Any = None
    seen_user_email: str | None = None
    seen_metadata: SerializableBaseModel | None = None

    def on_start(self) -> None:
        """Capture identity and metadata from inside the spawn."""
        self.seen_user_id = self._user_id
        self.seen_user_email = self._user_email
        # ``on_start`` runs on THIS actor's thread, so a blocking ask on the
        # orchestrator is safe here. It would NOT be safe in ``__init__``,
        # which pykka runs on the *parent's* thread: asking the orchestrator
        # from there, while the orchestrator is itself inside ``createActor``,
        # deadlocks.
        self.seen_metadata = self.orchestrator_proxy_ask.get_metadata()
        self.spawn_recorded = True


def _make_member(
    name: str,
    role: str = "TestRole",
    members: list[TeamCardMember] | None = None,
) -> TeamCardMember:
    return TeamCardMember(
        card=AgentCard(
            role=role,
            description=f"Test: {role}",
            skills=["testing"],
            agent_class=IdentityAgent,
            config=BaseConfig(name=name, role=role),
            routes_to=[],
        ),
        members=members or [],
    )


def _make_team_card() -> TeamCard:
    """An entry point with a subordinate — two spawn layers, one card."""
    return TeamCard(
        name="acme-support",
        description="Test team",
        entry_point=_make_member("lead", "Lead", members=[_make_member("helper", "Helper")]),
        metadata_type=AcmeMetadata,
    )


def _spawn_record(actor_system: ActorSystem, addr: ActorAddress) -> IdentityAgent:
    """Return a proxy to *addr* once its ``on_start`` has finished recording.

    ``on_start`` runs asynchronously on the actor's own thread, so poll with a
    deadline rather than sleeping a guessed interval. Written locally on
    purpose: the equivalent helper in ``tests/integration/conftest.py`` lives in
    a suite this package never collects.
    """
    deadline = time.monotonic() + SPAWN_RECORD_TIMEOUT
    while time.monotonic() < deadline:
        proxy: IdentityAgent = actor_system.proxy_ask(addr, IdentityAgent)
        if proxy.spawn_recorded:
            return proxy
        time.sleep(0.05)
    msg = f"Agent at {addr} never recorded its spawn within {SPAWN_RECORD_TIMEOUT}s"
    raise AssertionError(msg)


def _hire_through(actor_system: ActorSystem, member_addr: ActorAddress) -> ActorAddress:
    """Spawn a new agent through a LIVE member, as the hire mechanism does.

    Hiring itself is ``akgentic-tool``'s surface, but the propagation path it
    takes is exactly this: ``Akgent.createActor`` on an already-running member.
    """
    member_proxy: Akgent[Any, Any] = actor_system.proxy_ask(member_addr, Akgent)
    addr = member_proxy.createActor(
        IdentityAgent,
        config=BaseConfig(name="hired", role="Hired"),
    )
    assert addr is not None
    return addr


def _assert_identity(actor_system: ActorSystem, addr: ActorAddress) -> None:
    """Assert the agent at *addr* saw the owning user's identity at spawn."""
    record = _spawn_record(actor_system, addr)
    assert record.seen_user_id == USER_ID
    assert record.seen_user_email == USER_EMAIL


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
    """Provide a fresh event store per test."""
    return InMemoryEventStore()


@pytest.fixture()
def manager(actor_system: ActorSystem, event_store: InMemoryEventStore) -> TeamManager:
    """Provide a TeamManager with the default NullServiceRegistry."""
    return TeamManager(actor_system=actor_system, event_store=event_store)


@pytest.fixture()
def running_team(manager: TeamManager) -> TeamRuntime:
    """Create a team owned by ``alice``, carrying metadata."""
    return manager.create_team(
        _make_team_card(),
        user_id=USER_ID,
        user_email=USER_EMAIL,
        metadata=AcmeMetadata(tenant="acme"),
    )


# ---------------------------------------------------------------------------
# AC1 / AC2 — the identity reaches every agent, on create and on resume
# ---------------------------------------------------------------------------


class TestIdentityReachesEveryAgent:
    """AC1, AC2: create and resume must agree on who the team belongs to."""

    def test_entry_point_and_subordinate_carry_the_identity(
        self, actor_system: ActorSystem, running_team: TeamRuntime
    ) -> None:
        """AC1: propagation through the orchestrator and down one more layer.

        Read off the real spawned actors, not off ``build``'s arguments: the
        defect this closes was an argument that was never forwarded, which an
        argument-level assertion would have reproduced rather than caught.
        """
        _assert_identity(actor_system, running_team.entry_addr)
        _assert_identity(actor_system, running_team.addrs["helper"])

    def test_an_agent_hired_later_carries_the_identity(
        self, actor_system: ActorSystem, running_team: TeamRuntime
    ) -> None:
        """AC1: the identity is on the tree, not on the build call."""
        hired_addr = _hire_through(actor_system, running_team.entry_addr)

        _assert_identity(actor_system, hired_addr)

    def test_identity_survives_a_restart(
        self, actor_system: ActorSystem, manager: TeamManager, running_team: TeamRuntime
    ) -> None:
        """AC2: every restored agent reports the identity from the Process.

        Asserted over the whole restored roster rather than over named agents:
        resume rebuilds whatever the event log holds — the hired agent
        included — and an agent that came back without an identity must fail
        this whichever branch of the tree it sits on.
        """
        _hire_through(actor_system, running_team.entry_addr)
        manager.stop_team(running_team.id)

        resumed = manager.resume_team(running_team.id)

        assert set(resumed.addrs) == {"lead", "helper", "hired"}
        for addr in resumed.addrs.values():
            _assert_identity(actor_system, addr)


# ---------------------------------------------------------------------------
# AC3 / AC4 — the metadata is readable AT spawn time, on both paths
# ---------------------------------------------------------------------------


class TestMetadataIsVisibleAtSpawnTime:
    """AC3, AC4: a member consults the metadata while it is being spawned."""

    def test_metadata_is_set_before_the_first_member_spawns(
        self, actor_system: ActorSystem, running_team: TeamRuntime
    ) -> None:
        """AC3: the create path's ordering, observed from inside the spawn.

        ``seen_metadata`` was read by the member itself, during ``on_start``,
        while ``TeamFactory.build`` was still walking the tree. A create path
        that pushes after ``build`` returns leaves this ``None`` while every
        after-the-fact assertion still passes.
        """
        for addr in (running_team.entry_addr, running_team.addrs["helper"]):
            record = _spawn_record(actor_system, addr)
            assert record.seen_metadata == AcmeMetadata(tenant="acme")

    def test_metadata_is_set_before_agents_respawn_on_resume(
        self, actor_system: ActorSystem, manager: TeamManager, running_team: TeamRuntime
    ) -> None:
        """AC4: resume was already correct (step 2b-bis) and stays correct."""
        manager.stop_team(running_team.id)

        resumed = manager.resume_team(running_team.id)

        for addr in resumed.addrs.values():
            record = _spawn_record(actor_system, addr)
            assert record.seen_metadata == AcmeMetadata(tenant="acme")

    def test_a_team_without_metadata_spawns_with_none(
        self, actor_system: ActorSystem, manager: TeamManager
    ) -> None:
        """AC3: no metadata is not an error — the member simply sees None."""
        runtime = manager.create_team(_make_team_card(), user_id=USER_ID, user_email=USER_EMAIL)

        record = _spawn_record(actor_system, runtime.entry_addr)
        assert record.seen_metadata is None


# ---------------------------------------------------------------------------
# AC7 — the new parameters are optional
# ---------------------------------------------------------------------------


class TestBuildWithoutTheNewArguments:
    """AC7: existing direct callers of TeamFactory.build keep working."""

    def test_positional_call_still_builds_a_team(self, actor_system: ActorSystem) -> None:
        """A caller that knows nothing of identity or metadata is unaffected.

        The agents come up with no identity and no metadata — the pre-story
        behaviour, deliberately preserved for direct ``build`` callers that have
        neither value to give.
        """
        runtime = TeamFactory.build(_make_team_card(), actor_system)

        assert isinstance(runtime, TeamRuntime)
        record = _spawn_record(actor_system, runtime.entry_addr)
        assert record.seen_user_id is None
        assert record.seen_user_email is None
        assert record.seen_metadata is None

    def test_team_id_stays_positional(self, actor_system: ActorSystem) -> None:
        """The three new parameters are keyword-only, so nothing shifted."""
        team_id = uuid.uuid4()

        runtime = TeamFactory.build(_make_team_card(), actor_system, None, team_id)

        assert runtime.id == team_id
