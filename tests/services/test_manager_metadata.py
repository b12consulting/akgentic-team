"""Tests for the TeamManager metadata write paths — create and update.

The ordering under test is validate -> single database write of the value and
its re-derived index -> best-effort orchestrator push while RUNNING (ADR-24
§D7). The failed-push tests are the ones that pin that ordering: an
implementation that pushed before writing could not pass them.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.agent_card import AgentCard
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.orchestrator import Orchestrator
from akgentic.core.utils.serializer import SerializableBaseModel
from pydantic import Field, ValidationError

from akgentic.team.manager import TeamManager
from akgentic.team.metadata import TeamMetadata
from akgentic.team.models import Process, TeamCard, TeamCardMember, TeamRuntime, TeamStatus
from akgentic.team.ports import NullServiceRegistry
from tests.services.conftest import InMemoryEventStore

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class StubAgent(Akgent[BaseConfig, BaseState]):
    """Minimal agent so a real team can be built and torn down."""


class AcmeCaseMetadata(TeamMetadata):
    """Business metadata for the acme deployment: two required indexed fields,
    one optional indexed field, one unindexed field."""

    tenant: str = Field(json_schema_extra={"indexed": True})
    channel: str = Field(json_schema_extra={"indexed": True})
    case_ref: str | None = Field(default=None, json_schema_extra={"indexed": True})
    note: str = ""


class ContosoMetadata(TeamMetadata):
    """A different concrete subclass — a value of this type must be rejected by
    a card that declares ``AcmeCaseMetadata``."""

    region: str = Field(json_schema_extra={"indexed": True})


class CountingEventStore(InMemoryEventStore):
    """InMemoryEventStore that counts ``save_team`` calls.

    The single-write invariant ("value and index are never written apart") is
    only observable by counting writes — the final state alone looks identical
    whether one write or two produced it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.save_team_calls = 0

    def save_team(self, process: Process) -> None:
        """Record the write, then delegate."""
        self.save_team_calls += 1
        super().save_team(process)


class FailingOrchestratorTell:
    """Orchestrator tell-proxy whose ``set_metadata`` raises.

    Every other attribute delegates to the real proxy, so a team built while
    this wrapper is installed still behaves normally — only the metadata push
    fails, which is exactly the fault the ordering is designed to survive.

    ``on_push`` fires immediately before the failure, giving a test a window to
    observe the database *at the moment the push is attempted*. That window is
    what separates database-first from actor-first: a swallowed push failure
    leaves the same final state either way, so the final state alone cannot
    tell the two orderings apart.
    """

    def __init__(self, inner: Any, on_push: Callable[[], None] | None = None) -> None:
        self._inner = inner
        self._on_push = on_push

    def set_metadata(self, metadata: Any) -> None:
        """Observe the database, then simulate an unreachable orchestrator."""
        del metadata
        if self._on_push is not None:
            self._on_push()
        msg = "orchestrator unreachable"
        raise RuntimeError(msg)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _break_metadata_push(
    actor_system: ActorSystem,
    monkeypatch: pytest.MonkeyPatch,
    on_push: Callable[[], None] | None = None,
) -> None:
    """Make every orchestrator ``set_metadata`` tell raise, leaving all else live.

    Args:
        actor_system: The system whose ``proxy_tell`` is wrapped.
        monkeypatch: Fixture used to restore ``proxy_tell`` after the test.
        on_push: Optional callback fired at push time, before the failure.
    """
    real_proxy_tell = actor_system.proxy_tell

    def failing_proxy_tell(addr: Any, agent_class: Any) -> Any:
        proxy = real_proxy_tell(addr, agent_class)
        if agent_class is Orchestrator:
            return FailingOrchestratorTell(proxy, on_push)
        return proxy

    monkeypatch.setattr(actor_system, "proxy_tell", failing_proxy_tell)


def _index_recorder(
    event_store: InMemoryEventStore, team_id: uuid.UUID, sink: list[list[str] | None]
) -> Callable[[], None]:
    """Build a push-time callback that snapshots the persisted index.

    ``None`` is recorded when no ``Process`` is stored yet — which is exactly
    what an actor-first create would produce.
    """

    def record() -> None:
        stored = event_store.load_team(team_id)
        sink.append(None if stored is None else list(stored.metadata_indexes))

    return record


def _make_member(name: str, role: str = "TestRole") -> TeamCardMember:
    return TeamCardMember(
        card=AgentCard(
            role=role,
            description=f"Test: {role}",
            skills=["testing"],
            agent_class=StubAgent,
            config=BaseConfig(name=name, role=role),
            routes_to=[],
        ),
    )


def _make_team_card(
    metadata_type: type[SerializableBaseModel] | None = AcmeCaseMetadata,
    name: str = "acme-support",
) -> TeamCard:
    return TeamCard(
        name=name,
        description="Test team",
        entry_point=_make_member("lead", "Lead"),
        metadata_type=metadata_type,
    )


def _read_orchestrator_metadata(
    actor_system: ActorSystem, runtime: TeamRuntime
) -> SerializableBaseModel | None:
    """Read the live orchestrator's metadata through the public proxy API."""
    proxy: Orchestrator = actor_system.proxy_ask(runtime.orchestrator_addr, Orchestrator)
    return proxy.get_metadata()


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
def event_store() -> CountingEventStore:
    """Provide a fresh write-counting event store per test."""
    return CountingEventStore()


@pytest.fixture()
def manager(actor_system: ActorSystem, event_store: CountingEventStore) -> TeamManager:
    """Provide a TeamManager with default NullServiceRegistry."""
    return TeamManager(actor_system=actor_system, event_store=event_store)


# ---------------------------------------------------------------------------
# Create path
# ---------------------------------------------------------------------------


class TestCreateTeamMetadata:
    """AC 1-7: create_team validates, persists value + index, then pushes."""

    def test_metadata_persisted_with_derived_indexes(
        self, manager: TeamManager, event_store: CountingEventStore
    ) -> None:
        """AC 4: the persisted Process carries the value and its derived index."""
        metadata = AcmeCaseMetadata(tenant="acme", channel="email", case_ref="c-1")

        runtime = manager.create_team(_make_team_card(), metadata=metadata)

        process = event_store.load_team(runtime.id)
        assert process is not None
        assert type(process.metadata) is AcmeCaseMetadata
        assert process.metadata.tenant == "acme"
        assert process.metadata_indexes == ["tenant|acme", "channel|email", "case_ref|c-1"]

    def test_metadata_and_index_land_in_a_single_write(
        self, manager: TeamManager, event_store: CountingEventStore
    ) -> None:
        """AC 4: one save_team — never a second write that adds the other field."""
        metadata = AcmeCaseMetadata(tenant="acme", channel="email")

        manager.create_team(_make_team_card(), metadata=metadata)

        assert event_store.save_team_calls == 1

    def test_metadata_pushed_to_live_orchestrator(
        self, manager: TeamManager, actor_system: ActorSystem
    ) -> None:
        """AC 5: the orchestrator hands the concrete subclass back, not a base copy."""
        metadata = AcmeCaseMetadata(tenant="acme", channel="chat")

        runtime = manager.create_team(_make_team_card(), metadata=metadata)

        live = _read_orchestrator_metadata(actor_system, runtime)
        assert type(live) is AcmeCaseMetadata
        assert live.channel == "chat"

    def test_metadata_rejected_when_card_declares_no_type(
        self, actor_system: ActorSystem, event_store: CountingEventStore
    ) -> None:
        """AC 2: rejected before build and before any write — no half-created team."""
        registry = MagicMock(spec=NullServiceRegistry)
        mgr = TeamManager(
            actor_system=actor_system, event_store=event_store, service_registry=registry
        )

        with pytest.raises(ValueError, match="declares no metadata_type"):
            mgr.create_team(
                _make_team_card(metadata_type=None),
                metadata=AcmeCaseMetadata(tenant="acme", channel="email"),
            )

        assert event_store.teams == {}
        assert event_store.save_team_calls == 0
        registry.register_team.assert_not_called()

    def test_invalid_metadata_rejected_with_nothing_persisted(
        self, actor_system: ActorSystem, event_store: CountingEventStore
    ) -> None:
        """AC 3: a value of the wrong concrete type creates nothing at all."""
        registry = MagicMock(spec=NullServiceRegistry)
        mgr = TeamManager(
            actor_system=actor_system, event_store=event_store, service_registry=registry
        )

        with pytest.raises(ValidationError):
            mgr.create_team(_make_team_card(), metadata=ContosoMetadata(region="eu"))

        assert event_store.teams == {}
        assert event_store.save_team_calls == 0
        registry.register_team.assert_not_called()

    def test_without_metadata_persists_none_and_empty_index(
        self, manager: TeamManager, event_store: CountingEventStore, actor_system: ActorSystem
    ) -> None:
        """AC 7: the default keeps every existing call site working unchanged."""
        runtime = manager.create_team(_make_team_card())

        process = event_store.load_team(runtime.id)
        assert process is not None
        assert process.metadata is None
        assert process.metadata_indexes == []
        assert _read_orchestrator_metadata(actor_system, runtime) is None

    def test_without_metadata_performs_no_push(self, manager: TeamManager) -> None:
        """AC 7: no metadata means no orchestrator round-trip at all."""
        with patch.object(TeamManager, "_push_metadata", autospec=True) as push:
            manager.create_team(_make_team_card())

        push.assert_not_called()

    def test_failed_push_still_returns_a_runtime_with_the_database_correct(
        self,
        manager: TeamManager,
        actor_system: ActorSystem,
        event_store: CountingEventStore,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """AC 6: the push is best-effort — the persisted Process is unaffected.

        ``pushed_indexes`` is the ordering assertion: the Process must already
        be on disk, index and all, when the push is attempted. Asserting only on
        the final state would pass for an actor-first create too, since the
        swallowed failure leaves the same end state either way.
        """
        team_id = uuid.uuid4()
        pushed_indexes: list[list[str] | None] = []
        _break_metadata_push(
            actor_system, monkeypatch, _index_recorder(event_store, team_id, pushed_indexes)
        )
        metadata = AcmeCaseMetadata(tenant="acme", channel="email")

        with caplog.at_level(logging.WARNING, logger="akgentic.team.manager"):
            runtime = manager.create_team(_make_team_card(), team_id=team_id, metadata=metadata)

        assert pushed_indexes == [["tenant|acme", "channel|email"]]
        assert "Failed to push metadata" in caplog.text
        assert isinstance(runtime, TeamRuntime)
        process = event_store.load_team(runtime.id)
        assert process is not None
        assert process.metadata == metadata
        assert process.metadata_indexes == ["tenant|acme", "channel|email"]


# ---------------------------------------------------------------------------
# Update path
# ---------------------------------------------------------------------------


class TestUpdateTeamMetadata:
    """AC 8-16: update_team_metadata validates, writes once, then pushes."""

    def test_returns_persisted_process_with_new_value_and_index(
        self, manager: TeamManager, event_store: CountingEventStore
    ) -> None:
        """AC 8, 10: the returned Process is the one that was written."""
        runtime = manager.create_team(
            _make_team_card(), metadata=AcmeCaseMetadata(tenant="acme", channel="email")
        )

        result = manager.update_team_metadata(
            runtime.id, AcmeCaseMetadata(tenant="acme", channel="chat")
        )

        stored = event_store.load_team(runtime.id)
        assert stored is not None
        assert result.metadata == stored.metadata
        assert result.metadata_indexes == stored.metadata_indexes
        assert stored.metadata_indexes == ["tenant|acme", "channel|chat"]

    def test_index_reflects_the_newest_value_only(
        self, manager: TeamManager, event_store: CountingEventStore
    ) -> None:
        """AC 10: two updates leave no trace of the intermediate value."""
        runtime = manager.create_team(
            _make_team_card(), metadata=AcmeCaseMetadata(tenant="acme", channel="email")
        )

        manager.update_team_metadata(runtime.id, AcmeCaseMetadata(tenant="acme", channel="chat"))
        manager.update_team_metadata(runtime.id, AcmeCaseMetadata(tenant="acme", channel="voice"))

        stored = event_store.load_team(runtime.id)
        assert stored is not None
        assert stored.metadata_indexes == ["tenant|acme", "channel|voice"]

    def test_replace_semantics_drop_an_unset_optional_field(
        self, manager: TeamManager, event_store: CountingEventStore
    ) -> None:
        """AC 11: replace, not merge — an omitted field is gone from value and index."""
        runtime = manager.create_team(
            _make_team_card(),
            metadata=AcmeCaseMetadata(tenant="acme", channel="email", case_ref="c-1"),
        )

        manager.update_team_metadata(runtime.id, AcmeCaseMetadata(tenant="acme", channel="email"))

        stored = event_store.load_team(runtime.id)
        assert stored is not None
        assert isinstance(stored.metadata, AcmeCaseMetadata)
        assert stored.metadata.case_ref is None
        assert stored.metadata_indexes == ["tenant|acme", "channel|email"]

    def test_none_clears_the_metadata_and_its_index(
        self, manager: TeamManager, event_store: CountingEventStore
    ) -> None:
        """AC 11: clearing is the degenerate case of replace."""
        runtime = manager.create_team(
            _make_team_card(), metadata=AcmeCaseMetadata(tenant="acme", channel="email")
        )

        manager.update_team_metadata(runtime.id, None)

        stored = event_store.load_team(runtime.id)
        assert stored is not None
        assert stored.metadata is None
        assert stored.metadata_indexes == []

    def test_writes_exactly_once(
        self, manager: TeamManager, event_store: CountingEventStore
    ) -> None:
        """AC 10: value and index are written together or not at all."""
        runtime = manager.create_team(
            _make_team_card(), metadata=AcmeCaseMetadata(tenant="acme", channel="email")
        )
        writes_after_create = event_store.save_team_calls

        manager.update_team_metadata(runtime.id, AcmeCaseMetadata(tenant="acme", channel="chat"))

        assert event_store.save_team_calls == writes_after_create + 1

    def test_carries_every_other_process_field_forward(
        self, manager: TeamManager, event_store: CountingEventStore
    ) -> None:
        """AC 12: an update touches metadata and updated_at, nothing else."""
        runtime = manager.create_team(
            _make_team_card(),
            user_id="u-1",
            user_email="u@acme.test",
            catalog_namespace="ns-1",
            metadata=AcmeCaseMetadata(tenant="acme", channel="email"),
        )
        before = event_store.load_team(runtime.id)
        assert before is not None

        after = manager.update_team_metadata(
            runtime.id, AcmeCaseMetadata(tenant="acme", channel="chat")
        )

        assert after.team_id == before.team_id
        assert after.team_card.name == before.team_card.name
        assert after.status == before.status
        assert after.user_id == "u-1"
        assert after.user_email == "u@acme.test"
        assert after.created_at == before.created_at
        assert after.catalog_namespace == "ns-1"
        assert after.updated_at >= before.updated_at

    def test_pushes_to_the_live_orchestrator_when_running(
        self, manager: TeamManager, actor_system: ActorSystem
    ) -> None:
        """AC 13: a running team's orchestrator sees the new value."""
        runtime = manager.create_team(
            _make_team_card(), metadata=AcmeCaseMetadata(tenant="acme", channel="email")
        )

        manager.update_team_metadata(runtime.id, AcmeCaseMetadata(tenant="acme", channel="chat"))

        live = _read_orchestrator_metadata(actor_system, runtime)
        assert type(live) is AcmeCaseMetadata
        assert live.channel == "chat"

    def test_failed_push_leaves_the_database_correct_and_the_call_successful(
        self,
        manager: TeamManager,
        actor_system: ActorSystem,
        event_store: CountingEventStore,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """AC 14: the database-first ordering test.

        Two things are asserted, and both are needed. ``pushed_indexes`` records
        the persisted index *at the instant the push is attempted*: it must
        already be the new one, which is what makes this a test of ordering
        rather than of error handling — an actor-first update would record the
        stale index here while still reaching the same final state, because the
        push failure is swallowed either way. The rest asserts the failure is
        non-fatal: it must not roll back, mask or fail the write that already
        succeeded, so team listing stays truthful and the orchestrator
        repopulates on the next resume.
        """
        runtime = manager.create_team(
            _make_team_card(), metadata=AcmeCaseMetadata(tenant="acme", channel="email")
        )
        pushed_indexes: list[list[str] | None] = []
        _break_metadata_push(
            actor_system, monkeypatch, _index_recorder(event_store, runtime.id, pushed_indexes)
        )
        new_metadata = AcmeCaseMetadata(tenant="acme", channel="chat")

        with caplog.at_level(logging.WARNING, logger="akgentic.team.manager"):
            result = manager.update_team_metadata(runtime.id, new_metadata)

        assert pushed_indexes == [["tenant|acme", "channel|chat"]]
        assert "Failed to push metadata" in caplog.text
        assert result.metadata == new_metadata
        stored = event_store.load_team(runtime.id)
        assert stored is not None
        assert stored.metadata == new_metadata
        assert stored.metadata_indexes == ["tenant|acme", "channel|chat"]

    def test_stopped_team_writes_the_database_and_skips_the_push(
        self, manager: TeamManager, event_store: CountingEventStore
    ) -> None:
        """AC 15: no live orchestrator to push to — the write still happens."""
        runtime = manager.create_team(
            _make_team_card(), metadata=AcmeCaseMetadata(tenant="acme", channel="email")
        )
        manager.stop_team(runtime.id)

        with patch.object(TeamManager, "_push_metadata", autospec=True) as push:
            result = manager.update_team_metadata(
                runtime.id, AcmeCaseMetadata(tenant="acme", channel="chat")
            )

        push.assert_not_called()
        assert result.status == TeamStatus.STOPPED
        stored = event_store.load_team(runtime.id)
        assert stored is not None
        assert stored.metadata_indexes == ["tenant|acme", "channel|chat"]

    def test_running_team_with_no_tracked_runtime_writes_and_skips_the_push(
        self, manager: TeamManager, event_store: CountingEventStore
    ) -> None:
        """AC 13/15: RUNNING in the database but untracked here is not an error.

        This is the worker-restart shape, and the shape of a team owned by
        another replica: there is no orchestrator address to push to, so the
        write must stand on its own and the push must be skipped rather than
        attempted against a stale runtime.
        """
        runtime = manager.create_team(
            _make_team_card(), metadata=AcmeCaseMetadata(tenant="acme", channel="email")
        )
        manager.stop_team(runtime.id)  # drops the tracked runtime
        stopped = event_store.load_team(runtime.id)
        assert stopped is not None
        event_store.save_team(stopped.model_copy(update={"status": TeamStatus.RUNNING}))

        with patch.object(TeamManager, "_push_metadata", autospec=True) as push:
            result = manager.update_team_metadata(
                runtime.id, AcmeCaseMetadata(tenant="acme", channel="chat")
            )

        push.assert_not_called()
        assert result.status == TeamStatus.RUNNING
        stored = event_store.load_team(runtime.id)
        assert stored is not None
        assert stored.metadata_indexes == ["tenant|acme", "channel|chat"]

    def test_unknown_team_raises(self, manager: TeamManager) -> None:
        """AC 16: an unknown team id is a caller error, not a silent no-op."""
        with pytest.raises(ValueError, match="not found"):
            manager.update_team_metadata(
                uuid.uuid4(), AcmeCaseMetadata(tenant="acme", channel="email")
            )

    def test_deleted_team_raises_and_writes_nothing(
        self, manager: TeamManager, event_store: CountingEventStore
    ) -> None:
        """AC 16: a deleted team cannot be resurrected through the metadata path."""
        team_id = uuid.uuid4()
        now = datetime.now(UTC)
        event_store.save_team(
            Process(
                team_id=team_id,
                team_card=_make_team_card(),
                status=TeamStatus.DELETED,
                created_at=now,
                updated_at=now,
            )
        )
        writes_before = event_store.save_team_calls

        with pytest.raises(ValueError, match="has been deleted"):
            manager.update_team_metadata(team_id, AcmeCaseMetadata(tenant="acme", channel="email"))

        assert event_store.save_team_calls == writes_before

    def test_rejected_when_card_declares_no_metadata_type(
        self, manager: TeamManager, event_store: CountingEventStore
    ) -> None:
        """AC 9: a card with no declared type accepts no metadata, ever."""
        runtime = manager.create_team(_make_team_card(metadata_type=None))
        writes_before = event_store.save_team_calls

        with pytest.raises(ValueError, match="declares no metadata_type"):
            manager.update_team_metadata(
                runtime.id, AcmeCaseMetadata(tenant="acme", channel="email")
            )

        assert event_store.save_team_calls == writes_before

    def test_wrong_concrete_type_rejected_and_nothing_written(
        self, manager: TeamManager, event_store: CountingEventStore
    ) -> None:
        """AC 9: metadata_type cannot be changed through the update path.

        The previously persisted value and index survive a rejected update
        untouched — a rejection must not leave a partially applied write. The
        snapshot is taken BEFORE the rejected call: the store hands back the
        same object it was given, so a dump taken afterwards would compare the
        state to itself and pass even if the Process had been mutated in place.
        """
        original = AcmeCaseMetadata(tenant="acme", channel="email", case_ref="c-1")
        runtime = manager.create_team(_make_team_card(), metadata=original)
        before = event_store.load_team(runtime.id)
        assert before is not None
        snapshot = before.model_dump()
        writes_before = event_store.save_team_calls

        with pytest.raises(ValidationError):
            manager.update_team_metadata(runtime.id, ContosoMetadata(region="eu"))

        after = event_store.load_team(runtime.id)
        assert after is not None
        assert after.model_dump() == snapshot
        assert event_store.save_team_calls == writes_before


# ---------------------------------------------------------------------------
# Existing write paths must not drop metadata
# ---------------------------------------------------------------------------


class TestLifecycleCarriesMetadata:
    """AC 17: stop and resume rebuild a Process — both must carry the pair."""

    def test_stop_then_resume_preserves_metadata_and_indexes(
        self, manager: TeamManager, event_store: CountingEventStore
    ) -> None:
        metadata = AcmeCaseMetadata(tenant="acme", channel="email", case_ref="c-1")
        runtime = manager.create_team(_make_team_card(), metadata=metadata)
        expected_indexes = ["tenant|acme", "channel|email", "case_ref|c-1"]

        manager.stop_team(runtime.id)
        stopped = event_store.load_team(runtime.id)
        assert stopped is not None
        assert stopped.metadata == metadata
        assert stopped.metadata_indexes == expected_indexes

        manager.resume_team(runtime.id)

        resumed = event_store.load_team(runtime.id)
        assert resumed is not None
        assert resumed.metadata == metadata
        assert resumed.metadata_indexes == expected_indexes
