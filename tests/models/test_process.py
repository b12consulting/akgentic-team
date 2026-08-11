"""Tests for persistence models: TeamStatus, Process, PersistedEvent, AgentStateSnapshot."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from akgentic.core.agent_state import BaseState
from akgentic.core.messages.message import Message, UserMessage

from akgentic.team.models import (
    AgentStateSnapshot,
    PersistedEvent,
    Process,
    TeamStatus,
)

from .conftest import (
    AcmeTeamMetadata,
    SampleAgentState,
    make_agent_state_snapshot,
    make_persisted_event,
    make_process,
    make_team_card,
)


class TestTeamStatus:
    """Tests for TeamStatus enum."""

    def test_has_exactly_three_values(self) -> None:
        assert len(TeamStatus) == 3

    def test_values_are_running_stopped_deleted(self) -> None:
        assert set(TeamStatus) == {
            TeamStatus.RUNNING,
            TeamStatus.STOPPED,
            TeamStatus.DELETED,
        }

    def test_each_value_is_str(self) -> None:
        for status in TeamStatus:
            assert isinstance(status, str)

    def test_string_comparison_running(self) -> None:
        assert TeamStatus.RUNNING.value == "running"

    def test_string_comparison_stopped(self) -> None:
        assert TeamStatus.STOPPED.value == "stopped"

    def test_string_comparison_deleted(self) -> None:
        assert TeamStatus.DELETED.value == "deleted"


class TestProcess:
    """Tests for Process model."""

    def test_construction_with_all_fields(self) -> None:
        tid = uuid.uuid4()
        tc = make_team_card()
        now = datetime.now(UTC)
        process = Process(
            team_id=tid,
            team_card=tc,
            status=TeamStatus.RUNNING,
            user_id="admin",
            user_email="admin@example.com",
            created_at=now,
            updated_at=now,
        )
        assert process.team_id == tid
        assert process.team_card.name == tc.name
        assert process.status == TeamStatus.RUNNING
        assert process.user_id == "admin"
        assert process.user_email == "admin@example.com"
        assert process.created_at == now
        assert process.updated_at == now

    def test_construction_with_defaults(self) -> None:
        process = make_process()
        assert process.user_id == "cli"
        assert process.user_email == ""

    def test_serialization_round_trip(self) -> None:
        process = make_process()
        data = process.model_dump()
        restored = Process.model_validate(data)
        assert restored.team_id == process.team_id
        assert restored.team_card.name == process.team_card.name
        assert restored.status == process.status
        assert restored.user_id == process.user_id
        assert restored.user_email == process.user_email
        assert restored.created_at == process.created_at
        assert restored.updated_at == process.updated_at

    def test_round_trip_with_stopped_status(self) -> None:
        process = make_process(status=TeamStatus.STOPPED)
        data = process.model_dump()
        restored = Process.model_validate(data)
        assert restored.status == TeamStatus.STOPPED

    def test_round_trip_with_deleted_status(self) -> None:
        process = make_process(status=TeamStatus.DELETED)
        data = process.model_dump()
        restored = Process.model_validate(data)
        assert restored.status == TeamStatus.DELETED

    def test_catalog_namespace_defaults_to_none(self) -> None:
        process = make_process()
        assert process.catalog_namespace is None

    def test_catalog_namespace_round_trip(self) -> None:
        process = make_process(catalog_namespace="ns-abc")
        data = process.model_dump()
        restored = Process.model_validate(data)
        assert restored.catalog_namespace == "ns-abc"

    def test_catalog_namespace_default_round_trip(self) -> None:
        process = make_process()
        data = process.model_dump()
        restored = Process.model_validate(data)
        assert restored.catalog_namespace is None

    def test_legacy_persisted_data_without_catalog_namespace(self) -> None:
        """Pre-existing persisted teams without the field validate cleanly."""
        process = make_process()
        data = process.model_dump()
        data.pop("catalog_namespace", None)
        restored = Process.model_validate(data)
        assert restored.catalog_namespace is None


class TestProcessMetadata:
    """Tests for Process.metadata and Process.metadata_indexes."""

    def test_metadata_and_indexes_default_to_none_and_empty(self) -> None:
        process = make_process()
        assert process.metadata is None
        assert process.metadata_indexes == []

    def test_metadata_reloads_as_the_concrete_subclass(self) -> None:
        """Polymorphic reload: the value comes back typed, not as the base or a dict."""
        process = make_process(metadata=AcmeTeamMetadata(tenant="acme", case_ref="c-1"))
        restored = Process.model_validate(process.model_dump())
        assert type(restored.metadata) is AcmeTeamMetadata
        assert restored.metadata.tenant == "acme"
        assert restored.metadata.case_ref == "c-1"

    def test_metadata_indexes_round_trip_verbatim(self) -> None:
        process = make_process(
            metadata=AcmeTeamMetadata(tenant="acme"),
            metadata_indexes=["tenant|acme"],
        )
        restored = Process.model_validate(process.model_dump())
        assert restored.metadata_indexes == ["tenant|acme"]

    def test_metadata_indexes_are_never_derived_on_construction(self) -> None:
        """Setting metadata alone does not populate the index -- write paths do that.

        Deriving here would fire on every read of a persisted document and mask a
        write path that forgot to update the index alongside the value.
        """
        process = make_process(metadata=AcmeTeamMetadata(tenant="acme"))
        assert process.metadata_indexes == []

    def test_metadata_indexes_are_never_derived_on_read(self) -> None:
        """A persisted document with a value but a stale-empty index stays empty."""
        process = make_process(metadata=AcmeTeamMetadata(tenant="acme"))
        restored = Process.model_validate(process.model_dump())
        assert restored.metadata_indexes == []

    def test_legacy_payload_without_any_metadata_key_loads(self) -> None:
        """A document persisted before this change loads with no migration."""
        process = make_process()
        data = process.model_dump()
        data.pop("metadata", None)
        data.pop("metadata_indexes", None)
        data["team_card"].pop("metadata_type", None)
        restored = Process.model_validate(data)
        assert restored.metadata is None
        assert restored.metadata_indexes == []
        assert restored.team_card.metadata_type is None


class TestPersistedEvent:
    """Tests for PersistedEvent model."""

    def test_construction(self) -> None:
        event = make_persisted_event()
        assert event.sequence == 0
        assert isinstance(event.event, UserMessage)

    def test_serialization_round_trip(self) -> None:
        event = make_persisted_event(sequence=42)
        data = event.model_dump()
        restored = PersistedEvent.model_validate(data)
        assert restored.team_id == event.team_id
        assert restored.sequence == 42
        assert restored.timestamp == event.timestamp

    def test_polymorphic_deserialization_preserves_message_type(self) -> None:
        user_msg = UserMessage(content="hello world")
        event = make_persisted_event(event=user_msg)
        data = event.model_dump()
        restored = PersistedEvent.model_validate(data)
        assert type(restored.event) is UserMessage
        assert restored.event.content == "hello world"

    def test_polymorphic_event_is_not_base_message(self) -> None:
        user_msg = UserMessage(content="specific")
        event = make_persisted_event(event=user_msg)
        data = event.model_dump()
        restored = PersistedEvent.model_validate(data)
        assert type(restored.event) is not Message


class TestAgentStateSnapshot:
    """Tests for AgentStateSnapshot model."""

    def test_construction(self) -> None:
        snapshot = make_agent_state_snapshot()
        assert snapshot.agent_id == "test-agent"
        assert isinstance(snapshot.state, SampleAgentState)

    def test_serialization_round_trip(self) -> None:
        snapshot = make_agent_state_snapshot(agent_id="worker-1")
        data = snapshot.model_dump()
        restored = AgentStateSnapshot.model_validate(data)
        assert restored.team_id == snapshot.team_id
        assert restored.agent_id == "worker-1"
        assert restored.updated_at == snapshot.updated_at

    def test_polymorphic_deserialization_preserves_state_type(self) -> None:
        state = SampleAgentState(task_count=10)
        snapshot = make_agent_state_snapshot(state=state)
        data = snapshot.model_dump()
        restored = AgentStateSnapshot.model_validate(data)
        assert type(restored.state) is SampleAgentState
        assert restored.state.task_count == 10

    def test_polymorphic_state_is_not_base_state(self) -> None:
        state = SampleAgentState(task_count=3)
        snapshot = make_agent_state_snapshot(state=state)
        data = snapshot.model_dump()
        restored = AgentStateSnapshot.model_validate(data)
        assert type(restored.state) is not BaseState
