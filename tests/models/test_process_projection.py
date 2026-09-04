"""Tests for the Process structural projection: the ref models and the fields.

The derivation that fills these fields lives in ``tests/services/test_projection.py``;
this module covers the models themselves — their shape, their round trip, and the
referential-integrity validator that makes an unresolvable ref impossible to
persist or to read back.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from akgentic.core.messages.message import UserMessage
from pydantic import ValidationError

from akgentic.team.models import (
    AgentCardRef,
    AgentRef,
    Process,
    TeamCardMember,
    TeamStatus,
    spawned_names,
)

from .conftest import (
    AcmeTeamMetadata,
    make_agent_card,
    make_process,
    make_team_card,
)


class TestAgentRef:
    """AgentRef: a spawned agent identity plus the role it was spawned from."""

    def test_carries_exactly_name_and_role(self) -> None:
        assert set(AgentRef.model_fields) == {"name", "role"}

    def test_both_fields_are_required(self) -> None:
        with pytest.raises(ValidationError):
            AgentRef(name="@Lead")  # type: ignore[call-arg]
        with pytest.raises(ValidationError):
            AgentRef(role="Lead")  # type: ignore[call-arg]

    def test_round_trips(self) -> None:
        ref = AgentRef(name="@Worker_2", role="Worker")
        restored = AgentRef.model_validate(ref.model_dump())
        assert restored == ref


class TestAgentCardRef:
    """AgentCardRef: a role, the hash of its card, and its hireability."""

    def test_carries_role_hash_and_hireability(self) -> None:
        assert set(AgentCardRef.model_fields) == {"role", "card_hash", "can_be_hired"}

    def test_can_be_hired_defaults_to_false(self) -> None:
        ref = AgentCardRef(role="Worker", card_hash="abc")
        assert ref.can_be_hired is False

    def test_round_trips_with_the_flag_set(self) -> None:
        ref = AgentCardRef(role="Worker", card_hash="abc", can_be_hired=True)
        restored = AgentCardRef.model_validate(ref.model_dump())
        assert restored == ref
        assert restored.can_be_hired is True


class TestSpawnedNames:
    """The one rule that expands a member's headcount into spawned names."""

    def test_headcount_one_yields_the_bare_card_name(self) -> None:
        member = TeamCardMember(card=make_agent_card(name="@Lead", role="Lead"))
        assert spawned_names(member) == ["@Lead"]

    def test_headcount_three_yields_three_indexed_names_in_order(self) -> None:
        member = TeamCardMember(
            card=make_agent_card(name="@Worker", role="Worker"), headcount=3
        )
        assert spawned_names(member) == ["@Worker_0", "@Worker_1", "@Worker_2"]


class TestProcessProjectionFields:
    """The seven projection fields on Process."""

    def test_the_seven_fields_exist_with_the_declared_defaults(self) -> None:
        process = make_process()
        assert process.team_name == "test-team"
        assert process.team_description is None
        assert isinstance(process.entry_point, AgentRef)
        assert process.supervisors == []
        assert [c.role for c in process.agent_cards] == ["Lead"]
        assert process.message_types == []
        assert process.metadata_type is None

    def test_entry_point_is_required(self) -> None:
        """An unmigrated document — nested card, no projection — fails loudly.

        A default here would let such a document load half-formed and fail later
        inside a resume with a confusing error; the migration is mandatory.
        """
        with pytest.raises(ValidationError, match="entry_point"):
            Process(
                team_id=uuid.uuid4(),
                team_card=make_team_card(),
                status=TeamStatus.RUNNING,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
            )

    def test_round_trip_preserves_message_types_as_concrete_classes(self) -> None:
        process = make_process(team_card=make_team_card(message_types=[UserMessage]))
        restored = Process.model_validate(process.model_dump())
        assert restored.message_types == [UserMessage]

    def test_round_trip_preserves_metadata_type(self) -> None:
        tc = make_team_card()
        tc.metadata_type = AcmeTeamMetadata
        process = make_process(team_card=tc)
        restored = Process.model_validate(process.model_dump())
        assert restored.metadata_type is AcmeTeamMetadata

    def test_round_trip_preserves_the_refs_and_the_hireable_flag(self) -> None:
        tc = make_team_card(
            member_names=["@Worker"],
            member_roles=["Worker"],
        )
        tc.agent_profiles = [make_agent_card(name="@Helper", role="Helper")]
        process = make_process(team_card=tc)
        restored = Process.model_validate(process.model_dump())

        assert restored.entry_point == process.entry_point
        assert restored.supervisors == process.supervisors
        assert restored.agent_cards == process.agent_cards
        hireable = {c.role: c.can_be_hired for c in restored.agent_cards}
        assert hireable == {"Lead": False, "Worker": False, "Helper": True}


class TestProcessReferentialIntegrity:
    """Every ref must resolve against agent_cards — on write and on read."""

    @staticmethod
    def _process_kwargs() -> dict[str, object]:
        now = datetime.now(UTC)
        return {
            "team_id": uuid.uuid4(),
            "team_card": make_team_card(),
            "status": TeamStatus.RUNNING,
            "created_at": now,
            "updated_at": now,
            "team_name": "test-team",
        }

    def test_resolvable_refs_validate_cleanly(self) -> None:
        process = Process(
            **self._process_kwargs(),  # type: ignore[arg-type]
            entry_point=AgentRef(name="@Lead", role="Lead"),
            supervisors=[AgentRef(name="@Worker", role="Worker")],
            agent_cards=[
                AgentCardRef(role="Lead", card_hash="h1"),
                AgentCardRef(role="Worker", card_hash="h2"),
            ],
        )
        assert process.entry_point.role == "Lead"

    def test_unresolvable_entry_point_is_rejected_and_named(self) -> None:
        with pytest.raises(ValidationError, match="Ghost"):
            Process(
                **self._process_kwargs(),  # type: ignore[arg-type]
                entry_point=AgentRef(name="@Ghost", role="Ghost"),
                agent_cards=[AgentCardRef(role="Lead", card_hash="h1")],
            )

    def test_unresolvable_supervisor_is_rejected_and_named(self) -> None:
        with pytest.raises(ValidationError, match="Ghost"):
            Process(
                **self._process_kwargs(),  # type: ignore[arg-type]
                entry_point=AgentRef(name="@Lead", role="Lead"),
                supervisors=[AgentRef(name="@Ghost", role="Ghost")],
                agent_cards=[AgentCardRef(role="Lead", card_hash="h1")],
            )

    def test_the_check_fires_on_model_validate_too(self) -> None:
        """The same payload must be rejected on read, not only in the constructor.

        A document can reach ``model_validate`` without ever passing through the
        constructor — that is exactly the path a corrupted or half-migrated
        record takes.
        """
        good = Process(
            **self._process_kwargs(),  # type: ignore[arg-type]
            entry_point=AgentRef(name="@Lead", role="Lead"),
            agent_cards=[AgentCardRef(role="Lead", card_hash="h1")],
        )
        payload = good.model_dump()
        payload["entry_point"]["role"] = "Ghost"

        with pytest.raises(ValidationError, match="Ghost"):
            Process.model_validate(payload)

    def test_a_supervisor_ref_is_rejected_on_read_too(self) -> None:
        good = Process(
            **self._process_kwargs(),  # type: ignore[arg-type]
            entry_point=AgentRef(name="@Lead", role="Lead"),
            supervisors=[AgentRef(name="@Worker", role="Worker")],
            agent_cards=[
                AgentCardRef(role="Lead", card_hash="h1"),
                AgentCardRef(role="Worker", card_hash="h2"),
            ],
        )
        payload = good.model_dump()
        payload["agent_cards"] = [payload["agent_cards"][0]]

        with pytest.raises(ValidationError, match="Worker"):
            Process.model_validate(payload)
