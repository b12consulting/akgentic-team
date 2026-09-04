"""The backend-agnostic projection migration — story 31-5, AC 1-14 and AC 18.

Every spec here drives :func:`akgentic.team.migration.migrate_documents`
directly, against ``InMemoryEventStore`` or a ``tmp_path`` YAML store. The
per-script specs (argument parsing, configuration, exit codes) live beside this
module, one file per backend.

Nothing in this module claims that ``can_be_hired`` is enforced anywhere. It is
a value the projection carries and the store normalises away; no guard reads it.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.messages.message import UserMessage
from akgentic.core.utils.serializer import SerializableBaseModel, serialize_type

import akgentic.team.migration as migration
from akgentic.team.migration import (
    MigrationReport,
    migrate_document,
    migrate_documents,
)
from akgentic.team.models import Process, TeamCard, TeamCardMember, TeamStatus
from akgentic.team.projection import derive_team_projection, resolve_agent_cards
from akgentic.team.repositories.yaml import YamlEventStore
from akgentic.team.restorer import TeamRestorer
from tests.conftest import projection_kwargs
from tests.models.conftest import (
    AcmeTeamMetadata,
    make_process,
    to_legacy_document,
)
from tests.services.conftest import InMemoryEventStore
from tests.services.test_restorer import (
    RecordingSubscriber,
    StubAgent,
    _make_member,
    _make_team_card,
    _populate_stopped_team,
    _sent_recipients,
)

# The projection field names, read off the one helper that produces them, so
# these specs cannot fall behind a projection that grows a field.
PROJECTION_FIELDS = tuple(projection_kwargs(_make_team_card()))


class ProcessWithAFieldAddedLater(Process):
    """``Process`` as it will look once someone adds the next field to it.

    Module level, not nested in a test: the serializer records a model by its
    importable ``module.ClassName`` path, and a class defined inside a function
    has no such path.
    """

    added_later: str = "the-default"


@pytest.fixture
def store() -> InMemoryEventStore:
    """A fresh dict-backed store per test."""
    return InMemoryEventStore()


@pytest.fixture
def actor_system() -> ActorSystem:  # type: ignore[misc]
    """An ActorSystem that shuts down after each test."""
    system = ActorSystem()
    yield system  # type: ignore[misc]
    system.shutdown()


def _team_card(
    name: str = "migration-team",
    members: list[TeamCardMember] | None = None,
) -> TeamCard:
    """A team card whose agent classes survive a store round trip."""
    return TeamCard(
        name=name,
        description="A team stored before the projection existed",
        entry_point=_make_member("lead", "Lead", StubAgent),
        members=members if members is not None else [_make_member("worker", "Worker", StubAgent)],
        message_types=[UserMessage],
    )


def _legacy_document(
    team_card: TeamCard | None = None,
    team_id: uuid.UUID | None = None,
    **process_overrides: Any,
) -> dict[str, Any]:
    """One pre-projection stored document, as a backend would hold it."""
    card = team_card or _team_card()
    process = make_process(team_id=team_id, team_card=card)
    if process_overrides:
        process = process.model_copy(update=process_overrides)
    return to_legacy_document(process, card)


class TestTheMigrationCore:
    """AC 1-9: what one conversion does, and what a run reports."""

    def test_a_legacy_document_migrates_to_what_create_team_would_produce(
        self, store: InMemoryEventStore
    ) -> None:
        """AC 1: compared against the derivation, never a hand-written expectation."""
        card = _team_card()
        document = _legacy_document(card)
        expected = derive_team_projection(card)

        report = migrate_documents([document], store)

        assert report == MigrationReport(converted=1, skipped=0, failed=0)
        migrated = store.load_team(uuid.UUID(str(document["team_id"])))
        assert migrated is not None
        for field in PROJECTION_FIELDS:
            assert getattr(migrated, field) == getattr(expected, field), field

    def test_the_derivation_is_imported_not_reimplemented(
        self, store: InMemoryEventStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC 2: the projection reaches the document through the ONE function.

        A second walk of the ``TeamCard`` would produce an identical result
        today and drift the first time the derivation changes — which is exactly
        why the call, not the value, is what is asserted.
        """
        calls: list[TeamCard] = []
        real = derive_team_projection

        def _spy(team_card: TeamCard) -> Any:
            calls.append(team_card)
            return real(team_card)

        # Patched on the MODULE OBJECT, not by dotted path: the dotted form
        # resolves through the ``akgentic.team`` namespace package, which does
        # not always carry the submodule as an attribute.
        monkeypatch.setattr(migration, "derive_team_projection", _spy)
        migrate_documents([_legacy_document()], store)

        assert len(calls) == 1
        assert calls[0].name == "migration-team"

    def test_cards_are_written_before_the_document(self, store: InMemoryEventStore) -> None:
        """AC 3 (FR13): assert the ORDER — both orders leave identical storage."""
        migrate_documents([_legacy_document()], store)

        assert store.write_calls == ["save_agent_cards", "save_team"]

    def test_every_non_projection_field_of_the_source_document_survives(
        self, store: InMemoryEventStore
    ) -> None:
        """AC 4 (Golden Rule #12): the migrated mapping is a COPY, not a rebuild.

        Every field is set to a distinctive non-default value, and the surviving
        set is computed from ``Process.model_fields`` rather than listed here,
        so a field added to ``Process`` is checked without an edit.
        """
        card = _team_card()
        created = datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC)
        updated = datetime(2021, 6, 7, 8, 9, 10, tzinfo=UTC)
        metadata = AcmeTeamMetadata(tenant="contoso", case_ref="C-77", department="ops")
        source = make_process(
            team_card=card,
            status=TeamStatus.STOPPED,
            user_id="operator-9",
            catalog_namespace="ns-migration",
            metadata=metadata,
            metadata_indexes=["tenant|contoso", "case_ref|c-77"],
        ).model_copy(
            update={
                "user_email": "operator-9@example.test",
                "created_at": created,
                "updated_at": updated,
            }
        )
        document = to_legacy_document(source, card)

        migrate_documents([document], store)

        migrated = store.load_team(source.team_id)
        assert migrated is not None
        carried = [
            name
            for name in Process.model_fields
            if name in document and name not in PROJECTION_FIELDS
        ]
        # Guards the guard: an empty list would make every assertion below vacuous.
        assert set(carried) >= {
            "team_id",
            "status",
            "user_id",
            "user_email",
            "created_at",
            "updated_at",
            "catalog_namespace",
            "metadata",
            "metadata_indexes",
        }
        for name in carried:
            assert getattr(migrated, name) == getattr(source, name), name

    def test_a_field_the_migration_has_never_heard_of_survives(
        self, store: InMemoryEventStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC 4, the half a whole-model comparison cannot reach.

        Comparing every field that exists TODAY is green for an enumerated
        rebuild that names every field that exists today — which is what a
        developer would actually write, and what silently destroys the
        **eleventh** field the day someone adds it. Measured, not argued: that
        mutation passes the comparison above and fails this one.

        So the field this exercises is one the write path has never heard of.
        ``ProcessWithAFieldAddedLater`` stands in for "``Process`` a release
        from now", and the source document carries a value for it. A dict-merge
        carries it through by construction; an enumerated rebuild cannot name
        what does not exist yet and hands back the default.
        """
        monkeypatch.setattr(migration, "Process", ProcessWithAFieldAddedLater)
        document = _legacy_document()
        # A stored document names its own class, and a document written by the
        # later release would name the later class.
        document["__model__"] = serialize_type(ProcessWithAFieldAddedLater)
        document["added_later"] = "carried-through"

        migrate_documents([document], store)

        migrated = store.load_team(uuid.UUID(str(document["team_id"])))
        assert isinstance(migrated, ProcessWithAFieldAddedLater)
        assert migrated.added_later == "carried-through"

    def test_a_second_run_over_a_migrated_store_writes_nothing(
        self, store: InMemoryEventStore
    ) -> None:
        """AC 5: idempotent — assert the WRITE CALLS, not only the documents."""
        document = _legacy_document()
        migrate_documents([document], store)
        cards_after_first = dict(store.agent_cards)
        store.write_calls.clear()

        migrated_document = store.load_team(uuid.UUID(str(document["team_id"])))
        assert migrated_document is not None
        second = migrate_documents([migrated_document.model_dump()], store)

        assert second == MigrationReport(converted=0, skipped=1, failed=0)
        assert store.write_calls == []
        assert store.agent_cards == cards_after_first

    def test_an_already_migrated_document_is_recognised_before_team_card(
        self, store: InMemoryEventStore
    ) -> None:
        """AC 6: a hybrid carrying BOTH keys counts as migrated, not as legacy.

        That is ``Process.reject_unmigrated_document``'s own rule — legacy is
        ``team_card`` present AND ``entry_point`` absent — so an interrupted run
        cannot re-flatten a document from a card that may since have changed.
        """
        card = _team_card()
        hybrid = _legacy_document(card)
        hybrid.update(
            {
                key: value
                for key, value in derive_team_projection(card).model_dump().items()
                if key in PROJECTION_FIELDS
            }
        )
        assert "team_card" in hybrid
        assert "entry_point" in hybrid

        report = migrate_documents([hybrid], store)

        assert report == MigrationReport(converted=0, skipped=1, failed=0)
        assert store.write_calls == []

    def test_a_document_without_a_team_card_is_skipped_not_failed(
        self, store: InMemoryEventStore
    ) -> None:
        """AC 7: nothing to derive from is not a failure; the run continues."""
        document = _legacy_document()
        document.pop("team_card")

        report = migrate_documents([document, _legacy_document()], store)

        assert report == MigrationReport(converted=1, skipped=1, failed=0)

    def test_a_malformed_document_is_counted_and_does_not_abort_the_run(
        self, store: InMemoryEventStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AC 8: the first and third still convert, and the failure names the team."""
        first = _legacy_document()
        broken = _legacy_document()
        broken["team_card"] = {"not": "a team card"}
        third = _legacy_document()

        with caplog.at_level(logging.ERROR, logger="akgentic.team.migration"):
            report = migrate_documents([first, broken, third], store)

        assert report == MigrationReport(converted=2, skipped=0, failed=1)
        assert store.load_team(uuid.UUID(str(first["team_id"]))) is not None
        assert store.load_team(uuid.UUID(str(third["team_id"]))) is not None
        assert store.load_team(uuid.UUID(str(broken["team_id"]))) is None
        assert str(broken["team_id"]) in caplog.text

    def test_a_document_that_is_not_a_mapping_is_a_failure(
        self, store: InMemoryEventStore
    ) -> None:
        """AC 8: a backend can hand back a scalar or ``None``; the run survives it."""
        report = migrate_documents([None, "not a document", _legacy_document()], store)

        assert report == MigrationReport(converted=1, skipped=0, failed=2)

    def test_the_report_is_a_model_with_three_counts(self) -> None:
        """AC 9: a typed report, not a mapping (Golden Rule #1)."""
        report = MigrationReport(converted=2, skipped=1, failed=3)

        assert isinstance(report, SerializableBaseModel)
        assert (report.converted, report.skipped, report.failed) == (2, 1, 3)
        assert set(MigrationReport.model_fields) == {"converted", "skipped", "failed"}


class TestHeadcountIsExpandedByTheMigration:
    """AC 10 (FR8a): the stored card is the DECLARED shape; refs are spawned names."""

    def test_a_headcount_supervisor_migrates_to_one_ref_per_instance(
        self, store: InMemoryEventStore
    ) -> None:
        """Three refs named ``worker_0..2``; the bare declared name appears nowhere.

        Free rather than a second implementation: ``derive_team_projection``
        already expands through ``spawned_names``, and the old stored card still
        carries ``headcount``, so no event log is needed.
        """
        card = _team_card(members=[_make_member("worker", "Worker", StubAgent, headcount=3)])
        document = _legacy_document(card)

        migrate_documents([document], store)

        migrated = store.load_team(uuid.UUID(str(document["team_id"])))
        assert migrated is not None
        assert [ref.name for ref in migrated.supervisors] == [
            "worker_0",
            "worker_1",
            "worker_2",
        ]
        assert "worker" not in {ref.name for ref in migrated.supervisors}


class TestTheMigrationPopulatesTheCardStore:
    """AC 11, 12 (FR11-FR13): cards land in the content-addressed store."""

    def test_every_ref_on_the_migrated_process_resolves(
        self, store: InMemoryEventStore
    ) -> None:
        """AC 11: one card per ref, resolved through the one resolution function."""
        document = _legacy_document()

        migrate_documents([document], store)

        migrated = store.load_team(uuid.UUID(str(document["team_id"])))
        assert migrated is not None
        resolved = resolve_agent_cards(migrated.agent_cards, store)
        assert len(resolved) == len(migrated.agent_cards)
        assert {card.role for card in resolved} == {ref.role for ref in migrated.agent_cards}

    def test_two_teams_sharing_a_card_produce_one_blob(
        self, store: InMemoryEventStore
    ) -> None:
        """AC 12: the second team writes nothing the first did not already write."""
        card = _team_card()
        first = _legacy_document(card)
        second = _legacy_document(card)

        migrate_documents([first], store)
        after_first = dict(store.agent_cards)
        report = migrate_documents([second], store)

        assert report.converted == 1
        assert store.agent_cards == after_first
        assert len(store.agent_cards) == len(derive_team_projection(card).cards)


class TestAMigratedTeamResumes:
    """AC 13, 14: the point of the whole story.

    If a ``headcount=3`` resume fails here, the fixture is the first suspect —
    the member-tree walk synthesises one ``StartMessage`` per SPAWNED actor, not
    per declared slot (story 31-4). The restore path has no expansion step of
    its own by design (FR3a).
    """

    @staticmethod
    def _migrate_in_place(
        store: InMemoryEventStore, team_card: TeamCard, process: Process
    ) -> Process:
        """Roll the stored team back to the legacy shape, then migrate it forward."""
        document = to_legacy_document(process, team_card)
        store.teams.clear()
        store.agent_cards.clear()
        store.write_calls.clear()

        report = migrate_documents([document], store)
        assert report == MigrationReport(converted=1, skipped=0, failed=0)

        migrated = store.load_team(process.team_id)
        assert migrated is not None
        return migrated

    def test_a_migrated_team_resumes_and_reaches_every_supervisor(
        self, actor_system: ActorSystem, store: InMemoryEventStore
    ) -> None:
        """AC 13: entry point and supervisors resolve; str and Message both route."""
        card = _make_team_card(
            members=[_make_member("alpha", "Alpha"), _make_member("beta", "Beta")],
        )
        card = card.model_copy(update={"message_types": [UserMessage]})
        _team_id, process = _populate_stopped_team(store, card)

        migrated = self._migrate_in_place(store, card, process)

        recording = RecordingSubscriber()
        runtime = TeamRestorer(actor_system, store).restore(migrated, subscribers=[recording])

        assert set(runtime.supervisor_addrs) == {"alpha", "beta"}
        baseline = len(recording.messages)
        runtime.send("after a migration")
        sent = _sent_recipients(recording, baseline, expected=2)
        assert len(sent) == 2
        assert {m.recipient.name for m in sent} == {"alpha", "beta"}

        baseline = len(recording.messages)
        runtime.send(UserMessage(content="a pre-formed message"))
        sent = _sent_recipients(recording, baseline, expected=2)
        assert len(sent) == 2

    def test_a_migrated_headcount_team_resumes_with_all_three_reachable(
        self, actor_system: ActorSystem, store: InMemoryEventStore
    ) -> None:
        """AC 14: three refs, three addresses, three deliveries."""
        crew = _make_member("worker", "Worker", headcount=3)
        card = TeamCard(
            name="headcount-team",
            description="One multi-instance supervisor",
            entry_point=_make_member("lead", "Lead"),
            members=[crew],
            message_types=[UserMessage],
        )
        expected = {"worker_0", "worker_1", "worker_2"}
        _team_id, process = _populate_stopped_team(store, card)

        migrated = self._migrate_in_place(store, card, process)

        assert {ref.name for ref in migrated.supervisors} == expected
        recording = RecordingSubscriber()
        runtime = TeamRestorer(actor_system, store).restore(migrated, subscribers=[recording])

        assert set(runtime.supervisor_addrs) == expected
        baseline = len(recording.messages)
        runtime.send("after a migration")
        sent = _sent_recipients(recording, baseline, expected=3)
        assert len(sent) == 3
        assert {m.recipient.name for m in sent} == expected


class TestTheMigrationReadsRawDocuments:
    """AC 18: the public read path cannot see what the migration converts."""

    def test_list_teams_and_load_team_are_blind_to_an_unmigrated_store(
        self, tmp_path: Path
    ) -> None:
        """A migration built on them converts nothing and reports success.

        Both refuse the legacy shape by design — ``load_team`` returns ``None``
        for a team that exists, and ``list_teams`` returns nothing at all — so
        the raw read is a requirement, not a shortcut.
        """
        store = YamlEventStore(tmp_path)
        document = _legacy_document()
        team_id = uuid.UUID(str(document["team_id"]))
        team_dir = tmp_path / str(team_id)
        team_dir.mkdir(parents=True)
        with open(team_dir / "team.yaml", "w") as handle:
            yaml.dump(document, handle, default_flow_style=False)

        assert store.list_teams() == []
        assert store.load_team(team_id) is None

        # The raw read sees it, and the migration makes both answer properly.
        report = migrate_documents([document], store)
        assert report == MigrationReport(converted=1, skipped=0, failed=0)
        assert store.load_team(team_id) is not None
        assert [p.team_id for p in store.list_teams()] == [team_id]


class TestMigrateDocumentRaisesRatherThanCounting:
    """The single-document entry point is deliberately loud; the loop counts."""

    def test_a_team_card_that_does_not_validate_raises(
        self, store: InMemoryEventStore
    ) -> None:
        """``migrate_documents`` is what converts this into a counted failure."""
        document = _legacy_document()
        document["team_card"] = {"not": "a team card"}

        with pytest.raises(ValueError):
            migrate_document(document, store)
