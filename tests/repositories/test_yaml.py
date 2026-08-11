"""YAML-specific tests for ``YamlEventStore``.

Behavioural Protocol coverage (round-trip, upsert, list, sequencing,
max sequence, cascading delete, polymorphic round-trips) lives in the
shared ``tests/repositories/test_event_store_contract.py`` and runs
once per backend. This module retains only YAML-specific invariants:

* Protocol structural-typing check.
* On-disk directory-layout and lazy-creation behaviour.
* List-teams skipping non-UUID directories.
* List-teams filtering the raw parsed mapping ahead of validation —
  a YAML-only property, since the other backends push the filter into
  the query.
* Corrupted-file resilience (YAML parser errors → ``None`` / ``[]`` /
  skip rather than raise — this is the YamlEventStore contract).
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from akgentic.team.models import Process, TeamStatus
from akgentic.team.repositories.yaml import YamlEventStore

if TYPE_CHECKING:
    from akgentic.team.ports import EventStore

from tests.models.conftest import (
    AcmeTeamMetadata,
    make_agent_state_snapshot,
    make_indexed_process,
    make_persisted_event,
    make_process,
)

_DELETE_KEY = object()
"""Parametrize control token meaning "remove the key" rather than "store this value".

A dedicated object rather than a string: the cases it sits beside ARE arbitrary
values, one of them already a bare string, so a string token would share their
value space and a future case could silently mean deletion instead.
"""


@pytest.fixture
def yaml_store(tmp_path: Path) -> YamlEventStore:
    """Create a YamlEventStore backed by a temporary directory."""
    return YamlEventStore(tmp_path)


class TestYamlEventStoreYamlSpecific:
    """YAML-only invariants — see contract suite for behavioural coverage."""

    # --- Protocol compliance ------------------------------------------------

    def test_satisfies_event_store_protocol(self, tmp_path: Path) -> None:
        """``YamlEventStore`` satisfies ``EventStore`` Protocol structurally."""
        store: EventStore = YamlEventStore(tmp_path)
        assert store is not None

    # --- On-disk layout / directory creation --------------------------------

    def test_directory_creation_is_automatic(self, yaml_store: YamlEventStore) -> None:
        """Per-team directories are created on demand, not eagerly."""
        team_id = uuid.uuid4()
        # Save event without pre-creating any dirs
        event = make_persisted_event(team_id=team_id, sequence=1)
        yaml_store.save_event(event)  # should not raise

        loaded = yaml_store.load_events(team_id)
        assert len(loaded) == 1

    def test_list_teams_ignores_non_team_directories(
        self, yaml_store: YamlEventStore, tmp_path: Path
    ) -> None:
        """``list_teams`` skips non-UUID directories like ``.gitkeep``."""
        p1 = make_process()
        yaml_store.save_team(p1)
        # Create non-team entries
        (tmp_path / ".gitkeep").touch()
        (tmp_path / "__pycache__").mkdir()

        result = yaml_store.list_teams()
        assert len(result) == 1
        assert result[0].team_id == p1.team_id

    # --- list_teams filters before validating -------------------------------

    def test_list_teams_validates_only_the_teams_it_returns(
        self, yaml_store: YamlEventStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A filtered ``list_teams`` hydrates only the teams it returns.

        This is the whole point of the pre-validation filter: a skip
        applied *after* ``load_team`` returns identical results while
        still paying for ``Process.model_validate`` — the expensive half,
        which builds the full ``TeamCard`` graph — on every team it is
        about to discard. YAML is the community tier's boot path, so that
        cost is the one this filter exists to avoid.

        The unfiltered assertion at the end keeps the filtered one from
        passing vacuously: a store that validated nothing at all would
        satisfy ``len(calls) == 1`` by accident.
        """
        # Seed BEFORE installing the spy — save_team uses model_dump, so
        # seeding cannot pollute the model_validate count.
        yaml_store.save_team(make_process(status=TeamStatus.RUNNING))
        for _ in range(3):
            yaml_store.save_team(make_process(status=TeamStatus.STOPPED))

        calls: list[object] = []
        original = Process.model_validate

        def counting(data: object, *args: object, **kwargs: object) -> Process:
            calls.append(data)
            return original(data, *args, **kwargs)

        monkeypatch.setattr(Process, "model_validate", counting)

        running = yaml_store.list_teams(status=TeamStatus.RUNNING)
        assert len(running) == 1
        assert len(calls) == 1  # NOT 4 — the stopped teams are never hydrated

        calls.clear()
        assert len(yaml_store.list_teams()) == 4
        assert len(calls) == 4

    def test_list_teams_metadata_filter_validates_only_the_teams_it_returns(
        self, yaml_store: YamlEventStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The metadata filter is applied BEFORE hydration, not after it.

        This is the entire point of the story. A metadata filter applied
        after ``load_team`` returns identical results while still paying
        ``Process.model_validate`` — which builds the full ``TeamCard``
        object graph — for every team it is about to discard. YAML is the
        community tier's boot-path scan, so that is the cost this exists to
        avoid.

        The unfiltered assertion at the end keeps the filtered one from
        passing vacuously: a store that hydrated nothing at all would
        satisfy ``len(calls) == 1`` by accident.
        """
        # Seed BEFORE installing the spy — save_team uses model_dump, so
        # seeding cannot pollute the model_validate count.
        yaml_store.save_team(make_indexed_process(AcmeTeamMetadata(tenant="acme")))
        for _ in range(3):
            yaml_store.save_team(make_indexed_process(AcmeTeamMetadata(tenant="contoso")))

        calls: list[object] = []
        original = Process.model_validate

        def counting(data: object, *args: object, **kwargs: object) -> Process:
            calls.append(data)
            return original(data, *args, **kwargs)

        monkeypatch.setattr(Process, "model_validate", counting)

        matched = yaml_store.list_teams(metadata={"tenant": "acme"})
        assert len(matched) == 1
        assert len(calls) == 1  # NOT 4 — the contoso teams are never hydrated

        calls.clear()
        assert len(yaml_store.list_teams()) == 4
        assert len(calls) == 4

    def test_list_teams_metadata_and_status_together_still_skip_hydration(
        self, yaml_store: YamlEventStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Combining filters must not reintroduce hydration of discarded teams.

        A conjunction evaluated across two passes — one before validation
        and one after — would return the right teams while hydrating every
        team the first pass let through. The call count is what catches it.
        """
        yaml_store.save_team(
            make_indexed_process(AcmeTeamMetadata(tenant="acme"), status=TeamStatus.RUNNING)
        )
        yaml_store.save_team(
            make_indexed_process(AcmeTeamMetadata(tenant="acme"), status=TeamStatus.STOPPED)
        )
        yaml_store.save_team(
            make_indexed_process(AcmeTeamMetadata(tenant="contoso"), status=TeamStatus.RUNNING)
        )
        yaml_store.save_team(
            make_indexed_process(AcmeTeamMetadata(tenant="contoso"), status=TeamStatus.STOPPED)
        )

        calls: list[object] = []
        original = Process.model_validate

        def counting(data: object, *args: object, **kwargs: object) -> Process:
            calls.append(data)
            return original(data, *args, **kwargs)

        monkeypatch.setattr(Process, "model_validate", counting)

        matched = yaml_store.list_teams(
            status=TeamStatus.RUNNING, metadata={"tenant": "acme"}
        )
        assert len(matched) == 1
        assert len(calls) == 1  # NOT 2 (status-only) and NOT 4 (unfiltered)

        calls.clear()
        assert len(yaml_store.list_teams()) == 4
        assert len(calls) == 4

    @pytest.mark.parametrize(
        "corrupt",
        [
            pytest.param(_DELETE_KEY, id="key-missing"),
            pytest.param(None, id="null-value"),
            pytest.param({"not": "a list"}, id="mapping-not-a-list"),
            pytest.param("tenant|acme", id="bare-string-not-a-list"),
            pytest.param([{"nested": "entry"}], id="list-of-non-strings"),
        ],
    )
    def test_list_teams_skips_team_with_malformed_metadata_indexes(
        self, yaml_store: YamlEventStore, tmp_path: Path, corrupt: object
    ) -> None:
        """A wrong-shaped ``metadata_indexes`` is a non-match, never a raise.

        A healthy matching team is seeded alongside the broken one so the
        assertion pins an exact survivor — otherwise a ``list_teams`` broken
        to always return nothing would satisfy it.
        """
        good = make_indexed_process(AcmeTeamMetadata(tenant="acme"))
        bad = make_indexed_process(AcmeTeamMetadata(tenant="acme"))
        yaml_store.save_team(good)
        yaml_store.save_team(bad)
        team_path = tmp_path / str(bad.team_id) / "team.yaml"
        data = yaml.safe_load(team_path.read_text())
        if corrupt is _DELETE_KEY:
            del data["metadata_indexes"]
        else:
            data["metadata_indexes"] = corrupt
        team_path.write_text(yaml.dump(data))

        result = yaml_store.list_teams(metadata={"tenant": "acme"})
        assert [p.team_id for p in result] == [good.team_id]

    def test_list_teams_without_metadata_filter_still_returns_malformed_team(
        self, yaml_store: YamlEventStore, tmp_path: Path
    ) -> None:
        """A missing ``metadata_indexes`` key changes nothing for a call that ignores it.

        Teams persisted before the metadata contract existed carry no such
        key. They must keep listing exactly as they did — the new filter is
        additive, so it can only affect calls that ask for it.
        """
        legacy = make_process(status=TeamStatus.RUNNING)
        yaml_store.save_team(legacy)
        team_path = tmp_path / str(legacy.team_id) / "team.yaml"
        data = yaml.safe_load(team_path.read_text())
        del data["metadata_indexes"]
        team_path.write_text(yaml.dump(data))

        assert [p.team_id for p in yaml_store.list_teams()] == [legacy.team_id]
        assert [p.team_id for p in yaml_store.list_teams(status=TeamStatus.RUNNING)] == [
            legacy.team_id
        ]
        assert yaml_store.list_teams(metadata={"tenant": "acme"}) == []

    def test_list_teams_metadata_filter_skips_document_that_is_not_a_mapping(
        self, yaml_store: YamlEventStore, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A metadata filter must not break the no-filter fast path (AC 15).

        ``_matches`` returns True for a non-mapping document when NO filter
        is requested, deliberately ahead of its ``isinstance`` guard, so an
        unfiltered call still routes the document to validation and skips it
        with the corrupted-document log. Widening that condition to include
        ``entries`` must keep that property: the ``caplog`` assertion is what
        fails if the fast path stops firing, since every result set here
        would be unchanged either way.
        """
        good = make_indexed_process(AcmeTeamMetadata(tenant="acme"))
        yaml_store.save_team(good)
        team_dir = tmp_path / str(uuid.uuid4())
        team_dir.mkdir()
        (team_dir / "team.yaml").write_text("- just\n- a\n- list\n")

        filtered = yaml_store.list_teams(metadata={"tenant": "acme"})
        assert [p.team_id for p in filtered] == [good.team_id]

        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="akgentic.team.repositories.yaml"):
            assert [p.team_id for p in yaml_store.list_teams()] == [good.team_id]
        assert any("Corrupted team.yaml" in record.message for record in caplog.records)

    def test_list_teams_skips_team_with_missing_status_key(
        self, yaml_store: YamlEventStore, tmp_path: Path
    ) -> None:
        """A ``team.yaml`` with no ``status`` key is skipped, not raised on.

        A healthy team is seeded alongside it so both assertions pin an
        exact survivor rather than ``[]`` — otherwise a ``list_teams`` that
        returned nothing at all would satisfy them.
        """
        good = make_process(status=TeamStatus.RUNNING)
        yaml_store.save_team(good)
        process = make_process(status=TeamStatus.RUNNING)
        yaml_store.save_team(process)
        team_path = tmp_path / str(process.team_id) / "team.yaml"
        data = yaml.safe_load(team_path.read_text())
        del data["status"]
        team_path.write_text(yaml.dump(data))

        filtered = yaml_store.list_teams(status=TeamStatus.RUNNING)
        assert [p.team_id for p in filtered] == [good.team_id]
        # Unfiltered it is absent too — it fails validation, exactly as today.
        assert [p.team_id for p in yaml_store.list_teams()] == [good.team_id]

    def test_list_teams_skips_team_with_non_scalar_status(
        self, yaml_store: YamlEventStore, tmp_path: Path
    ) -> None:
        """A ``status`` that parses to a mapping matches nothing and does not raise."""
        good = make_process(status=TeamStatus.RUNNING)
        yaml_store.save_team(good)
        process = make_process(status=TeamStatus.RUNNING)
        yaml_store.save_team(process)
        team_path = tmp_path / str(process.team_id) / "team.yaml"
        data = yaml.safe_load(team_path.read_text())
        data["status"] = {"nested": "mapping"}
        team_path.write_text(yaml.dump(data))

        filtered = yaml_store.list_teams(status=TeamStatus.RUNNING)
        assert [p.team_id for p in filtered] == [good.team_id]
        assert [p.team_id for p in yaml_store.list_teams()] == [good.team_id]

    def test_list_teams_skips_document_that_is_not_a_mapping(
        self, yaml_store: YamlEventStore, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Valid YAML of the wrong shape cannot match a filter and is skipped.

        The ``caplog`` assertion is the point of this test, not decoration.
        ``_matches`` returns True for a non-mapping when NO filter is given,
        deliberately ahead of its ``isinstance`` guard, so an unfiltered
        call still routes the document to validation and skips it with the
        corrupted-document log. Reordering those two lines would leave every
        result set here unchanged and silently drop that log line — this
        assertion is what fails if anyone does.
        """
        good = make_process(status=TeamStatus.RUNNING)
        yaml_store.save_team(good)
        team_dir = tmp_path / str(uuid.uuid4())
        team_dir.mkdir()
        (team_dir / "team.yaml").write_text("- just\n- a\n- list\n")

        filtered = yaml_store.list_teams(status=TeamStatus.RUNNING)
        assert [p.team_id for p in filtered] == [good.team_id]
        by_user = yaml_store.list_teams(user_id=good.user_id)
        assert [p.team_id for p in by_user] == [good.team_id]

        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="akgentic.team.repositories.yaml"):
            assert [p.team_id for p in yaml_store.list_teams()] == [good.team_id]
        assert any("Corrupted team.yaml" in record.message for record in caplog.records)

    # --- Corrupted-file resilience ------------------------------------------

    def test_load_team_returns_none_for_corrupted_yaml(self, tmp_path: Path) -> None:
        """Corrupted ``team.yaml`` returns None instead of raising."""
        store = YamlEventStore(tmp_path)
        team_id = uuid.uuid4()
        team_dir = tmp_path / str(team_id)
        team_dir.mkdir()
        (team_dir / "team.yaml").write_text("{{invalid: yaml: [}")
        assert store.load_team(team_id) is None

    def test_load_team_returns_none_for_undecodable_bytes(self, tmp_path: Path) -> None:
        """A ``team.yaml`` that is not valid UTF-8 returns None instead of raising.

        The decode happens inside ``yaml.safe_load``'s read of the text
        stream and surfaces as ``UnicodeDecodeError`` — a ``ValueError``
        subclass, not a ``yaml.YAMLError``. Unreadable bytes are an
        unparseable file like any other and must not escape the store.
        """
        store = YamlEventStore(tmp_path)
        team_id = uuid.uuid4()
        team_dir = tmp_path / str(team_id)
        team_dir.mkdir()
        (team_dir / "team.yaml").write_bytes(b"status: \xff\xfe\x00running\n")
        assert store.load_team(team_id) is None

    def test_list_teams_skips_team_with_undecodable_bytes(
        self, yaml_store: YamlEventStore, tmp_path: Path
    ) -> None:
        """An unreadable ``team.yaml`` is skipped by ``list_teams``, not raised out of.

        Filtered and unfiltered alike: one bad file on disk must never
        break a whole list call for the teams that are readable.
        """
        good = make_process(status=TeamStatus.RUNNING)
        yaml_store.save_team(good)
        bad_dir = tmp_path / str(uuid.uuid4())
        bad_dir.mkdir()
        (bad_dir / "team.yaml").write_bytes(b"status: \xff\xfe\x00running\n")

        assert [p.team_id for p in yaml_store.list_teams()] == [good.team_id]
        assert [p.team_id for p in yaml_store.list_teams(status=TeamStatus.RUNNING)] == [
            good.team_id
        ]

    def test_load_events_returns_empty_for_corrupted_yaml(self, tmp_path: Path) -> None:
        """Corrupted ``events.yaml`` returns empty list instead of raising."""
        store = YamlEventStore(tmp_path)
        team_id = uuid.uuid4()
        team_dir = tmp_path / str(team_id)
        team_dir.mkdir()
        (team_dir / "events.yaml").write_text("{{invalid: yaml: [}")
        assert store.load_events(team_id) == []

    def test_load_agent_states_skips_corrupted_files(self, tmp_path: Path) -> None:
        """A corrupted state file is skipped; valid ones are still loaded."""
        store = YamlEventStore(tmp_path)
        team_id = uuid.uuid4()
        # Save a valid state first
        snap = make_agent_state_snapshot(team_id=team_id, agent_id="good-agent")
        store.save_agent_state(snap)
        # Write a corrupted state file
        states_dir = tmp_path / str(team_id) / "states"
        (states_dir / "bad-agent.yaml").write_text("{{invalid: yaml: [}")
        loaded = store.load_agent_states(team_id)
        assert len(loaded) == 1
        assert loaded[0].agent_id == "good-agent"
