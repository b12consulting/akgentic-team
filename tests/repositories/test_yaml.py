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
* Corrupted-file resilience for teams, states and cards (YAML parser
  errors → ``None`` / skip rather than raise — this is the YamlEventStore
  contract). The EVENT LOG is the deliberate exception: a log that exists
  and will not parse raises ``EventLogUnreadableError``, because ``[]``
  there is read by every caller as "this team has no history".
* The writer and the reader speak one dialect — a value the safe loader
  could not construct fails the write instead of landing on disk.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml
from akgentic.core.agent import Akgent
from akgentic.core.agent_card import AgentCard
from akgentic.core.agent_state import BaseState
from akgentic.core.messages.message import Message
from pydantic import BaseModel, Field

from akgentic.team.models import PersistedEvent, Process, TeamStatus
from akgentic.team.ports import EventLogUnreadableError, EventNotFoundError
from akgentic.team.projection import hash_agent_card, storable_agent_card
from akgentic.team.repositories.yaml import (
    CARD_ENVELOPE_KEY,
    CARD_FIRST_SEEN_KEY,
    CARDS_DIRNAME,
    YamlEventStore,
)

if TYPE_CHECKING:
    from akgentic.team.ports import EventStore

from tests.models.conftest import (
    AcmeTeamMetadata,
    make_agent_card,
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


def _card_fixture() -> AgentCard:
    """An AgentCard whose ``agent_class`` resolves on the way back out of YAML.

    ``AgentCard`` resolves ``agent_class`` only when ``config`` arrives as a
    dict — that is, on the way out of storage — so the suite's default
    ``tests.fixtures.MockAgent`` placeholder constructs fine and then fails to
    read back. The card store reads cards back.
    """
    return make_agent_card(name="lead", role="Lead", agent_class=Akgent)


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

        matched = yaml_store.list_teams(metadata={"tenant": ["acme"]})
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
            status=TeamStatus.RUNNING, metadata={"tenant": ["acme"]}
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

        result = yaml_store.list_teams(metadata={"tenant": ["acme"]})
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
        assert yaml_store.list_teams(metadata={"tenant": ["acme"]}) == []

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

        filtered = yaml_store.list_teams(metadata={"tenant": ["acme"]})
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

    def test_an_unmigrated_team_yaml_logs_the_legible_reason(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AC 13: the log line is the only place a resumer learns what happened.

        ``load_team`` swallows any ``ValueError`` from validation into a
        log-and-skip, so the caller sees ``None`` either way. The ``caplog``
        assertion is the point of this test, not decoration: without the
        before-validator the same line carries Pydantic's generic
        "Field required" for ``entry_point``, which names neither the cause nor
        the remedy.
        """
        store = YamlEventStore(tmp_path)
        team_id = uuid.uuid4()
        team_dir = tmp_path / str(team_id)
        team_dir.mkdir()
        (team_dir / "team.yaml").write_text(
            yaml.safe_dump(
                {
                    "team_id": str(team_id),
                    "team_card": {"name": "pre-projection-team"},
                    "status": TeamStatus.STOPPED.value,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            )
        )

        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="akgentic.team.repositories.yaml"):
            assert store.load_team(team_id) is None

        corrupted = [r for r in caplog.records if "Corrupted team.yaml" in r.getMessage()]
        assert corrupted
        assert any("predates the structural projection" in r.getMessage() for r in corrupted)

    def test_an_unmigrated_team_yaml_does_not_break_list_teams(self, tmp_path: Path) -> None:
        """One unmigrated document must not take the whole listing down.

        This is why the guard raises ``ValueError`` rather than something that
        escapes the store's handler: ``list_teams`` walks every team directory
        through the same validation, so an escaping error would turn a one-team
        problem into a broken listing for the whole store.

        Pinned for YAML, and Mongo catches the same pair — Postgres validates
        inline and has no such handler, which is a backend-parity gap recorded
        in ``backlog.md`` rather than something this story changes.
        """
        store = YamlEventStore(tmp_path)
        healthy = make_process()
        store.save_team(healthy)

        stale_id = uuid.uuid4()
        stale_dir = tmp_path / str(stale_id)
        stale_dir.mkdir()
        (stale_dir / "team.yaml").write_text(
            yaml.safe_dump(
                {
                    "team_id": str(stale_id),
                    "team_card": {"name": "pre-projection-team"},
                    "status": TeamStatus.STOPPED.value,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                }
            )
        )

        listed = store.list_teams()
        assert [p.team_id for p in listed] == [healthy.team_id]

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

    def test_load_events_raises_for_corrupted_yaml(self, tmp_path: Path) -> None:
        """Corrupted ``events.yaml`` raises instead of answering with an empty list.

        The inverse of what this test used to assert. Its old docstring —
        "returns empty list instead of raising" — was the defect written down as
        a specification: a 186 KB log of 40 events answered ``[]``, the restorer
        found no orchestrator in it, and the operator was handed an error about
        the orchestrator for a fault in the event store.
        """
        store = YamlEventStore(tmp_path)
        team_id = uuid.uuid4()
        team_dir = tmp_path / str(team_id)
        team_dir.mkdir()
        (team_dir / "events.yaml").write_text("{{invalid: yaml: [}")
        with pytest.raises(EventLogUnreadableError, match=str(team_id)):
            store.load_events(team_id)

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


class TestYamlMetadataPredicateConstruction:
    """What the walk is handed, not only what it returns.

    A result-level assertion cannot tell "no metadata predicate" from "a
    predicate that happens to match everything", so the empty-term rule is
    pinned on the argument ``_matches`` actually receives.
    """

    @pytest.mark.parametrize(
        "metadata",
        [
            None,
            {},
            {"tenant": []},
            {"tenant": [""]},
            {"tenant": ["", ""]},
            {"tenant": [""], "case_ref": []},
        ],
        ids=[
            "none",
            "empty-dict",
            "empty-term-list",
            "one-blank-term",
            "two-blank-terms",
            "blank-term-and-empty-list",
        ],
    )
    def test_no_effective_term_reaches_the_walk_with_no_predicate(
        self,
        yaml_store: YamlEventStore,
        monkeypatch: pytest.MonkeyPatch,
        metadata: dict[str, list[str]] | None,
    ) -> None:
        """Every "no effective term" spelling leaves ``_matches`` with no prefix.

        ``{"tenant": []}`` and ``{"tenant": [""]}`` are the ones an outer
        truthiness gate on ``metadata`` misses — the mapping itself is truthy.
        """
        acme = make_indexed_process(AcmeTeamMetadata(tenant="acme"))
        bare = make_process()
        yaml_store.save_team(acme)
        yaml_store.save_team(bare)

        seen: list[list[list[str]]] = []
        original = YamlEventStore._matches

        def spy(
            data: object, user_id: object, status: object, groups: list[list[str]]
        ) -> bool:
            seen.append(groups)
            return bool(original(data, user_id, status, groups))  # type: ignore[arg-type]

        monkeypatch.setattr(YamlEventStore, "_matches", staticmethod(spy))

        result = yaml_store.list_teams(metadata=metadata)

        assert seen, "the walk must have run at all"
        assert all(groups == [] for groups in seen), seen
        assert {p.team_id for p in result} == {acme.team_id, bare.team_id}

    def test_a_real_term_does_reach_the_walk(
        self, yaml_store: YamlEventStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The mirror image, so the spec above cannot pass by spying on nothing."""
        yaml_store.save_team(make_indexed_process(AcmeTeamMetadata(tenant="acme")))

        seen: list[list[list[str]]] = []
        original = YamlEventStore._matches

        def spy(
            data: object, user_id: object, status: object, groups: list[list[str]]
        ) -> bool:
            seen.append(groups)
            return bool(original(data, user_id, status, groups))  # type: ignore[arg-type]

        monkeypatch.setattr(YamlEventStore, "_matches", staticmethod(spy))

        yaml_store.list_teams(metadata={"tenant": ["AcM"], "case_ref": ["C-"]})

        # One GROUP per key — not a flat list. The grouping is what carries the
        # combination rule down to the predicate: terms inside a group OR,
        # groups AND.
        assert seen == [[["tenant|acm"], ["case_ref|c-"]]]

    def test_a_key_whose_terms_render_away_reaches_the_walk_as_no_group(
        self, yaml_store: YamlEventStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An emptied key drops out entirely — not as an empty group.

        An empty group is the YAML-side twin of Mongo's ``$or: []``. Here it
        would not raise, it would quietly evaluate ``any(...)`` over nothing —
        which is ``False`` — and so match no team at all, turning "I sent a
        blank for this facet" into "return nothing".
        """
        yaml_store.save_team(make_indexed_process(AcmeTeamMetadata(tenant="acme")))

        seen: list[list[list[str]]] = []
        original = YamlEventStore._matches

        def spy(
            data: object, user_id: object, status: object, groups: list[list[str]]
        ) -> bool:
            seen.append(groups)
            return bool(original(data, user_id, status, groups))  # type: ignore[arg-type]

        monkeypatch.setattr(YamlEventStore, "_matches", staticmethod(spy))

        yaml_store.list_teams(metadata={"tenant": [], "case_ref": ["C-"], "other": [""]})

        assert seen == [[["case_ref|c-"]]]
        assert all([] not in groups for groups in seen), seen

    def test_a_bare_string_is_rejected_before_the_directory_is_read(
        self, tmp_path: Path
    ) -> None:
        """The guard runs even when the data directory does not exist.

        The early return for a missing directory is exactly where a lazily
        rendered term would slip past and answer ``[]`` instead of raising.
        """
        store = YamlEventStore(tmp_path / "does-not-exist")

        with pytest.raises(TypeError, match="tenant"):
            store.list_teams(metadata={"tenant": "acme"})  # type: ignore[dict-item]


class TestYamlAgentCardStoreLayout:
    """YAML-only invariants of the content-addressed card store."""

    def test_cards_live_beside_the_team_directories_not_inside_them(
        self, yaml_store: YamlEventStore, tmp_path: Path
    ) -> None:
        """FR13 is unsatisfiable by construction if the cards sit inside a team.

        ``delete_team`` removes a team directory with ``shutil.rmtree``, so a
        card store nested under one would go with the first team that
        referenced it — however carefully ``delete_team`` were written.
        """
        card = _card_fixture()
        process = make_process()

        yaml_store.save_agent_cards([card])
        yaml_store.save_team(process)

        cards_dir = tmp_path / CARDS_DIRNAME
        assert cards_dir.is_dir()
        assert cards_dir.parent == tmp_path
        assert not (tmp_path / str(process.team_id) / CARDS_DIRNAME).exists()

    def test_each_card_is_one_file_named_by_its_hash(
        self, yaml_store: YamlEventStore, tmp_path: Path
    ) -> None:
        card = _card_fixture()
        yaml_store.save_agent_cards([card])

        files = sorted((tmp_path / CARDS_DIRNAME).iterdir())
        assert [f.name for f in files] == [f"{hash_agent_card(card)}.yaml"]

    def test_a_re_save_rewrites_rather_than_appends(
        self, yaml_store: YamlEventStore, tmp_path: Path
    ) -> None:
        """A content-addressed file holds the bytes its name promises, once."""
        card = _card_fixture()
        yaml_store.save_agent_cards([card])
        first = (tmp_path / CARDS_DIRNAME / f"{hash_agent_card(card)}.yaml").read_text()

        yaml_store.save_agent_cards([card])
        second = (tmp_path / CARDS_DIRNAME / f"{hash_agent_card(card)}.yaml").read_text()

        assert first == second

    def test_saving_no_cards_creates_no_directory(
        self, yaml_store: YamlEventStore, tmp_path: Path
    ) -> None:
        yaml_store.save_agent_cards([])
        assert not (tmp_path / CARDS_DIRNAME).exists()

    # --- list_teams must not warn about the card directory ------------------

    def test_list_teams_does_not_warn_about_the_card_directory(
        self,
        yaml_store: YamlEventStore,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The warning fires per CALL, not per team — it would log forever.

        Asserting on the returned teams alone passes whether or not the warning
        is emitted, which is exactly why this asserts on the log record.
        """
        yaml_store.save_agent_cards([_card_fixture()])
        yaml_store.save_team(make_process())

        with caplog.at_level(logging.WARNING, logger="akgentic.team.repositories.yaml"):
            teams = yaml_store.list_teams()

        assert len(teams) == 1
        assert [r for r in caplog.records if "non-team directory" in r.getMessage()] == []

    def test_a_genuinely_unexpected_directory_still_warns(
        self,
        yaml_store: YamlEventStore,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The skip is by name, not a blanket silencing of the warning."""
        (tmp_path / "not-a-team").mkdir()

        with caplog.at_level(logging.WARNING, logger="akgentic.team.repositories.yaml"):
            yaml_store.list_teams()

        assert [r for r in caplog.records if "non-team directory" in r.getMessage()]

    def test_the_card_directory_is_not_mistaken_for_a_team(
        self, yaml_store: YamlEventStore
    ) -> None:
        yaml_store.save_agent_cards([_card_fixture()])
        assert yaml_store.list_teams() == []

    # --- corrupted card files -----------------------------------------------

    def test_a_corrupted_card_file_is_skipped_not_raised(
        self,
        yaml_store: YamlEventStore,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """It surfaces as FR14's loud failure at resolution, not as an escape.

        The same treatment corrupted team and state files already get: logged,
        absent from the result. Raising out of the store would bypass the error
        that names the role.
        """
        card = _card_fixture()
        card_hash = hash_agent_card(card)
        yaml_store.save_agent_cards([card])
        (tmp_path / CARDS_DIRNAME / f"{card_hash}.yaml").write_text("{[not: yaml")

        with caplog.at_level(logging.ERROR, logger="akgentic.team.repositories.yaml"):
            loaded = yaml_store.load_agent_cards([card_hash])

        assert loaded == {}
        assert [r for r in caplog.records if "corrupted agent card" in r.getMessage()]

    # --- the envelope, and the bare files that predate it --------------------

    def test_a_saved_card_file_is_an_envelope_around_the_card(
        self, yaml_store: YamlEventStore, tmp_path: Path
    ) -> None:
        """The card sits UNDER a key, with the stamp beside it, not merged into it.

        Merging the stamp into the card's own mapping would put a key on disk
        that ``AgentCard`` does not declare, and ``load_agent_cards`` validates
        that mapping — so the blob would stop loading, on every backend at once.
        """
        card = _card_fixture()
        yaml_store.save_agent_cards([card])

        on_disk = yaml.safe_load(
            (tmp_path / CARDS_DIRNAME / f"{hash_agent_card(card)}.yaml").read_text()
        )

        assert set(on_disk) == {CARD_ENVELOPE_KEY, CARD_FIRST_SEEN_KEY}
        assert on_disk[CARD_ENVELOPE_KEY] == storable_agent_card(card).model_dump()
        assert isinstance(on_disk[CARD_FIRST_SEEN_KEY], datetime)
        assert on_disk[CARD_FIRST_SEEN_KEY].tzinfo is not None

    def test_a_re_save_carries_the_stamp_on_disk_forward_verbatim(
        self, yaml_store: YamlEventStore, tmp_path: Path
    ) -> None:
        """``_atomic_write`` rewrites the file WHOLESALE, so the save must read first.

        This is the YAML expression of ``$setOnInsert``. Asserted on the bytes on
        disk rather than through the enumeration, because the enumeration reads
        whatever the save wrote and cannot tell a carried-forward stamp from one
        the reader invented.
        """
        card = _card_fixture()
        card_path = tmp_path / CARDS_DIRNAME / f"{hash_agent_card(card)}.yaml"
        yaml_store.save_agent_cards([card])
        first = yaml.safe_load(card_path.read_text())[CARD_FIRST_SEEN_KEY]

        yaml_store.save_agent_cards([card])

        assert yaml.safe_load(card_path.read_text())[CARD_FIRST_SEEN_KEY] == first

    def test_a_bare_pre_existing_card_file_loads_without_a_complaint(
        self,
        yaml_store: YamlEventStore,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Every card file in every deployment today is a BARE card, not an envelope.

        A reader that assumed the envelope would turn all of them into "corrupted
        document, skipped" — FR14's loud failure arriving for a reason that is
        not the card's fault. The ``caplog`` half is the one that catches it: a
        reader that routes legacy files through the corrupted path still returns
        ``{}`` quietly on some spellings and a log is the only witness.
        """
        card = _card_fixture()
        card_hash = hash_agent_card(card)
        cards_dir = tmp_path / CARDS_DIRNAME
        cards_dir.mkdir(parents=True, exist_ok=True)
        # Exactly the bytes ``save_agent_cards`` wrote before the envelope.
        with open(cards_dir / f"{card_hash}.yaml", "w") as handle:
            yaml.dump(storable_agent_card(card).model_dump(), handle, default_flow_style=False)

        with caplog.at_level(logging.WARNING, logger="akgentic.team.repositories.yaml"):
            loaded = yaml_store.load_agent_cards([card_hash])
            entries = yaml_store.list_agent_card_entries()

        assert set(loaded) == {card_hash}
        assert hash_agent_card(loaded[card_hash]) == card_hash
        assert [(e.card_hash, e.first_seen_at) for e in entries] == [(card_hash, None)]
        assert caplog.records == [], [r.getMessage() for r in caplog.records]

    def test_a_file_the_store_did_not_write_is_not_an_entry(
        self, yaml_store: YamlEventStore, tmp_path: Path
    ) -> None:
        """The enumeration applies ``load_agent_cards``' own name guard.

        A stray file is not a blob this store holds, and handing its name to a
        consumer that deletes what nothing claims is how an unrelated file gets
        reclaimed.
        """
        yaml_store.save_agent_cards([_card_fixture()])
        (tmp_path / CARDS_DIRNAME / "notes.yaml").write_text("just: a file\n")
        (tmp_path / CARDS_DIRNAME / "README.md").write_text("not yaml at all\n")

        entries = yaml_store.list_agent_card_entries()

        assert [e.card_hash for e in entries] == [hash_agent_card(_card_fixture())]

    def test_an_unparseable_card_file_is_skipped_by_the_enumeration(
        self,
        yaml_store: YamlEventStore,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Logged and skipped, as everywhere else in this module."""
        card = _card_fixture()
        yaml_store.save_agent_cards([card])
        (tmp_path / CARDS_DIRNAME / f"{hash_agent_card(card)}.yaml").write_text("{[not: yaml")

        with caplog.at_level(logging.ERROR, logger="akgentic.team.repositories.yaml"):
            entries = yaml_store.list_agent_card_entries()

        assert entries == []
        assert [r for r in caplog.records if "unreadable agent card" in r.getMessage()]

    def test_enumerating_an_absent_card_directory_is_empty_not_an_error(
        self, yaml_store: YamlEventStore, tmp_path: Path
    ) -> None:
        """A store that has never saved a card has no directory to list."""
        assert not (tmp_path / CARDS_DIRNAME).exists()
        assert yaml_store.list_agent_card_entries() == []

    def test_a_save_over_an_unreadable_file_heals_it_without_inventing_a_stamp(
        self,
        yaml_store: YamlEventStore,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A half-written file heals on the next save, but its age stays unknown.

        The file exists, so this is not a first sight and the save must not claim
        one; the stamp it held (if any) is unreadable, so the honest answer is
        ``None``. Stamping ``now`` here would date the blob from the crash that
        damaged it.
        """
        card = _card_fixture()
        card_hash = hash_agent_card(card)
        cards_dir = tmp_path / CARDS_DIRNAME
        cards_dir.mkdir(parents=True, exist_ok=True)
        (cards_dir / f"{card_hash}.yaml").write_text("{[not: yaml")

        with caplog.at_level(logging.WARNING, logger="akgentic.team.repositories.yaml"):
            yaml_store.save_agent_cards([card])

        assert set(yaml_store.load_agent_cards([card_hash])) == {card_hash}
        assert [(e.card_hash, e.first_seen_at) for e in yaml_store.list_agent_card_entries()] == [
            (card_hash, None)
        ]
        assert [r for r in caplog.records if "Could not read existing card file" in r.getMessage()]

    def test_loading_from_a_store_with_no_card_directory_returns_empty(
        self, yaml_store: YamlEventStore
    ) -> None:
        assert yaml_store.load_agent_cards(["0" * 64]) == {}

    def test_a_hash_that_is_not_a_hex_digest_never_reaches_the_filesystem(
        self,
        yaml_store: YamlEventStore,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """This backend turns a hash into a PATH, so a bad key must not name a file.

        The hashes reaching ``load_agent_cards`` come off a persisted
        ``Process``, and ``../`` is a perfectly valid ``str``. A SHA-256 hex
        digest is the only key the store ever writes, so anything else is a
        clean miss rather than a stray read outside the card directory.
        """
        # A genuinely VALID card outside the store, so the assertion below fails
        # on the escape itself rather than on the escaped file being unparseable.
        outside = tmp_path / "escaped.yaml"
        outside.write_text(yaml.dump(_card_fixture().model_dump()))
        yaml_store.save_agent_cards([_card_fixture()])

        with caplog.at_level(logging.ERROR, logger="akgentic.team.repositories.yaml"):
            loaded = yaml_store.load_agent_cards([f"../{outside.stem}", "not-a-hash"])

        assert loaded == {}
        assert len(
            [r for r in caplog.records if "malformed agent card hash" in r.getMessage()]
        ) == 2

    def test_a_real_digest_is_still_resolved(self, yaml_store: YamlEventStore) -> None:
        """The guard rejects by shape, and a genuine key has that shape."""
        card = _card_fixture()
        yaml_store.save_agent_cards([card])

        assert set(yaml_store.load_agent_cards([hash_agent_card(card)])) == {
            hash_agent_card(card)
        }


class _MatchKind(StrEnum):
    """A ``StrEnum`` standing in for the one that triggered this in the field.

    The specific enum is incidental and already fixed at its source. What
    matters is the shape: a value that is an instance of a class PyYAML has no
    safe representer for, which the unsafe dumper happily writes as
    ``!!python/object/apply:`` and the safe loader then refuses to construct.
    """

    EXACT = "exact"


class _PlainNested(BaseModel):
    """A plain ``BaseModel`` — deliberately NOT a ``SerializableBaseModel``.

    That distinction is the leak. ``serialize()`` converts UUID, datetime,
    ``ActorAddress`` and bytes to plain forms, but for a nested ``BaseModel`` it
    calls ``value.model_dump()`` in *python* mode and returns the result as-is,
    so anything exotic inside survives into the dict handed to the dumper.
    """

    kind: _MatchKind = _MatchKind.EXACT


class _LeakyMessage(Message):
    """A Message whose ``model_dump()`` genuinely carries a non-plain value."""

    payload: _PlainNested = Field(default_factory=_PlainNested)


class _LeakyState(BaseState):
    """The same leak, on the other write path.

    ``save_event`` is not the only writer: ``_atomic_write`` backs team.yaml,
    ``states/`` and ``agent_cards/``, and AC#1 covers it too. A snapshot is the
    shortest public route to it that can carry a value the safe dumper refuses.
    """

    payload: _PlainNested = Field(default_factory=_PlainNested)


def _leaky_event(team_id: uuid.UUID, sequence: int = 1) -> PersistedEvent:
    return PersistedEvent(
        team_id=team_id,
        sequence=sequence,
        event=_LeakyMessage(),
        timestamp=datetime.now(UTC),
    )


_PYTHON_TAGGED_LOG = """---
team_id: 6a3d1f9c-6f1e-4a2e-9a1a-2d4e6f8a0b1c
sequence: 1
event: !!python/object/apply:builtins.getattr
- !!python/name:builtins.str ''
- upper
timestamp: '2026-09-15T10:00:00+00:00'
"""
"""The exact shape the field logs carried: a tag ``safe_load_all`` refuses.

Written by the unsafe dumper, unreadable by the safe loader — the asymmetry this
story closes from the writing side.
"""


class TestTheWriterRefusesWhatTheReaderWouldRefuse:
    """The write path and the read path must agree on one YAML dialect.

    ``yaml.dump`` emits a ``!!python/`` tag for any value it has no representer
    for; ``yaml.safe_load_all`` refuses to construct one. The pair could write a
    file it could not read back, and every layer above compounded that rather
    than containing it: the whole log answered ``[]``, the restorer reported a
    missing orchestrator, and the operator was pointed at the wrong subsystem.
    """

    # --- (a) Write-time refusal ---------------------------------------------

    def test_the_leak_this_guard_stands_on_is_real(self) -> None:
        """Pin the premise, so the refusal guard below cannot go vacuous.

        If ``serialize()`` ever learns to flatten a nested plain ``BaseModel``,
        this fails first and says so — rather than leaving the refusal guard
        passing because there is nothing left to refuse.
        """
        dumped = _leaky_event(uuid.uuid4()).model_dump()

        leaked = dumped["event"]["payload"]["kind"]
        assert type(leaked) is _MatchKind, (
            "the nested plain BaseModel no longer leaks a non-plain value; "
            "rebuild this guard on a shape that still does"
        )

    def test_an_unrepresentable_value_fails_the_write(
        self, yaml_store: YamlEventStore, tmp_path: Path
    ) -> None:
        """``save_event`` raises rather than writing what it cannot read back."""
        team_id = uuid.uuid4()

        with pytest.raises(yaml.YAMLError):
            yaml_store.save_event(_leaky_event(team_id))

        assert not (tmp_path / str(team_id) / "events.yaml").exists()

    def test_a_refused_write_leaves_the_log_byte_identical(
        self, yaml_store: YamlEventStore, tmp_path: Path
    ) -> None:
        """AC#2's negative, and the reason the dump happens before the open.

        Dumping into an already-open handle would have appended ``"---\\n"`` —
        and possibly half a document — before the refusal, leaving a stray
        separator on an append-only log that no later read can distinguish from
        a real one.
        """
        team_id = uuid.uuid4()
        good = make_persisted_event(team_id=team_id, sequence=1)
        yaml_store.save_event(good)

        events_path = tmp_path / str(team_id) / "events.yaml"
        before = events_path.read_bytes()

        with pytest.raises(yaml.YAMLError):
            yaml_store.save_event(_leaky_event(team_id, sequence=2))

        assert events_path.read_bytes() == before
        assert [e.sequence for e in yaml_store.load_events(team_id)] == [1]

    def test_no_file_this_store_writes_carries_a_python_tag(
        self, yaml_store: YamlEventStore, tmp_path: Path
    ) -> None:
        """AC#1 across all four file kinds, asserted on the bytes on disk.

        ``_atomic_write`` backs team.yaml, states/ and agent_cards/; ``save_event``
        backs events.yaml. Both dialects are checked where it is observable.
        """
        process = make_process(status=TeamStatus.RUNNING)
        yaml_store.save_team(process)
        yaml_store.save_event(make_persisted_event(team_id=process.team_id, sequence=1))
        yaml_store.save_agent_state(
            make_agent_state_snapshot(team_id=process.team_id, agent_id="a1")
        )
        yaml_store.save_agent_cards([_card_fixture()])

        written = sorted(tmp_path.rglob("*.yaml"))
        assert len(written) == 4, [str(p) for p in written]
        for path in written:
            assert "!!python/" not in path.read_text(), path

    def test_the_atomic_write_path_refuses_it_too(
        self, yaml_store: YamlEventStore, tmp_path: Path
    ) -> None:
        """AC#1's other half: ``_atomic_write``, not ``save_event``.

        The guard above cannot see which dumper ``_atomic_write`` uses — every
        payload the suite writes through it is representable, and the two
        dumpers agree on all of those. Measured: reverting that one call to
        ``yaml.dump`` and leaving ``save_event`` alone left the whole package
        suite green. So this write site had no guard at all, for exactly the
        reason AC#8 gives about the round-trip one.

        It also pins what Task 3 asserted only in a prose comment: the existing
        ``except BaseException`` leaves the previous good document in place and
        unlinks the temp, so a refusal costs nothing that was already on disk.
        """
        team_id = uuid.uuid4()
        yaml_store.save_agent_state(make_agent_state_snapshot(team_id=team_id, agent_id="a1"))
        state_path = tmp_path / str(team_id) / "states" / "a1.yaml"
        before = state_path.read_bytes()

        with pytest.raises(yaml.YAMLError):
            yaml_store.save_agent_state(
                make_agent_state_snapshot(team_id=team_id, agent_id="a1", state=_LeakyState())
            )

        assert state_path.read_bytes() == before
        assert not list(state_path.parent.glob("*.tmp"))

    # --- (b) Round trip — the guard Task 9 mutates ---------------------------

    def test_anything_the_writer_accepts_the_reader_reads_back(
        self, yaml_store: YamlEventStore
    ) -> None:
        """The story's title, stated as one falsifiable property.

        Not "every event survives" — an event carrying an unrepresentable value
        is *supposed* to be refused, and a guard that only writes representable
        payloads cannot tell the two dumpers apart: they agree on everything a
        plain ``UserMessage`` contains. That guard passes with ``yaml.dump``
        restored, which makes it a test of the model rather than of the pair.

        The property that does separate them is the asymmetry itself: **a write
        the store ACCEPTS must be a write the store can read back.** Refusing is
        a legal outcome; accepting-then-failing-to-read is not. Restore
        ``yaml.dump`` and the leaky event is accepted, the log gains a
        ``!!python/`` tag, and ``load_events`` raises instead of returning it.
        """
        team_id = uuid.uuid4()
        candidates = [
            make_persisted_event(team_id=team_id, sequence=1),
            _leaky_event(team_id, sequence=2),
            make_persisted_event(team_id=team_id, sequence=3),
        ]

        accepted: list[int] = []
        for event in candidates:
            try:
                yaml_store.save_event(event)
            except yaml.YAMLError:
                # Refused at write time, while the data was still in hand.
                continue
            accepted.append(event.sequence)

        # Whatever the store took, it must hand back — through the real reader.
        assert [e.sequence for e in yaml_store.load_events(team_id)] == accepted
        assert accepted == [1, 3]

    def test_written_events_read_back_through_the_real_pair(
        self, yaml_store: YamlEventStore
    ) -> None:
        """Write through ``save_event``, read through ``load_events``, lose nothing.

        Deliberately the real writer and the real reader, never a model
        comparison: the defect lived *between* them, in the dialect mismatch, and
        a model round-trip cannot see it.
        """
        team_id = uuid.uuid4()
        written = [make_persisted_event(team_id=team_id, sequence=n) for n in (1, 2, 3)]
        for event in written:
            yaml_store.save_event(event)

        loaded = yaml_store.load_events(team_id)

        assert [e.sequence for e in loaded] == [1, 2, 3]
        assert [str(e.event.id) for e in loaded] == [str(e.event.id) for e in written]

    # --- (c) A log that will not parse raises --------------------------------

    @pytest.mark.parametrize(
        "content",
        [_PYTHON_TAGGED_LOG, "{{invalid: yaml: [}"],
        ids=["python-tag", "malformed"],
    )
    def test_an_unparseable_log_raises_on_both_paths(self, tmp_path: Path, content: str) -> None:
        """With a cursor and without one — the no-cursor path is the resume path.

        ``ports.py`` already stated the rule for the cursor path ("MUST fail
        loudly, never silently degrade"). The no-cursor path did exactly what
        that forbids, at the one call site where it matters.
        """
        store = YamlEventStore(tmp_path)
        team_id = uuid.uuid4()
        team_dir = tmp_path / str(team_id)
        team_dir.mkdir()
        (team_dir / "events.yaml").write_text(content)

        with pytest.raises(EventLogUnreadableError, match=str(team_id)):
            store.load_events(team_id)
        with pytest.raises(EventLogUnreadableError, match=str(team_id)):
            store.load_events(team_id, after_event_id=uuid.uuid4())

    def test_an_unparseable_log_is_not_a_stale_cursor(self, tmp_path: Path) -> None:
        """The new error must NOT be catchable as ``EventNotFoundError``.

        The infra read path reads ``EventNotFoundError`` as "your cursor is
        stale, resync from the top". An unreadable log answering with that type
        sends a client into a resync loop against a log that will never parse —
        the failure mode this story exists to remove, wearing a different mask.
        """
        store = YamlEventStore(tmp_path)
        team_id = uuid.uuid4()
        team_dir = tmp_path / str(team_id)
        team_dir.mkdir()
        (team_dir / "events.yaml").write_text(_PYTHON_TAGGED_LOG)

        assert not issubclass(EventLogUnreadableError, EventNotFoundError)
        assert not issubclass(EventLogUnreadableError, LookupError)
        assert not issubclass(EventLogUnreadableError, ValueError)
        with pytest.raises(EventLogUnreadableError):
            store.load_events(team_id, after_event_id=uuid.uuid4())

    def test_get_max_sequence_propagates_rather_than_answering_zero(self, tmp_path: Path) -> None:
        """Answering 0 would restart numbering over a log still on disk."""
        store = YamlEventStore(tmp_path)
        team_id = uuid.uuid4()
        team_dir = tmp_path / str(team_id)
        team_dir.mkdir()
        (team_dir / "events.yaml").write_text(_PYTHON_TAGGED_LOG)

        with pytest.raises(EventLogUnreadableError):
            store.get_max_sequence(team_id)

    # --- (d) Regression surface, both halves ---------------------------------

    def test_an_absent_log_is_still_an_empty_log(
        self, yaml_store: YamlEventStore, tmp_path: Path
    ) -> None:
        """An absent file and an unparseable one must not share an answer."""
        team_id = uuid.uuid4()
        (tmp_path / str(team_id)).mkdir()

        assert yaml_store.load_events(team_id) == []
        with pytest.raises(EventNotFoundError):
            yaml_store.load_events(team_id, after_event_id=uuid.uuid4())

    def test_an_intact_log_still_loads_every_event_in_sequence_order(
        self, yaml_store: YamlEventStore
    ) -> None:
        """Written out of order, read back in order, nothing dropped."""
        team_id = uuid.uuid4()
        for sequence in (3, 1, 4, 2):
            yaml_store.save_event(make_persisted_event(team_id=team_id, sequence=sequence))

        assert [e.sequence for e in yaml_store.load_events(team_id)] == [1, 2, 3, 4]

    def test_a_document_that_parses_but_fails_validation_is_still_skipped(
        self, yaml_store: YamlEventStore, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """AC#5: per-document tolerance is deliberate and survives this change.

        One event lost rather than the whole log. This is a *different* fact
        from a log that will not parse, and it keeps its own answer.
        """
        team_id = uuid.uuid4()
        yaml_store.save_event(make_persisted_event(team_id=team_id, sequence=1))
        events_path = tmp_path / str(team_id) / "events.yaml"
        with open(events_path, "a") as handle:
            handle.write("---\nsequence: not-an-int\n")
        yaml_store.save_event(make_persisted_event(team_id=team_id, sequence=2))

        with caplog.at_level(logging.WARNING, logger="akgentic.team.repositories.yaml"):
            loaded = yaml_store.load_events(team_id)

        assert [e.sequence for e in loaded] == [1, 2]
        assert any("Skipping corrupted event" in r.getMessage() for r in caplog.records)
