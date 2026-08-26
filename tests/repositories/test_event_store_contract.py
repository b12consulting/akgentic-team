"""Shared parametrized contract suite for ``EventStore`` implementations.

Every behavioural test in this module runs once per backend (yaml,
mongo, postgres) via the ``event_store`` fixture defined in
``conftest.py``. The intent (ADR-15 §9) is that a Protocol drift cannot
land in one backend without breaking the others — this module is the
"non-negotiable first gate" for ``EventStore`` conformance.

Backend-specific assertions (collection names, index specs, exception
TYPES on duplicate keys, on-disk file layout, schema-drift /
payload-authority invariants) stay in the per-backend modules under
``tests/repositories/{yaml,mongo,postgres}/``.
"""

from __future__ import annotations

import uuid

import pytest
from akgentic.core.messages.message import UserMessage

from akgentic.team.models import Process, TeamStatus
from akgentic.team.ports import EventNotFoundError, EventStore
from tests.models.conftest import (
    AcmeTeamMetadata,
    SampleAgentState,
    make_agent_state_snapshot,
    make_indexed_process,
    make_persisted_event,
    make_process,
)


def build_metadata_fixture_set() -> dict[str, Process]:
    """Seed teams for the shared metadata-filter matrix, keyed by label.

    Built through ``make_indexed_process`` so the value and its derived index
    are seeded together exactly as a real write path leaves them — an index
    written by hand would let a query-side folding or escaping bug pass.

    The set is deliberately adversarial. It carries a mixed-case value (the
    casefold), a value holding the separator (the escaping), one value per
    regex and ``LIKE`` metacharacter, and for the two metacharacters that would
    match something if left unescaped a **decoy** the unescaped pattern would
    wrongly reach: ``axb`` for ``a.b`` and ``a_b``, ``5012`` for ``50%``.
    """
    return {
        "azefr": make_indexed_process(AcmeTeamMetadata(tenant="AzeFR")),
        "azejyt": make_indexed_process(AcmeTeamMetadata(tenant="azejyt")),
        "acme": make_indexed_process(AcmeTeamMetadata(tenant="acme", case_ref="C-1")),
        "contoso": make_indexed_process(AcmeTeamMetadata(tenant="contoso", case_ref="C-1")),
        "piped": make_indexed_process(AcmeTeamMetadata(tenant="acme|corp")),
        "dot": make_indexed_process(AcmeTeamMetadata(tenant="a.b")),
        "star": make_indexed_process(AcmeTeamMetadata(tenant="a*b")),
        "caret": make_indexed_process(AcmeTeamMetadata(tenant="a^b")),
        "dollar": make_indexed_process(AcmeTeamMetadata(tenant="a$b")),
        "bracket": make_indexed_process(AcmeTeamMetadata(tenant="a[b")),
        "percent": make_indexed_process(AcmeTeamMetadata(tenant="50%")),
        "underscore": make_indexed_process(AcmeTeamMetadata(tenant="a_b")),
        "backslash": make_indexed_process(AcmeTeamMetadata(tenant="a\\b")),
        "axb": make_indexed_process(AcmeTeamMetadata(tenant="axb")),
        "fifty_twelve": make_indexed_process(AcmeTeamMetadata(tenant="5012")),
        "bare": make_process(),
    }


ALL_LABELS = frozenset(build_metadata_fixture_set())
"""Every seeded label — the answer any call that constrains nothing must give."""

_MetadataCase = tuple[str, dict[str, list[str]] | None, set[str] | frozenset[str]]

METADATA_FILTER_MATRIX: list[_MetadataCase] = [
    # --- AC5: anchored prefix -------------------------------------------
    ("prefix reaches longer values", {"tenant": ["aze"]}, {"azefr", "azejyt"}),
    ("a longer prefix narrows", {"tenant": ["azef"]}, {"azefr"}),
    ("the whole value is still a prefix", {"tenant": ["azefr"]}, {"azefr"}),
    ("not anchored at the start", {"tenant": ["zef"]}, set()),
    ("longer than the stored value", {"tenant": ["azefrx"]}, set()),
    ("a term under another key", {"case_ref": ["aze"]}, set()),
    # --- AC6: the direction is stored-starts-with-term -------------------
    # A reversed predicate satisfies every case above and makes this one red:
    # the crafted term STARTS WITH the stored ``tenant|acme`` entry.
    ("a crafted term cannot span two entries", {"tenant": ["acme|case_ref|C-1"]}, set()),
    # --- AC7: case-insensitive without a case-insensitive query ----------
    ("an upper-case term", {"tenant": ["AZE"]}, {"azefr", "azejyt"}),
    ("a mixed-case term", {"tenant": ["AzE"]}, {"azefr", "azejyt"}),
    ("a mixed-case whole value", {"tenant": ["AzEfR"]}, {"azefr"}),
    # --- AC8: same key ORs, different keys AND ----------------------------
    # The load-bearing case: under the AND rule this returned NOTHING, because
    # no team's tenant can start with both "acme" and "contoso". It is what
    # separates the disjunction from the conjunction, so the OR->AND mutation
    # has something to go red on.
    ("terms for one key OR-combine", {"tenant": ["acme", "contoso"]}, {"acme", "piped", "contoso"}),
    # Terms reaching disjoint parts of the store — a conjunction cannot produce
    # this set at all, whichever way it is written.
    (
        "terms for one key reach disjoint values",
        {"tenant": ["aze", "50"]},
        {"azefr", "azejyt", "percent", "fifty_twelve"},
    ),
    # Nested prefixes: OR and AND happen to agree here, which is precisely why
    # this case cannot be the only multi-term one. Kept as evidence that the
    # redundant spelling is idempotent rather than an error.
    ("nested terms for one key are idempotent", {"tenant": ["ac", "acm"]}, {"acme", "piped"}),
    ("distinct keys AND-combine", {"tenant": ["acme"], "case_ref": ["C-"]}, {"acme"}),
    ("a key pair no team carries", {"tenant": ["acme"], "case_ref": ["C-9"]}, set()),
    # Both rules at once: the tenant terms OR, and the result is then ANDed
    # against the case_ref key. `piped` carries no case_ref and drops out.
    (
        "keys AND while their own terms OR",
        {"tenant": ["acme", "contoso"], "case_ref": ["C-1"]},
        {"acme", "contoso"},
    ),
    # --- AC9: an empty term contributes no constraint --------------------
    ("no metadata at all", None, ALL_LABELS),
    ("an empty mapping", {}, ALL_LABELS),
    ("an empty term list", {"tenant": []}, ALL_LABELS),
    ("a single blank term", {"tenant": [""]}, ALL_LABELS),
    ("two blank terms", {"tenant": ["", ""]}, ALL_LABELS),
    ("a blank term beside an empty list", {"tenant": [""], "case_ref": []}, ALL_LABELS),
    # Under a DISJUNCTION this is the sharp one: a blank surviving into the
    # group would match every entry for the key and widen the answer to
    # everything, instead of merely failing to narrow it.
    ("a blank term does not widen a real one", {"tenant": ["", "aze"]}, {"azefr", "azejyt"}),
    # An emptied key must contribute no arm at all — not an empty disjunction,
    # which MongoDB rejects — while a real key beside it still constrains.
    ("an emptied key beside a real one", {"tenant": [], "case_ref": ["C-1"]}, {"acme", "contoso"}),
    # --- AC11: metacharacters are matched literally ----------------------
    ("the regex dot is literal", {"tenant": ["a.b"]}, {"dot"}),
    ("the regex star is literal", {"tenant": ["a*b"]}, {"star"}),
    ("the regex caret is literal", {"tenant": ["a^b"]}, {"caret"}),
    ("the regex dollar is literal", {"tenant": ["a$b"]}, {"dollar"}),
    ("the regex bracket is literal", {"tenant": ["a[b"]}, {"bracket"}),
    ("the LIKE wildcard is literal", {"tenant": ["50%"]}, {"percent"}),
    ("the LIKE single-char wildcard is literal", {"tenant": ["a_b"]}, {"underscore"}),
    ("a backslash is literal", {"tenant": ["a\\b"]}, {"backslash"}),
    ("the separator is literal", {"tenant": ["acme|corp"]}, {"piped"}),
    # The decoys are reachable, so "not returned above" is evidence and not
    # an artefact of them being absent from the store.
    ("the dot decoy is present", {"tenant": ["axb"]}, {"axb"}),
    ("the wildcard decoy is present", {"tenant": ["5012"]}, {"fifty_twelve"}),
]
"""One shared matrix of (seeded teams x filter input) covering AC5-AC11.

Held as data so the parametrized contract suite and the ``InMemoryEventStore``
conformance suite provably run the SAME cases — ``EventStore`` is not
``@runtime_checkable`` and CI mypy does not reach ``tests/``, so a backend or a
fake left on whole-entry equality has no other gate.
"""

METADATA_FILTER_IDS = [case[0] for case in METADATA_FILTER_MATRIX]


class TestEventStoreContract:
    """Behavioural contract every ``EventStore`` backend must satisfy.

    Each test takes the parametrized ``event_store`` fixture and runs
    once per backend. Test IDs end in ``[yaml]`` / ``[mongo]`` /
    ``[postgres]`` so a per-backend regression is immediately obvious in
    pytest output.
    """

    # --- save_team / load_team round-trip ---------------------------------

    def test_save_and_load_team_round_trip(self, event_store: EventStore) -> None:
        """``save_team(p)`` then ``load_team(p.team_id)`` returns deep-equal Process."""
        process = make_process()
        event_store.save_team(process)
        loaded = event_store.load_team(process.team_id)

        assert loaded is not None
        assert loaded.team_id == process.team_id
        assert loaded.status == process.status
        assert loaded.team_card.name == process.team_card.name
        assert loaded.created_at == process.created_at

    def test_load_team_returns_none_when_missing(self, event_store: EventStore) -> None:
        """Protocol contract: missing team yields ``None``."""
        assert event_store.load_team(uuid.uuid4()) is None

    def test_save_team_is_upsert(self, event_store: EventStore) -> None:
        """``save_team`` is upsert-by-team_id; second insert replaces payload."""
        process = make_process(status=TeamStatus.RUNNING)
        event_store.save_team(process)

        updated = make_process(team_id=process.team_id, status=TeamStatus.STOPPED)
        event_store.save_team(updated)

        loaded = event_store.load_team(process.team_id)
        assert loaded is not None
        assert loaded.status == TeamStatus.STOPPED

    def test_catalog_namespace_round_trips(self, event_store: EventStore) -> None:
        """Story 18.1: ``Process.catalog_namespace`` persists through save/load."""
        process = make_process(catalog_namespace="ns-contract")
        event_store.save_team(process)
        loaded = event_store.load_team(process.team_id)
        assert loaded is not None
        assert loaded.catalog_namespace == "ns-contract"

    def test_catalog_namespace_default_round_trips(self, event_store: EventStore) -> None:
        """Story 18.1: default ``None`` catalog_namespace also survives a round trip."""
        process = make_process()
        event_store.save_team(process)
        loaded = event_store.load_team(process.team_id)
        assert loaded is not None
        assert loaded.catalog_namespace is None

    # --- list_teams -------------------------------------------------------

    def test_list_teams_returns_all(self, event_store: EventStore) -> None:
        """``list_teams`` returns every persisted process (order-independent)."""
        p1 = make_process()
        p2 = make_process()
        p3 = make_process()
        event_store.save_team(p1)
        event_store.save_team(p2)
        event_store.save_team(p3)

        result = event_store.list_teams()
        assert len(result) == 3
        assert {p.team_id for p in result} == {p1.team_id, p2.team_id, p3.team_id}

    def test_list_teams_user_id_none_equals_no_arg(self, event_store: EventStore) -> None:
        """``list_teams(user_id=None)`` is equivalent to ``list_teams()`` (no arg)."""
        p1 = make_process()
        p2 = make_process()
        p3 = make_process()
        event_store.save_team(p1)
        event_store.save_team(p2)
        event_store.save_team(p3)

        no_arg = event_store.list_teams()
        with_none = event_store.list_teams(user_id=None)
        assert {p.team_id for p in with_none} == {p.team_id for p in no_arg}

    def test_list_teams_filters_by_user_id_returns_only_matching(
        self, event_store: EventStore
    ) -> None:
        """``list_teams(user_id=...)`` returns only snapshots whose user_id matches."""
        p1 = make_process(user_id="u1")
        p2 = make_process(user_id="u2")
        p3 = make_process(user_id="u1")
        event_store.save_team(p1)
        event_store.save_team(p2)
        event_store.save_team(p3)

        u1_result = event_store.list_teams(user_id="u1")
        assert {p.team_id for p in u1_result} == {p1.team_id, p3.team_id}
        assert len(u1_result) == 2

        u2_result = event_store.list_teams(user_id="u2")
        assert {p.team_id for p in u2_result} == {p2.team_id}
        assert len(u2_result) == 1

    def test_list_teams_filters_by_user_id_returns_empty_for_unknown(
        self, event_store: EventStore
    ) -> None:
        """``list_teams(user_id="nonexistent")`` returns ``[]`` when nothing matches."""
        event_store.save_team(make_process(user_id="u1"))
        event_store.save_team(make_process(user_id="u1"))

        assert event_store.list_teams(user_id="nonexistent") == []

    def test_list_teams_filters_by_user_id_empty_string_is_literal(
        self, event_store: EventStore
    ) -> None:
        """``user_id=""`` is a literal match, not an alias for "all teams" (ADR-16 §8)."""
        p_u1 = make_process(user_id="u1")
        p_empty = make_process(user_id="")
        event_store.save_team(p_u1)
        event_store.save_team(p_empty)

        result = event_store.list_teams(user_id="")
        assert {p.team_id for p in result} == {p_empty.team_id}
        assert len(result) == 1

    def test_list_teams_status_none_equals_no_arg_and_includes_deleted(
        self, event_store: EventStore
    ) -> None:
        """``status=None`` means *no status filter*, not "not deleted".

        A backend that quietly excluded ``DELETED`` from the unfiltered
        result would move the default result set — the one thing the
        additive parameter must never do. A caller that wants only live
        teams asks for them explicitly.
        """
        running = make_process(status=TeamStatus.RUNNING)
        stopped = make_process(status=TeamStatus.STOPPED)
        deleted = make_process(status=TeamStatus.DELETED)
        for process in (running, stopped, deleted):
            event_store.save_team(process)

        no_arg_ids = {p.team_id for p in event_store.list_teams()}
        with_none_ids = {p.team_id for p in event_store.list_teams(status=None)}

        assert with_none_ids == no_arg_ids
        assert deleted.team_id in with_none_ids
        assert with_none_ids == {running.team_id, stopped.team_id, deleted.team_id}

    def test_list_teams_filters_by_status(self, event_store: EventStore) -> None:
        """``list_teams(status=...)`` returns exactly the snapshots in that state."""
        r1 = make_process(status=TeamStatus.RUNNING)
        r2 = make_process(status=TeamStatus.RUNNING)
        stopped = make_process(status=TeamStatus.STOPPED)
        deleted = make_process(status=TeamStatus.DELETED)
        for process in (r1, r2, stopped, deleted):
            event_store.save_team(process)

        running_result = event_store.list_teams(status=TeamStatus.RUNNING)
        assert {p.team_id for p in running_result} == {r1.team_id, r2.team_id}
        assert all(p.status == TeamStatus.RUNNING for p in running_result)

        deleted_result = event_store.list_teams(status=TeamStatus.DELETED)
        assert {p.team_id for p in deleted_result} == {deleted.team_id}
        assert len(deleted_result) == 1

    def test_list_teams_user_id_still_accepts_a_positional_argument(
        self, event_store: EventStore
    ) -> None:
        """``status`` was appended, not inserted: ``list_teams("u1")`` still works.

        The whole point of appending the parameter is that no existing
        call site moves. Pinned across all three backends because stories
        26.2 / 26.3 / 26.5 rewrite each of these method bodies — reordering
        the parameters, or making them keyword-only, would break positional
        callers silently and no other test would notice.
        """
        p1 = make_process(user_id="u1")
        p2 = make_process(user_id="u2")
        event_store.save_team(p1)
        event_store.save_team(p2)

        positional = event_store.list_teams("u1")
        keyword = event_store.list_teams(user_id="u1")

        assert {p.team_id for p in positional} == {p.team_id for p in keyword}
        assert {p.team_id for p in positional} == {p1.team_id}

    def test_list_teams_user_id_and_status_return_the_intersection(
        self, event_store: EventStore
    ) -> None:
        """``user_id`` and ``status`` together select the intersection.

        Each user owns one running and one stopped team, so every single
        term matches two teams and only the conjunction narrows to one. A
        backend that ORed the terms, or honoured just one of them, would
        return two or three teams here.
        """
        u1_running = make_process(user_id="u1", status=TeamStatus.RUNNING)
        u1_stopped = make_process(user_id="u1", status=TeamStatus.STOPPED)
        u2_running = make_process(user_id="u2", status=TeamStatus.RUNNING)
        u2_stopped = make_process(user_id="u2", status=TeamStatus.STOPPED)
        for process in (u1_running, u1_stopped, u2_running, u2_stopped):
            event_store.save_team(process)

        result = event_store.list_teams(user_id="u1", status=TeamStatus.RUNNING)
        assert {p.team_id for p in result} == {u1_running.team_id}

        stopped = event_store.list_teams(user_id="u2", status=TeamStatus.STOPPED)
        assert {p.team_id for p in stopped} == {u2_stopped.team_id}

    def test_list_teams_user_id_and_status_with_no_common_team_returns_empty(
        self, event_store: EventStore
    ) -> None:
        """A combination no single team satisfies returns ``[]``, never a union.

        ``u2`` owns a team and a running team exists, so both terms match
        something on their own — but no one team matches both. AND yields
        ``[]``; OR would yield two.
        """
        event_store.save_team(make_process(user_id="u1", status=TeamStatus.RUNNING))
        event_store.save_team(make_process(user_id="u2", status=TeamStatus.STOPPED))

        assert event_store.list_teams(user_id="u2", status=TeamStatus.RUNNING) == []

    def test_list_teams_metadata_and_combines_across_keys(
        self, event_store: EventStore
    ) -> None:
        """A two-entry metadata filter requires BOTH entries, not either.

        The partial team carries the ``tenant`` half, so an implementation
        that ORed the entries — or honoured only the first — would return it
        alongside the match.
        """
        both = make_indexed_process(AcmeTeamMetadata(tenant="acme", case_ref="C-1"))
        tenant_only = make_indexed_process(AcmeTeamMetadata(tenant="acme"))
        other = make_indexed_process(AcmeTeamMetadata(tenant="contoso", case_ref="C-1"))
        for process in (both, tenant_only, other):
            event_store.save_team(process)

        result = event_store.list_teams(metadata={"tenant": ["acme"], "case_ref": ["C-1"]})
        assert {p.team_id for p in result} == {both.team_id}

    def test_list_teams_metadata_and_combines_with_user_id(
        self, event_store: EventStore
    ) -> None:
        """``metadata`` and ``user_id`` intersect: each term alone matches two teams."""
        u1_acme = make_indexed_process(AcmeTeamMetadata(tenant="acme"), user_id="u1")
        u1_contoso = make_indexed_process(AcmeTeamMetadata(tenant="contoso"), user_id="u1")
        u2_acme = make_indexed_process(AcmeTeamMetadata(tenant="acme"), user_id="u2")
        for process in (u1_acme, u1_contoso, u2_acme):
            event_store.save_team(process)

        result = event_store.list_teams(user_id="u1", metadata={"tenant": ["acme"]})
        assert {p.team_id for p in result} == {u1_acme.team_id}

    def test_list_teams_metadata_and_combines_with_status(
        self, event_store: EventStore
    ) -> None:
        """``metadata`` and ``status`` intersect, and ``DELETED`` is reachable."""
        running = make_indexed_process(
            AcmeTeamMetadata(tenant="acme"), status=TeamStatus.RUNNING
        )
        deleted = make_indexed_process(
            AcmeTeamMetadata(tenant="acme"), status=TeamStatus.DELETED
        )
        running_contoso = make_indexed_process(
            AcmeTeamMetadata(tenant="contoso"), status=TeamStatus.RUNNING
        )
        for process in (running, deleted, running_contoso):
            event_store.save_team(process)

        result = event_store.list_teams(status=TeamStatus.RUNNING, metadata={"tenant": ["acme"]})
        assert {p.team_id for p in result} == {running.team_id}

        # metadata alone does not implicitly constrain status.
        unconstrained = event_store.list_teams(metadata={"tenant": ["acme"]})
        assert {p.team_id for p in unconstrained} == {running.team_id, deleted.team_id}

    def test_list_teams_all_three_filters_combine_with_and(
        self, event_store: EventStore
    ) -> None:
        """Every filter is an independent conjunct; only the full match survives.

        Each of the three decoys differs from the wanted team in exactly one
        dimension, so every single term still matches three of the four teams
        and only the conjunction narrows to one.
        """
        wanted = make_indexed_process(
            AcmeTeamMetadata(tenant="acme"), user_id="u1", status=TeamStatus.RUNNING
        )
        wrong_metadata = make_indexed_process(
            AcmeTeamMetadata(tenant="contoso"), user_id="u1", status=TeamStatus.RUNNING
        )
        wrong_status = make_indexed_process(
            AcmeTeamMetadata(tenant="acme"), user_id="u1", status=TeamStatus.STOPPED
        )
        wrong_user = make_indexed_process(
            AcmeTeamMetadata(tenant="acme"), user_id="u2", status=TeamStatus.RUNNING
        )
        for process in (wanted, wrong_metadata, wrong_status, wrong_user):
            event_store.save_team(process)

        result = event_store.list_teams(
            user_id="u1", status=TeamStatus.RUNNING, metadata={"tenant": ["acme"]}
        )
        assert {p.team_id for p in result} == {wanted.team_id}

    def test_list_teams_metadata_combination_no_team_satisfies_returns_empty(
        self, event_store: EventStore
    ) -> None:
        """A combination no single team satisfies returns ``[]``, never a union.

        Both entries match something on their own — they just never match the
        same team. AND yields ``[]``; OR would yield two.
        """
        event_store.save_team(make_indexed_process(AcmeTeamMetadata(tenant="acme")))
        event_store.save_team(
            make_indexed_process(AcmeTeamMetadata(tenant="contoso", case_ref="C-9"))
        )

        assert event_store.list_teams(metadata={"tenant": ["acme"], "case_ref": ["C-9"]}) == []

    def test_list_teams_metadata_unknown_key_returns_empty(
        self, event_store: EventStore
    ) -> None:
        """A key no team carries matches nothing — including an unindexed field.

        ``department`` is declared on the metadata model but not marked
        indexed, so it never reaches the derived index. Filtering on it
        matching nothing is what proves the filter reads the index rather
        than the stored value.
        """
        event_store.save_team(
            make_indexed_process(AcmeTeamMetadata(tenant="acme", department="ops"))
        )

        assert event_store.list_teams(metadata={"unknown_key": ["acme"]}) == []
        assert event_store.list_teams(metadata={"department": ["ops"]}) == []

    def test_list_teams_empty_metadata_dict_equals_no_metadata_filter(
        self, event_store: EventStore
    ) -> None:
        """``metadata={}`` is an empty conjunction and matches everything.

        Pinned explicitly so the behaviour is a decision rather than an
        accident: the alternative reading — "only teams with no metadata" —
        is defensible but inconsistent with how ``user_id=None`` and
        ``status=None`` behave.
        """
        acme = make_indexed_process(AcmeTeamMetadata(tenant="acme"))
        bare = make_process()
        event_store.save_team(acme)
        event_store.save_team(bare)

        expected = {acme.team_id, bare.team_id}
        assert {p.team_id for p in event_store.list_teams(metadata={})} == expected
        assert {p.team_id for p in event_store.list_teams(metadata=None)} == expected
        assert {p.team_id for p in event_store.list_teams()} == expected

    def test_list_teams_metadata_does_not_widen_user_id_scope(
        self, event_store: EventStore
    ) -> None:
        """SECURITY: a metadata filter can only narrow, never reach past ``user_id``.

        Both users own a team carrying byte-identical metadata. Metadata is
        caller-supplied and non-secret, so it must never become a way to
        enumerate another owner's teams — the ``user_id`` scope is applied
        server-side and stands regardless of what metadata is asked for.
        """
        u1 = make_indexed_process(AcmeTeamMetadata(tenant="acme"), user_id="u1")
        u2 = make_indexed_process(AcmeTeamMetadata(tenant="acme"), user_id="u2")
        event_store.save_team(u1)
        event_store.save_team(u2)

        result = event_store.list_teams(user_id="u1", metadata={"tenant": ["acme"]})
        assert [p.team_id for p in result] == [u1.team_id]

        # And the unfiltered-by-metadata call is scoped identically, so the
        # metadata term is what changed nothing about visibility.
        assert [p.team_id for p in event_store.list_teams(user_id="u1")] == [u1.team_id]

    def test_list_teams_teams_without_metadata_are_unaffected(
        self, event_store: EventStore
    ) -> None:
        """A team carrying no metadata is returned by any call that does not filter on it.

        This is the backwards-compatibility case: teams persisted before the
        metadata contract existed carry no index and must keep listing
        exactly as they did.
        """
        bare = make_process(user_id="u1", status=TeamStatus.RUNNING)
        event_store.save_team(bare)

        assert [p.team_id for p in event_store.list_teams()] == [bare.team_id]
        assert [p.team_id for p in event_store.list_teams(user_id="u1")] == [bare.team_id]
        assert [p.team_id for p in event_store.list_teams(status=TeamStatus.RUNNING)] == [
            bare.team_id
        ]
        assert event_store.list_teams(metadata={"tenant": ["acme"]}) == []

    def test_list_teams_metadata_value_containing_the_separator_matches_itself(
        self, event_store: EventStore
    ) -> None:
        """Escaping is symmetric: a ``|`` in a value matches, and cannot forge an entry.

        Query terms are built with the same ``make_index_entry`` the
        derivation uses. A second, hand-rolled implementation in the query
        path would drift on exactly this input — the value would stop
        matching itself — and a crafted value could otherwise span two
        separate entries.

        Under prefix matching the shorter term legitimately reaches BOTH
        teams: ``tenant|acme\\|corp`` starts with ``tenant|acme``. That is the
        intended widening, not a leak — the two stay distinguishable by the
        longer term, which is the first assertion here.
        """
        piped = make_indexed_process(AcmeTeamMetadata(tenant="acme|corp"))
        plain = make_indexed_process(AcmeTeamMetadata(tenant="acme"))
        event_store.save_team(piped)
        event_store.save_team(plain)

        assert [p.team_id for p in event_store.list_teams(metadata={"tenant": ["acme|corp"]})] == [
            piped.team_id
        ]
        assert {p.team_id for p in event_store.list_teams(metadata={"tenant": ["acme"]})} == {
            piped.team_id,
            plain.team_id,
        }

    def test_list_teams_metadata_cannot_match_across_two_entries(
        self, event_store: EventStore
    ) -> None:
        """A crafted value cannot span the boundary between two index entries.

        The team stores ``tenant|acme`` and ``case_ref|C-1`` as two separate
        entries. A query for a tenant literally named ``acme|case_ref|C-1``
        must not match by concatenation — entries are compared whole.
        """
        team = make_indexed_process(AcmeTeamMetadata(tenant="acme", case_ref="C-1"))
        event_store.save_team(team)

        assert event_store.list_teams(metadata={"tenant": ["acme|case_ref|C-1"]}) == []
        # The honest query for the same team still works.
        result = event_store.list_teams(metadata={"tenant": ["acme"], "case_ref": ["C-1"]})
        assert [p.team_id for p in result] == [team.team_id]

    # --- the shared prefix-matching matrix --------------------------------

    @pytest.mark.parametrize(
        "case_label,metadata,expected_labels",
        METADATA_FILTER_MATRIX,
        ids=METADATA_FILTER_IDS,
    )
    def test_list_teams_metadata_filter_matrix(
        self,
        event_store: EventStore,
        case_label: str,
        metadata: dict[str, list[str]] | None,
        expected_labels: set[str] | frozenset[str],
    ) -> None:
        """AC5-AC11 on every backend, from one shared matrix.

        Runs once per backend via the parametrized ``event_store`` fixture, so
        a backend left on whole-entry equality fails a named spec of its own
        rather than hiding behind the others.
        """
        teams = build_metadata_fixture_set()
        for process in teams.values():
            event_store.save_team(process)

        found = {p.team_id for p in event_store.list_teams(metadata=metadata)}
        expected = {teams[label].team_id for label in expected_labels}
        assert found == expected, case_label

    def test_list_teams_rejects_a_bare_string_metadata_value(
        self, event_store: EventStore
    ) -> None:
        """A bare ``str`` raises rather than filtering on one term per character.

        ``str`` IS a ``Sequence[str]``, so an un-migrated caller would otherwise
        filter ``"acme"`` as ``["a", "c", "m", "e"]`` — four prefixes, each of
        which matches something — and get plausible wrong rows back with no
        error anywhere. The annotation stops a caller mypy covers; this stops
        the rest.
        """
        event_store.save_team(make_indexed_process(AcmeTeamMetadata(tenant="acme")))

        with pytest.raises(TypeError, match="tenant"):
            event_store.list_teams(metadata={"tenant": "acme"})  # type: ignore[dict-item]

    def test_list_teams_rejects_a_bare_string_on_an_empty_store(
        self, event_store: EventStore
    ) -> None:
        """The rejection does not depend on there being anything to return.

        A backend that renders its terms lazily — after an early return for an
        empty store, or inside the row loop — would answer ``[]`` here and look
        entirely reasonable doing it.
        """
        with pytest.raises(TypeError):
            event_store.list_teams(metadata={"tenant": "acme"})  # type: ignore[dict-item]

    def test_list_teams_casefolding_costs_no_information_in_the_value(
        self, event_store: EventStore
    ) -> None:
        """Only the derived index folds; ``Process.metadata`` reports what was written."""
        process = make_indexed_process(AcmeTeamMetadata(tenant="AzeFR", case_ref="C-1"))
        event_store.save_team(process)

        found = event_store.list_teams(metadata={"tenant": ["aze"]})
        assert [p.team_id for p in found] == [process.team_id]
        metadata = AcmeTeamMetadata.model_validate(found[0].metadata)
        assert metadata.tenant == "AzeFR"
        assert metadata.case_ref == "C-1"

    # --- save_event / load_events ordering --------------------------------

    def test_save_and_load_events_in_sequence_order(self, event_store: EventStore) -> None:
        """Events inserted out of order are returned ascending by sequence."""
        team_id = uuid.uuid4()
        for seq in (3, 1, 2, 5, 4):
            event_store.save_event(make_persisted_event(team_id=team_id, sequence=seq))

        loaded = event_store.load_events(team_id)
        assert [e.sequence for e in loaded] == [1, 2, 3, 4, 5]

    def test_load_events_returns_empty_list_when_missing(self, event_store: EventStore) -> None:
        """Protocol contract: no events for a team yields ``[]``."""
        assert event_store.load_events(uuid.uuid4()) == []

    # --- load_events(after_event_id) cursor baseline -----------------------

    def test_load_events_after_event_id_none_equals_no_arg(self, event_store: EventStore) -> None:
        """``load_events(t, after_event_id=None)`` is equivalent to ``load_events(t)``."""
        team_id = uuid.uuid4()
        for seq in (1, 2, 3):
            event_store.save_event(make_persisted_event(team_id=team_id, sequence=seq))

        no_arg = event_store.load_events(team_id)
        with_none = event_store.load_events(team_id, after_event_id=None)
        assert [e.event.id for e in with_none] == [e.event.id for e in no_arg]
        assert [e.sequence for e in with_none] == [1, 2, 3]

    def test_load_events_after_event_id_excludes_anchor(self, event_store: EventStore) -> None:
        """A resolving anchor yields the strict tail — the anchor itself is excluded."""
        team_id = uuid.uuid4()
        for seq in (1, 2, 3, 4, 5):
            event_store.save_event(make_persisted_event(team_id=team_id, sequence=seq))

        anchor = event_store.load_events(team_id)[2]
        assert anchor.sequence == 3

        tail = event_store.load_events(team_id, after_event_id=anchor.event.id)
        assert [e.sequence for e in tail] == [4, 5]
        assert anchor.event.id not in [e.event.id for e in tail]

    def test_load_events_unknown_after_event_id_raises(self, event_store: EventStore) -> None:
        """An anchor that was never persisted raises rather than returning ``[]``.

        Returning ``[]`` would be indistinguishable from the legitimate
        "you are already up to date" answer (ADR-21 §2).
        """
        team_id = uuid.uuid4()
        for seq in (1, 2, 3):
            event_store.save_event(make_persisted_event(team_id=team_id, sequence=seq))

        with pytest.raises(EventNotFoundError):
            event_store.load_events(team_id, after_event_id=uuid.uuid4())

    def test_load_events_first_event_anchor_returns_all_but_first(
        self, event_store: EventStore
    ) -> None:
        """The first event as anchor yields every event except that one."""
        team_id = uuid.uuid4()
        for seq in (1, 2, 3, 4, 5):
            event_store.save_event(make_persisted_event(team_id=team_id, sequence=seq))

        first = event_store.load_events(team_id)[0]
        tail = event_store.load_events(team_id, after_event_id=first.event.id)
        assert [e.sequence for e in tail] == [2, 3, 4, 5]

    def test_load_events_last_event_anchor_returns_empty(self, event_store: EventStore) -> None:
        """The last event as anchor yields ``[]`` — the caller is up to date."""
        team_id = uuid.uuid4()
        for seq in (1, 2, 3):
            event_store.save_event(make_persisted_event(team_id=team_id, sequence=seq))

        last = event_store.load_events(team_id)[-1]
        assert event_store.load_events(team_id, after_event_id=last.event.id) == []

    def test_load_events_anchor_from_other_team_raises(self, event_store: EventStore) -> None:
        """An anchor belonging to another team raises — the lookup is team-scoped.

        Returning team A's whole log because the caller passed team B's
        cursor is the silent degradation ADR-21 §2 exists to prevent.
        """
        team_a = uuid.uuid4()
        team_b = uuid.uuid4()
        for seq in (1, 2, 3):
            event_store.save_event(make_persisted_event(team_id=team_a, sequence=seq))
            event_store.save_event(make_persisted_event(team_id=team_b, sequence=seq))

        foreign_anchor = event_store.load_events(team_b)[0]
        with pytest.raises(EventNotFoundError):
            event_store.load_events(team_a, after_event_id=foreign_anchor.event.id)

    def test_load_events_after_event_id_on_empty_store_raises(
        self, event_store: EventStore
    ) -> None:
        """An anchor queried against a team with no events raises, never ``[]``."""
        with pytest.raises(EventNotFoundError):
            event_store.load_events(uuid.uuid4(), after_event_id=uuid.uuid4())

    def test_load_events_cursor_round_trip(self, event_store: EventStore) -> None:
        """The incremental-reader usage pattern: poll, append, poll again.

        This is the case that would catch a silent regression back to a
        full-log return — the second poll must yield exactly the one new
        event, not the whole history.
        """
        team_id = uuid.uuid4()
        for seq in (1, 2, 3):
            event_store.save_event(make_persisted_event(team_id=team_id, sequence=seq))

        cursor = event_store.load_events(team_id)[-1].event.id
        assert event_store.load_events(team_id, after_event_id=cursor) == []

        event_store.save_event(make_persisted_event(team_id=team_id, sequence=4))

        fresh = event_store.load_events(team_id, after_event_id=cursor)
        assert [e.sequence for e in fresh] == [4]

    # --- get_max_sequence -------------------------------------------------

    def test_get_max_sequence_returns_zero_when_empty(self, event_store: EventStore) -> None:
        """Protocol contract: no events yields max sequence ``0``."""
        assert event_store.get_max_sequence(uuid.uuid4()) == 0

    def test_get_max_sequence_returns_largest_after_inserts(self, event_store: EventStore) -> None:
        """``get_max_sequence`` returns the largest sequence ever written."""
        team_id = uuid.uuid4()
        for seq in (1, 2, 7, 3):
            event_store.save_event(make_persisted_event(team_id=team_id, sequence=seq))
        assert event_store.get_max_sequence(team_id) == 7

    # --- save_agent_state / load_agent_states -----------------------------

    def test_save_and_load_agent_state_round_trip(self, event_store: EventStore) -> None:
        """``save_agent_state(s)`` puts ``s`` into ``load_agent_states(team_id)``."""
        snap = make_agent_state_snapshot(agent_id="round-trip-agent")
        event_store.save_agent_state(snap)

        loaded = event_store.load_agent_states(snap.team_id)
        assert len(loaded) == 1
        assert loaded[0].agent_id == "round-trip-agent"
        assert isinstance(loaded[0].state, SampleAgentState)

    def test_save_and_load_agent_state_uuid_and_name_round_trip(
        self, event_store: EventStore
    ) -> None:
        """AC #5: a UUID ``agent_id`` and a non-None ``name`` survive the round-trip.

        Runs once per backend (yaml/mongo/postgres) via the parametrized
        ``event_store`` fixture -- the per-backend round-trip coverage.
        """
        agent_uuid = str(uuid.uuid4())
        snap = make_agent_state_snapshot(agent_id=agent_uuid, name="@SomeAgent")
        event_store.save_agent_state(snap)

        loaded = event_store.load_agent_states(snap.team_id)
        assert len(loaded) == 1
        assert loaded[0].agent_id == agent_uuid
        assert loaded[0].name == "@SomeAgent"
        assert isinstance(loaded[0].state, SampleAgentState)

    def test_save_agent_state_is_upsert(self, event_store: EventStore) -> None:
        """``save_agent_state`` is upsert on ``(team_id, agent_id)``."""
        team_id = uuid.uuid4()
        first = make_agent_state_snapshot(
            team_id=team_id,
            agent_id="agent-a",
            state=SampleAgentState(task_count=1),
        )
        event_store.save_agent_state(first)

        second = make_agent_state_snapshot(
            team_id=team_id,
            agent_id="agent-a",
            state=SampleAgentState(task_count=99),
        )
        event_store.save_agent_state(second)

        loaded = event_store.load_agent_states(team_id)
        assert len(loaded) == 1
        assert isinstance(loaded[0].state, SampleAgentState)
        assert loaded[0].state.task_count == 99

    def test_load_agent_states_returns_empty_list_when_missing(
        self, event_store: EventStore
    ) -> None:
        """Protocol contract: no agent states for a team yields ``[]``."""
        assert event_store.load_agent_states(uuid.uuid4()) == []

    # --- delete_team ------------------------------------------------------

    def test_delete_team_cascades_across_three_kinds(self, event_store: EventStore) -> None:
        """``delete_team`` removes the team's process, events, and agent states."""
        process = make_process()
        for seq in range(1, 6):
            event_store.save_event(make_persisted_event(team_id=process.team_id, sequence=seq))
        for agent_id in ("a1", "a2", "a3"):
            event_store.save_agent_state(
                make_agent_state_snapshot(team_id=process.team_id, agent_id=agent_id)
            )
        event_store.save_team(process)

        event_store.delete_team(process.team_id)

        assert event_store.load_team(process.team_id) is None
        assert event_store.load_events(process.team_id) == []
        assert event_store.load_agent_states(process.team_id) == []

    def test_delete_team_isolates_other_teams(self, event_store: EventStore) -> None:
        """``delete_team`` only purges the requested team_id; others survive."""
        team_a = make_process()
        team_b = make_process()
        event_store.save_team(team_a)
        event_store.save_team(team_b)

        for seq in range(1, 4):
            event_store.save_event(make_persisted_event(team_id=team_a.team_id, sequence=seq))
            event_store.save_event(make_persisted_event(team_id=team_b.team_id, sequence=seq))
        event_store.save_agent_state(
            make_agent_state_snapshot(team_id=team_a.team_id, agent_id="a1")
        )
        event_store.save_agent_state(
            make_agent_state_snapshot(team_id=team_b.team_id, agent_id="b1")
        )

        event_store.delete_team(team_a.team_id)

        assert event_store.load_team(team_b.team_id) is not None
        assert len(event_store.load_events(team_b.team_id)) == 3
        assert len(event_store.load_agent_states(team_b.team_id)) == 1

    def test_delete_team_is_idempotent(self, event_store: EventStore) -> None:
        """Deleting a non-existent team is a no-op (no exception)."""
        ghost_id = uuid.uuid4()
        event_store.delete_team(ghost_id)
        event_store.delete_team(ghost_id)  # second call also a no-op

    # --- Polymorphic round-trip (Message / BaseState) ---------------------

    def test_polymorphic_message_round_trip_through_event(self, event_store: EventStore) -> None:
        """A polymorphic ``Message`` subtype survives the persist/hydrate cycle."""
        team_id = uuid.uuid4()
        msg = UserMessage(content="hello from polymorphic test")
        event_store.save_event(make_persisted_event(team_id=team_id, sequence=1, event=msg))

        loaded = event_store.load_events(team_id)
        assert len(loaded) == 1
        assert isinstance(loaded[0].event, UserMessage)
        assert loaded[0].event.content == "hello from polymorphic test"

    def test_polymorphic_basestate_round_trip_through_agent_state(
        self, event_store: EventStore
    ) -> None:
        """A polymorphic ``BaseState`` subtype survives the persist/hydrate cycle."""
        team_id = uuid.uuid4()
        state = SampleAgentState(task_count=42)
        event_store.save_agent_state(
            make_agent_state_snapshot(team_id=team_id, agent_id="poly", state=state)
        )

        loaded = event_store.load_agent_states(team_id)
        assert len(loaded) == 1
        assert isinstance(loaded[0].state, SampleAgentState)
        assert loaded[0].state.task_count == 42

    # --- Validation failure on corrupted payload --------------------------

    def test_validation_failure_propagates_pydantic_error(
        self,
        event_store: EventStore,
        request: pytest.FixtureRequest,
    ) -> None:
        """A corrupted stored payload triggers Pydantic validation handling.

        Implementation note (per AC #5, last bullet): all three current
        backends DELIBERATELY swallow ``pydantic.ValidationError`` on
        corrupted payloads (logging + ``None`` / ``[]`` semantics) — see
        the per-backend ``test_load_*_corrupted_*`` tests for the
        bespoke coverage. None expose a clean seam to assert raw
        propagation without leaking internals into the contract suite.

        Per the AC's escape hatch ("If a backend cannot expose a seam
        without leaking implementation, **skip on that backend** with a
        clear message — do NOT loosen the assertion"), this test skips
        on every backend until a future EventStore implementation
        propagates ``ValidationError`` natively. The skip preserves the
        contract slot so the moment such a backend lands, the test slot
        is already there to be flipped on.
        """
        backend = request.node.callspec.params["event_store"]
        pytest.skip(
            f"{backend!r} swallows ValidationError by design (resilient "
            "load semantics); per-backend corrupted-payload coverage "
            "lives in tests/repositories/{yaml,mongo,postgres}/."
        )
