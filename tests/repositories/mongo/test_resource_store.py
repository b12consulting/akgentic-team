"""Specs for ``MongoResourceStore`` — core's ``ResourceStore`` on the team database.

What is proven against what, because this package has blurred it before:

* **Against ``mongomock``** (the ``mongo_db`` fixture): ``None`` on a miss, the concrete state
  class on a hit, the encode/decode symmetry of the member key, kind separation, the untouched
  field surviving a delta, the unknown-field subclass round trip, the corrupted document and the
  unresolvable class both answering ``None`` with a log line.
* **Against the emitted command** (a spy on ``update_one``): one write, ``$set`` and ``$unset``
  in one update document, ``upsert=True``, nothing issued for an empty delta, a conflicting
  ``unset`` dropped. Atomicity and path-conflict rejection are server behaviours a double does
  not reproduce, so the command is the only thing a unit spec can assert.
* **Against a real server**: exactly one spec, ``integration``-marked and skipped without
  ``MONGO_TEST_URI``. The package's default ``addopts`` deselect it, so it is **not** part of
  the gate and does not count as covering the round trip for anyone who has not run it.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from akgentic.core import Akgent, BaseConfig, BaseState, ResourceStore, StateDelta
from pydantic import JsonValue
from pymongo.errors import PyMongoError

from akgentic.team.repositories.mongo_resource import (
    RESOURCES_COLLECTION,
    MongoResourceStore,
    _decode_member,
    _delta_path,
    _encode_member,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_LOGGER = "akgentic.team.repositories.mongo_resource"
_REAL_SERVER_ENV = "MONGO_TEST_URI"


# --- Test resource kinds ------------------------------------------------------


class _WorkspaceLikeState(BaseState):
    """The shape the first real consumer persists: two path-keyed maps and a scalar."""

    documents: dict[str, JsonValue] = {}
    rag_index: dict[str, str] = {}
    counter: int = 0


class _AlphaActor(Akgent[BaseConfig, _WorkspaceLikeState]):
    """One resource kind."""


class _BetaState(BaseState):
    notes: dict[str, str] = {}
    label: str = "beta-default"


class _BetaActor(Akgent[BaseConfig, _BetaState]):
    """A second kind with a different state shape — the two-kinds guard needs both."""


class _StateWithExtraField(_WorkspaceLikeState):
    """A field the store has never heard of (Golden Rule 12 guard)."""

    extra_field: str = "sentinel"


class _ExtraActor(Akgent[BaseConfig, _StateWithExtraField]):
    """Declares the subclass, so the resolver hands the store a class with an unknown field."""


class _NoneStateActor(Akgent[BaseConfig, None]):
    """A real shape in this repository: the resolver answers ``None`` for it."""


def _twin_from_module_a() -> type[Akgent[Any, Any]]:
    """A class whose bare ``__name__`` collides with :func:`_twin_from_module_b`'s."""

    class Twin(Akgent[BaseConfig, _WorkspaceLikeState]):
        pass

    return Twin


def _twin_from_module_b() -> type[Akgent[Any, Any]]:
    """Same ``__name__`` as the other twin, different ``__qualname__``."""

    class Twin(Akgent[BaseConfig, _BetaState]):
        pass

    return Twin


# --- Helpers ------------------------------------------------------------------


def _raw_documents(mongo_db: Any, scope: str) -> list[dict[str, Any]]:
    """Every stored document at *scope*, straight off the collection."""
    return list(mongo_db[RESOURCES_COLLECTION].find({"scope": scope}))


def _records(caplog: pytest.LogCaptureFixture, level: int) -> list[logging.LogRecord]:
    """Records from the store's own logger at exactly *level* — never a substring over all."""
    return [r for r in caplog.records if r.name == _LOGGER and r.levelno == level]


@pytest.fixture
def store(mongo_db: Any) -> MongoResourceStore:
    """A store over a fresh mongomock database."""
    return MongoResourceStore(mongo_db)


@pytest.fixture
def update_spy(store: MongoResourceStore) -> Iterator[Any]:
    """Wrap the collection's ``update_one`` so the emitted command is observable.

    ``wraps`` keeps the real effect, so a spec can assert both the command and its
    outcome on the double in one place.
    """
    collection = store._resources
    with patch.object(collection, "update_one", wraps=collection.update_one) as spy:
        yield spy


# --- Construction ------------------------------------------------------------


class TestConstruction:
    def test_it_satisfies_the_protocol_by_declaration_and_at_runtime(
        self, store: MongoResourceStore
    ) -> None:
        """Explicit inheritance is what makes signature drift a ``mypy src/`` failure.

        ``runtime_checkable`` checks method presence only, so this assertion is the weak
        half; the strong half is the ``mypy`` gate, which this package's CI runs on ``src/``.
        """
        declared: ResourceStore = store
        assert isinstance(declared, ResourceStore)
        assert issubclass(MongoResourceStore, ResourceStore)

    def test_construction_creates_a_unique_index_on_kind_and_scope(
        self, store: MongoResourceStore, mongo_db: Any
    ) -> None:
        info = mongo_db[RESOURCES_COLLECTION].index_information()
        matching = [e for e in info.values() if e["key"] == [("kind", 1), ("scope", 1)]]
        assert matching, f"(kind, scope) index missing: {info}"
        assert matching[0].get("unique") is True

    def test_an_index_rejection_warns_and_construction_still_succeeds(
        self, mongo_db: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Refusing to construct the store over an index is the worse outcome."""
        collection = mongo_db[RESOURCES_COLLECTION]
        caplog.clear()
        with (
            caplog.at_level(logging.WARNING, logger=_LOGGER),
            patch.object(collection, "create_index", side_effect=PyMongoError("nope")),
        ):
            built = MongoResourceStore(mongo_db)
        assert built is not None
        warnings = _records(caplog, logging.WARNING)
        assert len(warnings) == 1
        assert RESOURCES_COLLECTION in warnings[0].getMessage()


# --- load ---------------------------------------------------------------------


class TestLoad:
    def test_nothing_stored_answers_none_without_raising(
        self, store: MongoResourceStore, mongo_db: Any
    ) -> None:
        assert _raw_documents(mongo_db, "P") == []
        assert store.load(_AlphaActor, "P") is None

    def test_a_delta_loads_back_as_the_concrete_state_class(
        self, store: MongoResourceStore
    ) -> None:
        store.apply(
            _AlphaActor,
            "P",
            StateDelta(set={"documents.readme": {"pages": 1}, "counter": 7}),
        )
        loaded = store.load(_AlphaActor, "P")
        assert isinstance(loaded, _WorkspaceLikeState)
        assert type(loaded) is _WorkspaceLikeState, "must be the declared class, not BaseState"
        assert loaded.counter == 7
        assert loaded.documents == {"readme": {"pages": 1}}

    def test_a_corrupted_document_answers_none_and_is_logged_at_error_naming_the_scope(
        self, store: MongoResourceStore, mongo_db: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A clean document at another scope, so the negative half is not vacuous.
        store.apply(_AlphaActor, "clean-scope", StateDelta(set={"counter": 1}))
        corrupted = dict(_raw_documents(mongo_db, "clean-scope")[0])
        corrupted.pop("_id")
        corrupted["scope"] = "broken-scope"
        corrupted["state"] = {"counter": "not-an-int"}
        mongo_db[RESOURCES_COLLECTION].insert_one(corrupted)

        caplog.clear()
        with caplog.at_level(logging.ERROR, logger=_LOGGER):
            assert store.load(_AlphaActor, "broken-scope") is None
            errors = _records(caplog, logging.ERROR)
            assert len(errors) == 1
            assert "broken-scope" in errors[0].getMessage()

            caplog.clear()
            assert isinstance(store.load(_AlphaActor, "clean-scope"), _WorkspaceLikeState)
            assert _records(caplog, logging.ERROR) == []

    def test_a_state_field_holding_a_non_mapping_is_still_a_corrupted_document(
        self, store: MongoResourceStore, mongo_db: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The decoder must not raise on a shape it cannot walk; the read rule still holds."""
        store.apply(_AlphaActor, "P", StateDelta(set={"counter": 1}))
        mongo_db[RESOURCES_COLLECTION].update_one({"scope": "P"}, {"$set": {"state": ["x"]}})
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger=_LOGGER):
            assert store.load(_AlphaActor, "P") is None
        assert len(_records(caplog, logging.ERROR)) == 1

    def test_an_unresolvable_actor_class_answers_none_rather_than_a_base_state(
        self, store: MongoResourceStore, mongo_db: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The arm is reachable: a document exists under the class, and load still says None.

        Paired with the positive — the same delta under a resolvable class loads — so a store
        that never reads at all cannot pass this spec.
        """
        store.apply(_NoneStateActor, "P", StateDelta(set={"counter": 3}))
        store.apply(_AlphaActor, "P", StateDelta(set={"counter": 3}))
        assert len(_raw_documents(mongo_db, "P")) == 2, "apply needs only the kind key"

        caplog.clear()
        with caplog.at_level(logging.ERROR, logger=_LOGGER):
            assert store.load(_NoneStateActor, "P") is None
        errors = _records(caplog, logging.ERROR)
        assert len(errors) == 1
        assert "P" in errors[0].getMessage()

        positive = store.load(_AlphaActor, "P")
        assert isinstance(positive, _WorkspaceLikeState)
        assert positive.counter == 3


# --- The encoding -------------------------------------------------------------


class TestEncoding:
    @pytest.mark.parametrize(
        "member",
        ["notes/a.pdf", "a.b.c", "$dollar", "100%", "%2E-literal", "sp ace", "ünïcödé.txt", ""],
    )
    def test_encode_and_decode_are_inverse(self, member: str) -> None:
        assert _decode_member(_encode_member(member)) == member

    def test_the_encoded_form_carries_no_dot_and_no_dollar(self) -> None:
        encoded = _encode_member("notes/a.pdf$")
        assert "." not in encoded
        assert "$" not in encoded
        assert "/" not in encoded

    def test_encoding_is_injective_where_percent_encoding_alone_is_not(self) -> None:
        """``%2E`` written literally and ``.`` must not land on one stored key."""
        assert _encode_member("%2E") != _encode_member(".")

    def test_the_path_splits_on_the_first_dot_only(self) -> None:
        assert _delta_path("counter") == "state.counter"
        assert _delta_path("documents.notes/a.pdf") == "state.documents." + _encode_member(
            "notes/a.pdf"
        )

    def test_a_member_key_with_a_dot_and_a_slash_survives_the_round_trip(
        self, store: MongoResourceStore, mongo_db: Any
    ) -> None:
        """One map entry comes back, not a two-level nested sub-document (mongomock half)."""
        store.apply(_AlphaActor, "P", StateDelta(set={"documents.notes/a.pdf": {"pages": 3}}))

        loaded = store.load(_AlphaActor, "P")
        assert isinstance(loaded, _WorkspaceLikeState)
        assert loaded.documents, "an empty map would make the entry assertions vacuous"
        assert loaded.documents == {"notes/a.pdf": {"pages": 3}}
        assert list(loaded.documents) == ["notes/a.pdf"]

    def test_the_stored_path_is_the_encoded_member_not_a_nested_document(
        self, store: MongoResourceStore, mongo_db: Any
    ) -> None:
        """The command half: what actually sits on disk, read raw off the collection."""
        store.apply(_AlphaActor, "P", StateDelta(set={"documents.notes/a.pdf": {"pages": 3}}))

        raw = _raw_documents(mongo_db, "P")
        assert len(raw) == 1
        documents = raw[0]["state"]["documents"]
        assert list(documents) == [_encode_member("notes/a.pdf")]
        assert "notes/a" not in documents
        assert documents[_encode_member("notes/a.pdf")] == {"pages": 3}

    def test_a_non_mapping_state_field_is_left_untouched_on_load(
        self, store: MongoResourceStore
    ) -> None:
        store.apply(_BetaActor, "P", StateDelta(set={"label": "100%2E"}))
        loaded = store.load(_BetaActor, "P")
        assert isinstance(loaded, _BetaState)
        assert loaded.label == "100%2E"


# --- apply --------------------------------------------------------------------


class TestApply:
    def test_one_update_carrying_set_and_unset_together_with_upsert(
        self, store: MongoResourceStore, update_spy: Any
    ) -> None:
        store.apply(
            _AlphaActor,
            "P",
            StateDelta(set={"documents.a": 1, "counter": 2}, unset=["rag_index.old"]),
        )

        assert update_spy.call_count == 1
        (filter_doc, update_doc), kwargs = update_spy.call_args
        assert kwargs == {"upsert": True}
        assert filter_doc["scope"] == "P"
        assert set(update_doc) == {"$set", "$unset"}
        assert update_doc["$set"] == {"state.documents.a": 1, "state.counter": 2}
        assert update_doc["$unset"] == {"state.rag_index.old": ""}

    def test_an_empty_delta_issues_no_command(
        self, store: MongoResourceStore, update_spy: Any, mongo_db: Any
    ) -> None:
        store.apply(_AlphaActor, "P", StateDelta())
        assert update_spy.call_count == 0
        assert _raw_documents(mongo_db, "P") == []

        store.apply(_AlphaActor, "P", StateDelta(set={"counter": 1}))
        assert update_spy.call_count == 1, "the spy sees a real write, so zero above meant zero"

    def test_a_delta_of_only_unsets_issues_no_set_operator(
        self, store: MongoResourceStore, update_spy: Any
    ) -> None:
        """``$set: {}`` is rejected by the server, so an operator is present only when non-empty."""
        store.apply(_AlphaActor, "P", StateDelta(unset=["documents.gone"]))
        assert update_spy.call_count == 1
        (_, update_doc), _ = update_spy.call_args
        assert set(update_doc) == {"$unset"}

    def test_an_unset_equal_to_or_a_prefix_of_a_set_path_is_dropped(
        self, store: MongoResourceStore, update_spy: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``set`` wins; a survivor proves the filter is a filter and not a blanket drop."""
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger=_LOGGER):
            store.apply(
                _AlphaActor,
                "P",
                StateDelta(
                    set={"documents.a": 1},
                    unset=["documents.a", "documents", "rag_index.survivor"],
                ),
            )
        (_, update_doc), _ = update_spy.call_args
        assert update_doc["$set"] == {"state.documents.a": 1}
        assert update_doc["$unset"] == {"state.rag_index.survivor": ""}
        dropped = [r for r in _records(caplog, logging.DEBUG) if "dropped" in r.getMessage()]
        assert len(dropped) == 2

    def test_an_unset_below_a_whole_field_set_is_dropped_too(
        self, store: MongoResourceStore, update_spy: Any
    ) -> None:
        """The other direction of the same conflict: the server refuses it just the same."""
        store.apply(
            _AlphaActor,
            "P",
            StateDelta(set={"documents": {"a": 1}}, unset=["documents.a"]),
        )
        (_, update_doc), _ = update_spy.call_args
        assert set(update_doc) == {"$set"}

    def test_a_conflict_that_empties_the_unsets_still_writes_the_sets(
        self, store: MongoResourceStore, update_spy: Any
    ) -> None:
        store.apply(_AlphaActor, "P", StateDelta(set={"counter": 5}, unset=["counter"]))
        assert update_spy.call_count == 1
        loaded = store.load(_AlphaActor, "P")
        assert isinstance(loaded, _WorkspaceLikeState)
        assert loaded.counter == 5

    def test_a_field_the_delta_did_not_mention_is_not_destroyed(
        self, store: MongoResourceStore
    ) -> None:
        store.apply(
            _AlphaActor,
            "P",
            StateDelta(set={"counter": 3, "rag_index.k": "v", "documents.keep": {"pages": 1}}),
        )
        store.apply(
            _AlphaActor,
            "P",
            StateDelta(set={"documents.new": {"pages": 2}}, unset=["documents.keep"]),
        )

        loaded = store.load(_AlphaActor, "P")
        assert isinstance(loaded, _WorkspaceLikeState)
        assert loaded.counter == 3
        assert loaded.rag_index == {"k": "v"}
        assert loaded.documents == {"new": {"pages": 2}}

    def test_an_unset_of_a_whole_field_restores_its_default_on_load(
        self, store: MongoResourceStore
    ) -> None:
        store.apply(_AlphaActor, "P", StateDelta(set={"counter": 3, "rag_index.k": "v"}))
        store.apply(_AlphaActor, "P", StateDelta(unset=["counter"]))
        loaded = store.load(_AlphaActor, "P")
        assert isinstance(loaded, _WorkspaceLikeState)
        assert loaded.counter == 0
        assert loaded.rag_index == {"k": "v"}


# --- Golden Rule 12 -----------------------------------------------------------


class TestNeverRebuiltByEnumeration:
    def test_an_unknown_field_survives_the_round_trip_as_the_subclass(
        self, store: MongoResourceStore
    ) -> None:
        """A whole-model comparison is not enough; the subclass with an unknown field is.

        ``extra_field`` is written to a non-default value on purpose: a rebuild that named
        every field existing today would still produce the right class and would pass an
        assertion on the default.
        """
        store.apply(
            _ExtraActor,
            "P",
            StateDelta(set={"extra_field": "written", "counter": 4, "documents.a": {"n": 1}}),
        )
        # A second delta, so an apply that read-rebuilt-wrote would get its chance to drop it.
        store.apply(_ExtraActor, "P", StateDelta(set={"rag_index.k": "v"}))

        loaded = store.load(_ExtraActor, "P")
        assert isinstance(loaded, _StateWithExtraField)
        assert loaded.extra_field == "written"
        assert loaded.counter == 4
        assert loaded.documents == {"a": {"n": 1}}
        assert loaded.rag_index == {"k": "v"}


# --- Kind separation ----------------------------------------------------------


class TestTwoKindsAtOneScope:
    def test_two_actor_classes_at_one_scope_are_two_independent_documents(
        self, store: MongoResourceStore, mongo_db: Any
    ) -> None:
        """A store that ignored the class could not tell these apart; one kind cannot prove it."""
        store.apply(_AlphaActor, "P", StateDelta(set={"counter": 9, "documents.a": {"n": 1}}))
        assert store.load(_BetaActor, "P") is None, "Beta must not see Alpha's document"

        store.apply(_BetaActor, "P", StateDelta(set={"label": "written-by-beta"}))
        assert len(_raw_documents(mongo_db, "P")) == 2

        beta = store.load(_BetaActor, "P")
        assert isinstance(beta, _BetaState)
        assert beta.label == "written-by-beta"
        assert beta.notes == {}

        alpha = store.load(_AlphaActor, "P")
        assert isinstance(alpha, _WorkspaceLikeState)
        assert alpha.counter == 9
        assert alpha.documents == {"a": {"n": 1}}

    def test_two_classes_sharing_a_bare_name_are_still_two_documents(
        self, store: MongoResourceStore, mongo_db: Any
    ) -> None:
        """The kind key is derived from the qualified name; ``__name__`` alone would collide."""
        twin_a, twin_b = _twin_from_module_a(), _twin_from_module_b()
        assert twin_a.__name__ == twin_b.__name__ == "Twin"

        store.apply(twin_a, "P", StateDelta(set={"counter": 1}))
        assert store.load(twin_b, "P") is None
        store.apply(twin_b, "P", StateDelta(set={"label": "b"}))

        kinds = {doc["kind"] for doc in _raw_documents(mongo_db, "P")}
        assert len(kinds) == 2

        loaded_a = store.load(twin_a, "P")
        assert isinstance(loaded_a, _WorkspaceLikeState)
        assert loaded_a.counter == 1


# --- The one real-server spec ---------------------------------------------------


@pytest.mark.integration
def test_a_real_server_accepts_the_encoded_path_and_returns_the_same_value() -> None:
    """AC3 against a real MongoDB: the encoded path is a legal update path and reads back.

    Skipped when ``MONGO_TEST_URI`` is unset. Deselected by the package's default
    ``addopts`` — running it is a deliberate act, and until someone does, the round trip is
    proven on the double only.
    """
    uri = os.environ.get(_REAL_SERVER_ENV)
    if not uri:
        pytest.skip(f"{_REAL_SERVER_ENV} not set — needs a real MongoDB server")

    import pymongo

    client: pymongo.MongoClient[Any] = pymongo.MongoClient(uri, serverSelectionTimeoutMS=5000)
    db = client["akgentic_team_test_resources"]
    db.drop_collection(RESOURCES_COLLECTION)
    try:
        real_store = MongoResourceStore(db)
        real_store.apply(
            _AlphaActor,
            "P",
            StateDelta(
                set={"documents.notes/a.pdf": {"pages": 3}, "documents.b$.md": {"pages": 1}},
            ),
        )
        real_store.apply(_AlphaActor, "P", StateDelta(unset=["documents.b$.md"]))
        loaded = real_store.load(_AlphaActor, "P")
        assert isinstance(loaded, _WorkspaceLikeState)
        assert loaded.documents == {"notes/a.pdf": {"pages": 3}}
    finally:
        db.drop_collection(RESOURCES_COLLECTION)
        client.close()
