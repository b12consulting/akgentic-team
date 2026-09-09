"""MongoResourceStore: core's ``ResourceStore`` on the team database, via the ``[mongo]`` extra.

A hosted actor — one that outlives every team that reaches it — has no team stream to
checkpoint to. Core's ``ResourceHost`` restores its state through ``ResourceStore.load`` on a
get-or-create miss and absorbs every ``StateDelta`` it reports through ``ResourceStore.apply``.
This module is the one implementation of that Protocol; the deployment package registers it.

Collection layout::

    resources          # One document per (kind, scope) -- {kind, scope, state: {...}}

``kind`` is derived from the resource's actor class, ``scope`` is the host's registry key
(``config.name``, verbatim, never parsed), and ``state`` is assembled by accretion from
delta paths. It is **not** a dumped model and carries no root type marker, which is why the
class is needed to rebuild it — see :func:`akgentic.core.resolve_state_type`.

**The dotted delta key and the encoding.** A delta key is ``<field>.<member>``, split on the
**first** dot only: the left half is a field of the state model, the right half is a key inside
that mapping. In a MongoDB update every dot is a path separator and the member here is usually a
filename, so the member is percent-encoded on the way in — ``documents.notes/a.pdf`` is stored
at ``state.documents.notes%2Fa%2Epdf`` — and decoded symmetrically on the way out, for every
state field whose stored value is a mapping. Only those two levels are addressable. A value is
never inspected, so a whole mapping written in one ``set`` lands verbatim; its keys are still
decoded on load, which is why an emitter addresses members one key at a time.

This is a second, independent port that happens to share a database with ``MongoEventStore``.
Nothing here touches the event store's collections, and ``delete_team`` deliberately never
touches this one: a resource document is keyed by scope, not by team, and one tree is shared by
every team on it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote

try:
    import pymongo  # noqa: F401
except ImportError as exc:
    raise ImportError(
        "pymongo is required for MongoResourceStore. Install with: pip install akgentic-team[mongo]"
    ) from exc

from pymongo.errors import PyMongoError

from akgentic.core import Akgent, BaseState, ResourceStore, StateDelta, resolve_state_type

if TYPE_CHECKING:
    from pydantic import JsonValue
    from pymongo.collection import Collection
    from pymongo.database import Database

logger = logging.getLogger(__name__)

RESOURCES_COLLECTION = "resources"
"""Collection holding one document per hosted resource. Public: tests assert the layout."""

_KIND_KEY = "kind"
_SCOPE_KEY = "scope"
_STATE_FIELD = "state"
_KIND_SCOPE_INDEX = "resources_kind_scope_idx"


def _kind_of(actor_class: type[Akgent[Any, Any]]) -> str:
    """The namespace key for *actor_class*: module plus qualified name.

    Core passes the class object and has no opinion about the key; the derivation is the
    store's. The qualified name rather than the bare ``__name__``, which collides across
    packages and across nested classes sharing a leaf name.
    """
    return f"{actor_class.__module__}.{actor_class.__qualname__}"


def _encode_member(member: str) -> str:
    """Percent-encode a member key so no ``.`` and no ``$`` reaches an update path.

    ``quote`` treats ``.`` as always-safe whatever ``safe`` says, hence the explicit second
    replacement. ``%`` itself is encoded, so the pair with :func:`_decode_member` is injective.
    """
    return quote(member, safe="").replace(".", "%2E")


def _decode_member(encoded: str) -> str:
    """The inverse of :func:`_encode_member`."""
    return unquote(encoded)


def _delta_path(key: str) -> str:
    """Turn a delta key into the update path under the ``state`` sub-document.

    Split on the first dot only. No dot names a whole state field; a dot names one member of
    a mapping-valued field, with the member encoded.
    """
    field, dot, member = key.partition(".")
    if not dot:
        return f"{_STATE_FIELD}.{field}"
    return f"{_STATE_FIELD}.{field}.{_encode_member(member)}"


def _decode_state(document: dict[str, Any]) -> dict[str, Any]:
    """Decode the member keys of every mapping-valued state field; leave the rest untouched."""
    decoded: dict[str, Any] = {}
    for field, value in document.items():
        if isinstance(value, dict):
            decoded[field] = {_decode_member(key): member for key, member in value.items()}
        else:
            decoded[field] = value
    return decoded


def _conflicts(unset_path: str, set_paths: dict[str, JsonValue]) -> bool:
    """Whether *unset_path* equals, contains or sits inside a ``$set`` path.

    MongoDB refuses an update whose ``$set`` and ``$unset`` paths overlap in either direction,
    and the store must not turn a benign delta into a raised error. ``set`` wins.
    """
    for set_path in set_paths:
        if (
            unset_path == set_path
            or set_path.startswith(unset_path + ".")
            or unset_path.startswith(set_path + ".")
        ):
            return True
    return False


class MongoResourceStore(ResourceStore):
    """Core's ``ResourceStore`` on a MongoDB ``resources`` collection.

    Inherits the Protocol explicitly, unlike ``MongoEventStore`` and ``NullServiceRegistry``,
    which satisfy theirs structurally. That is a deliberate divergence: ``runtime_checkable``
    checks method presence only, so a parameter drift on a structural implementation passes
    every ``isinstance`` and is caught by nothing this package's CI runs on ``tests/``.
    Declaring the conformance is what makes a drift a ``mypy src/`` failure.

    Args:
        db: A pymongo ``Database`` connected to the target server — the same one the team's
            ``MongoEventStore`` uses, or another; the store has no opinion.
    """

    def __init__(self, db: Database[Any]) -> None:
        self._resources: Collection[Any] = db[RESOURCES_COLLECTION]
        # Guarded for the same reason the card-hash index is: the store's own writes upsert
        # on this exact key, so correctness does not depend on the index existing, and
        # refusing to construct the store over it would be the worse outcome.
        try:
            self._resources.create_index(
                [(_KIND_KEY, 1), (_SCOPE_KEY, 1)], name=_KIND_SCOPE_INDEX, unique=True
            )
        except PyMongoError:
            logger.warning(
                "Could not create index '%s' on '%s'; resource lookups fall back to a "
                "collection scan. Results stay correct.",
                _KIND_SCOPE_INDEX,
                RESOURCES_COLLECTION,
                exc_info=True,
            )
        logger.debug("Initialized MongoResourceStore with database '%s'", db.name)

    def load(self, actor_class: type[Akgent[Any, Any]], scope: str) -> BaseState | None:
        """Restore the state stored under *actor_class* and *scope*.

        Absent answers ``None``. A class the resolver cannot name a concrete state for, or a
        stored document that does not validate, also answers ``None`` — logged at ERROR naming
        the scope and never raised, which is this package's rule for every read path.

        Args:
            actor_class: The hosted resource's class; namespaces the document and names the
                state class to rebuild into.
            scope: The host's registry key for the resource.

        Returns:
            The stored state as the actor's declared state class, or ``None``.
        """
        kind = _kind_of(actor_class)
        document = self._resources.find_one({_KIND_KEY: kind, _SCOPE_KEY: scope})
        if document is None:
            return None
        state_type = resolve_state_type(actor_class)
        if state_type is None:
            logger.error(
                "No concrete state class for %s at scope %s; the stored document is skipped",
                kind,
                scope,
            )
            return None
        raw = document.get(_STATE_FIELD, {})
        payload = _decode_state(raw) if isinstance(raw, dict) else raw
        try:
            state = state_type.model_validate(payload)
        except (ValueError, TypeError) as exc:
            logger.error("Corrupted resource document for %s at scope %s: %s", kind, scope, exc)
            return None
        logger.debug("Loaded resource %s at scope %s", kind, scope)
        return state

    def apply(self, actor_class: type[Akgent[Any, Any]], scope: str, delta: StateDelta) -> None:
        """Absorb *delta* into the document under *actor_class* and *scope*.

        Exactly one ``update_one`` carrying ``$set`` and ``$unset`` together, upserting — the
        server applies both atomically on the one document. An empty delta issues nothing; an
        operator is present only when it has paths, since ``$set: {}`` is rejected. An
        ``unset`` path that conflicts with a ``set`` path is dropped and logged at DEBUG.

        Args:
            actor_class: The hosted resource's class, namespacing the document.
            scope: The host's registry key for the resource.
            delta: The fields to set and the field keys to remove, as dotted keys.
        """
        set_paths: dict[str, JsonValue] = {
            _delta_path(key): value for key, value in delta.set.items()
        }
        unset_paths: dict[str, str] = {}
        for key in delta.unset:
            path = _delta_path(key)
            if _conflicts(path, set_paths):
                logger.debug("unset %s dropped at scope %s: a set path wins", path, scope)
                continue
            unset_paths[path] = ""
        if not set_paths and not unset_paths:
            return
        update: dict[str, dict[str, Any]] = {}
        if set_paths:
            update["$set"] = set_paths
        if unset_paths:
            update["$unset"] = unset_paths
        self._resources.update_one(
            {_KIND_KEY: _kind_of(actor_class), _SCOPE_KEY: scope}, update, upsert=True
        )
        logger.debug(
            "Applied delta at scope %s: %d set, %d unset", scope, len(set_paths), len(unset_paths)
        )
