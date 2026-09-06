"""The one conversion from a pre-projection team document to the projection.

A ``Process`` written before the structural projection carries a nested
``team_card`` and none of the seven projection fields. ``Process`` refuses such
a document by design (``Process.reject_unmigrated_document``), so every team
already in a deployed store stops resuming the moment the new code starts —
``resume_team`` reports ``Team {id} not found`` for a team nobody deleted. This
module converts those documents.

It is deliberately **backend-agnostic**: it takes an iterable of raw stored
documents and an ``EventStore``, and returns a report. Each of the three scripts
under ``akgentic.team.scripts`` owns only its reader — opening the backend and
yielding raw mappings — so there is one conversion behind three readers rather
than three conversions that can drift.

Two properties are load-bearing and neither is visible in the end state:

* **The derivation is imported, never re-implemented.**
  ``akgentic.team.projection.derive_team_projection`` is the same function
  ``TeamManager.create_team`` calls, so a migrated team and a freshly created
  one cannot disagree — including on ``headcount``, which that function already
  expands into one ref per spawned instance.
* **Cards are written before the document that references them** (FR13). Both
  orders leave identical storage once they have run; only the call sequence
  tells them apart, which is why FR13 is a requirement rather than an
  observation.

Reading is deliberately *raw*. ``load_team`` and ``list_teams`` cannot see what
this migration exists to convert: they skip (YAML, Mongo, and Postgres since
this story) exactly the documents that carry the legacy shape. A migration built
on the public read path converts nothing and reports success.

Implements ADR-26 §Migration (FR8, FR8a, FR13).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from pydantic import Field

from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.team.models import Process, TeamCard
from akgentic.team.projection import derive_team_projection

if TYPE_CHECKING:
    from akgentic.team.ports import EventStore

logger = logging.getLogger(__name__)

LEGACY_CARD_KEY = "team_card"
"""The key a pre-projection document carries and a migrated one does not."""

PROJECTION_MARKER_KEY = "entry_point"
"""The key whose presence means "already migrated".

Consulted BEFORE ``team_card``, matching ``Process.reject_unmigrated_document``'s
recognition rule — legacy is ``team_card`` present *and* ``entry_point`` absent.
A hybrid left behind by an interrupted run is therefore treated as migrated
rather than re-flattened from a card that may since have been edited.
"""

_PROJECTION_ONLY_KEYS = frozenset({"cards", "__model__"})
"""Keys of a ``TeamProjection`` dump that must NOT land on the document.

``cards`` is the card blobs themselves, which go to the card store rather than
onto the ``Process``. ``__model__`` is the serializer's class marker: every
``SerializableBaseModel`` dump carries one, so overlaying the projection's dump
wholesale would relabel a ``Process`` document as a ``TeamProjection`` and the
deserializer would then try to rebuild it as one.

Everything else in the dump is copied across unnamed, so a field added to both
``TeamProjection`` and ``Process`` migrates without an edit here.
"""

ROLLOUT_ORDERING_NOTE = (
    "Rollout ordering: run this migration BETWEEN stopping the old version and "
    "starting the new one. Old code cannot read a migrated document, and new "
    "code cannot resume an unmigrated one, so there is no window in which both "
    "versions can serve the same store."
)
"""The FR10 constraint every script repeats in its ``--help``."""


class MigrationReport(SerializableBaseModel):
    """What one migration run did, as three counts.

    A Pydantic model rather than a mapping so the three numbers cross the
    script boundary with names and types attached.

    Attributes:
        converted: Documents rewritten to the projection.
        skipped: Documents that needed no work — already migrated, or carrying
            no ``team_card`` to derive from.
        failed: Documents the run could not convert. Each one is logged with
            its ``team_id``; the run continues past them, which is why the
            count exists: not aborting is not the same as succeeding.
    """

    converted: int = Field(default=0, description="Documents rewritten to the projection")
    skipped: int = Field(default=0, description="Documents that needed no work")
    failed: int = Field(default=0, description="Documents that could not be converted")


def _document_team_id(document: Mapping[str, Any]) -> str:
    """Return *document*'s ``team_id`` for a log line, or a placeholder.

    A document malformed enough to be missing its own id is exactly the one
    worth logging, so this never raises.
    """
    raw = document.get("team_id")
    return "<unknown>" if raw is None else str(raw)


def migrate_document(document: Mapping[str, Any], store: EventStore) -> None:
    """Convert ONE legacy document and write it — cards first, document second.

    The migrated mapping is built by **copying the source document and
    overriding only what changes**: drop ``team_card``, overlay the projection's
    fields. Naming ``Process``'s fields one by one instead would be correct on
    the day it was written and would silently destroy the next field anyone adds
    to ``Process``, on a path that runs unattended over a customer's stored
    teams (Golden Rule #12).

    Keys the source document carries that are not ``Process`` fields are dropped
    by ``model_validate``, exactly as every other re-save in this package drops
    them. That is accepted: the write path is a ``Process``.

    Args:
        document: The raw stored mapping, carrying ``team_card``.
        store: The store to write the cards and the rewritten document into.

    Raises:
        ValueError: If ``team_card`` does not validate as a ``TeamCard``, or the
            derived document does not validate as a ``Process``. Pydantic's
            ``ValidationError`` is a ``ValueError``.
        TypeError: If the stored ``team_card`` is not a shape Pydantic can read.
        KeyError: If *document* carries no ``team_card`` at all. Callers reach
            this through :func:`migrate_documents`, which classifies such a
            document as skipped before it gets here.
        ImportError: If the stored card names a module that cannot be imported.
        AttributeError: If that module carries no class of the stored name.
        Exception: The list above is what this function is *known* to raise, not
            a closed set. ``TeamCard.model_validate`` runs the deserializer over
            an arbitrary stored payload, which imports and constructs
            third-party classes; anything they raise surfaces here. That is why
            :func:`migrate_documents` catches broadly rather than repeating a
            tuple that has to grow every time a deserializer gains a failure
            mode.
    """
    team_card = TeamCard.model_validate(document[LEGACY_CARD_KEY])
    projection = derive_team_projection(team_card)

    # Cards FIRST, document second (FR13) — a stored Process must never
    # reference a blob that is not there. The store normalises the hireable flag
    # off each card; it belongs to the AgentCardRef, not to the shared blob.
    store.save_agent_cards(projection.cards)

    overlay = projection.model_dump()
    for key in _PROJECTION_ONLY_KEYS:
        overlay.pop(key, None)
    migrated = {key: value for key, value in document.items() if key != LEGACY_CARD_KEY}
    migrated.update(overlay)
    store.save_team(Process.model_validate(migrated))


def migrate_documents(documents: Iterable[Any], store: EventStore) -> MigrationReport:
    """Migrate every document in *documents*, counting the outcome.

    Idempotent by classification rather than by comparison: a document already
    carrying ``entry_point`` is skipped before ``team_card`` is even consulted,
    so a second run over a migrated store performs no writes at all.

    One document's failure is counted and logged; the run continues. A store
    holding a single malformed team must still migrate every other team in it —
    including the common real case of a card naming an agent class that has
    since been renamed, moved, or removed with its package.

    Args:
        documents: Raw stored mappings, in any order. A non-mapping is counted
            as a failure rather than raising — it is a corrupted document like
            any other.
        store: The store to write into.

    Returns:
        The three counts. ``failed > 0`` is what each script turns into a
        non-zero exit code.
    """
    report = MigrationReport()
    for document in documents:
        if not isinstance(document, Mapping):
            logger.error(
                "Skipping a stored team document that is not a mapping (%s)",
                type(document).__name__,
            )
            report.failed += 1
            continue
        team_id = _document_team_id(document)
        if PROJECTION_MARKER_KEY in document:
            logger.debug("Team %s already carries the projection; skipping", team_id)
            report.skipped += 1
            continue
        if LEGACY_CARD_KEY not in document:
            logger.warning(
                "Team %s carries no '%s' to derive from; skipping",
                team_id,
                LEGACY_CARD_KEY,
            )
            report.skipped += 1
            continue
        try:
            migrate_document(document, store)
        except Exception as exc:
            # Deliberately every exception, not a tuple. This loop's whole
            # contract is that one bad document must not stop the run, and
            # migrate_document deserializes an arbitrary stored payload —
            # import_class alone raises ImportError for a module that has moved
            # and AttributeError for a class that has been renamed, and the
            # third-party classes it constructs can raise anything at all. A
            # named tuple can only list the failures somebody has already seen;
            # the one it omits aborts a fleet migration inside the FR10 window,
            # where there is no version that can serve the store while it is
            # fixed. Nothing is swallowed: every failure is logged with its
            # team_id and counted, and a non-zero ``failed`` is what each script
            # turns into a non-zero exit code.
            logger.error("Failed to migrate team %s: %s", team_id, exc)
            report.failed += 1
            continue
        report.converted += 1
        logger.info("Migrated team %s to the structural projection", team_id)
    return report


def log_report(report: MigrationReport) -> None:
    """Log *report* at INFO — the one line every script ends with."""
    logger.info(
        "Migration finished: converted=%d skipped=%d failed=%d",
        report.converted,
        report.skipped,
        report.failed,
    )
