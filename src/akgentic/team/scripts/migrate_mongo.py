"""Migrate a MongoDB team store to the structural ``Process`` projection.

Reads the ``teams`` collection raw, converts the pre-projection documents
through :func:`akgentic.team.migration.migrate_documents`, and writes the agent
cards and the rewritten documents back through ``MongoEventStore``.

Invoked as::

    python -m akgentic.team.scripts.migrate_mongo

Configuration reuses the names ``akgentic.team.scripts.init_mongo`` already
uses — ``MONGO_URI`` and ``MONGO_DB`` — so one ``.env`` drives both. Either may
be overridden on the command line.

Idempotent: a document that already carries the projection is skipped without a
write, so re-running after a partial run costs nothing and changes nothing.

The read is deliberately raw, and the cursor is materialised before the first
write. ``list_teams`` returns *nothing* for an unmigrated store — it skips
exactly the documents this script exists to convert — and rewriting documents
while iterating a live cursor could hand the same document back twice.

Exit codes:
    0 — every document converted or skipped
    1 — at least one document could not be converted, the backend is
        unavailable, or the server could not be reached
    2 — ``MONGO_URI`` or ``MONGO_DB`` not set and not passed
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

from akgentic.team.migration import ROLLOUT_ORDERING_NOTE, log_report, migrate_documents

logger = logging.getLogger(__name__)

_URI_ENV = "MONGO_URI"
_DB_ENV = "MONGO_DB"


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser, whose ``--help`` carries the rollout ordering."""
    parser = argparse.ArgumentParser(
        prog="python -m akgentic.team.scripts.migrate_mongo",
        description=(
            "Migrate a MongoDB team store to the structural Process projection. "
            "Idempotent: already-migrated documents are skipped."
        ),
        epilog=ROLLOUT_ORDERING_NOTE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mongo-uri",
        default=None,
        help=f"MongoDB connection URI. Defaults to the {_URI_ENV} environment variable.",
    )
    parser.add_argument(
        "--mongo-db",
        default=None,
        help=f"Database holding the team collections. Defaults to {_DB_ENV}.",
    )
    return parser


def _strip_mongo_id(document: dict[str, Any]) -> dict[str, Any]:
    """Return *document* without Mongo's ``_id``, which is not a ``Process`` field."""
    document.pop("_id", None)
    return document


def main(argv: list[str] | None = None) -> int:
    """Migrate the configured MongoDB store.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Process exit code — 0 on success, 1 on a failed document or an
        unusable backend, 2 on missing configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = build_parser().parse_args(argv)
    uri = args.mongo_uri or os.environ.get(_URI_ENV)
    if not uri:
        logger.error("%s is not set; cannot migrate the team store", _URI_ENV)
        return 2
    db_name = args.mongo_db or os.environ.get(_DB_ENV)
    if not db_name:
        logger.error("%s is not set; cannot migrate the team store", _DB_ENV)
        return 2

    try:
        import pymongo

        from akgentic.team.repositories.mongo import TEAMS_COLLECTION, MongoEventStore
    except ImportError as exc:
        logger.error("Mongo backend unavailable: %s", exc)
        return 1

    try:
        with pymongo.MongoClient(uri) as client:  # type: ignore[var-annotated]
            # Reach the server before touching anything. MongoClient connects
            # lazily, so without this an unreachable or authentication-rejected
            # server would migrate nothing and exit 0 having read no documents.
            client.admin.command("ping")
            db = client[db_name]
            # Materialised before the first write: rewriting documents while a
            # cursor over the same collection is open can hand one back twice.
            documents = [_strip_mongo_id(doc) for doc in db[TEAMS_COLLECTION].find({})]
            report = migrate_documents(documents, MongoEventStore(db))
    except Exception:
        logger.exception("Mongo migration failed")
        return 1

    log_report(report)
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
