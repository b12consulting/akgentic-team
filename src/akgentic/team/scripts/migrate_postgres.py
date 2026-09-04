"""Migrate a PostgreSQL team store to the structural ``Process`` projection.

``NagraEventStore.save_team`` writes the whole ``Process`` as JSON into the
``data`` column, so a Postgres document carries a nested ``team_card`` exactly
as a YAML or Mongo one does. There is no shape-specific work here: the same
derivation behind a third reader/writer pair.

Invoked as::

    python -m akgentic.team.scripts.migrate_postgres

Configuration reuses the name ``akgentic.team.scripts.init_db`` already uses —
``DB_CONN_STRING_PERSISTENCE`` — so one ``.env`` drives both. It may be
overridden on the command line.

Run ``python -m akgentic.team.scripts.init_db`` first if the store predates the
content-addressed card table: the migration writes into ``agent_card_entries``,
which ``init_db`` provisions.

Idempotent: a document that already carries the projection is skipped without a
write, so re-running after a partial run costs nothing and changes nothing.

The read is deliberately raw, and every row is fetched before the first write.
``list_teams`` skips exactly the documents this script exists to convert, so a
migration built on the public read path would convert nothing and report
success.

Exit codes:
    0 — every document converted or skipped
    1 — at least one document could not be converted, or the backend is
        unavailable or unreachable
    2 — ``DB_CONN_STRING_PERSISTENCE`` not set and not passed
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

from akgentic.team.migration import ROLLOUT_ORDERING_NOTE, log_report, migrate_documents

logger = logging.getLogger(__name__)

_ENV_VAR = "DB_CONN_STRING_PERSISTENCE"

_SELECT_TEAMS = "SELECT id, data FROM team_process_entries"


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser, whose ``--help`` carries the rollout ordering."""
    parser = argparse.ArgumentParser(
        prog="python -m akgentic.team.scripts.migrate_postgres",
        description=(
            "Migrate a PostgreSQL team store to the structural Process projection. "
            "Idempotent: already-migrated documents are skipped."
        ),
        epilog=ROLLOUT_ORDERING_NOTE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--conn-string",
        default=None,
        help=f"Nagra-compatible Postgres connection string. Defaults to {_ENV_VAR}.",
    )
    return parser


def read_documents(conn_string: str) -> list[Any]:
    """Return every stored team document, decoded but not validated.

    The whole result set is fetched before the caller writes anything: the
    migration rewrites rows of the very table being read.
    """
    from nagra import Transaction  # type: ignore[import-untyped]

    from akgentic.team.repositories.postgres._queries import decode_jsonb_column

    with Transaction(conn_string) as trn:
        rows = trn.execute(_SELECT_TEAMS).fetchall()
    return [decode_jsonb_column(row[1]) for row in rows]


def main(argv: list[str] | None = None) -> int:
    """Migrate the configured PostgreSQL store.

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
    conn_string = args.conn_string or os.environ.get(_ENV_VAR)
    if not conn_string:
        logger.error("%s is not set; cannot migrate the team store", _ENV_VAR)
        return 2

    try:
        from akgentic.team.repositories.postgres import NagraEventStore
    except ImportError as exc:
        logger.error("Postgres backend unavailable: %s", exc)
        return 1

    try:
        store = NagraEventStore(conn_string)
        documents = read_documents(conn_string)
        report = migrate_documents(documents, store)
    except Exception:
        logger.exception("Postgres migration failed")
        return 1

    log_report(report)
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
