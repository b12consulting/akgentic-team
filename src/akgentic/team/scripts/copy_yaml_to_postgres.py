"""Copy a YAML team store into PostgreSQL.

Invoked as::

    python -m akgentic.team.scripts.copy_yaml_to_postgres --data-dir data/event_store

The target DSN comes from ``DB_CONN_STRING_PERSISTENCE`` — the variable
``init_db`` and ``migrate_postgres`` already read — or ``--conn-string``.

Before running:

1. Stop the server: the YAML store must not change during the copy.
2. ``python -m akgentic.team.scripts.migrate_yaml`` on the source, so every
   document has the current shape (documents that do not load are skipped).
3. ``python -m akgentic.team.scripts.init_db`` on the target.

Safe to re-run: see :mod:`akgentic.team.store_copy`. ``--dry-run`` reads
both stores and reports what would be copied, writing nothing.

The YAML store is only read; delete it yourself once the copy is verified.

Exit codes:
    0 — every team copied (or would be, with ``--dry-run``)
    1 — at least one team failed, or the backend is unavailable or unreachable
    2 — ``DB_CONN_STRING_PERSISTENCE`` not set and not passed, or the source
        directory does not exist
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from akgentic.team.repositories.yaml import YamlEventStore
from akgentic.team.store_copy import copy_store, log_copy_report

logger = logging.getLogger(__name__)

_ENV_VAR = "DB_CONN_STRING_PERSISTENCE"


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m akgentic.team.scripts.copy_yaml_to_postgres",
        description=(
            "Copy every team, with its events, agent states and agent cards, from a "
            "YAML event store into PostgreSQL. Safe to re-run."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory of the YAML event store (AKGENTIC_EVENT_STORE_PATH).",
    )
    parser.add_argument(
        "--conn-string",
        default=None,
        help=f"postgresql:// DSN of the target. Defaults to {_ENV_VAR}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read both stores and report what would be copied; write nothing.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Copy the YAML store at ``--data-dir`` into the configured Postgres.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Process exit code — see the module docstring.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = build_parser().parse_args(argv)

    conn_string = args.conn_string or os.environ.get(_ENV_VAR)
    if not conn_string:
        logger.error("%s is not set and --conn-string was not passed", _ENV_VAR)
        return 2
    if not conn_string.startswith("postgresql://"):
        # Nagra rejects any other scheme with an error that prints the DSN.
        logger.error("The target DSN must start with 'postgresql://'")
        return 2
    if not args.data_dir.is_dir():
        logger.error("YAML store not found: %s is not a directory", args.data_dir)
        return 2

    try:
        from akgentic.team.repositories.postgres import NagraEventStore
    except ImportError as exc:
        logger.error("Postgres backend unavailable: %s", exc)
        return 1

    try:
        report = copy_store(
            YamlEventStore(args.data_dir),
            NagraEventStore(conn_string),
            dry_run=args.dry_run,
        )
    except Exception:
        # Per-team failures — the target unreachable or missing its tables
        # included — are recorded in the report. Reaching here means the
        # source itself could not be listed.
        logger.exception("Copy failed")
        return 1

    log_copy_report(report)
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
