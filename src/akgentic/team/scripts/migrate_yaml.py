"""Migrate a YAML team store to the structural ``Process`` projection.

Reads every ``{data_dir}/{team_uuid}/team.yaml`` raw, converts the
pre-projection ones through :func:`akgentic.team.migration.migrate_documents`,
and writes the agent cards and the rewritten document back through
``YamlEventStore``.

Invoked as::

    python -m akgentic.team.scripts.migrate_yaml --data-dir /var/lib/akgentic/teams

Idempotent: a document that already carries the projection is skipped without a
write, so re-running after a partial run costs nothing and changes nothing.

The read is deliberately raw. ``list_teams`` returns *nothing* for an unmigrated
store — it skips exactly the documents this script exists to convert — so a
migration built on the public read path would convert nothing and report
success.

Exit codes:
    0 — every document converted or skipped
    1 — at least one document could not be converted (each is logged)
    2 — ``--data-dir`` missing (argparse) or not a directory
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from akgentic.team.migration import ROLLOUT_ORDERING_NOTE, log_report, migrate_documents
from akgentic.team.repositories.yaml import CARDS_DIRNAME, YamlEventStore

logger = logging.getLogger(__name__)

_TEAM_FILENAME = "team.yaml"


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser, whose ``--help`` carries the rollout ordering."""
    parser = argparse.ArgumentParser(
        prog="python -m akgentic.team.scripts.migrate_yaml",
        description=(
            "Migrate a YAML team store to the structural Process projection. "
            "Idempotent: already-migrated documents are skipped."
        ),
        epilog=ROLLOUT_ORDERING_NOTE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root directory of the YAML store — the one YamlEventStore was built with.",
    )
    return parser


def _read_team_document(team_path: Path, team_id: str) -> Any:
    """Parse one ``team.yaml`` without validating it.

    An unreadable or unparseable file yields ``None`` rather than raising: the
    migration counts it as a failure and carries on to the next team, which is
    the whole point of counting failures instead of aborting.
    """
    try:
        with open(team_path) as handle:
            return yaml.safe_load(handle)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        # ValueError covers a file that is not valid UTF-8, whose
        # UnicodeDecodeError surfaces out of the text stream rather than as a
        # yaml.YAMLError — the same pair YamlEventStore._load_team_data handles.
        logger.error("Unreadable %s for team %s: %s", _TEAM_FILENAME, team_id, exc)
        return None


def read_documents(data_dir: Path) -> Iterator[Any]:
    """Yield the raw stored document of every team directory under *data_dir*.

    ``agent_cards/`` is skipped **by name** — it is a deliberate sibling of the
    per-team directories, not a team — and any other non-UUID directory is
    skipped with a warning, matching ``YamlEventStore.list_teams``.
    """
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir() or child.name == CARDS_DIRNAME:
            continue
        try:
            uuid.UUID(child.name)
        except ValueError:
            logger.warning("Skipping non-team directory: %s", child.name)
            continue
        team_path = child / _TEAM_FILENAME
        if not team_path.exists():
            continue
        yield _read_team_document(team_path, child.name)


def main(argv: list[str] | None = None) -> int:
    """Migrate the configured YAML store.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.

    Returns:
        Process exit code — 0 on success, 1 when a document failed, 2 when the
        data directory is unusable.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        logger.error("%s is not a directory; nothing to migrate", data_dir)
        return 2

    try:
        store = YamlEventStore(data_dir)
        report = migrate_documents(read_documents(data_dir), store)
    except Exception:
        logger.exception("YAML migration failed")
        return 1

    log_report(report)
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
