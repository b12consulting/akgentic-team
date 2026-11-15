"""Copy every team from one ``EventStore`` to another.

Backend-agnostic: it reads and writes only through the ``EventStore`` Protocol,
so the same function moves a YAML store into Postgres or Mongo, or between two
databases. ``akgentic.team.scripts.copy_yaml_to_postgres`` is the command-line
entry point for the case the community tier needs.

Per team, in this order:

1. **Agent cards** the team references (``Process.agent_cards``), then a check
   that every one of them resolves on the target. A team is never written
   pointing at a blob the target does not hold (FR13).
2. **Events** with a ``sequence`` above the target's current maximum, so a
   re-run appends only what is missing and never trips the
   ``(team_id, sequence)`` uniqueness of a database target.
3. **Agent states**, upserted.
4. **The team record** (``Process``), last. A team appears on the target only
   once everything it points at is there; an interrupted run leaves no
   half-copied team listed, and running again finishes it.

Re-running is safe: cards are content-addressed, events are skipped by
sequence, states and the team record are upserts. Teams are copied with the
status they have — ``RUNNING``, ``STOPPED`` and retired (``DELETED``) alike.

**What it does not copy.** Documents the source cannot load are skipped by the
source's own ``list_teams`` (logged there, naming the team); run the source's
``migrate_*`` script first so every document has the current shape. Card blobs
no team references are left behind — they are exactly what the card-store sweep
reclaims. A card's ``first_seen_at`` is stamped anew on the target, which makes
it younger, never older, than it was: the safe direction for that sweep.

Stop the server before copying. The source must not change underneath the copy.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from akgentic.team.models import Process, TeamStatus
from akgentic.team.ports import EventStore

__all__ = ["CopyFailure", "CopyReport", "copy_store", "log_copy_report"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CopyFailure:
    """One team that could not be copied, and why."""

    team_id: uuid.UUID
    reason: str


@dataclass
class CopyReport:
    """What a copy did. With ``dry_run`` the counts are what it *would* write."""

    dry_run: bool = False
    teams: int = 0
    events: int = 0
    agent_states: int = 0
    agent_cards: int = 0
    failed: list[CopyFailure] = field(default_factory=list)


def copy_store(source: EventStore, target: EventStore, *, dry_run: bool = False) -> CopyReport:
    """Copy every team, with its events, agent states and cards, from ``source`` to ``target``.

    A team that fails is recorded in ``CopyReport.failed`` and the copy moves on
    to the next one; a failed team is never written to the target's team table,
    so it does not appear there half-copied.

    Args:
        source: The store to read from. Never written.
        target: The store to write to. Its tables must already exist.
        dry_run: Read and count, write nothing.

    Returns:
        The counts of what was copied (or would be) and the teams that failed.
    """
    report = CopyReport(dry_run=dry_run)
    for process in _all_teams(source):
        try:
            _copy_team(process, source, target, report, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001 — one team must not stop the copy
            logger.error("Could not copy team %s: %s", process.team_id, exc)
            report.failed.append(CopyFailure(team_id=process.team_id, reason=str(exc)))
    return report


def log_copy_report(report: CopyReport) -> None:
    """Log the report the way the ``migrate_*`` scripts log theirs."""
    verb = "Would copy" if report.dry_run else "Copied"
    logger.info(
        "%s %d team(s), %d event(s), %d agent state(s), %d agent card(s); %d team(s) failed",
        verb,
        report.teams,
        report.events,
        report.agent_states,
        report.agent_cards,
        len(report.failed),
    )
    for failure in report.failed:
        logger.error("Failed: team %s — %s", failure.team_id, failure.reason)


def _all_teams(store: EventStore) -> list[Process]:
    """Every team the store can load, retired ones included.

    ``list_teams()`` hides ``DELETED`` under either delete mode, so retired
    teams are asked for separately.
    """
    return [*store.list_teams(), *store.list_teams(status=TeamStatus.DELETED)]


def _copy_team(
    process: Process,
    source: EventStore,
    target: EventStore,
    report: CopyReport,
    *,
    dry_run: bool,
) -> None:
    team_id = process.team_id

    # Read everything first: a source that fails to read fails the team before
    # anything of it is written.
    hashes = sorted({ref.card_hash for ref in process.agent_cards})
    cards = source.load_agent_cards(hashes)
    missing_on_source = [h for h in hashes if h not in cards]
    if missing_on_source:
        raise LookupError(f"source holds no card for hash(es) {', '.join(missing_on_source)}")
    events = source.load_events(team_id)
    states = source.load_agent_states(team_id)

    already = target.get_max_sequence(team_id)
    new_events = [e for e in events if e.sequence > already]

    if not dry_run:
        target.save_agent_cards(list(cards.values()))
        resolved = target.load_agent_cards(hashes)
        unresolved = [h for h in hashes if h not in resolved]
        if unresolved:
            # A legacy blob whose bytes no longer hash to its key lands under a
            # different hash on the target; the Process would point at nothing.
            raise LookupError(
                f"card(s) {', '.join(unresolved)} do not resolve on the target after copying"
            )
        for event in new_events:
            target.save_event(event)
        for state in states:
            target.save_agent_state(state)
        target.save_team(process)

    report.teams += 1
    report.agent_cards += len(cards)
    report.events += len(new_events)
    report.agent_states += len(states)
    logger.info(
        "%s team %s: %d new event(s), %d agent state(s), %d card(s)",
        "Would copy" if dry_run else "Copied",
        team_id,
        len(new_events),
        len(states),
        len(cards),
    )
