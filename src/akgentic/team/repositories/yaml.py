"""YamlEventStore: file-based EventStore with per-team directory layout.

Persists team data to YAML files in a per-team directory structure using
PyYAML for serialization and pathlib for filesystem operations. Satisfies
the EventStore protocol via structural subtyping (no explicit inheritance).

File layout per team::

    {data_dir}/
      agent_cards/          # Content-addressed card store, SHARED by every team
        {card_hash}.yaml    #   one blob per card (overwrite; immutable content)
      {team_uuid}/
        team.yaml           # Process metadata (overwrite)
        events.yaml         # Append-only event log (multi-document YAML)
        states/
          {agent_id}.yaml   # Latest agent state snapshot (overwrite)

``agent_cards/`` is a **sibling** of the per-team directories, not a child of
one. ``delete_team`` removes a whole team directory with ``shutil.rmtree``, so a
card store nested inside one would be deleted with the first team that referenced
it — making "cards are never deleted" (FR13) unsatisfiable by construction.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from akgentic.core.agent_card import AgentCard
from akgentic.team.metadata import make_index_prefix_groups
from akgentic.team.models import (
    AgentCardEntry,
    AgentStateSnapshot,
    PersistedEvent,
    Process,
    TeamStatus,
)
from akgentic.team.ports import EventLogUnreadableError, EventNotFoundError
from akgentic.team.projection import hash_agent_card, storable_agent_card

logger = logging.getLogger(__name__)

CARDS_DIRNAME = "agent_cards"
"""Name of the shared card directory under ``data_dir``.

Public because ``list_teams`` must skip it by name and tests assert the layout.
A team directory is always a UUID, so the two namespaces cannot collide.
"""

CARD_ENVELOPE_KEY = "card"
"""Key holding the card payload inside a card file's envelope.

Also the **discriminator** between the two shapes a card file can have. A file
written before the envelope existed *is* the bare card, whose top-level keys are
``AgentCard``'s own — ``agent_class``, ``skills``, ``description``, ``config``,
``can_be_hired``, ``metadata`` and the ``__model__`` tag. None of them is
``card``, so its presence cannot be confused with a bare card. Public because
the YAML layout specs assert it.
"""

CARD_FIRST_SEEN_KEY = "first_seen_at"
"""Key holding the first-seen stamp inside a card file's envelope.

Written once, when the file is created, and carried forward verbatim by every
later save. Public for the same reason as :data:`CARD_ENVELOPE_KEY`.

The file's mtime is NOT a substitute: ``_atomic_write`` rewrites the file
wholesale on every save, so the mtime moves exactly when this must not.
"""

_HEX_DIGITS = frozenset("0123456789abcdef")
_CARD_HASH_LENGTH = 64


def _read_card_document(document: object) -> tuple[object, datetime | None]:
    """Split one loaded card file into its card payload and its stamp.

    THE reader for both shapes a card file can have, used by
    ``load_agent_cards`` and ``list_agent_card_entries`` alike so the two can
    never disagree about what a legacy file means:

    * an **envelope** — ``{card: ..., first_seen_at: ...}`` — which every save
      since the stamp landed writes;
    * a **bare card**, which is every card file written before it and therefore
      every card file in every deployment today. It is NOT corrupted, and must
      not be logged as such; it simply has no stamp, so its age is unknown.

    A stamp that is not a datetime — absent, or some other type from a foreign
    writer — reads as ``None``, never as an epoch: unknown age, which a consumer
    must treat as too young to reclaim.

    Returns:
        ``(card_payload, first_seen_at)``. The payload is handed on unvalidated;
        the caller decides whether it needs to be an ``AgentCard``.
    """
    if not isinstance(document, Mapping) or CARD_ENVELOPE_KEY not in document:
        return document, None
    stamp = document.get(CARD_FIRST_SEEN_KEY)
    if not isinstance(stamp, datetime):
        return document[CARD_ENVELOPE_KEY], None
    # PyYAML hands back a tz-aware value for the offset this module writes; a
    # naive one can only come from a file written by something else, and UTC is
    # the only zone this store ever stamps in.
    return document[CARD_ENVELOPE_KEY], stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)


def _is_card_hash(value: str) -> bool:
    """Return whether *value* has the shape ``hash_agent_card`` produces.

    This backend is the one that turns a hash into a **path**, so a hash it did
    not compute must not be allowed to name a file: ``../../secrets`` is a valid
    ``str`` and reads outside the store. The hashes reaching ``load_agent_cards``
    come off a persisted ``Process``, which this module already treats as
    untrusted enough to guard corrupted payloads against.

    A SHA-256 hex digest is the only key the store ever writes, so constraining
    reads to that shape costs nothing and makes a malformed key a clean miss
    (``AgentCardNotFoundError`` naming the role) rather than a stray filesystem
    read.
    """
    return len(value) == _CARD_HASH_LENGTH and _HEX_DIGITS.issuperset(value)


class YamlEventStore:
    """File-based EventStore using YAML serialization with per-team directories.

    Satisfies the ``EventStore`` protocol via structural subtyping without
    inheriting from it. All filesystem directories are created on demand
    (not eagerly at instantiation time).

    Args:
        data_dir: Root directory for all persisted team data.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    @staticmethod
    def _atomic_write(path: Path, data: dict[str, object]) -> None:
        """Write YAML data atomically using write-to-temp-then-rename.

        Prevents corrupted partial files if the process crashes mid-write.

        Serializes through ``yaml.safe_dump``, which is the same dialect
        ``yaml.safe_load`` reads: the unsafe dumper emits a ``!!python/object``
        tag for any value it has no representer for, and the safe loader every
        read path here uses then refuses to construct it — a pair that can write
        a file it cannot read back. A value the reader could not reconstruct now
        fails the write instead, while the data is still in hand.

        The existing ``except BaseException`` gives that refusal its no-partial-
        write property for free: the dump raises into the temp file, the temp is
        unlinked, and ``path`` is never touched, so the previous good document
        survives intact.

        Args:
            path: Destination file path.
            data: Dictionary to serialize as YAML.

        Raises:
            yaml.YAMLError: If *data* holds a value the safe dumper cannot
                represent. ``path`` is left byte-identical.
        """
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with open(fd, "w") as f:
                yaml.safe_dump(data, f, default_flow_style=False)
            Path(tmp).replace(path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def _team_dir(self, team_id: uuid.UUID) -> Path:
        """Return the directory path for a specific team.

        Args:
            team_id: Unique identifier of the team.

        Returns:
            Path to the team's directory under the data root.
        """
        return self._data_dir / str(team_id)

    def _cards_dir(self) -> Path:
        """Return the shared card directory — a sibling of the team directories.

        Deliberately NOT under a team directory: ``delete_team`` rmtrees those,
        and cards outlive every team that references them (FR13).
        """
        return self._data_dir / CARDS_DIRNAME

    def save_team(self, process: Process) -> None:
        """Persist a team process snapshot to team.yaml.

        Creates the team directory if it does not exist, then writes
        (or overwrites) the serialized Process to ``team.yaml``.

        Args:
            process: The team process snapshot to persist.
        """
        team_dir = self._team_dir(process.team_id)
        team_dir.mkdir(parents=True, exist_ok=True)
        team_path = team_dir / "team.yaml"
        self._atomic_write(team_path, process.model_dump())
        logger.debug("Saved team %s to %s", process.team_id, team_path)

    def _load_team_data(self, team_id: uuid.UUID) -> Any:
        """Read and parse team.yaml WITHOUT validating it.

        Splitting the read from the validation is what lets ``list_teams``
        decide whether it wants a team before paying for
        ``Process.model_validate`` — see ADR-23 §3.

        Args:
            team_id: Unique identifier of the team.

        Returns:
            The raw parsed document — normally a mapping, but any YAML shape
            is possible — or None if the file is absent or unparseable.
        """
        team_path = self._team_dir(team_id) / "team.yaml"
        if not team_path.exists():
            return None
        try:
            with open(team_path) as f:
                return yaml.safe_load(f)
        except (yaml.YAMLError, ValueError) as exc:
            # ValueError is load-bearing, not defensive padding: a file that
            # is not valid UTF-8 fails in the text stream's decode inside
            # safe_load and surfaces as UnicodeDecodeError, a ValueError that
            # is NOT a yaml.YAMLError. Together with the ValueError clause in
            # _validate_team_data this reproduces exactly the one
            # (yaml.YAMLError, ValueError) clause load_team used to have, so
            # unreadable bytes stay a skip and never escape a list_teams call.
            logger.error("Corrupted team.yaml for team %s: %s", team_id, exc)
            return None

    def _validate_team_data(self, team_id: uuid.UUID, data: Any) -> Process | None:
        """Hydrate a raw parsed document into a Process.

        Args:
            team_id: Unique identifier of the team, for the error log.
            data: Raw parsed document as returned by ``_load_team_data``.

        Returns:
            The validated Process, or None if the document is corrupted.
            Pydantic's ValidationError is a ValueError subclass, which is
            why the clause is not narrowed to ValidationError.
        """
        try:
            return Process.model_validate(data)
        except ValueError as exc:
            logger.error("Corrupted team.yaml for team %s: %s", team_id, exc)
            return None

    @staticmethod
    def _matches(
        data: Any,
        user_id: str | None,
        status: TeamStatus | None,
        prefix_groups: list[list[str]],
    ) -> bool:
        """Test a RAW parsed team.yaml mapping against the requested filters.

        ``TeamStatus`` is a ``StrEnum`` and ``save_team`` persists
        ``model_dump()`` output, so the stored values are plain strings that
        already compare equal to the enum members — no round-trip needed.
        A document that is not a mapping cannot match a filter, so it is
        skipped one step ahead of the corrupted-document skip that catches
        it today. With no filter at all it is passed through untouched, so
        the unfiltered result set — corrupted-document log line included —
        does not move (ADR-23 §3).

        Each filter is an independent guard, so a further one is another two
        lines and never disturbs the ones already here.

        Args:
            data: Raw parsed document as returned by ``_load_team_data``.
            user_id: Owning-user filter, or None for no user filter.
            status: Lifecycle-state filter, or None for no status filter.
            prefix_groups: One group of rendered ``"key|value"`` prefixes per
                filtered key; empty for no metadata filter. **Prefixes within a
                group are a disjunction and the groups are a conjunction** —
                same key ORs, different keys AND. Matching is anchored prefix in
                the direction *stored entry starts with prefix*, never the
                reverse, which would let a term crafted to span two entries
                match (ADR-28 §D3). A stored ``metadata_indexes`` that is
                missing or not a list is a non-match, never a raise: a team
                written before the metadata contract existed simply carries
                nothing to match (ADR-24 §D5).

        Returns:
            True if the document should be hydrated and returned.
        """
        if user_id is None and status is None and not prefix_groups:
            return True
        if not isinstance(data, dict):
            return False
        if user_id is not None and data.get("user_id") != user_id:
            return False
        if status is not None and data.get("status") != status:
            return False
        if prefix_groups:
            stored = data.get("metadata_indexes")
            if not isinstance(stored, list):
                return False
            entries = [e for e in stored if isinstance(e, str)]
            if not all(
                any(e.startswith(p) for p in group for e in entries) for group in prefix_groups
            ):
                return False
        return True

    def load_team(self, team_id: uuid.UUID) -> Process | None:
        """Load a team process snapshot from team.yaml.

        Args:
            team_id: Unique identifier of the team.

        Returns:
            The deserialized Process, or None if no team.yaml exists or the
            document is corrupted.
        """
        data = self._load_team_data(team_id)
        if data is None:
            # Also the empty-file case: an empty team.yaml parses to None.
            # It used to reach Process.model_validate and emit the corrupted-
            # document error log before returning None. The return value is
            # unchanged; only that log line is gone. Deliberate, ADR-23 §6.
            return None
        process = self._validate_team_data(team_id, data)
        if process is None:
            return None
        logger.debug("Loaded team %s from %s", team_id, self._team_dir(team_id) / "team.yaml")
        return process

    def list_teams(
        self,
        user_id: str | None = None,
        status: TeamStatus | None = None,
        metadata: Mapping[str, list[str]] | None = None,
    ) -> list[Process]:
        """Load matching team process snapshots from the data directory.

        Iterates subdirectories of ``data_dir``, attempts to parse each
        directory name as a UUID, and reads the team snapshot for valid
        team directories. Non-UUID directories are skipped with a warning.

        ALL filters are evaluated on the raw parsed mapping, ahead of
        ``Process.model_validate``, so a team that will not be returned is
        never hydrated into a full ``TeamCard`` object graph (ADR-23 §3,
        ADR-24 §D5). That is why this reads through ``_load_team_data``
        instead of calling ``load_team``: going back through ``load_team``
        for the survivors would re-read and re-parse each file. The walk
        itself is still O(total teams) — this is a constant-factor win,
        nothing more.

        Args:
            user_id: If provided, return only snapshots whose
                ``Process.user_id`` matches. If ``None`` (default), return all
                snapshots. See ADR-16 §1.
            status: If provided, return only snapshots whose
                ``Process.status`` matches. If ``None`` (default), every
                lifecycle state is returned, including ``DELETED``. See
                ADR-23 §1.
            metadata: Mapping of indexed field name to a list of prefix terms.
                Terms for one key OR-combine; distinct keys AND-combine. Empty
                terms drop out, so ``{}``, ``{"tenant": []}``, ``{"tenant": [""]}``
                and ``None`` all behave alike and leave the walk with no metadata
                predicate at all. See ADR-24 §D5 and ADR-28 §D3/§D7.

        The three filters are independent terms combining as a conjunction;
        one left at ``None`` constrains nothing.

        Returns:
            List of all loadable Process snapshots matching every filter
            that was provided.

        Raises:
            TypeError: If a ``metadata`` value is a bare ``str``.
        """
        # Rendered BEFORE the early return, so a bare-``str`` value is rejected
        # on an empty data directory exactly as it is on a populated one.
        # Translated once, before the walk — never per team. An all-empty
        # mapping renders to [], which is "no metadata filter", and a key whose
        # terms all render away contributes no group rather than an empty one.
        prefix_groups = make_index_prefix_groups(metadata)
        if not self._data_dir.exists():
            return []
        teams: list[Process] = []
        for child in sorted(self._data_dir.iterdir()):
            if not child.is_dir():
                continue
            if child.name == CARDS_DIRNAME:
                # The shared card store is a deliberate sibling of the team
                # directories, so it is skipped SILENTLY — the warning below
                # fires once per list_teams call, not once per team, and would
                # otherwise log on every single call for the rest of time.
                continue
            try:
                team_id = uuid.UUID(child.name)
            except ValueError:
                logger.warning("Skipping non-team directory: %s", child.name)
                continue
            data = self._load_team_data(team_id)
            if data is None or not self._matches(data, user_id, status, prefix_groups):
                continue
            # A survivor that fails validation is still dropped rather than
            # raised on, exactly as it was when load_team returned None.
            process = self._validate_team_data(team_id, data)
            if process is None:
                continue
            teams.append(process)
        return teams

    def save_event(self, event: PersistedEvent) -> None:
        """Append a persisted event to events.yaml.

        Uses multi-document YAML format (documents separated by ``---``)
        for append-only semantics. Creates the team directory if needed.

        Serialized to a string FIRST, then written in one call — not dumped into
        an open handle. Two reasons, and only the second is about this story:

        * ``safe_dump`` refuses a value ``safe_load_all`` could not construct.
          Dumping into the handle would have already written ``"---\\n"`` and
          possibly half a document before that refusal, leaving a stray
          separator on an append-only log that no later read can tell from a
          real one.
        * One ``write`` of separator-plus-body is also one append, so a
          concurrent appender cannot interleave into the middle of a document.

        The failure therefore happens before the file is opened at all, and
        ``events.yaml`` is byte-identical to what it was.

        Args:
            event: The event to append.

        Raises:
            yaml.YAMLError: If the event's ``model_dump()`` holds a value the
                safe dumper cannot represent. Nothing is appended.

                This propagates to ``PersistenceSubscriber`` and out to the
                orchestrator, which logs one ERROR per failed subscriber call
                and keeps the team running — one dropped event, not a dead
                team, and above all not a log that reads back as ``[]``. The
                subscriber has already consumed the sequence number, so a
                refused write leaves a **gap**; that is harmless here (reads
                sort by ``sequence``, ``get_max_sequence`` takes the max) where
                a duplicate would not be.
        """
        team_dir = self._team_dir(event.team_id)
        team_dir.mkdir(parents=True, exist_ok=True)
        events_path = team_dir / "events.yaml"
        body = yaml.safe_dump(event.model_dump(), default_flow_style=False)
        with open(events_path, "a") as f:
            f.write(f"---\n{body}")
        logger.debug("Appended event seq=%d for team %s", event.sequence, event.team_id)

    @staticmethod
    def _absent_log(team_id: uuid.UUID, after_event_id: uuid.UUID | None) -> list[PersistedEvent]:
        """Result for a team that has NO events.yaml at all.

        A team that has never persisted an event has an empty log, and an empty
        log is ``[]``. This is emphatically NOT the answer for a log that exists
        and will not parse — that raises ``EventLogUnreadableError`` at the
        parse site. The two used to share this helper, which is how a storage
        fault came to be indistinguishable from a brand-new team.

        Raises:
            EventNotFoundError: If a cursor was passed — an absent log cannot
                resolve an anchor, and ``[]`` would be read by the caller as
                "you are already up to date".
        """
        if after_event_id is not None:
            raise EventNotFoundError(f"Event {after_event_id} not found for team {team_id}")
        return []

    def load_events(
        self, team_id: uuid.UUID, after_event_id: uuid.UUID | None = None
    ) -> list[PersistedEvent]:
        """Load persisted events for a team from events.yaml, ordered by sequence.

        Args:
            team_id: Unique identifier of the team.
            after_event_id: If provided, return only events after the matching
                event — anchor excluded. If ``None`` (default), the full log.

        Returns:
            List of PersistedEvent ordered by sequence, or empty list if no
            events file exists and no cursor was passed.

        Raises:
            EventNotFoundError: If ``after_event_id`` does not resolve to an
                event of this team, including when the events file is absent.
            EventLogUnreadableError: If events.yaml exists but will not parse,
                on both the cursor and the no-cursor path. Returning ``[]`` here
                is what turned a 186 KB log into ``200 {"events": []}`` and then
                into a "no orchestrator" error against a team whose orchestrator
                was on disk the whole time.

        A document that parses but fails ``PersistedEvent.model_validate`` is a
        different case again and stays a per-document WARNING skip: one event
        lost rather than the log. That tolerance is deliberate.
        """
        events_path = self._team_dir(team_id) / "events.yaml"
        if not events_path.exists():
            return self._absent_log(team_id, after_event_id)
        try:
            with open(events_path) as f:
                docs = list(yaml.safe_load_all(f))
        except yaml.YAMLError as exc:
            # The ERROR line stays — it names the team and is what an operator
            # greps for — but it is no longer the ONLY signal. Raising is.
            logger.error("Corrupted events.yaml for team %s: %s", team_id, exc)
            raise EventLogUnreadableError(
                f"events.yaml for team {team_id} exists but could not be parsed: {exc}"
            ) from exc
        events: list[PersistedEvent] = []
        for doc in docs:
            if doc is None:
                continue
            try:
                events.append(PersistedEvent.model_validate(doc))
            except ValueError as exc:
                logger.warning(
                    "Skipping corrupted event for team %s: %s", team_id, exc
                )
        logger.debug("Loaded %d events for team %s", len(events), team_id)
        ordered = sorted(events, key=lambda e: e.sequence)
        if after_event_id is None:
            return ordered
        # Interim in-memory slice; YAML keeps it permanently (ADR-21 §4).
        # event.id is persisted as a string, so compare stringified ids.
        for index, event in enumerate(ordered):
            if str(event.event.id) == str(after_event_id):
                return ordered[index + 1 :]
        raise EventNotFoundError(f"Event {after_event_id} not found for team {team_id}")

    def get_max_sequence(self, team_id: uuid.UUID) -> int:
        """Return the highest event sequence number for a team, or 0.

        Loads all events and computes the max in Python. This is acceptable
        for a file-based store; database-backed stores should use an
        efficient query instead.

        Args:
            team_id: Unique identifier of the team.

        Returns:
            The highest sequence number, or 0 if no events exist.

        Raises:
            EventLogUnreadableError: Propagated from ``load_events`` when the
                log exists and will not parse. Deliberately not caught:
                answering 0 would restart the sequence at 1 and overwrite the
                numbering of a log that is still on disk.
        """
        events = self.load_events(team_id)
        return max((e.sequence for e in events), default=0)

    def save_agent_state(self, snapshot: AgentStateSnapshot) -> None:
        """Persist an agent state snapshot to states/{agent_id}.yaml.

        Creates the states directory if it does not exist, then writes
        (or overwrites) the serialized snapshot.

        Args:
            snapshot: The agent state snapshot to persist.
        """
        states_dir = self._team_dir(snapshot.team_id) / "states"
        states_dir.mkdir(parents=True, exist_ok=True)
        state_path = states_dir / f"{snapshot.agent_id}.yaml"
        self._atomic_write(state_path, snapshot.model_dump())
        logger.debug(
            "Saved agent state %s for team %s", snapshot.agent_id, snapshot.team_id
        )

    def load_agent_states(self, team_id: uuid.UUID) -> list[AgentStateSnapshot]:
        """Load all agent state snapshots for a team from states/ directory.

        Args:
            team_id: Unique identifier of the team.

        Returns:
            List of AgentStateSnapshot, or empty list if no states
            directory exists.
        """
        states_dir = self._team_dir(team_id) / "states"
        if not states_dir.exists():
            return []
        snapshots: list[AgentStateSnapshot] = []
        for state_path in sorted(states_dir.glob("*.yaml")):
            try:
                with open(state_path) as f:
                    data = yaml.safe_load(f)
                snapshots.append(AgentStateSnapshot.model_validate(data))
            except (yaml.YAMLError, ValueError) as exc:
                logger.warning(
                    "Skipping corrupted state file %s for team %s: %s",
                    state_path.name,
                    team_id,
                    exc,
                )
        logger.debug("Loaded %d agent states for team %s", len(snapshots), team_id)
        return snapshots

    def delete_team(self, team_id: uuid.UUID) -> None:
        """Delete all persisted data for a team.

        Removes the entire team directory and all contents. If the directory
        does not exist, this is a no-op (no error raised).

        Args:
            team_id: Unique identifier of the team to delete.
        """
        team_dir = self._team_dir(team_id)
        if team_dir.exists():
            shutil.rmtree(team_dir)
            logger.debug("Deleted team directory %s", team_dir)

    def save_agent_cards(self, cards: list[AgentCard]) -> None:
        """Persist agent cards as ``{data_dir}/agent_cards/{hash}.yaml``.

        Content-addressed, so the write is naturally idempotent: the same card
        always lands on the same path with the same bytes, whether it arrives
        once or from ten teams. Rewritten rather than skipped, through
        ``_atomic_write``, so a half-written file left by an earlier crash heals
        on the next save instead of being trusted forever.

        The file is an **envelope** — the card under :data:`CARD_ENVELOPE_KEY`
        and a first-seen stamp beside it. Because ``_atomic_write`` rewrites the
        file wholesale, the save must READ the existing stamp and carry it
        forward; only a file that does not exist yet is stamped with ``now``.
        That is this backend's expression of Mongo's ``$setOnInsert`` and
        Postgres' survives-by-omission ``DO UPDATE``, and it is what keeps the
        stamp first-seen rather than last-written.

        A file that exists but carries no readable stamp — every card file
        written before the envelope, and any file too damaged to parse — is
        rewritten WITHOUT one. Stamping it here would be a backfill by the back
        door, claiming a first sight the store never had.

        Args:
            cards: The cards to persist. An empty list touches no filesystem.
        """
        if not cards:
            return
        cards_dir = self._cards_dir()
        cards_dir.mkdir(parents=True, exist_ok=True)
        for card in cards:
            storable = storable_agent_card(card)
            card_path = cards_dir / f"{hash_agent_card(storable)}.yaml"
            first_seen = (
                self._existing_first_seen(card_path)
                if card_path.exists()
                else datetime.now(UTC)
            )
            self._atomic_write(
                card_path,
                {
                    CARD_ENVELOPE_KEY: storable.model_dump(),
                    CARD_FIRST_SEEN_KEY: first_seen,
                },
            )
        logger.debug("Saved %d agent cards to %s", len(cards), cards_dir)

    @staticmethod
    def _existing_first_seen(card_path: Path) -> datetime | None:
        """Return the stamp already on disk at *card_path*, or ``None``.

        ``None`` covers a bare pre-envelope file, an envelope whose stamp is
        absent, and a file that will not parse at all — all of which are ages
        this store does not know. Unknown fails closed downstream (too young to
        reclaim), so guessing here would be the one unsafe direction.
        """
        try:
            with open(card_path) as f:
                document = yaml.safe_load(f)
        except (OSError, yaml.YAMLError, ValueError) as exc:
            logger.warning("Could not read existing card file %s: %s", card_path.name, exc)
            return None
        return _read_card_document(document)[1]

    def load_agent_cards(self, hashes: list[str]) -> dict[str, AgentCard]:
        """Resolve card hashes against the card directory, one read per hash.

        The batch is a directory of files, so "one round trip" is one directory;
        there is no query to push down. A hash the directory does not hold is
        simply absent from the mapping, and a corrupted card file is skipped
        with a log exactly as a corrupted team or state file is — either way it
        surfaces as ``AgentCardNotFoundError`` at resolution rather than as an
        exception escaping the store.

        Both file shapes resolve, through :func:`_read_card_document`: the
        envelope this backend writes today, and the bare card every deployment's
        store is full of. A bare file is NOT corrupted and is not logged as
        such — a reader that assumed the envelope would turn every existing card
        into FR14's loud failure for a reason that is not the card's fault.

        Args:
            hashes: The content hashes to resolve; empty returns ``{}``.

        Returns:
            Mapping of hash to card for every hash the store holds.
        """
        if not hashes:
            return {}
        cards_dir = self._cards_dir()
        if not cards_dir.exists():
            return {}
        resolved: dict[str, AgentCard] = {}
        for card_hash in hashes:
            if not _is_card_hash(card_hash):
                # A key this store cannot have written. Skipped rather than
                # joined onto a path — see ``_is_card_hash``.
                logger.error("Skipping malformed agent card hash %r", card_hash)
                continue
            card_path = cards_dir / f"{card_hash}.yaml"
            if not card_path.exists():
                continue
            try:
                with open(card_path) as f:
                    payload, _ = _read_card_document(yaml.safe_load(f))
                resolved[card_hash] = AgentCard.model_validate(payload)
            except (yaml.YAMLError, ValueError) as exc:
                # ValueError covers both a non-UTF-8 file (UnicodeDecodeError
                # out of the text stream) and Pydantic's ValidationError, the
                # same pair _load_team_data and load_agent_states handle.
                logger.error("Skipping corrupted agent card %s: %s", card_hash, exc)
        logger.debug("Resolved %d of %d agent cards", len(resolved), len(hashes))
        return resolved

    def list_agent_card_entries(self) -> list[AgentCardEntry]:
        """Enumerate the card directory, reading each file for its stamp only.

        "One round trip" is one directory listing plus one small read per file;
        there is no query to push down. The card payload is never validated —
        :func:`_read_card_document` hands it back untouched and it is dropped —
        so a blob whose card no longer parses still enumerates, which is exactly
        the kind a sweep exists to reclaim.

        File names are filtered through :func:`_is_card_hash`, the same guard
        ``load_agent_cards`` applies: a file this store did not write is not a
        blob it holds, and must not become an entry a consumer then tries to
        reclaim.

        Returns:
            One entry per card file, unordered; ``[]`` when the directory is
            absent or holds none. A bare pre-envelope file reports
            ``first_seen_at=None`` and is not a failure.
        """
        cards_dir = self._cards_dir()
        if not cards_dir.exists():
            return []
        entries: list[AgentCardEntry] = []
        for card_path in cards_dir.glob("*.yaml"):
            if not _is_card_hash(card_path.stem):
                continue
            try:
                with open(card_path) as f:
                    document = yaml.safe_load(f)
            except (OSError, yaml.YAMLError, ValueError) as exc:
                logger.error("Skipping unreadable agent card file %s: %s", card_path.name, exc)
                continue
            entries.append(
                AgentCardEntry(
                    card_hash=card_path.stem,
                    first_seen_at=_read_card_document(document)[1],
                )
            )
        logger.debug("Enumerated %d agent card entries in %s", len(entries), cards_dir)
        return entries
