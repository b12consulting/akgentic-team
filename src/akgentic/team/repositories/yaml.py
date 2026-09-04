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
from pathlib import Path
from typing import Any

import yaml

from akgentic.core.agent_card import AgentCard
from akgentic.team.metadata import make_index_prefix_groups
from akgentic.team.models import AgentStateSnapshot, PersistedEvent, Process, TeamStatus
from akgentic.team.ports import EventNotFoundError
from akgentic.team.projection import hash_agent_card, storable_agent_card

logger = logging.getLogger(__name__)

CARDS_DIRNAME = "agent_cards"
"""Name of the shared card directory under ``data_dir``.

Public because ``list_teams`` must skip it by name and tests assert the layout.
A team directory is always a UUID, so the two namespaces cannot collide.
"""


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

        Args:
            path: Destination file path.
            data: Dictionary to serialize as YAML.
        """
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with open(fd, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
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

        Args:
            event: The event to append.
        """
        team_dir = self._team_dir(event.team_id)
        team_dir.mkdir(parents=True, exist_ok=True)
        events_path = team_dir / "events.yaml"
        data = event.model_dump()
        with open(events_path, "a") as f:
            f.write("---\n")
            yaml.dump(data, f, default_flow_style=False)
        logger.debug("Appended event seq=%d for team %s", event.sequence, event.team_id)

    @staticmethod
    def _unreadable_log(
        team_id: uuid.UUID, after_event_id: uuid.UUID | None
    ) -> list[PersistedEvent]:
        """Result for a team whose events.yaml is absent or unparseable.

        Raises:
            EventNotFoundError: If a cursor was passed — an unreadable log
                cannot resolve an anchor, and ``[]`` would be read by the
                caller as "you are already up to date".
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
                event of this team, including when the events file is absent
                or unparseable.
        """
        events_path = self._team_dir(team_id) / "events.yaml"
        if not events_path.exists():
            return self._unreadable_log(team_id, after_event_id)
        try:
            with open(events_path) as f:
                docs = list(yaml.safe_load_all(f))
        except yaml.YAMLError as exc:
            logger.error("Corrupted events.yaml for team %s: %s", team_id, exc)
            return self._unreadable_log(team_id, after_event_id)
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
            self._atomic_write(card_path, storable.model_dump())
        logger.debug("Saved %d agent cards to %s", len(cards), cards_dir)

    def load_agent_cards(self, hashes: list[str]) -> dict[str, AgentCard]:
        """Resolve card hashes against the card directory, one read per hash.

        The batch is a directory of files, so "one round trip" is one directory;
        there is no query to push down. A hash the directory does not hold is
        simply absent from the mapping, and a corrupted card file is skipped
        with a log exactly as a corrupted team or state file is — either way it
        surfaces as ``AgentCardNotFoundError`` at resolution rather than as an
        exception escaping the store.

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
            card_path = cards_dir / f"{card_hash}.yaml"
            if not card_path.exists():
                continue
            try:
                with open(card_path) as f:
                    resolved[card_hash] = AgentCard.model_validate(yaml.safe_load(f))
            except (yaml.YAMLError, ValueError) as exc:
                # ValueError covers both a non-UTF-8 file (UnicodeDecodeError
                # out of the text stream) and Pydantic's ValidationError, the
                # same pair _load_team_data and load_agent_states handle.
                logger.error("Skipping corrupted agent card %s: %s", card_hash, exc)
        logger.debug("Resolved %d of %d agent cards", len(resolved), len(hashes))
        return resolved
