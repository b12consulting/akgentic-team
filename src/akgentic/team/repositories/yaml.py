"""YamlEventStore: file-based EventStore with per-team directory layout.

Persists team data to YAML files in a per-team directory structure using
PyYAML for serialization and pathlib for filesystem operations. Satisfies
the EventStore protocol via structural subtyping (no explicit inheritance).

File layout per team::

    {data_dir}/
      {team_uuid}/
        team.yaml           # Process metadata (overwrite)
        events.yaml         # Append-only event log (multi-document YAML)
        states/
          {agent_id}.yaml   # Latest agent state snapshot (overwrite)
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

import yaml

from akgentic.team.models import AgentStateSnapshot, PersistedEvent, Process

logger = logging.getLogger(__name__)


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

    def _team_dir(self, team_id: uuid.UUID) -> Path:
        """Return the directory path for a specific team.

        Args:
            team_id: Unique identifier of the team.

        Returns:
            Path to the team's directory under the data root.
        """
        return self._data_dir / str(team_id)

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
        data = process.model_dump()
        with open(team_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        logger.debug("Saved team %s to %s", process.team_id, team_path)

    def load_team(self, team_id: uuid.UUID) -> Process | None:
        """Load a team process snapshot from team.yaml.

        Args:
            team_id: Unique identifier of the team.

        Returns:
            The deserialized Process, or None if no team.yaml exists.
        """
        team_path = self._team_dir(team_id) / "team.yaml"
        if not team_path.exists():
            return None
        with open(team_path) as f:
            data = yaml.safe_load(f)
        return Process.model_validate(data)

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

    def load_events(self, team_id: uuid.UUID) -> list[PersistedEvent]:
        """Load all persisted events for a team from events.yaml.

        Reads multi-document YAML and reconstructs each document as a
        PersistedEvent, returned sorted by sequence number.

        Args:
            team_id: Unique identifier of the team.

        Returns:
            List of PersistedEvent ordered by sequence, or empty list if
            no events file exists.
        """
        events_path = self._team_dir(team_id) / "events.yaml"
        if not events_path.exists():
            return []
        with open(events_path) as f:
            docs = list(yaml.safe_load_all(f))
        events = [PersistedEvent.model_validate(doc) for doc in docs if doc is not None]
        return sorted(events, key=lambda e: e.sequence)

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
        data = snapshot.model_dump()
        with open(state_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
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
            with open(state_path) as f:
                data = yaml.safe_load(f)
            snapshots.append(AgentStateSnapshot.model_validate(data))
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
