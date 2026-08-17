"""Nagra-backed ``EventStore`` implementation.

Implements the nine :class:`~akgentic.team.ports.EventStore` Protocol methods
against PostgreSQL using Nagra's :class:`~nagra.Transaction` wrapper. Each
public method opens its own transaction (per-method ownership);
:meth:`NagraEventStore.delete_team` is the one exception that spans a single
transaction across three ``DELETE`` statements for atomic cascade semantics.

The class satisfies the ``EventStore`` Protocol via structural subtyping — it
does NOT inherit from the Protocol explicitly (mirrors ``YamlEventStore`` and
``MongoEventStore``).

Implements ADR-15 Nagra-based PostgreSQL EventStore §§2, 3, 4, 8, 9, 10.
"""

from __future__ import annotations

import json
import uuid

from nagra import Transaction  # type: ignore[import-untyped]

from akgentic.team.metadata import make_index_entry
from akgentic.team.models import AgentStateSnapshot, PersistedEvent, Process, TeamStatus
from akgentic.team.ports import EventNotFoundError
from akgentic.team.repositories.postgres._queries import decode_jsonb_column


class NagraEventStore:
    """Nagra-backed ``EventStore`` implementation.

    Constructor calls :func:`_ensure_schema_loaded` so instances are always
    safe to build, but does NOT call :func:`init_db` — operators must
    invoke ``python -m akgentic.team.scripts.init_db`` once per deployment.

    Args:
        conn_string: Nagra-compatible Postgres connection string used to
            open per-method transactions.
    """

    def __init__(self, conn_string: str) -> None:
        # Local import keeps the constructor cheap and avoids a circular
        # import on package initialisation (``__init__`` re-exports this
        # class after defining ``_ensure_schema_loaded``).
        from akgentic.team.repositories.postgres import _ensure_schema_loaded

        _ensure_schema_loaded()
        self._conn_string = conn_string

    # --- team process (team_process_entries) -------------------------------

    def save_team(self, process: Process) -> None:
        """Upsert a team process snapshot keyed by ``team_id``.

        Writes the whole ``Process`` as JSON into ``data`` AND its derived
        index into the promoted ``metadata_indexes`` column, in one
        statement. The ``ON CONFLICT`` branch refreshes both: setting only
        ``data`` would leave a stale index that :meth:`list_teams` then
        queries as though it were current.

        The index is persisted as handed over, never re-derived here.
        ``derive_metadata_indexes`` has exactly one call site per write
        path, in the manager; a second derivation in the repository is the
        drift the single-derivation rule exists to prevent, and it would
        also overwrite an index a caller deliberately planted.
        """
        data = json.dumps(process.model_dump())
        with Transaction(self._conn_string) as trn:
            trn.execute(
                "INSERT INTO team_process_entries (id, data, metadata_indexes) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data, "
                "metadata_indexes = EXCLUDED.metadata_indexes",
                (str(process.team_id), data, list(process.metadata_indexes)),
            )

    def load_team(self, team_id: uuid.UUID) -> Process | None:
        """Load a team process snapshot by id; return ``None`` if absent."""
        with Transaction(self._conn_string) as trn:
            cursor = trn.execute(
                "SELECT data FROM team_process_entries WHERE id = %s",
                (str(team_id),),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return Process.model_validate(decode_jsonb_column(row[0]))

    def list_teams(
        self,
        user_id: str | None = None,
        status: TeamStatus | None = None,
        metadata: dict[str, str] | None = None,
    ) -> list[Process]:
        """Return persisted team processes. Order is unspecified.

        Two of the three filters are pushed down into a single ``WHERE``
        clause; the third is applied after hydration:

        * ``user_id`` → ``(data ->> 'user_id') = %s``, backed by the
          functional expression index ``team_process_user_id_idx``. See
          ADR-16 §4.
        * ``metadata`` → ``metadata_indexes @> %s``, an array-containment
          term backed by the GIN index ``team_process_metadata_indexes_idx``
          over the promoted ``TEXT[]`` column. See ADR-24 §D5.
        * ``status`` → in memory, on the hydrated rows. Its push-down needs
          a further expression index and is story 26.5, still DEFERRED, so
          this is the shipped behaviour of this backend rather than a
          placeholder. Both indexes above are created by :func:`init_db`.

        The pushed-down terms AND-combine in one statement, and ``user_id``
        is appended whenever it is given — never dropped, weakened, or made
        conditional on ``metadata`` being more selective. It is a trust
        boundary, not an optimisation (ADR-24 §D5).

        Every caller-supplied value travels as a bound ``%s`` parameter; the
        statement is assembled only from fixed fragments joined with
        ``AND``, so no value ever reaches SQL through an f-string or
        concatenation.

        Results never depend on an index existing. Dropping either index
        changes the access path the planner picks and nothing else.

        Args:
            user_id: If provided, return only snapshots whose
                ``Process.user_id`` matches. If ``None`` (default), return all
                snapshots. See ADR-16 §1.
            status: If provided, return only snapshots whose
                ``Process.status`` matches. If ``None`` (default), every
                lifecycle state is returned, including ``DELETED``. See
                ADR-23 §1.
            metadata: If provided, return only snapshots whose
                ``metadata_indexes`` contains an entry for EVERY key/value
                pair given. An empty dict matches everything, like ``None``.
                See ADR-24 §D5.

        All three filters are independent terms combining as a conjunction.
        """
        clauses: list[str] = []
        params: list[object] = []
        if user_id is not None:
            clauses.append("(data ->> 'user_id') = %s")
            params.append(user_id)
        # Truthiness, NOT ``is not None``: an empty mapping must add no term
        # at all. ``x @> '{}'`` is TRUE for any non-null array but NULL for
        # a row written before this column existed, so an ``is not None``
        # gate would silently stop returning every pre-migration team while
        # looking correct against a store populated after it.
        if metadata:
            # Equality-only containment, one entry per key, AND-combined by
            # the array operator itself. Entries are built with the shared
            # helper the write path derived with, so the ``|`` escaping
            # cannot drift between the two sides.
            clauses.append("metadata_indexes @> %s")
            params.append([make_index_entry(k, v) for k, v in metadata.items()])
        sql = "SELECT data FROM team_process_entries"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with Transaction(self._conn_string) as trn:
            cursor = trn.execute(sql, tuple(params))
            rows = cursor.fetchall()
        teams = [Process.model_validate(decode_jsonb_column(r[0])) for r in rows]
        # The one filter still evaluated in Python — see the docstring.
        if status is not None:
            teams = [t for t in teams if t.status == status]
        return teams

    def delete_team(self, team_id: uuid.UUID) -> None:
        """Cascade-delete a team across all three tables in ONE transaction.

        Order (dependency-safe): ``agent_state_entries`` → ``event_entries``
        → ``team_process_entries``. Idempotent: calling twice for the same
        id is a no-op on the second call (matches YAML / Mongo semantics).
        """
        tid = str(team_id)
        with Transaction(self._conn_string) as trn:
            trn.execute(
                "DELETE FROM agent_state_entries WHERE team_id = %s", (tid,)
            )
            trn.execute("DELETE FROM event_entries WHERE team_id = %s", (tid,))
            trn.execute(
                "DELETE FROM team_process_entries WHERE id = %s", (tid,)
            )

    # --- events (event_entries) -------------------------------------------

    def save_event(self, event: PersistedEvent) -> None:
        """Append a single event. No upsert — events are immutable."""
        data = json.dumps(event.model_dump())
        with Transaction(self._conn_string) as trn:
            trn.execute(
                "INSERT INTO event_entries (team_id, sequence, data) "
                "VALUES (%s, %s, %s)",
                (str(event.team_id), event.sequence, data),
            )

    def load_events(
        self, team_id: uuid.UUID, after_event_id: uuid.UUID | None = None
    ) -> list[PersistedEvent]:
        """Return events for a team ordered by ``sequence`` ASC.

        Args:
            team_id: Team whose events to load.
            after_event_id: If provided, return only events after the matching
                event — anchor excluded. If ``None`` (default), the full log.

        Raises:
            EventNotFoundError: If ``after_event_id`` does not resolve to an
                event of this team.
        """
        with Transaction(self._conn_string) as trn:
            cursor = trn.execute(
                "SELECT data FROM event_entries WHERE team_id = %s "
                "ORDER BY sequence ASC",
                (str(team_id),),
            )
            rows = cursor.fetchall()
        events = [PersistedEvent.model_validate(decode_jsonb_column(r[0])) for r in rows]
        if after_event_id is None:
            return events
        # Interim in-memory slice; the range-query push-down lands in story 24-4.
        # event.id is persisted as a string, so compare stringified ids.
        for index, event in enumerate(events):
            if str(event.event.id) == str(after_event_id):
                return events[index + 1 :]
        raise EventNotFoundError(f"Event {after_event_id} not found for team {team_id}")

    def get_max_sequence(self, team_id: uuid.UUID) -> int:
        """Return the largest sequence for a team, or ``0`` if empty.

        Uses ``COALESCE(MAX(sequence), 0)`` so the empty-team case is a
        single round-trip and the method always returns an ``int`` (never
        ``None``).
        """
        with Transaction(self._conn_string) as trn:
            cursor = trn.execute(
                "SELECT COALESCE(MAX(sequence), 0) FROM event_entries "
                "WHERE team_id = %s",
                (str(team_id),),
            )
            row = cursor.fetchone()
        assert row is not None  # aggregate always returns one row
        result: int = row[0]
        return result

    # --- agent states (agent_state_entries) --------------------------------

    def save_agent_state(self, snapshot: AgentStateSnapshot) -> None:
        """Upsert an agent state snapshot keyed by ``(team_id, agent_id)``."""
        data = json.dumps(snapshot.model_dump())
        with Transaction(self._conn_string) as trn:
            trn.execute(
                "INSERT INTO agent_state_entries (team_id, agent_id, data) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (team_id, agent_id) DO UPDATE SET data = EXCLUDED.data",
                (str(snapshot.team_id), snapshot.agent_id, data),
            )

    def load_agent_states(self, team_id: uuid.UUID) -> list[AgentStateSnapshot]:
        """Return every agent-state snapshot for a team. Order unspecified."""
        with Transaction(self._conn_string) as trn:
            cursor = trn.execute(
                "SELECT data FROM agent_state_entries WHERE team_id = %s",
                (str(team_id),),
            )
            rows = cursor.fetchall()
        return [
            AgentStateSnapshot.model_validate(decode_jsonb_column(r[0]))
            for r in rows
        ]
