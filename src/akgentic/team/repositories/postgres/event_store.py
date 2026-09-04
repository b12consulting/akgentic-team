"""Nagra-backed ``EventStore`` implementation.

Implements the eleven :class:`~akgentic.team.ports.EventStore` Protocol methods
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
from collections.abc import Mapping

from nagra import Transaction  # type: ignore[import-untyped]

from akgentic.core.agent_card import AgentCard
from akgentic.team.metadata import make_index_prefix_groups
from akgentic.team.models import AgentStateSnapshot, PersistedEvent, Process, TeamStatus
from akgentic.team.ports import EventNotFoundError
from akgentic.team.projection import hash_agent_card, storable_agent_card
from akgentic.team.repositories.postgres._queries import decode_jsonb_column

_LIKE_ESCAPE = "!"
"""``ESCAPE`` character for the metadata ``LIKE`` patterns — chosen, not default.

``LIKE``'s default escape character is the backslash, and a backslash is
already in play twice on this path: ``make_index_entry`` plants ``\\|`` for an
escaped separator, and ``re.escape`` emits backslashes on the Mongo side of the
very same rendered prefix. Taking the default would put three layers of
backslash in one string, where a value of ``a\\b`` has to be doubled to survive
and a miscount is invisible until a user types one. ``!`` is not produced by any
escaping layer, so a backslash in an entry passes through ``LIKE`` as the
ordinary character it is, and only ``!`` itself, ``%`` and ``_`` need escaping.

Declaring it explicitly also sidesteps ``ESCAPE '\\'``, which is not a
well-formed SQL literal when ``standard_conforming_strings`` is off.
"""

_LIKE_METACHARACTERS = ("%", "_")

_LIKE_TERM = f"entry LIKE %s ESCAPE '{_LIKE_ESCAPE}'"
"""One term inside an ``EXISTS``. Fixed fragment — the value rides the ``%s``."""


def _metadata_key_clause(term_count: int) -> str:
    """Build the ``EXISTS`` clause for ONE filtered metadata key.

    One ``EXISTS`` per key, its inner ``WHERE`` ORing that key's term patterns:
    terms for a key are a disjunction, and the per-key clauses are ``AND``-ed by
    the caller (ADR-28 §D7).

    The ``OR`` sits *inside* the ``EXISTS`` so the disjunction is per-element:
    one stored entry must satisfy one of the terms, rather than two different
    entries each satisfying a different term. **Today those two readings cannot
    differ** — indexed fields are scalars, so a key contributes at most one entry
    per team — and hoisting the ``OR`` outside into one ``EXISTS`` per term was
    mutation-tested to change no behaviour, only the statement. It is kept inside
    because it is the reading that stays correct if a key ever contributes more
    than one entry, and because it is one clause per key rather than per term.

    Assembled only from fixed fragments repeated ``term_count`` times; every
    caller value travels as a bound ``%s``. ``unnest`` is what makes the match
    per-entry — the GIN index over the column serves containment and cannot
    serve a prefix on an element, so this is a sequential scan by design
    (ADR-28 §D6).

    Args:
        term_count: How many terms this key carries. Always at least one:
            a key that renders no term yields no group and never reaches here.

    Returns:
        The clause, with one ``%s`` placeholder per term.
    """
    return (
        "EXISTS (SELECT 1 FROM unnest(metadata_indexes) AS e(entry) WHERE "
        + " OR ".join([_LIKE_TERM] * term_count)
        + ")"
    )


def _like_prefix_pattern(prefix: str) -> str:
    """Turn a rendered index prefix into a literal-matching ``LIKE`` pattern.

    The escape character is escaped FIRST — doing it after ``%``/``_`` would
    re-escape the escapes this function just wrote and corrupt every pattern
    containing a metacharacter.

    Args:
        prefix: A rendered ``"key|value"`` prefix, matched literally.

    Returns:
        The pattern, with a trailing ``%`` making it an anchored prefix match.
    """
    escaped = prefix.replace(_LIKE_ESCAPE, _LIKE_ESCAPE * 2)
    for metacharacter in _LIKE_METACHARACTERS:
        escaped = escaped.replace(metacharacter, _LIKE_ESCAPE + metacharacter)
    return escaped + "%"


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
        metadata: Mapping[str, list[str]] | None = None,
    ) -> list[Process]:
        """Return persisted team processes. Order is unspecified.

        Two of the three filters are pushed down into a single ``WHERE``
        clause; the third is applied after hydration:

        * ``user_id`` → ``(data ->> 'user_id') = %s``, backed by the
          functional expression index ``team_process_user_id_idx``. See
          ADR-16 §4.
        * ``metadata`` → one ``EXISTS (SELECT 1 FROM unnest(metadata_indexes)
          ... LIKE %s ESCAPE ...)`` clause per filtered KEY, its inner ``WHERE``
          ORing that key's terms, matching per array element. This is a
          sequential scan: the GIN index
          ``team_process_metadata_indexes_idx`` serves array *containment*
          and cannot serve a prefix on an element. ADR-28 §D6 records the
          remedy — a side table with a ``text_pattern_ops`` B-tree — as
          something to take when volume calls for it, not now.
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
            metadata: Mapping of indexed field name to a list of prefix terms.
                Terms for one key OR-combine; distinct keys AND-combine. Empty
                terms drop out, so ``{}``, ``{"tenant": []}``, ``{"tenant": [""]}``
                and ``None`` all leave the statement without a metadata clause
                — and a legacy row whose column is ``NULL`` keeps listing,
                because ``unnest(NULL)`` yields no rows and so is matched by no
                metadata term while being excluded by none either. See
                ADR-24 §D5 and ADR-28 §D3/§D7.

        All three filters are independent terms combining as a conjunction.

        Raises:
            TypeError: If a ``metadata`` value is a bare ``str``.
        """
        clauses: list[str] = []
        params: list[object] = []
        if user_id is not None:
            clauses.append("(data ->> 'user_id') = %s")
            params.append(user_id)
        # Gate on the RENDERED groups, not on ``metadata`` truthiness: a key
        # carrying an empty term list is itself truthy and must still add no
        # clause, or a caller who sent a blank gets a different answer from one
        # who sent nothing. A group is never empty, so no clause is ever built
        # with zero ``OR`` arms.
        for group in make_index_prefix_groups(metadata):
            clauses.append(_metadata_key_clause(len(group)))
            params.extend(_like_prefix_pattern(prefix) for prefix in group)
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

        ``agent_card_entries`` is deliberately NOT in the cascade: cards are
        content-addressed and shared, another team may reference the very rows
        this team did, and no refcount exists (FR13).
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

    # --- agent cards (agent_card_entries) ----------------------------------

    def save_agent_cards(self, cards: list[AgentCard]) -> None:
        """Upsert agent cards into ``agent_card_entries``, keyed by content hash.

        ``ON CONFLICT (card_hash) DO UPDATE`` rather than a plain ``INSERT``:
        the store is content-addressed, so a card arriving again — from a
        re-save or from a second team — writes the same bytes to the same row
        instead of violating the natural key. Nagra's ``create_tables()``
        provisions that key from ``schema.toml``, which is what makes the
        ``ON CONFLICT`` target valid.

        The whole batch runs in ONE transaction, so a team's cards land
        together or not at all.

        Args:
            cards: The cards to persist. An empty list opens no transaction.
        """
        if not cards:
            return
        with Transaction(self._conn_string) as trn:
            for card in cards:
                storable = storable_agent_card(card)
                trn.execute(
                    "INSERT INTO agent_card_entries (card_hash, data) "
                    "VALUES (%s, %s) "
                    "ON CONFLICT (card_hash) DO UPDATE SET data = EXCLUDED.data",
                    (hash_agent_card(storable), json.dumps(storable.model_dump())),
                )

    def load_agent_cards(self, hashes: list[str]) -> dict[str, AgentCard]:
        """Resolve card hashes with a single ``IN`` query.

        One statement for the whole batch. The ``IN`` list is built from a
        placeholder per hash — fixed ``%s`` fragments joined with commas — so
        every caller value still travels as a bound parameter and none reaches
        SQL through interpolation.

        Args:
            hashes: The content hashes to resolve; empty returns ``{}`` without
                opening a transaction (and avoids an ``IN ()``, which is not
                valid SQL).

        Returns:
            Mapping of hash to card for every hash the table holds. A hash the
            table does not hold is simply absent.
        """
        if not hashes:
            return {}
        placeholders = ", ".join(["%s"] * len(hashes))
        with Transaction(self._conn_string) as trn:
            cursor = trn.execute(
                "SELECT card_hash, data FROM agent_card_entries "
                f"WHERE card_hash IN ({placeholders})",
                tuple(hashes),
            )
            rows = cursor.fetchall()
        return {
            row[0]: AgentCard.model_validate(decode_jsonb_column(row[1])) for row in rows
        }
