"""Port protocols: EventStore, ServiceRegistry, and NullServiceRegistry default."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from akgentic.core.agent_card import AgentCard
from akgentic.team.models import (
    AgentStateSnapshot,
    PersistedEvent,
    Process,
    TeamStatus,
)


class EventNotFoundError(LookupError):
    """Raised when ``load_events(after_event_id=...)`` cannot resolve the anchor.

    Subclasses ``LookupError`` — never ``ValueError`` — so the corrupted-document
    handlers in the YAML and Mongo backends cannot swallow a stale cursor.
    """


class AgentCardNotFoundError(LookupError):
    """Raised when an ``AgentCardRef``'s hash resolves to no card in the store.

    Subclasses ``LookupError`` — never ``ValueError`` — for exactly the reason
    ``EventNotFoundError`` does: the corrupted-document handlers in the YAML and
    Mongo backends catch ``ValueError``, so a ``ValueError`` here would be
    swallowed on the very path it exists to fail loudly on. A team that cannot
    resolve one of its cards must refuse to restore, never restore with agents
    the orchestrator cannot describe (ADR-26 §Decision 5, FR14).

    The message names **both** the role and the hash. The store knows only the
    hash — the role comes from the ``AgentCardRef``, which is why this is raised
    by :func:`akgentic.team.projection.resolve_agent_cards` and never by a
    backend.
    """


class EventStore(Protocol):
    """Typed persistence surface for team event sourcing.

    Implementations provide durable storage for events, team process state,
    and agent state snapshots. Concrete implementations (e.g. YamlEventStore,
    MongoEventStore) satisfy this protocol via structural subtyping.
    """

    def save_event(self, event: PersistedEvent) -> None:
        """Persist a single domain event.

        Called by PersistenceSubscriber on each event received from the orchestrator.
        """
        ...

    def load_events(
        self, team_id: uuid.UUID, after_event_id: uuid.UUID | None = None
    ) -> list[PersistedEvent]:
        """Load persisted events for a team, ordered by ``sequence`` ASC.

        Args:
            team_id: Team whose events to load.
            after_event_id: If provided, return only events persisted *after*
                the event whose ``event.id`` matches — exclusive of the anchor
                itself. If ``None`` (default), return the full log, which is
                what TeamRestorer replays on resume.

        Raises:
            EventNotFoundError: If ``after_event_id`` is not an event of this
                team. A stale cursor MUST fail loudly, never silently degrade
                into a full-log return.

        Implementations MUST resolve the anchor to its ``sequence`` and push a
        ``sequence > N`` range filter down to the backend (SQL WHERE, Mongo find
        filter, list slice) rather than load-all-then-filter in Python — the
        infra read path calls this per client poll.

        See ADR-21 §1 for the additive, backwards-compatible Protocol change.
        """
        ...

    def save_team(self, process: Process) -> None:
        """Persist the current team process snapshot.

        Called by PersistenceSubscriber to checkpoint team state.
        """
        ...

    def load_team(self, team_id: uuid.UUID) -> Process | None:
        """Load a team process snapshot by ID.

        Called by TeamManager to retrieve team state. Returns None if no
        snapshot exists for the given team ID.
        """
        ...

    def delete_team(self, team_id: uuid.UUID) -> None:
        """Delete all persisted data for a team.

        Called by TeamManager during team deletion to purge stored state.
        """
        ...

    def save_agent_state(self, snapshot: AgentStateSnapshot) -> None:
        """Persist an agent state snapshot.

        Called by PersistenceSubscriber to checkpoint individual agent state.
        """
        ...

    def list_teams(
        self,
        user_id: str | None = None,
        status: TeamStatus | None = None,
        metadata: Mapping[str, list[str]] | None = None,
    ) -> list[Process]:
        """Load team process snapshots.

        Args:
            user_id: If provided, return only snapshots whose ``Process.user_id``
                matches. If ``None`` (default), do not filter on the owner.
            status: If provided, return only snapshots whose ``Process.status``
                matches. If ``None`` (default), do not filter on the lifecycle
                state — every status is returned, including ``DELETED``.
            metadata: Mapping of indexed field name to a list of **prefix
                terms**. Each term is rendered to a flattened ``"key|value"``
                form via :func:`akgentic.team.metadata.make_index_entry` — the
                same helper the write path derived the stored index with — and a
                snapshot matches when one of its ``Process.metadata_indexes``
                entries *starts with* that rendered term. The direction is
                always stored-starts-with-term, never the reverse, so a crafted
                term cannot span two entries.

                Matching is therefore an **anchored prefix**, not equality:
                ``{"tenant": ["aze"]}`` reaches a team storing ``tenant|azefr``.
                It is also **case-insensitive without a case-insensitive query**,
                because ``make_index_entry`` casefolds the value half on both
                sides — a term of ``AZE``, ``aze`` or ``AzE`` renders
                identically. Only fields the metadata model marks indexed are
                matchable; an unindexed field never reaches the index and so
                matches nothing.

                Terms for ONE key **OR**-combine; DISTINCT keys **AND**-combine.
                That is ordinary faceted search — same field ORs, different
                fields AND — and it is what repeating a query parameter means
                everywhere else. ``{"tenant": ["acme", "contoso"]}`` returns the
                teams matching *either* prefix, while
                ``{"tenant": ["acme"], "case_ref": ["C-"]}`` returns only teams
                matching *both* keys. Two terms on one key could not usefully
                AND: under prefix matching either one is a prefix of the other,
                making it redundant, or the pair is unsatisfiable.

                An **empty term contributes no constraint** for its key, because
                ``"k|"`` would otherwise prefix-match every entry under that key
                — and under a disjunction that *widens* the answer to everything
                rather than merely failing to narrow it. ``{}``,
                ``{"tenant": []}``, ``{"tenant": [""]}`` and ``None`` are
                therefore all equivalent, and an implementation MUST put no
                metadata term on its query at all in those cases. A key whose
                terms all render away contributes NO term — neither an empty
                disjunction nor an empty conjunction, which backends variously
                reject outright or read as "match nothing".

                A bare ``str`` value raises ``TypeError`` rather than filtering
                on one term per character.

        Every filter is an independent term and they combine as a conjunction:
        a filter left at ``None`` constrains nothing, and adding one can only
        narrow the result set, never widen it. In particular ``user_id`` scoping
        is applied server-side and is NEVER weakened by ``metadata`` — metadata
        is caller-supplied and non-secret, so it must not become a way to reach
        another owner's teams.

        Results MUST be identical whatever strategy an implementation uses;
        where the filter runs is a performance property, not a semantic one.
        Beyond that, implementations MUST push each filter down to the backend
        (SQL WHERE, Mongo find filter, directory-walk skip) wherever the backend
        allows it, rather than load-all-then-filter in Python — the infra boot
        path lists RUNNING teams across the whole store. A backend that cannot
        yet express a term says so at the filter and keeps the terms it can
        express pushed down; ``user_id`` scoping is the one term that is always
        applied server-side, because it is a trust boundary rather than an
        optimisation.

        Raises:
            TypeError: If a ``metadata`` value is a bare ``str`` instead of a
                list of terms.

        See ADR-23 §1 and ADR-24 §D5 for the additive, backwards-compatible
        Protocol changes, and ADR-28 for the move from equality to prefix.
        """
        ...

    def get_max_sequence(self, team_id: uuid.UUID) -> int:
        """Return the highest event sequence number for a team.

        Used by TeamManager.resume_team() to initialize PersistenceSubscriber
        so that new events continue monotonically after restore. Returns 0 if
        no events exist for the given team.

        Implementations backed by a database (e.g. MongoDB) SHOULD use an
        efficient query (e.g. ``find().sort("sequence", -1).limit(1)``)
        rather than loading all events into memory.
        """
        ...

    def load_agent_states(self, team_id: uuid.UUID) -> list[AgentStateSnapshot]:
        """Load all agent state snapshots for a team.

        Called by TeamRestorer to restore agent states during team resumption.
        """
        ...

    def save_agent_cards(self, cards: list[AgentCard]) -> None:
        """Persist agent cards into the content-addressed card store.

        Called by ``TeamManager.create_team`` with the whole card set of a team,
        **before** the ``Process`` that references it — a stored ``Process`` must
        never point at a blob that is not there (FR13).

        The contract every backend honours:

        * **Keyed by content hash.** The key is
          :func:`akgentic.team.projection.hash_agent_card` of the card, and
          nothing else. The store is shared across teams: a card reached by ten
          teams is one blob.
        * **Idempotent by content.** Saving the same card twice, or from two
          different teams, leaves ONE stored blob. An empty list is a no-op that
          touches no storage.
        * **Stored in normalised form.** ``can_be_hired`` is excluded from the
          hash, so the bytes are stored with the flag back at its default, via
          :func:`akgentic.team.projection.storable_agent_card`. Excluding the
          flag from the *hash* is necessary but not sufficient: without also
          normalising the *bytes*, two teams differing only in hireability write
          the same key with different content and last-write-wins silently
          decides what every other team reads back. Hireability is carried by
          the ``Process``'s ``AgentCardRef`` alone.
        * **Immutable.** A blob at a hash is the bytes that hash names, forever.
          A re-save rewrites the same bytes; it never appends and never mutates
          a card in place. Editing a catalog entry produces a *new* card with a
          *new* hash, which is what leaves every existing team pointing at
          exactly what it was built from (ADR-26 §Consequences, NFR2).
        * **Never deleted.** NO ``EventStore`` method removes a card,
          ``delete_team`` included. Another team may share it, no refcount is
          introduced, cards are small, and garbage collection is out of scope
          (FR13).

        Args:
            cards: The cards to persist. Order is irrelevant; duplicates are
                harmless. An empty list is a no-op.
        """
        ...

    def load_agent_cards(self, hashes: list[str]) -> dict[str, AgentCard]:
        """Resolve card hashes to cards, in ONE round trip.

        Args:
            hashes: The content hashes to resolve. An empty list returns ``{}``
                without a backend round trip.

        Returns:
            A mapping keyed by hash holding an entry for every hash the store
            holds, and NO entry for one it does not. A miss is an absent key,
            never a raise: the store knows the hash but not the role, and the
            role is half of what FR14's failure has to name. That failure is
            owned by :func:`akgentic.team.projection.resolve_agent_cards`,
            which raises ``AgentCardNotFoundError``.

        Implementations MUST resolve the whole batch in ONE backend query
        (Mongo ``$in``, SQL ``IN``, one directory read) rather than one per
        hash — resolving a team's card set per role is the N+1 this store
        exists to prevent, and it is invisible in the result.

        A card the backend holds but cannot parse is skipped the way corrupted
        team and state documents already are — logged, absent from the mapping —
        so it surfaces as FR14's loud failure at resolution rather than as an
        exception escaping the store.
        """
        ...


@runtime_checkable
class ServiceRegistry(Protocol):
    """Service discovery port for multi-worker deployments.

    Tracks which worker instances are active and which teams are hosted
    on each instance. Implementations satisfy this protocol via structural
    subtyping. In single-process mode, use NullServiceRegistry.
    """

    def register_instance(self, instance_id: uuid.UUID) -> None:
        """Register a worker instance as active."""
        ...

    def deregister_instance(self, instance_id: uuid.UUID) -> None:
        """Remove a worker instance from the active set."""
        ...

    def register_team(self, instance_id: uuid.UUID, team_id: uuid.UUID) -> None:
        """Associate a team with a worker instance."""
        ...

    def deregister_team(self, instance_id: uuid.UUID, team_id: uuid.UUID) -> None:
        """Disassociate a team from a worker instance."""
        ...

    def find_team(self, team_id: uuid.UUID) -> uuid.UUID | None:
        """Find the worker instance hosting a team.

        Returns the instance UUID if found, None otherwise.
        """
        ...

    def get_active_instances(self) -> list[uuid.UUID]:
        """Return all currently active worker instance IDs."""
        ...


class NullServiceRegistry:
    """No-op service registry for single-process mode.

    Satisfies the ServiceRegistry protocol via structural subtyping without
    inheriting from it. All mutating methods are no-ops; queries return
    empty results.
    """

    def register_instance(self, instance_id: uuid.UUID) -> None:
        """No-op: single-process mode has no instance tracking."""
        pass

    def deregister_instance(self, instance_id: uuid.UUID) -> None:
        """No-op: single-process mode has no instance tracking."""
        pass

    def register_team(self, instance_id: uuid.UUID, team_id: uuid.UUID) -> None:
        """No-op: single-process mode has no team-to-instance mapping."""
        pass

    def deregister_team(self, instance_id: uuid.UUID, team_id: uuid.UUID) -> None:
        """No-op: single-process mode has no team-to-instance mapping."""
        pass

    def find_team(self, team_id: uuid.UUID) -> uuid.UUID | None:
        """Always returns None: no team routing in single-process mode."""
        return None

    def get_active_instances(self) -> list[uuid.UUID]:
        """Always returns empty list: no instance tracking in single-process mode."""
        return []
