"""Test fixtures for team service tests."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from typing import Any

from akgentic.core.agent_card import AgentCard

from akgentic.team.metadata import make_index_prefix_groups
from akgentic.team.models import AgentStateSnapshot, PersistedEvent, Process, TeamStatus
from akgentic.team.ports import EventNotFoundError
from akgentic.team.projection import hash_agent_card, storable_agent_card

logger = logging.getLogger(__name__)


class InMemoryEventStore:
    """Dict-backed EventStore for test isolation.

    Satisfies the EventStore protocol via structural subtyping.

    Events are round-trip serialized (model_dump/model_validate) to match
    real persistence behaviour: ActorAddressImpl is serialized to dict on
    save, then deserialized as ActorAddressProxy on load, preventing stale
    weak-ref errors when actors are stopped between save and load.
    """

    def __init__(self) -> None:
        self.events: list[PersistedEvent] = []
        self._event_dicts: list[dict[str, Any]] = []
        self.teams: dict[uuid.UUID, Process] = {}
        self.agent_states: dict[tuple[uuid.UUID, str], AgentStateSnapshot] = {}
        # Content-addressed and SHARED across teams — deliberately not keyed by
        # team_id, and deliberately untouched by delete_team.
        self.agent_cards: dict[str, AgentCard] = {}
        self.load_agent_cards_calls = 0
        # Write-method names in call order. The projection migration must write
        # cards BEFORE the document that references them (FR13), and both orders
        # leave identical storage — only the sequence tells them apart. Recorded
        # here rather than in a forked fake so every suite using this store gets
        # the same instrument.
        self.write_calls: list[str] = []

    def save_event(self, event: PersistedEvent) -> None:
        """Persist a single domain event.

        Stores both the live object (for subscriber tests that inspect
        ``store.events`` directly) and a serialized dict (for load_events
        round-trip, matching real persistence behaviour).
        """
        self.events.append(event)
        try:
            self._event_dicts.append(event.model_dump())
        except Exception:
            # Subscriber tests may use mock senders that fail serialization;
            # those tests never call load_events, so a missing dict is safe.
            logger.debug("Event serialization skipped (mock sender): %s", type(event.event))

    def load_events(
        self, team_id: uuid.UUID, after_event_id: uuid.UUID | None = None
    ) -> list[PersistedEvent]:
        """Load persisted events for a team (deserialized from dicts).

        Round-trips through model_dump/model_validate so that
        ActorAddressImpl becomes ActorAddressProxy, matching real
        persistence backends (YAML, MongoDB). Honours the same tail-slice
        contract as the real backends, which slice a ``sequence``-ordered
        list — so sort before slicing rather than trusting insertion order.

        Raises:
            EventNotFoundError: If ``after_event_id`` does not resolve to an
                event of this team.
        """
        tid = str(team_id)
        events = sorted(
            (PersistedEvent.model_validate(d) for d in self._event_dicts if d["team_id"] == tid),
            key=lambda e: e.sequence,
        )
        if after_event_id is None:
            return events
        # event.id is persisted as a string, so compare stringified ids.
        for index, event in enumerate(events):
            if str(event.event.id) == str(after_event_id):
                return events[index + 1 :]
        raise EventNotFoundError(f"Event {after_event_id} not found for team {team_id}")

    def save_team(self, process: Process) -> None:
        """Persist team process snapshot."""
        self.write_calls.append("save_team")
        self.teams[process.team_id] = process

    def load_team(self, team_id: uuid.UUID) -> Process | None:
        """Load a team process snapshot by ID."""
        return self.teams.get(team_id)

    def delete_team(self, team_id: uuid.UUID) -> None:
        """Delete all persisted data for a team.

        ``agent_cards`` is deliberately NOT purged: cards are shared and no
        ``EventStore`` method deletes one (FR13). Purging them here would let
        this fake pass a test the real backends fail.
        """
        self.teams.pop(team_id, None)
        self.events = [e for e in self.events if e.team_id != team_id]
        tid = str(team_id)
        self._event_dicts = [d for d in self._event_dicts if d["team_id"] != tid]
        self.agent_states = {k: v for k, v in self.agent_states.items() if k[0] != team_id}

    def save_agent_state(self, snapshot: AgentStateSnapshot) -> None:
        """Persist an agent state snapshot."""
        self.agent_states[(snapshot.team_id, snapshot.agent_id)] = snapshot

    def list_teams(
        self,
        user_id: str | None = None,
        status: TeamStatus | None = None,
        metadata: Mapping[str, list[str]] | None = None,
    ) -> list[Process]:
        """Load team process snapshots, honouring every Protocol filter.

        Kept at the full current Protocol shape so this fake cannot drift
        from the real backends it stands in for: ``user_id``, ``status`` and
        ``metadata`` are independent and combine with AND, and ``None`` on
        any of them means "do not filter on that dimension" — so the no-arg
        call still returns every team, ``DELETED`` ones included.

        The metadata term is an **anchored prefix** on the stored entry, in the
        direction *stored starts with term*. Terms for one key OR-combine and
        distinct keys AND-combine, and empty terms drop out — so ``{}``,
        ``{"tenant": []}`` and ``{"tenant": [""]}`` all behave like ``None``.
        Terms go through ``make_index_prefix_groups`` rather than being grouped
        here, so the fake folds, escapes and combines exactly as the real
        backends do — and rejects a bare ``str`` — and cannot pass a test they
        would fail. ``EventStore`` is not ``@runtime_checkable`` and CI mypy does
        not cover ``tests/``, so nothing but this catches a drift.
        """
        teams = list(self.teams.values())
        if user_id is not None:
            teams = [t for t in teams if t.user_id == user_id]
        if status is not None:
            teams = [t for t in teams if t.status == status]
        prefix_groups = make_index_prefix_groups(metadata)
        if prefix_groups:
            teams = [
                t
                for t in teams
                if all(
                    any(e.startswith(p) for p in group for e in t.metadata_indexes)
                    for group in prefix_groups
                )
            ]
        return teams

    def get_max_sequence(self, team_id: uuid.UUID) -> int:
        """Return the highest event sequence number for a team, or 0."""
        events = self.load_events(team_id)
        return max((e.sequence for e in events), default=0)

    def load_agent_states(self, team_id: uuid.UUID) -> list[AgentStateSnapshot]:
        """Load all agent state snapshots for a team, detached from the store.

        The state comes back as a ``serializable_copy()`` -- a fresh,
        observer-free instance of the same class -- because that is what the
        real backends produce: they rebuild the state from storage on every
        load and never hand out the object they hold.

        Returning the stored object aliased it to the caller. Restore phase 2d
        passes the loaded state straight to ``Akgent.init_state``, which does
        ``state.observer(...)`` -- an in-place write of ``_observer`` and the
        ``_last_serialized`` baseline. Through the alias that landed on the
        store's own snapshot, so a later ``load_agent_states`` returned a state
        that no longer compared equal to the one saved (Pydantic v2 ``__eq__``
        compares private attrs).
        """
        return [
            snapshot.model_copy(update={"state": snapshot.state.serializable_copy()})
            for (snapshot_team_id, _), snapshot in self.agent_states.items()
            if snapshot_team_id == team_id
        ]

    def save_agent_cards(self, cards: list[AgentCard]) -> None:
        """Persist agent cards into the dict-backed content-addressed store.

        Keyed by ``hash_agent_card`` and normalised through
        ``storable_agent_card``, exactly as every real backend does — so the
        stored blob is a pure function of its key and two teams differing only
        in hireability cannot write different bytes to one hash. Saving the
        same card twice leaves one entry by construction.

        Recorded in ``write_calls`` — the card write and the document write are
        ordered against each other by FR13, and the end state cannot show it.
        """
        self.write_calls.append("save_agent_cards")
        for card in cards:
            storable = storable_agent_card(card)
            self.agent_cards[hash_agent_card(storable)] = storable

    def load_agent_cards(self, hashes: list[str]) -> dict[str, AgentCard]:
        """Resolve card hashes; a hash the store does not hold is simply absent.

        Counts its calls: the restore path must resolve a team's whole set in
        ONE call, and a per-role read returns the identical mapping — the count
        is the only thing that separates them.
        """
        self.load_agent_cards_calls += 1
        if not hashes:
            return {}
        return {h: self.agent_cards[h] for h in hashes if h in self.agent_cards}
