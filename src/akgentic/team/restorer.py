"""TeamRestorer: rebuild teams from EventStore data for crash recovery."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig
from akgentic.core.messages.orchestrator import StartMessage, StopMessage
from akgentic.core.orchestrator import EventSubscriber, Orchestrator
from akgentic.core.utils.deserializer import import_class
from akgentic.team.factory import TeamFactory
from akgentic.team.models import (
    AgentStateSnapshot,
    PersistedEvent,
    Process,
    TeamCard,
    TeamRuntime,
)
from akgentic.team.ports import EventStore
from akgentic.team.subscriber import PersistenceSubscriber

logger = logging.getLogger(__name__)


@dataclass
class _RebuildResult:
    """Internal result container for the agent-rebuild phase."""

    orchestrator_addr: ActorAddress
    orchestrator_proxy: Orchestrator
    persistence_sub: PersistenceSubscriber
    addrs: dict[str, ActorAddress] = field(default_factory=dict)


class TeamRestorer:
    """Rebuild a team from persisted EventStore data.

    Executes a 3-phase restore protocol:
      1. Load persisted events and agent state snapshots from EventStore.
      2. Rebuild agents from the event log (Orchestrator first, then others).
      3. Replay all events through the orchestrator to reconstruct state.

    TeamRestorer is a plain Python class (not Pydantic) with injected
    dependencies, following the same pattern as TeamManager and TeamFactory.
    """

    def __init__(
        self,
        actor_system: ActorSystem,
        event_store: EventStore,
        subscriber_factory: Callable[[uuid.UUID], list[EventSubscriber]] | None = None,
    ) -> None:
        """Initialize the restorer with injected dependencies.

        Args:
            actor_system: The actor system to host rebuilt actors.
            event_store: Persistence backend containing events and states.
            subscriber_factory: Optional callable returning additional
                EventSubscribers to register with the orchestrator.
        """
        self._actor_system = actor_system
        self._event_store = event_store
        self._subscriber_factory = subscriber_factory

    def restore(self, process: Process) -> tuple[TeamRuntime, PersistenceSubscriber]:
        """Execute the 3-phase restore protocol.

        Args:
            process: The Process record of the STOPPED team to restore.

        Returns:
            A tuple of (TeamRuntime, PersistenceSubscriber) for the rebuilt
            team. TeamManager needs PersistenceSubscriber for restoring-flag
            management.

        Raises:
            Exception: If any phase fails, all spawned actors are torn down
                and the original exception is re-raised.
        """
        team_id = process.team_id
        spawned_addrs: list[ActorAddress] = []

        try:
            # Phase 1: Load persisted data
            events, agent_states = self._load_persisted_data(team_id)

            # Phase 2: Rebuild agents from event log
            result = self._rebuild_agents(
                process, events, agent_states, spawned_addrs
            )

            # Phase 3: Replay events
            self._replay_events(
                team_id, result.orchestrator_proxy, result.persistence_sub, events
            )

            # Build and return TeamRuntime
            runtime = self._build_team_runtime(
                team_id, process.team_card, result.orchestrator_addr, result.addrs
            )

            logger.info("Team %s restored successfully", team_id)
            return runtime, result.persistence_sub

        except Exception:
            # Rollback: stop all spawned actors in reverse order
            for addr in reversed(spawned_addrs):
                try:
                    addr.stop()
                except Exception:
                    logger.warning("Failed to stop actor during rollback: %s", addr)
            raise

    def _load_persisted_data(
        self, team_id: uuid.UUID
    ) -> tuple[list[PersistedEvent], list[AgentStateSnapshot]]:
        """Phase 1: Load and sort persisted events and agent state snapshots.

        Args:
            team_id: The team identifier to load data for.

        Returns:
            A tuple of (sorted events, agent state snapshots).
        """
        logger.info("Restoring team %s: phase 1 -- loading persisted data", team_id)
        events = self._event_store.load_events(team_id)
        events.sort(key=lambda e: e.sequence)
        agent_states = self._event_store.load_agent_states(team_id)
        return events, agent_states

    def _determine_live_agents(
        self, events: list[PersistedEvent]
    ) -> tuple[StartMessage | None, list[StartMessage]]:
        """Filter events to determine live agents and separate orchestrator start.

        Scans StartMessage/StopMessage events to find agents that were started
        but not subsequently stopped. Separates the orchestrator StartMessage
        from regular agent StartMessages.

        Args:
            events: Sorted persisted events to scan.

        Returns:
            A tuple of (orchestrator_start, agent_starts) where orchestrator_start
            may be None if no orchestrator was found.
        """
        start_messages: list[StartMessage] = []
        stopped_agent_ids: set[uuid.UUID] = set()

        for pe in events:
            if isinstance(pe.event, StartMessage) and pe.event.sender is not None:
                start_messages.append(pe.event)
            elif isinstance(pe.event, StopMessage) and pe.event.sender is not None:
                stopped_agent_ids.add(pe.event.sender.agent_id)

        live_starts = [
            sm for sm in start_messages
            if sm.sender is not None and sm.sender.agent_id not in stopped_agent_ids
        ]

        orchestrator_class_name = (
            f"{Orchestrator.__module__}.{Orchestrator.__name__}"
        )
        orchestrator_start: StartMessage | None = None
        agent_starts: list[StartMessage] = []

        for sm in live_starts:
            if sm.sender is None:  # pragma: no cover – filtered earlier
                continue
            sender_type = sm.sender.serialize().get("__actor_type__", "")
            if sender_type == orchestrator_class_name:
                orchestrator_start = sm
            else:
                agent_starts.append(sm)

        return orchestrator_start, agent_starts

    def _create_orchestrator(
        self,
        orchestrator_start: StartMessage,
        team_id: uuid.UUID,
        spawned_addrs: list[ActorAddress],
    ) -> tuple[ActorAddress, Orchestrator]:
        """Create the orchestrator actor from its persisted StartMessage.

        Args:
            orchestrator_start: The StartMessage for the orchestrator.
            team_id: The team identifier.
            spawned_addrs: Shared list for rollback tracking.

        Returns:
            A tuple of (orchestrator_addr, orchestrator_proxy).

        Raises:
            ValueError: If the orchestrator StartMessage has no sender.
        """
        if orchestrator_start.sender is None:  # pragma: no cover
            msg = f"Orchestrator StartMessage has no sender for team {team_id}"
            raise ValueError(msg)

        orchestrator_addr = self._actor_system.createActor(
            Orchestrator,
            restoring=True,
            agent_id=orchestrator_start.sender.agent_id,
            team_id=team_id,
            config=BaseConfig(name="orchestrator", role="Orchestrator"),
        )
        spawned_addrs.append(orchestrator_addr)

        orchestrator_proxy: Orchestrator = self._actor_system.proxy_ask(
            orchestrator_addr, Orchestrator
        )
        return orchestrator_addr, orchestrator_proxy

    def _spawn_agents(
        self,
        agent_starts: list[StartMessage],
        orchestrator_addr: ActorAddress,
        spawned_addrs: list[ActorAddress],
    ) -> dict[str, ActorAddress]:
        """Spawn non-orchestrator agents from their persisted StartMessages.

        Agents are spawned through the orchestrator address so that
        ``_orchestrator`` and ``_parent`` propagate correctly from the
        hierarchy root.

        Args:
            agent_starts: StartMessages for agents to rebuild.
            orchestrator_addr: Address of the restored orchestrator to spawn through.
            spawned_addrs: Shared list for rollback tracking.

        Returns:
            A dict mapping agent names to their actor addresses.
        """
        addrs: dict[str, ActorAddress] = {}

        for sm in agent_starts:
            if sm.sender is None:  # pragma: no cover – filtered earlier
                continue
            sender_type = sm.sender.serialize().get("__actor_type__", "")
            agent_class: type[Akgent[Any, Any]] = import_class(sender_type)
            agent_name = sm.sender.name
            original_agent_id = sm.sender.agent_id

            config = sm.config.model_copy()

            addr = TeamFactory._spawn_through_parent(
                agent_class,
                orchestrator_addr,
                config=config,
                agent_id=original_agent_id,
            )
            spawned_addrs.append(addr)
            addrs[agent_name] = addr

        return addrs

    def _rebuild_agents(
        self,
        process: Process,
        events: list[PersistedEvent],
        agent_states: list[AgentStateSnapshot],
        spawned_addrs: list[ActorAddress],
    ) -> _RebuildResult:
        """Phase 2: Rebuild agents from the event log.

        Determines live agents via StartMessage/StopMessage filtering,
        rebuilds the Orchestrator first, registers subscribers, spawns
        remaining agents, restores agent states, and registers agent profiles.

        Args:
            process: The Process record containing the team card.
            events: Sorted persisted events from Phase 1.
            agent_states: Agent state snapshots from Phase 1.
            spawned_addrs: Shared list for rollback tracking; spawned actors
                are appended here so the caller can clean up on failure.

        Returns:
            A _RebuildResult containing orchestrator address, proxy,
            persistence subscriber, and agent address map.
        """
        team_id = process.team_id
        logger.info("Restoring team %s: phase 2 -- rebuilding agents", team_id)

        # 2a. Determine live agents
        orchestrator_start, agent_starts = self._determine_live_agents(events)

        if orchestrator_start is None:
            msg = f"No Orchestrator StartMessage found for team {team_id}"
            raise ValueError(msg)

        # 2b. Rebuild Orchestrator first
        orchestrator_addr, orchestrator_proxy = self._create_orchestrator(
            orchestrator_start, team_id, spawned_addrs
        )

        # 2c. Register subscribers
        persistence_sub = PersistenceSubscriber(team_id, self._event_store)
        persistence_sub.set_restoring(True)

        subscribers: list[EventSubscriber] = [persistence_sub]
        if self._subscriber_factory is not None:
            subscribers.extend(self._subscriber_factory(team_id))

        for sub in subscribers:
            orchestrator_proxy.subscribe(sub)

        # 2d. Spawn remaining agents through orchestrator
        addrs = self._spawn_agents(agent_starts, orchestrator_addr, spawned_addrs)

        # 2e. Restore agent states
        state_map: dict[str, AgentStateSnapshot] = {
            snap.agent_id: snap for snap in agent_states
        }
        for agent_name, addr in addrs.items():
            if agent_name in state_map:
                proxy: Akgent[Any, Any] = self._actor_system.proxy_ask(
                    addr, Akgent
                )
                proxy.init_state(state_map[agent_name].state)

        # 2f. Register agent profiles with orchestrator
        orchestrator_proxy.register_agent_profiles(
            list(process.team_card.agent_cards.values())
        )

        return _RebuildResult(
            orchestrator_addr=orchestrator_addr,
            orchestrator_proxy=orchestrator_proxy,
            persistence_sub=persistence_sub,
            addrs=addrs,
        )

    def _replay_events(
        self,
        team_id: uuid.UUID,
        orchestrator_proxy: Orchestrator,
        persistence_sub: PersistenceSubscriber,
        events: list[PersistedEvent],
    ) -> None:
        """Phase 3: Replay all persisted events through the orchestrator.

        Args:
            team_id: The team identifier (used for logging context).
            orchestrator_proxy: Proxy to the restored orchestrator actor.
            persistence_sub: The persistence subscriber to toggle restoring flag.
            events: Sorted persisted events to replay.
        """
        logger.info("Restoring team %s: phase 3 -- replaying %d events", team_id, len(events))

        for pe in events:
            orchestrator_proxy.restore_message(pe.event)

        orchestrator_proxy.end_restoration()
        persistence_sub.set_restoring(False)

    def _build_team_runtime(
        self,
        team_id: uuid.UUID,
        team_card: TeamCard,
        orchestrator_addr: ActorAddress,
        addrs: dict[str, ActorAddress],
    ) -> TeamRuntime:
        """Construct a TeamRuntime from restored components.

        Args:
            team_id: The team identifier.
            team_card: The declarative team definition.
            orchestrator_addr: Address of the restored orchestrator.
            addrs: Map of agent names to their actor addresses.

        Returns:
            A fully constructed TeamRuntime.

        Raises:
            ValueError: If the entry point agent is not found in addrs.
        """
        entry_name = team_card.entry_point.card.config.name

        entry_addr = addrs.get(entry_name)
        if entry_addr is None:
            msg = f"Entry point agent '{entry_name}' not found after restore"
            raise ValueError(msg)

        supervisor_addrs: dict[str, ActorAddress] = {}
        for card in team_card.supervisors:
            name = card.config.name
            if name in addrs:
                supervisor_addrs[name] = addrs[name]

        return TeamRuntime(
            id=team_id,
            team=team_card,
            actor_system=self._actor_system,
            orchestrator_addr=orchestrator_addr,
            entry_addr=entry_addr,
            supervisor_addrs=supervisor_addrs,
            addrs=addrs,
        )
