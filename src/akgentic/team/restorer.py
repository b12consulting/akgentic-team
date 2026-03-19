"""TeamRestorer: rebuild teams from EventStore data for crash recovery."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import Any

from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig
from akgentic.core.messages.orchestrator import StartMessage, StopMessage
from akgentic.core.orchestrator import EventSubscriber, Orchestrator
from akgentic.core.utils.deserializer import import_class
from akgentic.team.models import AgentStateSnapshot, Process, TeamRuntime
from akgentic.team.ports import EventStore
from akgentic.team.subscriber import PersistenceSubscriber

logger = logging.getLogger(__name__)


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
            # ------------------------------------------------------------------
            # Phase 1: Load persisted data
            # ------------------------------------------------------------------
            logger.info("Restoring team %s: phase 1 -- loading persisted data", team_id)
            events = self._event_store.load_events(team_id)
            events.sort(key=lambda e: e.sequence)
            agent_states = self._event_store.load_agent_states(team_id)

            # ------------------------------------------------------------------
            # Phase 2: Rebuild agents from event log
            # ------------------------------------------------------------------
            logger.info("Restoring team %s: phase 2 -- rebuilding agents", team_id)

            # 2a. Determine live agents via StartMessage/StopMessage filtering
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

            # 2b. Find and rebuild Orchestrator first
            orchestrator_class_name = (
                f"{Orchestrator.__module__}.{Orchestrator.__name__}"
            )
            orchestrator_start: StartMessage | None = None
            agent_starts: list[StartMessage] = []

            for sm in live_starts:
                assert sm.sender is not None  # noqa: S101
                sender_type = sm.sender.serialize().get("__actor_type__", "")
                if sender_type == orchestrator_class_name:
                    orchestrator_start = sm
                else:
                    agent_starts.append(sm)

            if orchestrator_start is None:
                msg = f"No Orchestrator StartMessage found for team {team_id}"
                raise ValueError(msg)

            assert orchestrator_start.sender is not None  # noqa: S101
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

            # 2c. Register subscribers
            persistence_sub = PersistenceSubscriber(team_id, self._event_store)
            persistence_sub.set_restoring(True)

            subscribers: list[EventSubscriber] = [persistence_sub]
            if self._subscriber_factory is not None:
                subscribers.extend(self._subscriber_factory(team_id))

            for sub in subscribers:
                orchestrator_proxy.subscribe(sub)

            # 2d. Rebuild remaining agents in sequence order
            addrs: dict[str, ActorAddress] = {}

            for sm in agent_starts:
                assert sm.sender is not None  # noqa: S101
                sender_type = sm.sender.serialize().get("__actor_type__", "")
                agent_class: type[Akgent[Any, Any]] = import_class(sender_type)
                agent_name = sm.sender.name
                original_agent_id = sm.sender.agent_id

                config = sm.config.model_copy()

                addr = self._actor_system.createActor(
                    agent_class,
                    restoring=True,
                    agent_id=original_agent_id,
                    team_id=team_id,
                    config=config,
                )
                spawned_addrs.append(addr)
                addrs[agent_name] = addr

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

            # ------------------------------------------------------------------
            # Phase 3: Replay events
            # ------------------------------------------------------------------
            logger.info("Restoring team %s: phase 3 -- replaying %d events", team_id, len(events))

            for pe in events:
                orchestrator_proxy.restore_message(pe.event)

            orchestrator_proxy.end_restoration()
            persistence_sub.set_restoring(False)

            # ------------------------------------------------------------------
            # Build TeamRuntime
            # ------------------------------------------------------------------
            team_card = process.team_card
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

            runtime = TeamRuntime(
                id=team_id,
                team=team_card,
                actor_system=self._actor_system,
                orchestrator_addr=orchestrator_addr,
                entry_addr=entry_addr,
                supervisor_addrs=supervisor_addrs,
                addrs=addrs,
            )

            logger.info("Team %s restored successfully", team_id)
            return runtime, persistence_sub

        except Exception:
            # Rollback: stop all spawned actors in reverse order
            for addr in reversed(spawned_addrs):
                try:
                    addr.stop()
                except Exception:
                    logger.warning("Failed to stop actor during rollback: %s", addr)
            raise
