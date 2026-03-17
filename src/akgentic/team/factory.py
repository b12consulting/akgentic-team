"""TeamFactory: build running teams from TeamCard + ActorSystem."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig
from akgentic.core.orchestrator import EventSubscriber, Orchestrator
from akgentic.team.models import TeamCard, TeamCardMember, TeamRuntime

logger = logging.getLogger(__name__)


class TeamFactory:
    """Build running teams from TeamCard + ActorSystem.

    Static builder: TeamFactory.build(team_card, actor_system, subscribers) -> TeamRuntime.
    Creates Orchestrator, spawns agents, wires routes_to, registers subscribers.
    Rollback on partial failure (tear down spawned actors if build fails).
    """

    @staticmethod
    def build(
        team_card: TeamCard,
        actor_system: ActorSystem,
        subscribers: list[EventSubscriber] | None = None,
    ) -> TeamRuntime:
        """Build a running team from a declarative TeamCard.

        Creates an Orchestrator actor, spawns all agents defined in the TeamCard
        member tree, registers subscribers and agent profiles, and returns a
        TeamRuntime handle to the running team.

        Args:
            team_card: Declarative team definition with entry point and members.
            actor_system: The actor system to host the team's actors.
            subscribers: Optional event subscribers to register with the orchestrator.

        Returns:
            A TeamRuntime with all actor addresses populated and proxies rebuilt.

        Raises:
            Exception: If any agent spawn fails, all already-spawned actors are
                torn down and the original exception is re-raised.
        """
        team_id = uuid.uuid4()
        spawned_addrs: list[ActorAddress] = []

        try:
            # 1. Create Orchestrator
            orchestrator_addr = actor_system.createActor(
                Orchestrator,
                config=BaseConfig(name="orchestrator", role="Orchestrator"),
                team_id=team_id,
            )
            spawned_addrs.append(orchestrator_addr)

            # 2. Get orchestrator proxy and register subscribers
            orchestrator_proxy: Orchestrator = actor_system.proxy_ask(
                orchestrator_addr, Orchestrator
            )
            for sub in subscribers or []:
                orchestrator_proxy.subscribe(sub)

            # 3. Walk TeamCard tree and spawn all agents
            addrs: dict[str, ActorAddress] = {}
            entry_addr: ActorAddress | None = None

            # Spawn entry point
            entry_addrs = TeamFactory._spawn_member(
                team_card.entry_point, actor_system, team_id, spawned_addrs
            )
            addrs.update(entry_addrs)
            # Entry point always has headcount=1, so use the card name
            entry_addr = entry_addrs[team_card.entry_point.card.config.name]

            # Spawn top-level members
            for member in team_card.members:
                member_addrs = TeamFactory._spawn_member(
                    member, actor_system, team_id, spawned_addrs
                )
                addrs.update(member_addrs)

            # 4. Register agent profiles with orchestrator
            orchestrator_proxy.register_agent_profiles(
                list(team_card.agent_cards.values())
            )

            # 5. Build supervisor_addrs
            supervisor_addrs: dict[str, ActorAddress] = {}
            for card in team_card.supervisors:
                name = card.config.name
                if name in addrs:
                    supervisor_addrs[name] = addrs[name]

            # 6. Build and return TeamRuntime
            return TeamRuntime(
                id=team_id,
                team=team_card,
                actor_system=actor_system,
                orchestrator_addr=orchestrator_addr,
                entry_addr=entry_addr,
                supervisor_addrs=supervisor_addrs,
                addrs=addrs,
            )

        except Exception:
            # Rollback: stop all already-spawned actors
            for addr in reversed(spawned_addrs):
                try:
                    addr.stop()
                except Exception:
                    logger.warning("Failed to stop actor during rollback: %s", addr)
            raise

    @staticmethod
    def _spawn_member(
        member: TeamCardMember,
        actor_system: ActorSystem,
        team_id: uuid.UUID,
        spawned_addrs: list[ActorAddress],
    ) -> dict[str, ActorAddress]:
        """Spawn a member and its subordinates recursively.

        Args:
            member: The TeamCardMember to spawn.
            actor_system: The actor system to host the actors.
            team_id: Team identifier for all spawned actors.
            spawned_addrs: Accumulator for rollback tracking.

        Returns:
            Dictionary mapping agent names to their spawned ActorAddresses.
        """
        result: dict[str, ActorAddress] = {}
        agent_class: type[Akgent[Any, Any]] = member.card.get_agent_class()
        name = member.card.config.name

        if member.headcount == 1:
            addr = actor_system.createActor(
                agent_class,
                config=member.card.get_config_copy(),
                team_id=team_id,
            )
            spawned_addrs.append(addr)
            result[name] = addr
        else:
            for i in range(member.headcount):
                indexed_name = f"{name}_{i}"
                config = member.card.get_config_copy()
                config.name = indexed_name
                addr = actor_system.createActor(
                    agent_class,
                    config=config,
                    team_id=team_id,
                )
                spawned_addrs.append(addr)
                result[indexed_name] = addr

        # Recurse into subordinates
        for child in member.members:
            child_addrs = TeamFactory._spawn_member(
                child, actor_system, team_id, spawned_addrs
            )
            result.update(child_addrs)

        return result
