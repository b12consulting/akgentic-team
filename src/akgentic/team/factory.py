"""TeamFactory: build running teams from TeamCard + ActorSystem."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.agent_config import BaseConfig
from akgentic.core.messages.orchestrator import SentMessage
from akgentic.core.orchestrator import STOP_TIMEOUT, EventSubscriber, Orchestrator
from akgentic.team.messages import WelcomeMessage
from akgentic.team.models import TeamCard, TeamCardMember, TeamRuntime, spawned_names
from akgentic.team.projection import derive_team_projection

logger = logging.getLogger(__name__)

# Grace period passed into the non-blocking ``Orchestrator.stop(grace_timeout)``
# on the rollback path (ADR-19 §2). Defaulted to core's ``STOP_TIMEOUT``; the
# returned event is ``.wait()``-ed with no caller-side timeout because core's
# backstop guarantees it is set within ~this many seconds.
GRACE_TIMEOUT_SECONDS = STOP_TIMEOUT


class TeamFactory:
    """Build running teams from TeamCard + ActorSystem.

    Static builder: TeamFactory.build(team_card, actor_system, subscribers) -> TeamRuntime.
    Creates the Orchestrator, spawns the member tree, registers the team's whole
    role-keyed card set with the orchestrator, registers subscribers, and rolls
    back on partial failure (tearing down every actor spawned so far).
    """

    @staticmethod
    def build(
        team_card: TeamCard,
        actor_system: ActorSystem,
        subscribers: list[EventSubscriber] | None = None,
        team_id: uuid.UUID | None = None,
    ) -> TeamRuntime:
        """Build a running team from a declarative TeamCard.

        Creates an Orchestrator actor, spawns all agents defined in the TeamCard
        member tree, registers subscribers and the team's whole role catalog,
        and returns a TeamRuntime handle to the running team.

        Catalog registration precedes the ``TeamRuntime`` construction and may
        not be reordered: the runtime resolves its entry-point agent class out
        of the catalog on construction.

        Every name the returned runtime records — ``entry_addr``, the keys of
        ``supervisor_addrs`` — comes from :func:`spawned_names`, the single
        statement of the naming rule ``_spawn_member`` performs. A supervisor
        declared with ``headcount=3`` therefore contributes its three spawned
        names to ``supervisor_addrs``, so ``TeamRuntime.send`` reaches all
        three, and those keys equal the ``Process.supervisors`` names
        ``derive_team_projection`` builds through the same function.

        Args:
            team_card: Declarative team definition with entry point and members.
            actor_system: The actor system to host the team's actors.
            subscribers: Optional event subscribers to register with the orchestrator.
            team_id: Optional pre-generated team identifier. If None, a new UUID
                is generated. Allows callers (e.g. TeamManager) to know the team_id
                before build completes.

        Returns:
            A TeamRuntime with all actor addresses populated and proxies rebuilt.

        Raises:
            Exception: If any agent spawn fails, all already-spawned actors are
                torn down and the original exception is re-raised.
        """
        team_id = team_id or uuid.uuid4()
        spawned_addrs: list[ActorAddress] = []

        if team_card.entry_point.headcount != 1:
            msg = (
                f"Entry point '{team_card.entry_point.card.config.name}' "
                f"must have headcount=1, got {team_card.entry_point.headcount}"
            )
            raise ValueError(msg)

        try:
            # 1. Create Orchestrator
            orchestrator_addr = actor_system.createActor(
                Orchestrator,
                config=BaseConfig(name="@Orchestrator", role="Orchestrator"),
                team_id=team_id,
            )
            spawned_addrs.append(orchestrator_addr)

            # 2. Get orchestrator proxy and register subscribers
            orchestrator_proxy: Orchestrator = actor_system.proxy_ask(
                orchestrator_addr, Orchestrator
            )
            TeamFactory._register_subscribers(orchestrator_proxy, subscribers)

            # 3. Walk TeamCard tree and spawn all agents, recording the first
            #    layer as it goes. ``supervisor_addrs`` is keyed by the names
            #    the spawn actually used, which ``spawned_names`` states once
            #    for both this and ``derive_team_projection``: matching on the
            #    DECLARED ``config.name`` dropped every ``headcount > 1``
            #    supervisor out of ``send()``'s fan-out silently.
            addrs: dict[str, ActorAddress] = {}
            supervisor_addrs: dict[str, ActorAddress] = {}
            entry_addr: ActorAddress | None = None

            # Spawn entry point through orchestrator
            entry_addrs = TeamFactory._spawn_member(
                team_card.entry_point,
                orchestrator_addr,
                actor_system,
                spawned_addrs,
            )
            addrs.update(entry_addrs)
            entry_addr = entry_addrs[spawned_names(team_card.entry_point)[0]]

            # Spawn top-level members through orchestrator
            for member in team_card.members:
                member_addrs = TeamFactory._spawn_member(
                    member,
                    orchestrator_addr,
                    actor_system,
                    spawned_addrs,
                )
                addrs.update(member_addrs)
                # Index the member's OWN result: ``_spawn_member`` returns the
                # member plus its whole subtree, and the first layer is what
                # ``supervisor_addrs`` is. Unguarded on purpose — a name the
                # rule produces that the spawn did not is a defect, not an
                # entry to skip.
                for name in spawned_names(member):
                    supervisor_addrs[name] = member_addrs[name]

            # 4. Register the team's whole role catalog with the orchestrator.
            TeamFactory._register_role_catalog(orchestrator_proxy, team_card)

            # 5. Announce the team's welcome message
            # Hand the orchestrator a SentMessage wrapping a WelcomeMessage.
            # The orchestrator's receiveMsg_SentMessage records it on the event
            # log and broadcasts it to every subscriber (CLI printer,
            # persistence, WebSocket -> frontend). Skipped when no greeting set.
            if team_card.welcome_message:
                actor_system.tell(
                    orchestrator_addr,
                    SentMessage(
                        message=WelcomeMessage(
                            content=team_card.welcome_message,
                            sender=orchestrator_addr,
                            team_id=team_id,
                        ),
                        recipient=entry_addr,
                    ),
                )

            # 6. Build and return TeamRuntime
            return TeamRuntime(
                id=team_id,
                team_name=team_card.name,
                message_types=list(team_card.message_types),
                actor_system=actor_system,
                orchestrator_addr=orchestrator_addr,
                entry_addr=entry_addr,
                supervisor_addrs=supervisor_addrs,
                addrs=addrs,
            )

        except Exception:
            TeamFactory._rollback_spawned(actor_system, spawned_addrs)
            raise

    @staticmethod
    def _register_role_catalog(
        orchestrator_proxy: Orchestrator,
        team_card: TeamCard,
    ) -> None:
        """Register one card per role reachable from *team_card*.

        The WHOLE roster — entry point, every member of the tree at every depth,
        and ``agent_profiles`` — not just the hireable subset the previous
        ``team_card.agent_profiles`` registration carried. That is the intended
        change (ADR-26 §Decision 5, FR1/FR2): an agent can only read a
        colleague's description and skills if the colleague is in the catalog,
        and ``TeamRuntime`` now resolves its entry-point agent class here too.

        It is NOT hireability-neutral today, and that is worth stating plainly
        rather than assuming. Each registered card carries the ``can_be_hired``
        value ``derive_team_projection`` gave it, but NOTHING in the framework
        reads that flag: the hire path takes any catalog card whose role matches
        (`akgentic-tool`, ``team/team.py``) and ``get_available_roles()``
        advertises every registered role back to the model. So until a hire
        guard exists, a team's already-live members are hirable by role and the
        model can spawn duplicates of them — exactly what the narrow
        registration this replaces was avoiding. The guard belongs to
        `akgentic-tool` (issue #321); Golden Rule 4 bars fixing it here.
        ``TeamRestorer._rebuild_agents`` carries the same note for the restore
        path, where the widening landed first.

        One precedence consequence, decided in the derivation and merely carried
        here: a role reachable from BOTH the member tree and ``agent_profiles``
        dedups to a single entry, and the PROFILE's card is the survivor. The
        registration this replaces carried only ``agent_profiles``, so for such a
        role it carried the profile's card too — the precedence is deliberately
        the pre-epic one. A profile declares what a newly hired agent of that
        role should be; the tree's card only records what one already-running
        member was built from, and it would be the wrong thing to hand to
        whoever asks the catalog what the role is.

        Derived through ``derive_team_projection`` rather than by a second walk
        of the card here: ``TeamManager.create_team`` derives the same
        projection for the ``Process`` it persists, and one pure function is
        what keeps the registered catalog and the stored record from
        disagreeing. Deliberately not threaded in from ``create_team`` — a
        direct ``TeamFactory.build`` caller must get a catalog too.

        Args:
            orchestrator_proxy: Proxy to the orchestrator actor.
            team_card: The declarative definition whose roles to register.
        """
        orchestrator_proxy.register_agent_profiles(derive_team_projection(team_card).cards)

    @staticmethod
    def _rollback_spawned(
        actor_system: ActorSystem,
        spawned_addrs: list[ActorAddress],
    ) -> None:
        """Tear down already-spawned actors after a partial-build failure.

        Stops ``reversed(spawned_addrs)`` so the orchestrator (spawned first) is
        stopped last — by then its team roster is empty, so its stop finalizes
        near-instantly. The orchestrator entry uses the non-blocking
        ``Orchestrator.stop(grace).wait()`` (akgentic-core ADR-012) so rollback
        does not return while a live orchestrator (subscribers attached, event
        stream open) lingers; agent entries keep the unchanged blocking
        ``Akgent.stop()`` (ADR-19 §2). Best-effort: per-actor failures are
        logged and do not abort the rollback.

        Args:
            actor_system: The actor system whose proxies drive the stops.
            spawned_addrs: Actors spawned so far, in spawn order
                (``spawned_addrs[0]`` is the orchestrator).
        """
        orchestrator_addr = spawned_addrs[0] if spawned_addrs else None
        for addr in reversed(spawned_addrs):
            try:
                if addr == orchestrator_addr:
                    actor_system.proxy_ask(addr, Orchestrator).stop(
                        GRACE_TIMEOUT_SECONDS
                    ).wait()
                else:
                    actor_system.proxy_ask(addr, Akgent).stop()
            except Exception:
                logger.warning("Failed to stop actor during rollback: %s", addr)

    @staticmethod
    def _register_subscribers(
        orchestrator_proxy: Orchestrator,
        subscribers: list[EventSubscriber] | None,
    ) -> None:
        """Register subscribers and replay missed orchestrator startup events.

        The orchestrator generates its own StartMessage during ``on_start()``,
        before any subscribers are registered. This method replays those
        startup events so subscribers capture the full event history.

        Args:
            orchestrator_proxy: Proxy to the orchestrator actor.
            subscribers: Optional list of event subscribers to register.
        """
        for sub in subscribers or []:
            orchestrator_proxy.subscribe(sub)

        if subscribers:
            for msg in orchestrator_proxy.get_messages():
                for sub in subscribers:
                    sub.on_message(msg)

    @staticmethod
    def _spawn_member(
        member: TeamCardMember,
        parent_addr: ActorAddress,
        actor_system: ActorSystem,
        spawned_addrs: list[ActorAddress],
    ) -> dict[str, ActorAddress]:
        """Spawn a member and its subordinates recursively via public proxy API.

        Uses ``actor_system.proxy_ask(parent_addr, Akgent).createActor()``
        to spawn children through the parent, ensuring context propagation
        (orchestrator, parent, user_id, team_id) is handled by ``createActor()``.

        Args:
            member: The TeamCardMember to spawn.
            parent_addr: Address of the parent actor to spawn through.
            actor_system: The actor system for creating proxies.
            spawned_addrs: Accumulator for rollback tracking.

        Returns:
            Dictionary mapping agent names to their spawned ActorAddresses.
        """
        result: dict[str, ActorAddress] = {}
        agent_class: type[Akgent[Any, Any]] = member.card.get_agent_class()
        name = member.card.config.name
        parent_proxy: Akgent[Any, Any] = actor_system.proxy_ask(parent_addr, Akgent)

        if member.headcount == 1:
            addr = parent_proxy.createActor(
                agent_class,
                config=member.card.get_config_copy(),
            )
            if addr is None:
                msg = f"Failed to spawn agent '{name}'"
                raise RuntimeError(msg)
            spawned_addrs.append(addr)
            result[name] = addr
        else:
            for i in range(member.headcount):
                indexed_name = f"{name}_{i}"
                config = member.card.get_config_copy()
                config.name = indexed_name
                addr = parent_proxy.createActor(
                    agent_class,
                    config=config,
                )
                if addr is None:
                    msg = f"Failed to spawn agent '{indexed_name}'"
                    raise RuntimeError(msg)
                spawned_addrs.append(addr)
                result[indexed_name] = addr

        # Recurse into subordinates using the spawned agent as parent
        if member.members:
            # For headcount == 1, use the single spawned agent as parent
            # For headcount > 1, use the last spawned instance as parent
            last_addr = next(reversed(result.values()))
            for child in member.members:
                child_addrs = TeamFactory._spawn_member(
                    child,
                    last_addr,
                    actor_system,
                    spawned_addrs,
                )
                result.update(child_addrs)

        return result
