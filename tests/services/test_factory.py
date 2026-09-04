"""Tests for TeamFactory.build — AC 1-10."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any
from unittest.mock import patch

import pytest
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.agent_card import AgentCard
from akgentic.core.agent_config import BaseConfig
from akgentic.core.agent_state import BaseState
from akgentic.core.messages.message import Message, UserMessage
from akgentic.core.messages.orchestrator import SentMessage
from akgentic.core.orchestrator import Orchestrator

from akgentic.team.factory import GRACE_TIMEOUT_SECONDS, TeamFactory
from akgentic.team.models import TeamCard, TeamCardMember, TeamRuntime, spawned_names
from akgentic.team.projection import derive_team_projection

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class StubAgent(Akgent[BaseConfig, BaseState]):
    """Minimal agent for factory tests."""

    pass


class FailingAgent(Akgent[BaseConfig, BaseState]):
    """Agent that raises during __init__ for rollback tests."""

    def __init__(self, **kwargs: Any) -> None:
        msg = "FailingAgent intentional error"
        raise RuntimeError(msg)


class StubSubscriber:
    """Minimal EventSubscriber for testing.

    Implements the team_id-aware ``EventSubscriber`` Protocol. The stub does
    not assert on ``team_id`` because it is reused across tests.
    """

    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.stopped: bool = False

    def on_stop(self, team_id: uuid.UUID) -> None:
        del team_id
        self.stopped = True

    def on_stop_request(self, team_id: uuid.UUID) -> None:
        del team_id

    def set_restoring(self, team_id: uuid.UUID, restoring: bool) -> None:  # noqa: FBT001
        del team_id, restoring

    def on_message(self, msg: Message) -> None:
        self.messages.append(msg)


def _make_card(
    name: str,
    role: str = "TestRole",
    agent_class: type[Akgent[Any, Any]] = StubAgent,
) -> AgentCard:
    return AgentCard(
        role=role,
        description=f"Test: {role}",
        skills=["testing"],
        agent_class=agent_class,
        config=BaseConfig(name=name, role=role),
    )


def _make_member(
    name: str,
    role: str = "TestRole",
    agent_class: type[Akgent[Any, Any]] = StubAgent,
    headcount: int = 1,
    members: list[TeamCardMember] | None = None,
) -> TeamCardMember:
    return TeamCardMember(
        card=_make_card(name, role, agent_class),
        headcount=headcount,
        members=members or [],
    )


def _make_team_card(
    entry_point: TeamCardMember | None = None,
    members: list[TeamCardMember] | None = None,
    name: str = "test-team",
) -> TeamCard:
    ep = entry_point or _make_member("lead", "Lead")
    return TeamCard(
        name=name,
        description="Test team",
        entry_point=ep,
        members=members or [],
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def actor_system() -> ActorSystem:  # type: ignore[misc]
    system = ActorSystem()
    yield system  # type: ignore[misc]
    system.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTeamFactoryBuild:
    """AC 1-8: TeamFactory.build creates a running team from a TeamCard."""

    # -- 3.1: Successful build with single agent -------------------------

    def test_build_single_agent(self, actor_system: ActorSystem) -> None:
        """AC 1,7: Build returns TeamRuntime with orchestrator, entry, addrs."""
        tc = _make_team_card()
        runtime = TeamFactory.build(tc, actor_system)

        assert isinstance(runtime, TeamRuntime)
        assert isinstance(runtime.id, uuid.UUID)
        assert runtime.orchestrator_addr is not None
        assert runtime.entry_addr is not None
        assert "lead" in runtime.addrs
        assert runtime.orchestrator_addr.is_alive()
        assert runtime.entry_addr.is_alive()

    # -- 3.2: Successful build with multiple agents (hierarchical) -------

    def test_build_hierarchical_team(self, actor_system: ActorSystem) -> None:
        """AC 2: All agents in TeamCard member tree are spawned."""
        worker = _make_member("worker", "Worker")
        supervisor = _make_member("supervisor", "Supervisor", members=[worker])
        tc = _make_team_card(members=[supervisor])

        runtime = TeamFactory.build(tc, actor_system)

        assert "lead" in runtime.addrs
        assert "supervisor" in runtime.addrs
        assert "worker" in runtime.addrs
        assert len(runtime.addrs) == 3
        for addr in runtime.addrs.values():
            assert addr.is_alive()

    # -- 3.3: Headcount > 1 spawns multiple instances --------------------

    def test_headcount_multiple_instances(self, actor_system: ActorSystem) -> None:
        """AC 3: Headcount > 1 spawns multiple actor instances."""
        multi = _make_member("worker", "Worker", headcount=3)
        tc = _make_team_card(members=[multi])

        runtime = TeamFactory.build(tc, actor_system)

        assert "worker_0" in runtime.addrs
        assert "worker_1" in runtime.addrs
        assert "worker_2" in runtime.addrs
        assert "worker" not in runtime.addrs
        # lead + 3 workers = 4
        assert len(runtime.addrs) == 4

    # -- 3.4: Subscriber registration -----------------------------------

    def test_subscriber_registration(self, actor_system: ActorSystem) -> None:
        """AC 5: Provided subscribers are registered with the orchestrator."""
        sub = StubSubscriber()
        tc = _make_team_card()

        runtime = TeamFactory.build(tc, actor_system, subscribers=[sub])

        # Verify subscriber is registered by stopping the orchestrator via proxy,
        # which calls on_stop on all subscribers.
        runtime.orchestrator_proxy.stop()
        # on_stop() fires asynchronously after the proxy stop() call returns;
        # wait for the actor thread to finish so on_stop() has been invoked.
        deadline = time.monotonic() + 2.0
        while runtime.orchestrator_addr.is_alive() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sub.stopped is True

    # -- 3.6: Partial failure rollback -----------------------------------

    def test_partial_failure_rollback(self, actor_system: ActorSystem) -> None:
        """AC 8: Partial failure tears down already-spawned actors."""
        # FailingAgent will raise on start -- this should trigger rollback
        failing = _make_member("failing", "Failing", agent_class=FailingAgent)
        tc = _make_team_card(members=[failing])

        with pytest.raises(Exception):
            TeamFactory.build(tc, actor_system)

    # -- 3.7: TeamRuntime.id equals orchestrator's team_id ---------------

    def test_runtime_id_is_team_id(self, actor_system: ActorSystem) -> None:
        """AC 7: TeamRuntime.id is the team_id assigned to all actors."""
        tc = _make_team_card()
        runtime = TeamFactory.build(tc, actor_system)

        # The orchestrator's team_id should match runtime.id
        assert runtime.orchestrator_addr.team_id == runtime.id
        assert runtime.entry_addr.team_id == runtime.id

    # -- 3.8: The full role catalog registered with the orchestrator -----

    def test_declared_profiles_are_in_the_catalog(self, actor_system: ActorSystem) -> None:
        """The roles named in agent_profiles reach the catalog, flagged hireable."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])
        tc.agent_profiles = list(tc.agent_cards.values())

        runtime = TeamFactory.build(tc, actor_system)

        catalog = runtime.orchestrator_proxy.get_agent_catalog()
        assert {c.role for c in catalog} == {"Lead", "Worker"}
        assert all(c.can_be_hired for c in catalog)

    # -- Additional edge-case tests --------------------------------------

    def test_build_with_no_subscribers(self, actor_system: ActorSystem) -> None:
        """Build works when subscribers is None."""
        tc = _make_team_card()
        runtime = TeamFactory.build(tc, actor_system, subscribers=None)
        assert runtime.orchestrator_addr.is_alive()

    def test_supervisor_addrs_populated(self, actor_system: ActorSystem) -> None:
        """Supervisor addresses are populated for first-layer members."""
        worker = _make_member("worker", "Worker")
        supervisor = _make_member("supervisor", "Supervisor", members=[worker])
        tc = _make_team_card(members=[supervisor])

        runtime = TeamFactory.build(tc, actor_system)

        # supervisor is a first-layer member, so it's in supervisor_addrs
        assert "supervisor" in runtime.supervisor_addrs
        assert runtime.supervisor_addrs["supervisor"] == runtime.addrs["supervisor"]
        # entry point (lead) is NOT in supervisor_addrs
        assert "lead" not in runtime.supervisor_addrs

    def test_entry_point_headcount_gt1_raises(self, actor_system: ActorSystem) -> None:
        """Entry point with headcount > 1 raises ValueError."""
        ep = _make_member("lead", "Lead", headcount=2)
        tc = _make_team_card(entry_point=ep)

        with pytest.raises(ValueError, match="must have headcount=1"):
            TeamFactory.build(tc, actor_system)

    def test_rollback_handles_stop_failure(self, actor_system: ActorSystem) -> None:
        """Rollback completes promptly even when stopping an actor raises.

        A broken ``Akgent.stop`` means the failed worker never emits a
        StopMessage, so the orchestrator's roster never empties and its stop
        finalizes only via the backstop timer. Drive the rollback grace down to
        a fraction of a second (patching the module-level ``GRACE_TIMEOUT_SECONDS``
        the rollback reads) so this best-effort cleanup path does not block for
        the full production grace on a wedged team. The rollback must still log
        the stop failure and re-raise the original build error.
        """
        failing = _make_member("failing", "Failing", agent_class=FailingAgent)
        tc = _make_team_card(members=[failing])

        with (
            patch("akgentic.team.factory.GRACE_TIMEOUT_SECONDS", 0.1),
            patch.object(Akgent, "stop", side_effect=RuntimeError("stop failed")),
        ):
            start = time.monotonic()
            with pytest.raises(RuntimeError) as excinfo:
                TeamFactory.build(tc, actor_system)
            elapsed = time.monotonic() - start

            # The original spawn failure from FailingAgent propagates as-is
            # (no wrapping). Test asserts on type + non-empty str, not a
            # fragile match on the fixture's own message text.
            assert str(excinfo.value)
            # Rollback returned promptly via the small grace backstop rather than
            # blocking for the full production grace on the wedged orchestrator.
            assert elapsed < 5.0, f"rollback blocked too long: {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# Tests: Hierarchy propagation (Story 10-1, AC 1, 2, 5)
# ---------------------------------------------------------------------------


class TestFactoryHierarchyPropagation:
    """AC 1,2,5: Spawned agents have _orchestrator and _parent set."""

    def test_orchestrator_set_on_spawned_agents(
        self, actor_system: ActorSystem
    ) -> None:
        """AC 1,5: _orchestrator is not None on all spawned agents."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])

        runtime = TeamFactory.build(tc, actor_system)

        for name, addr in runtime.addrs.items():
            proxy: Akgent[Any, Any] = actor_system.proxy_ask(addr, Akgent)
            orch = proxy.orchestrator
            assert orch is not None, f"Agent '{name}' has _orchestrator=None"
            assert orch.is_alive(), f"Agent '{name}' orchestrator is not alive"

    def test_parent_set_correctly_for_top_level_agents(
        self, actor_system: ActorSystem
    ) -> None:
        """AC 2,5: Top-level agents have orchestrator as parent."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])

        runtime = TeamFactory.build(tc, actor_system)

        # Both lead (entry point) and worker (top-level member) should have
        # the orchestrator as parent
        for name in ("lead", "worker"):
            addr = runtime.addrs[name]
            impl = addr  # ActorAddressImpl
            actor = impl._actor_ref._actor_weakref()  # type: ignore[union-attr]
            assert actor._parent is not None, f"Agent '{name}' has _parent=None"
            assert actor._parent.agent_id == runtime.orchestrator_addr.agent_id, (
                f"Agent '{name}' parent is not the orchestrator"
            )

    def test_parent_set_correctly_for_subordinates(
        self, actor_system: ActorSystem
    ) -> None:
        """AC 2,5: Subordinate agents have their supervisor as parent."""
        worker = _make_member("worker", "Worker")
        supervisor = _make_member("supervisor", "Supervisor", members=[worker])
        tc = _make_team_card(members=[supervisor])

        runtime = TeamFactory.build(tc, actor_system)

        # Worker should have supervisor as parent, not orchestrator
        worker_addr = runtime.addrs["worker"]
        worker_actor = worker_addr._actor_ref._actor_weakref()  # type: ignore[union-attr]
        supervisor_addr = runtime.addrs["supervisor"]

        assert worker_actor._parent is not None
        assert worker_actor._parent.agent_id == supervisor_addr.agent_id, (
            "Worker's parent should be the supervisor"
        )

        # Supervisor should have orchestrator as parent
        supervisor_actor = supervisor_addr._actor_ref._actor_weakref()  # type: ignore[union-attr]
        assert supervisor_actor._parent is not None
        assert supervisor_actor._parent.agent_id == runtime.orchestrator_addr.agent_id

    def test_orchestrator_set_on_headcount_agents(
        self, actor_system: ActorSystem
    ) -> None:
        """AC 5: _orchestrator is set on agents with headcount > 1."""
        multi = _make_member("worker", "Worker", headcount=2)
        tc = _make_team_card(members=[multi])

        runtime = TeamFactory.build(tc, actor_system)

        for name in ("worker_0", "worker_1"):
            proxy: Akgent[Any, Any] = actor_system.proxy_ask(
                runtime.addrs[name], Akgent
            )
            assert proxy.orchestrator is not None, (
                f"Agent '{name}' has _orchestrator=None"
            )


# ---------------------------------------------------------------------------
# Tests: Proxy-based spawning (Story 12.4, AC 3, 4)
# ---------------------------------------------------------------------------


class TestFactoryProxySpawning:
    """AC 3,4: Agents spawned through public createActor() API."""

    def test_build_creates_agents_through_public_api(
        self, actor_system: ActorSystem
    ) -> None:
        """AC 3: After build, all agents are alive and reachable via get_team()."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])

        runtime = TeamFactory.build(tc, actor_system)

        team = runtime.orchestrator_proxy.get_team()
        team_names = {addr.name for addr in team}
        assert "lead" in team_names
        assert "worker" in team_names
        for addr in team:
            assert addr.is_alive()

    def test_build_entry_point_not_in_supervisor_addrs_without_subordinates(
        self, actor_system: ActorSystem
    ) -> None:
        """Entry point without subordinates is NOT in supervisor_addrs."""
        tc = _make_team_card()  # lead has no subordinates

        runtime = TeamFactory.build(tc, actor_system)

        assert "lead" not in runtime.supervisor_addrs
        # Entry point is still reachable via entry_proxy
        assert runtime.entry_proxy is not None


# ---------------------------------------------------------------------------
# Tests: rollback waits on the orchestrator stop event (ADR-19 §2)
# ---------------------------------------------------------------------------


class TestFactoryRollbackWaitsOnOrchestrator:
    """AC2: build rollback waits on the orchestrator's non-blocking stop event.

    On a partial-build failure the rollback stops ``reversed(spawned_addrs)``.
    Agent entries keep the blocking ``Akgent.stop()``; the orchestrator entry
    (spawned first, stopped last) must use the non-blocking
    ``Orchestrator.stop(grace).wait()`` so rollback does not return while a
    live orchestrator lingers.
    """

    def test_factory_rollback_waits_on_orchestrator(
        self, actor_system: ActorSystem
    ) -> None:
        """AC2: rollback stops agents blocking, then waits on the orchestrator
        entry's stop event; the orchestrator is dead once build re-raises.

        We wrap the real ``Orchestrator.stop`` so it returns a wrapped event
        whose ``wait`` records the call order, and the real ``Akgent.stop`` so
        agent stops record their order. The orchestrator's ``wait()`` must run
        AFTER both agent stops (reversed order) and the orchestrator must be
        torn down by the time ``build`` re-raises.
        """
        good_worker = _make_member("good", "Good")
        failing = _make_member("failing", "Failing", agent_class=FailingAgent)
        tc = _make_team_card(members=[good_worker, failing])

        events: list[str] = []
        wait_calls: list[float] = []
        grace_args: list[float] = []

        real_orch_stop = Orchestrator.stop
        real_agent_stop = Akgent.stop

        def wrapped_orch_stop(
            self: Orchestrator, grace_timeout: float = GRACE_TIMEOUT_SECONDS
        ) -> threading.Event:
            grace_args.append(grace_timeout)
            evt = real_orch_stop(self, grace_timeout)

            class _RecordingEvent:
                def __init__(self, inner: threading.Event) -> None:
                    self._inner = inner

                def wait(self, timeout: float | None = None) -> bool:
                    events.append("orchestrator-wait")
                    wait_calls.append(time.monotonic())
                    return self._inner.wait(timeout)

            return _RecordingEvent(evt)  # type: ignore[return-value]

        def wrapped_agent_stop(self: Akgent[Any, Any]) -> None:
            # Only record non-orchestrator agent stops.
            if not isinstance(self, Orchestrator):
                events.append("agent-stop")
            return real_agent_stop(self)

        with (
            patch.object(Orchestrator, "stop", wrapped_orch_stop),
            patch.object(Akgent, "stop", wrapped_agent_stop),
        ):
            with pytest.raises(RuntimeError):
                TeamFactory.build(tc, actor_system)

        # Orchestrator stop received the single grace timeout, and its wait()
        # ran. Agent stops are blocking and occur BEFORE the orchestrator wait
        # (reversed spawn order → orchestrator last).
        assert grace_args == [GRACE_TIMEOUT_SECONDS]
        assert events.count("agent-stop") >= 1, f"no agent stops recorded: {events}"
        assert "orchestrator-wait" in events
        assert events.index("orchestrator-wait") == len(events) - 1, (
            f"orchestrator wait must be last (after agent stops): {events}"
        )


class TestBuildRegistersTheWholeRoleCatalog:
    """AC 1, 3-8 (31-2): the catalog is the team's full roster, not a subset.

    Note what these tests do NOT assert: nothing in the framework reads
    ``can_be_hired`` yet, so a card carrying it is carried, not enforced. The
    flag's value is asserted as a value.
    """

    @staticmethod
    def _three_level_card_with_two_profiles() -> TeamCard:
        """Entry point, a three-level tree, and two roles only in agent_profiles."""
        junior = _make_member("junior", "Junior")
        worker = _make_member("worker", "Worker", members=[junior])
        supervisor = _make_member("supervisor", "Supervisor", members=[worker])
        tc = _make_team_card(members=[supervisor])
        tc.agent_profiles = [
            _make_card("analyst", "Analyst"),
            _make_card("scribe", "Scribe"),
        ]
        return tc

    def test_every_reachable_role_gets_exactly_one_entry(
        self, actor_system: ActorSystem
    ) -> None:
        """AC1: entry point + every tree depth + every profile, one entry per role."""
        tc = self._three_level_card_with_two_profiles()

        runtime = TeamFactory.build(tc, actor_system)

        catalog = runtime.orchestrator_proxy.get_agent_catalog()
        roles = [c.role for c in catalog]
        assert sorted(roles) == [
            "Analyst",
            "Junior",
            "Lead",
            "Scribe",
            "Supervisor",
            "Worker",
        ]
        assert len(roles) == len(set(roles))

    def test_a_role_in_both_the_tree_and_the_profiles_yields_one_hireable_entry(
        self, actor_system: ActorSystem
    ) -> None:
        """AC3: no duplicate entry, and the surviving one is flagged hireable."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])
        tc.agent_profiles = [_make_card("worker-profile", "Worker")]

        runtime = TeamFactory.build(tc, actor_system)

        catalog = runtime.orchestrator_proxy.get_agent_catalog()
        workers = [c for c in catalog if c.role == "Worker"]
        assert len(workers) == 1
        assert workers[0].can_be_hired is True

    def test_the_profiles_card_reaches_the_catalog_for_a_dual_listed_role(
        self, actor_system: ActorSystem
    ) -> None:
        """AC 16, create path: WHICH card the catalog holds, not how many.

        The profile and the tree card for ``Worker`` differ in description and
        skills, so this fails under tree-wins precedence rather than passing
        either way — which the count-and-flag spec above cannot do.
        """
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])
        profile = _make_card("worker-profile", "Worker")
        profile.description = "Hired to work, not the one already working"
        profile.skills = ["onboarding"]
        tc.agent_profiles = [profile]

        runtime = TeamFactory.build(tc, actor_system)

        catalog = runtime.orchestrator_proxy.get_agent_catalog()
        entry = next(c for c in catalog if c.role == "Worker")
        assert entry.description == profile.description
        assert entry.skills == profile.skills
        assert entry.description != worker.card.description

    def test_only_profile_roles_carry_the_hireable_flag(
        self, actor_system: ActorSystem
    ) -> None:
        """AC4: every tree-only role in the catalog reads can_be_hired=False."""
        tc = self._three_level_card_with_two_profiles()

        runtime = TeamFactory.build(tc, actor_system)

        flags = {c.role: c.can_be_hired for c in runtime.orchestrator_proxy.get_agent_catalog()}
        assert flags == {
            "Lead": False,
            "Supervisor": False,
            "Worker": False,
            "Junior": False,
            "Analyst": True,
            "Scribe": True,
        }

    def test_a_team_with_no_profiles_still_registers_its_roster(
        self, actor_system: ActorSystem
    ) -> None:
        """AC5: the replaced code skipped registration entirely for such a team."""
        worker = _make_member("worker", "Worker")
        tc = _make_team_card(members=[worker])
        assert tc.agent_profiles == []

        runtime = TeamFactory.build(tc, actor_system)

        catalog = runtime.orchestrator_proxy.get_agent_catalog()
        assert {c.role for c in catalog} == {"Lead", "Worker"}
        assert not any(c.can_be_hired for c in catalog)

    def test_the_registered_cards_are_the_derivation_functions_own_output(
        self, actor_system: ActorSystem
    ) -> None:
        """AC6: a second walk of the card inside factory.py fails here."""
        tc = self._three_level_card_with_two_profiles()

        runtime = TeamFactory.build(tc, actor_system)

        assert runtime.orchestrator_proxy.get_agent_catalog() == derive_team_projection(tc).cards

    def test_the_callers_cards_are_never_flagged(self, actor_system: ActorSystem) -> None:
        """AC7: every card reachable from the INPUT card still reads False."""
        tc = self._three_level_card_with_two_profiles()

        TeamFactory.build(tc, actor_system)

        reachable = [
            tc.entry_point.card,
            *tc.agent_cards.values(),
            *tc.agent_profiles,
        ]
        assert [c.can_be_hired for c in reachable] == [False] * len(reachable)

    def test_a_hireable_catalog_entry_is_a_copy_not_the_callers_object(
        self, actor_system: ActorSystem
    ) -> None:
        """AC8: the flag is applied to a model_copy, never written back."""
        tc = self._three_level_card_with_two_profiles()
        analyst_input = tc.agent_profiles[0]

        runtime = TeamFactory.build(tc, actor_system)

        catalog = runtime.orchestrator_proxy.get_agent_catalog()
        analyst_registered = next(c for c in catalog if c.role == "Analyst")
        assert analyst_registered is not analyst_input
        assert analyst_registered.can_be_hired is True
        assert analyst_input.can_be_hired is False


class TestSpawnedNamesMatchesTheFactory:
    """AC 15: the projection's naming rule and the factory's spawning agree."""

    def test_every_name_the_rule_produces_is_a_live_agent(
        self, actor_system: ActorSystem
    ) -> None:
        """The projection records spawned names; a divergence must go red here.

        ``spawned_names`` states the ``headcount`` expansion once, and
        ``TeamFactory._spawn_member`` performs it. Nothing but this test holds
        the two together — 31-4 makes the factory call the function, and until
        then a drift would only surface as a supervisor silently missing from
        ``send()``'s fan-out at runtime.
        """
        entry = _make_member("lead", "Lead")
        crew = _make_member("worker", "Worker", headcount=3)
        solo = _make_member("scribe", "Scribe")
        tc = _make_team_card(entry_point=entry, members=[crew, solo])

        runtime = TeamFactory.build(tc, actor_system)

        for member in (entry, crew, solo):
            for name in spawned_names(member):
                assert name in runtime.addrs, (
                    f"spawned_names produced '{name}', which the factory never "
                    f"spawned: {sorted(runtime.addrs)}"
                )
        assert spawned_names(crew) == ["worker_0", "worker_1", "worker_2"]


def _sent_recipients(
    subscriber: StubSubscriber, since: int, expected: int
) -> list[SentMessage]:
    """Return the ``SentMessage``s recorded after *since*, once *expected* arrive.

    ``TeamRuntime.send`` routes through a fire-and-forget ``proxy_tell`` entry
    proxy and the orchestrator notification is asynchronous, so the count is
    polled rather than read once. Returning early on the expected count keeps a
    passing test fast; the deadline is what makes a failing one finite.

    On timeout this returns whatever DID arrive rather than failing, so every
    caller pins ``len(...)`` as well as the recipients — a set of one value is
    equal to a set of three identical ones, so a partial delivery reads as a
    pass without the count.
    """
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        sent = [m for m in subscriber.messages[since:] if isinstance(m, SentMessage)]
        if len(sent) >= expected:
            return sent
        time.sleep(0.01)
    return [m for m in subscriber.messages[since:] if isinstance(m, SentMessage)]


class TestASupervisorDeclaredWithHeadcountReachesTheFanOut:
    """FR7: ``supervisor_addrs`` is keyed by SPAWNED names, not declared ones.

    A supervisor declared ``headcount=3`` is spawned as ``worker_0..2``, so a
    construction that matches the bare ``config.name`` never finds it and
    ``send()`` silently never reaches it.
    """

    def test_a_headcount_supervisor_contributes_one_entry_per_instance(
        self, actor_system: ActorSystem
    ) -> None:
        """The three spawned names are the keys; the declared name is nowhere."""
        crew = _make_member("worker", "Worker", headcount=3)
        tc = _make_team_card(members=[crew])

        runtime = TeamFactory.build(tc, actor_system)

        assert set(runtime.supervisor_addrs) == {"worker_0", "worker_1", "worker_2"}
        for name in ("worker_0", "worker_1", "worker_2"):
            assert runtime.supervisor_addrs[name] == runtime.addrs[name]
        assert "worker" not in runtime.supervisor_addrs
        assert "worker" not in runtime.addrs

    def test_the_expansion_leaves_the_layer_boundary_where_it_was(
        self, actor_system: ActorSystem
    ) -> None:
        """Expanded, ``supervisor_addrs`` is still the first layer of members.

        The entry point is the sender, not a recipient, and a second-layer
        subordinate is internal to its supervisor's subtree. Neither joins the
        fan-out because a sibling member happens to be multi-instance.
        """
        crew = _make_member("worker", "Worker", headcount=3)
        junior = _make_member("junior", "Junior")
        solo = _make_member("scribe", "Scribe", members=[junior])
        tc = _make_team_card(members=[crew, solo])

        runtime = TeamFactory.build(tc, actor_system)

        assert set(runtime.supervisor_addrs) == {
            "worker_0",
            "worker_1",
            "worker_2",
            "scribe",
        }
        assert "lead" not in runtime.supervisor_addrs
        assert "junior" not in runtime.supervisor_addrs
        # The subordinate WAS spawned — it is excluded by layer, not missing.
        assert "junior" in runtime.addrs

    def test_send_reaches_every_instance_of_a_headcount_supervisor(
        self, actor_system: ActorSystem
    ) -> None:
        """The symptom: the send is RECORDED, not merely called.

        ``send`` routes through ``ActorSystem.proxy_tell``, which casts and
        discards the actor type it is handed, so a call that does not raise
        proves nothing. ``len(sent)`` is pinned alongside the recipient set
        because a set comparison alone cannot tell a full delivery from a
        partial one.
        """
        crew = _make_member("worker", "Worker", headcount=3)
        tc = _make_team_card(members=[crew])
        tc.message_types = [UserMessage]
        recording = StubSubscriber()

        runtime = TeamFactory.build(tc, actor_system, subscribers=[recording])
        baseline = len(recording.messages)
        runtime.send("staff the whole crew")

        sent = _sent_recipients(recording, baseline, expected=3)
        assert len(sent) == 3
        assert {m.recipient.name for m in sent} == {"worker_0", "worker_1", "worker_2"}
