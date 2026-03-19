"""Integration tests for TeamManager lifecycle: create, stop, resume.

Tests use real Akgent subclasses, real orchestrators, and real event flow.
"""

from __future__ import annotations

from typing import Any

from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.messages.message import UserMessage

from akgentic.team.manager import TeamManager
from akgentic.team.models import TeamCard, TeamCardMember, TeamStatus
from tests.integration.conftest import (
    RecordingAgent,
    StatefulAgent,
    get_actor_from_addr,
    make_integration_agent_card,
    wait_for_agent_state,
)
from tests.services.conftest import InMemoryEventStore


def _make_simple_team_card(
    agent_class: type[Akgent[Any, Any]] = RecordingAgent,
    name: str = "entry",
    role: str = "Entry",
) -> TeamCard:
    """Create a minimal team card with a single entry-point agent."""
    entry = TeamCardMember(
        card=make_integration_agent_card(
            name=name,
            role=role,
            agent_class=agent_class,
        ),
    )
    return TeamCard(
        name="simple-team",
        description="A single-agent team for lifecycle tests",
        entry_point=entry,
        members=[],
        message_types=[UserMessage],
    )


class TestLifecycleIntegration:
    """TeamManager lifecycle integration tests."""

    def test_create_team_is_functional(
        self,
        actor_system: ActorSystem,
    ) -> None:
        """A team created via TeamManager can receive messages."""
        event_store = InMemoryEventStore()
        manager = TeamManager(actor_system, event_store)
        team_card = _make_simple_team_card()

        runtime = manager.create_team(team_card)

        # Send a message directly to the entry agent
        actor_system.tell(
            runtime.entry_addr,
            UserMessage(content="lifecycle-test"),
        )

        reached = wait_for_agent_state(
            runtime.entry_addr,
            lambda state: "lifecycle-test" in getattr(state, "messages", []),
            timeout=3.0,
        )
        assert reached, "Message did not reach entry agent"

    def test_stop_team_persists_state(
        self,
        actor_system: ActorSystem,
    ) -> None:
        """Stopping a team persists Process and events to EventStore."""
        event_store = InMemoryEventStore()
        manager = TeamManager(actor_system, event_store)
        team_card = _make_simple_team_card()

        runtime = manager.create_team(team_card)
        team_id = runtime.id

        # Send messages to generate events
        actor_system.tell(
            runtime.entry_addr,
            UserMessage(content="msg-1"),
        )
        wait_for_agent_state(
            runtime.entry_addr,
            lambda state: getattr(state, "counter", 0) >= 1,
            timeout=3.0,
        )

        manager.stop_team(team_id)

        # Verify persistence
        process = event_store.load_team(team_id)
        assert process is not None, "Process not found after stop"
        assert process.status == TeamStatus.STOPPED, (
            f"Expected STOPPED, got {process.status}"
        )

        events = event_store.load_events(team_id)
        assert len(events) > 0, "No events persisted"

    def test_resume_team_is_functional(
        self,
        actor_system: ActorSystem,
    ) -> None:
        """A resumed team can send and receive NEW messages."""
        event_store = InMemoryEventStore()
        manager = TeamManager(actor_system, event_store)
        team_card = _make_simple_team_card()

        runtime = manager.create_team(team_card)
        team_id = runtime.id

        actor_system.tell(
            runtime.entry_addr,
            UserMessage(content="message-1"),
        )
        wait_for_agent_state(
            runtime.entry_addr,
            lambda state: "message-1" in getattr(state, "messages", []),
            timeout=3.0,
        )

        manager.stop_team(team_id)

        # Resume the team
        new_runtime = manager.resume_team(team_id)

        actor_system.tell(
            new_runtime.entry_addr,
            UserMessage(content="message-2"),
        )

        reached = wait_for_agent_state(
            new_runtime.entry_addr,
            lambda state: "message-2" in getattr(state, "messages", []),
            timeout=3.0,
        )
        assert reached, "Message 'message-2' did not reach entry agent on resumed team"

    def test_resume_team_preserves_agent_state(
        self,
        actor_system: ActorSystem,
    ) -> None:
        """Agent state survives stop/resume cycle."""
        event_store = InMemoryEventStore()
        manager = TeamManager(actor_system, event_store)
        team_card = _make_simple_team_card(
            agent_class=StatefulAgent,
            name="stateful",
            role="Stateful",
        )

        runtime = manager.create_team(team_card)
        team_id = runtime.id

        # Send 3 messages to increment counter to 3
        for i in range(3):
            actor_system.tell(
                runtime.entry_addr,
                UserMessage(content=f"msg-{i}"),
            )

        wait_for_agent_state(
            runtime.entry_addr,
            lambda state: getattr(state, "counter", 0) >= 3,
            timeout=3.0,
        )

        manager.stop_team(team_id)

        # Resume and verify counter preserved
        new_runtime = manager.resume_team(team_id)

        actor = get_actor_from_addr(new_runtime.entry_addr)
        assert actor.state.counter == 3, (
            f"Expected counter=3 after resume, got {actor.state.counter}"
        )

        # Send 1 more message, counter should become 4
        actor_system.tell(
            new_runtime.entry_addr,
            UserMessage(content="msg-extra"),
        )
        reached = wait_for_agent_state(
            new_runtime.entry_addr,
            lambda state: getattr(state, "counter", 0) >= 4,
            timeout=3.0,
        )
        assert reached, "Counter did not increment to 4 after resume"
