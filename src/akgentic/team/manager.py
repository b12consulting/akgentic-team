"""TeamManager: lifecycle facade for create, resume, stop, delete operations."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.orchestrator import EventSubscriber
from akgentic.team.factory import TeamFactory
from akgentic.team.models import Process, TeamCard, TeamRuntime, TeamStatus
from akgentic.team.ports import EventStore, NullServiceRegistry, ServiceRegistry
from akgentic.team.subscriber import PersistenceSubscriber

logger = logging.getLogger(__name__)


class TeamManager:
    """Single facade for team lifecycle: create, get, resume, stop, delete.

    Coordinates TeamFactory, PersistenceSubscriber, EventStore, and
    ServiceRegistry to provide a unified API for managing team instances
    with consistent state machine enforcement.

    The state machine enforces: RUNNING -> STOPPED -> DELETED.
    Operations that violate this flow raise ValueError.
    """

    def __init__(
        self,
        actor_system: ActorSystem,
        event_store: EventStore,
        service_registry: ServiceRegistry = NullServiceRegistry(),
        subscriber_factory: Callable[[uuid.UUID], list[EventSubscriber]] | None = None,
        instance_id: uuid.UUID | None = None,
    ) -> None:
        """Initialize TeamManager with injected dependencies.

        Args:
            actor_system: The actor system to host team actors.
            event_store: Persistence backend for team state and events.
            service_registry: Service discovery registry. Defaults to
                NullServiceRegistry for single-process mode.
            subscriber_factory: Optional callable that receives a team_id
                and returns additional EventSubscribers to register.
            instance_id: Worker instance identifier. Auto-generated if None.
        """
        self._actor_system = actor_system
        self._event_store = event_store
        self._service_registry = service_registry
        self._subscriber_factory = subscriber_factory
        self._instance_id = instance_id or uuid.uuid4()

    def create_team(
        self,
        team_card: TeamCard,
        user_id: str = "cli",
        user_email: str = "",
    ) -> TeamRuntime:
        """Create and start a new team from a TeamCard.

        Pre-generates a team_id, creates a PersistenceSubscriber (always first),
        appends any subscriber_factory results, then delegates to TeamFactory.build.
        On successful build, persists a Process with RUNNING status and registers
        the team with the ServiceRegistry.

        If build fails, the exception propagates without persisting any Process.

        Args:
            team_card: Declarative team definition.
            user_id: Identifier of the user creating the team.
            user_email: Email of the user creating the team.

        Returns:
            A TeamRuntime handle to the running team.

        Raises:
            ValueError: If the TeamCard is invalid (e.g. entry_point headcount != 1).
            Exception: Any exception from TeamFactory.build propagates unchanged.
        """
        team_id = uuid.uuid4()
        logger.info("Creating team '%s' with id %s", team_card.name, team_id)

        # Build subscriber list: PersistenceSubscriber always first
        persistence_sub = PersistenceSubscriber(team_id, self._event_store)
        subscribers: list[EventSubscriber] = [persistence_sub]
        if self._subscriber_factory is not None:
            subscribers.extend(self._subscriber_factory(team_id))

        # Build the team — if this raises, no Process is persisted
        runtime = TeamFactory.build(
            team_card, self._actor_system, subscribers, team_id=team_id
        )

        # Persist Process metadata
        now = datetime.now(UTC)
        process = Process(
            team_id=team_id,
            team_card=team_card,
            status=TeamStatus.RUNNING,
            user_id=user_id,
            user_email=user_email,
            created_at=now,
            updated_at=now,
        )
        self._event_store.save_team(process)

        # Register with service discovery
        self._service_registry.register_team(self._instance_id, team_id)

        logger.info("Team '%s' (%s) created successfully", team_card.name, team_id)
        return runtime

    def get_team(self, team_id: uuid.UUID) -> Process | None:
        """Retrieve Process metadata for a team.

        Args:
            team_id: The team identifier to look up.

        Returns:
            The Process if found, None otherwise.
        """
        return self._event_store.load_team(team_id)

    def delete_team(self, team_id: uuid.UUID) -> None:
        """Delete a stopped team, purging all persisted data.

        Enforces the state machine: only STOPPED teams can be deleted.

        Args:
            team_id: The team identifier to delete.

        Raises:
            ValueError: If the team is not found, is currently RUNNING,
                or is already DELETED.
        """
        process = self._event_store.load_team(team_id)
        if process is None:
            msg = f"Team {team_id} not found"
            raise ValueError(msg)
        if process.status == TeamStatus.RUNNING:
            msg = (
                f"Cannot delete team {team_id}: "
                f"team is currently running. Stop it first."
            )
            raise ValueError(msg)
        if process.status == TeamStatus.DELETED:
            msg = f"Team {team_id} is already deleted"
            raise ValueError(msg)

        # STOPPED — purge all data
        logger.info("Deleting team %s", team_id)
        self._event_store.delete_team(team_id)

    def resume_team(self, team_id: uuid.UUID) -> TeamRuntime:
        """Resume a stopped team. Implemented in story 3.2.

        Args:
            team_id: The team identifier to resume.

        Raises:
            NotImplementedError: Always — this is a stub for story 3.2.
        """
        raise NotImplementedError("Implemented in story 3.2")

    def stop_team(self, team_id: uuid.UUID) -> None:
        """Gracefully stop a running team. Implemented in story 4.2.

        Args:
            team_id: The team identifier to stop.

        Raises:
            NotImplementedError: Always — this is a stub for story 4.2.
        """
        raise NotImplementedError("Implemented in story 4.2")
