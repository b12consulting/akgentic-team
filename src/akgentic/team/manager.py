"""TeamManager: lifecycle facade for create, resume, stop, delete operations."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.orchestrator import STOP_TIMEOUT, EventSubscriber, Orchestrator
from akgentic.core.utils.serializer import SerializableBaseModel
from akgentic.team.factory import TeamFactory
from akgentic.team.metadata import derive_metadata_indexes
from akgentic.team.models import Process, TeamCard, TeamRuntime, TeamStatus
from akgentic.team.ports import EventStore, NullServiceRegistry, ServiceRegistry
from akgentic.team.restorer import TeamRestorer
from akgentic.team.subscriber import IdleStopSubscriber, PersistenceSubscriber

logger = logging.getLogger(__name__)

# Grace period passed into the non-blocking ``Orchestrator.stop(grace_timeout)``
# at every orchestrator-stop site. It is the single stop timeout (ADR-19 §4):
# core's backstop guarantees the returned event is set within ~this many seconds,
# so callers ``.wait()`` with NO separate caller-side timeout. Defaulted to core's
# ``STOP_TIMEOUT`` so the team grace period tracks the framework default.
GRACE_TIMEOUT_SECONDS = STOP_TIMEOUT


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
        service_registry: ServiceRegistry | None = None,
        subscribers: list[EventSubscriber] | None = None,
        instance_id: uuid.UUID | None = None,
    ) -> None:
        """Initialize TeamManager with injected dependencies.

        Args:
            actor_system: The actor system to host team actors.
            event_store: Persistence backend for team state and events.
            service_registry: Service discovery registry. Defaults to
                NullServiceRegistry for single-process mode.
            subscribers: Pre-instantiated list of long-lived EventSubscribers
                shared across all teams. These must be thread-safe since
                different teams' orchestrators may call on_message()
                concurrently from different actor threads.
                ``PersistenceSubscriber`` and ``IdleStopSubscriber`` are
                per-team and constructed internally by ``TeamManager`` in
                ``create_team`` / ``resume_team``.
            instance_id: Worker instance identifier. Auto-generated if None.
        """
        self._actor_system = actor_system
        self._event_store = event_store
        self._service_registry = service_registry or NullServiceRegistry()
        self._shared_subscribers = subscribers or []
        self._instance_id = instance_id or uuid.uuid4()
        self._runtimes: dict[uuid.UUID, TeamRuntime] = {}

    @staticmethod
    def _validate_metadata(
        team_card: TeamCard,
        metadata: SerializableBaseModel | None,
    ) -> SerializableBaseModel | None:
        """Validate a metadata value against the card's declared ``metadata_type``.

        The single validation site for both write paths, so create and update can
        never diverge on what a team's metadata is allowed to be (ADR-24 §D7).

        ``None`` is always accepted, including on a card that declares a type:
        metadata is optional, and a declared type constrains its shape, not its
        presence.

        Args:
            team_card: The card whose ``metadata_type`` is the contract. On the
                update path this is the card read back off the persisted
                ``Process`` — the type declared at creation, never an argument,
                so ``metadata_type`` cannot change for a live team.
            metadata: The candidate value, or ``None``.

        Returns:
            The validated value as an instance of the declared type, or ``None``.

        Raises:
            ValueError: If a value is supplied but the card declares no
                ``metadata_type``.
            pydantic.ValidationError: If the value does not validate against the
                declared type — propagates unchanged.
        """
        if metadata is None:
            return None
        if team_card.metadata_type is None:
            msg = (
                f"Team card '{team_card.name}' declares no metadata_type; "
                f"metadata cannot be supplied"
            )
            raise ValueError(msg)
        return team_card.metadata_type.model_validate(metadata)

    def _push_metadata(
        self,
        team_id: uuid.UUID,
        runtime: TeamRuntime,
        metadata: SerializableBaseModel | None,
    ) -> None:
        """Push a validated metadata value to the live orchestrator, best-effort.

        Called only AFTER the value and its re-derived index are persisted. A
        failure here is logged and swallowed: the database — which is what team
        listing filters on — stays truthful, and the orchestrator repopulates
        from the ``Process`` on the next resume. Raising instead would let a
        transient actor problem fail an operation that already succeeded.

        Args:
            team_id: The team whose metadata is being pushed.
            runtime: The tracked runtime holding the orchestrator address.
            metadata: The validated value, or ``None`` to clear it.
        """
        try:
            orchestrator_proxy: Orchestrator = self._actor_system.proxy_tell(
                runtime.orchestrator_addr, Orchestrator
            )
            orchestrator_proxy.set_metadata(metadata)
        except Exception:
            logger.warning(
                "Failed to push metadata to orchestrator for team %s",
                team_id,
                exc_info=True,
            )

    def create_team(
        self,
        team_card: TeamCard,
        user_id: str = "cli",
        user_email: str = "",
        team_id: uuid.UUID | None = None,
        catalog_namespace: str | None = None,
        *,
        metadata: SerializableBaseModel | None = None,
    ) -> TeamRuntime:
        """Create and start a new team from a TeamCard.

        Pre-generates a team_id, creates a PersistenceSubscriber (always first)
        and an IdleStopSubscriber behind it, appends shared subscribers, then
        delegates to TeamFactory.build.
        On successful build, persists a Process with RUNNING status and registers
        the team with the ServiceRegistry.

        If build fails, the exception propagates without persisting any Process.

        Args:
            team_card: Declarative team definition.
            user_id: Identifier of the user creating the team.
            user_email: Email of the user creating the team.
            team_id: Optional team identifier. Auto-generated if None.
            catalog_namespace: Optional opaque tag identifying the catalog
                namespace this team was instantiated from. Stored verbatim on
                the persisted ``Process``; ``akgentic-team`` does not interpret
                it. Consumers read it back via ``get_team(team_id)``.
            metadata: Optional business metadata, an instance of the card's
                declared ``metadata_type``. Validated FIRST, before any actor is
                started or any ``Process`` written, so a rejected value never
                leaves a half-created team behind. Persisted alongside its
                derived index, then pushed to the orchestrator best-effort.

        Returns:
            A TeamRuntime handle to the running team.

        Raises:
            ValueError: If the TeamCard is invalid (e.g. entry_point headcount != 1),
                or if ``metadata`` is supplied for a card declaring no ``metadata_type``.
            pydantic.ValidationError: If ``metadata`` does not validate against the
                card's declared ``metadata_type``.
            Exception: Any exception from TeamFactory.build propagates unchanged.
        """
        # Validate before anything is built or written — no half-created team.
        validated_metadata = self._validate_metadata(team_card, metadata)

        if team_id is None:
            team_id = uuid.uuid4()
        logger.info("Creating team '%s' with id %s", team_card.name, team_id)

        # Build subscriber list: per-team subscribers first, shared subscribers behind
        persistence_sub = PersistenceSubscriber(team_id, self._event_store)
        idle_stop_sub = IdleStopSubscriber(self, team_id)
        subscribers: list[EventSubscriber] = [
            persistence_sub,
            idle_stop_sub,
            *self._shared_subscribers,
        ]

        # Build the team — if this raises, no Process is persisted
        try:
            runtime = TeamFactory.build(team_card, self._actor_system, subscribers, team_id=team_id)
        except Exception:
            # The idle-stop countdown is armed in the subscriber's constructor,
            # above, but a team that never built has no orchestrator to dispatch
            # on_stop and cancel it. Without this the Timer thread outlives the
            # failed call for the whole delay and then fires stop_team on a team
            # that does not exist.
            idle_stop_sub.on_stop(team_id)
            raise

        # Track runtime for stop_team
        self._runtimes[team_id] = runtime

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
            catalog_namespace=catalog_namespace,
            metadata=validated_metadata,
            metadata_indexes=derive_metadata_indexes(validated_metadata),
        )
        self._event_store.save_team(process)

        # Database first, actor second — see _push_metadata
        if validated_metadata is not None:
            self._push_metadata(team_id, runtime, validated_metadata)

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
            logger.warning("Delete rejected: team %s not found", team_id)
            msg = f"Team {team_id} not found"
            raise ValueError(msg)
        if process.status == TeamStatus.RUNNING:
            logger.warning("Delete rejected: team %s is currently running", team_id)
            msg = f"Cannot delete team {team_id}: team is currently running. Stop it first."
            raise ValueError(msg)
        if process.status == TeamStatus.DELETED:
            logger.warning("Delete rejected: team %s is already deleted", team_id)
            msg = f"Team {team_id} is already deleted"
            raise ValueError(msg)

        # STOPPED — purge all data and deregister from service discovery
        logger.info("Deleting team %s", team_id)
        self._event_store.delete_team(team_id)
        self._service_registry.deregister_team(self._instance_id, team_id)

        # Cleanup runtime tracking
        self._runtimes.pop(team_id, None)

    def resume_team(self, team_id: uuid.UUID) -> TeamRuntime:
        """Resume a stopped team by restoring from persisted EventStore data.

        Loads the Process, validates state machine (only STOPPED teams may
        resume), delegates to TeamRestorer, then updates Process status to
        RUNNING and registers with ServiceRegistry.

        Args:
            team_id: The team identifier to resume.

        Returns:
            A TeamRuntime handle to the resumed team.

        Raises:
            ValueError: If the team is not found, is currently RUNNING,
                or is already DELETED.
        """
        process = self._event_store.load_team(team_id)
        if process is None:
            logger.warning("Resume rejected: team %s not found", team_id)
            msg = f"Team {team_id} not found"
            raise ValueError(msg)
        if process.status == TeamStatus.RUNNING:
            logger.warning("Resume rejected: team %s is currently running", team_id)
            msg = f"Cannot resume team {team_id}: team is currently running"
            raise ValueError(msg)
        if process.status == TeamStatus.DELETED:
            logger.warning("Resume rejected: team %s has been deleted", team_id)
            msg = f"Cannot resume team {team_id}: team has been deleted"
            raise ValueError(msg)

        restorer = TeamRestorer(self._actor_system, self._event_store)

        # Compute max existing sequence so new events continue monotonically
        max_seq = self._event_store.get_max_sequence(team_id)

        # Create per-team subscribers once — passed to restorer and tracked for stop
        persistence_sub = PersistenceSubscriber(
            team_id, self._event_store, initial_sequence=max_seq
        )
        idle_stop_sub = IdleStopSubscriber(self, team_id)
        all_subs: list[EventSubscriber] = [
            persistence_sub,
            idle_stop_sub,
            *self._shared_subscribers,
        ]

        # Toggle restoring guard on all subscribers.
        # Each subscriber decides independently whether to skip during restore.
        for sub in all_subs:
            sub.set_restoring(team_id, True)

        try:
            runtime = restorer.restore(process, subscribers=all_subs)
        except Exception:
            # Same reason as create_team: nothing dispatches on_stop to a
            # subscriber whose team never came back up, so cancel its countdown
            # here rather than leave it armed against a team that is not running.
            idle_stop_sub.on_stop(team_id)
            raise
        finally:
            for sub in all_subs:
                sub.set_restoring(team_id, False)

        # Track runtime for stop_team
        self._runtimes[team_id] = runtime

        # Copy-with-override rather than a hand-listed rebuild: a lifecycle write
        # changes the status and the timestamp and nothing else, so every other
        # field — including the next one added to Process — is carried forward by
        # construction instead of by remembering to list it. metadata_indexes
        # travels verbatim and is NOT re-derived: this path does not change the
        # value, and a second derivation site is how the index starts lying.
        updated_process = process.model_copy(
            update={"status": TeamStatus.RUNNING, "updated_at": datetime.now(UTC)}
        )
        self._event_store.save_team(updated_process)

        self._service_registry.register_team(self._instance_id, team_id)

        logger.info("Team '%s' (%s) resumed successfully", process.team_card.name, team_id)
        return runtime

    def _teardown_team(self, team_id: uuid.UUID, runtime: TeamRuntime) -> None:
        """Trigger graceful orchestrator stop via proxy and wait for completion.

        ``Orchestrator.stop(grace_timeout)`` is non-blocking (akgentic-core
        ADR-012): the ``proxy_ask`` resolves immediately and returns a
        ``threading.Event`` that core sets once teardown is complete (mailbox
        drained, subscribers' ``on_stop`` fired, actor deregistered). We
        ``.wait()`` on that event — on this caller/daemon thread, NOT the
        orchestrator's actor thread — so ``stop_team`` only persists ``STOPPED``
        and deregisters AFTER the actors are actually down.

        The wait takes no timeout: core's backstop guarantees the event is set
        within ~``GRACE_TIMEOUT_SECONDS``, so it cannot hang and always returns
        ``True``. There is therefore no ``if not stopped:`` branch — the
        "had to force it" WARNING lives in core's ``_force_stop``.

        The orchestrator's own ``on_stop`` fans out ``on_stop(team_id)`` to
        every attached subscriber *before* tearing actors down and clearing
        the subscriber list — see ``akgentic-core``
        ``Orchestrator.on_stop``. ``TeamManager`` therefore no longer
        unsubscribes subscribers itself; doing so would strip the list
        before the orchestrator could deliver the lifecycle signal.

        Best-effort: any failure is logged at WARNING but does not raise.

        Args:
            team_id: The team identifier being stopped.
            runtime: The active TeamRuntime containing actor addresses.
        """
        try:
            orchestrator_proxy: Orchestrator = self._actor_system.proxy_ask(
                runtime.orchestrator_addr, Orchestrator
            )
            orchestrator_proxy.stop(GRACE_TIMEOUT_SECONDS).wait()
        except Exception:
            logger.warning(
                "Failed to teardown orchestrator for team %s",
                team_id,
                exc_info=True,
            )

    def stop_team(self, team_id: uuid.UUID) -> None:
        """Gracefully stop a running team.

        Stops the orchestrator via proxy (which fans out ``on_stop`` to
        subscribers and recursively tears down all child actors), persists
        Process with STOPPED status, and deregisters from ServiceRegistry.

        Idempotent: calling stop on an already-STOPPED team is a no-op.

        Args:
            team_id: The team identifier to stop.

        Raises:
            ValueError: If the team is not found or is already DELETED.
        """
        process = self._event_store.load_team(team_id)
        if process is None:
            logger.warning("Stop rejected: team %s not found", team_id)
            msg = f"Team {team_id} not found"
            raise ValueError(msg)

        if process.status == TeamStatus.STOPPED:
            logger.info("Team %s is already stopped — no-op", team_id)
            return

        if process.status == TeamStatus.DELETED:
            logger.warning("Stop rejected: team %s no longer exists", team_id)
            msg = f"Team {team_id} no longer exists"
            raise ValueError(msg)

        # RUNNING — perform graceful shutdown
        runtime = self._runtimes.get(team_id)

        if runtime is not None:
            self._teardown_team(team_id, runtime)
        else:
            logger.warning(
                "Team %s is RUNNING but no runtime tracked — "
                "actors may already be dead. Updating state only.",
                team_id,
            )

        # Persist STOPPED status — copy-with-override for the same reason as
        # resume_team: status and timestamp change, everything else rides along.
        updated_process = process.model_copy(
            update={"status": TeamStatus.STOPPED, "updated_at": datetime.now(UTC)}
        )
        self._event_store.save_team(updated_process)

        # Deregister from service discovery
        self._service_registry.deregister_team(self._instance_id, team_id)

        # Cleanup runtime tracking
        self._runtimes.pop(team_id, None)

        logger.info("Team %s stopped successfully", team_id)

    def update_team_metadata(
        self,
        team_id: uuid.UUID,
        metadata: SerializableBaseModel | None,
    ) -> Process:
        """Replace a team's business metadata (ADR-24 §D7).

        Ordered validate -> single database write -> best-effort orchestrator
        push. The database is written first on purpose: it is what team listing
        filters on, so a failed push leaves the index truthful and self-heals on
        the next resume, whereas an actor-first write would leave listing wrong
        with no signal.

        **Replace, never merge.** ``metadata`` is a complete document that must
        validate on its own; a field set before and absent now is gone from both
        the stored value and its index. ``None`` clears the metadata entirely.

        The value is validated against the ``metadata_type`` the ``TeamCard``
        declared at creation — read off the persisted ``Process``, so the type
        cannot be changed for a live team.

        Args:
            team_id: The team whose metadata is being replaced.
            metadata: The complete new value, or ``None`` to clear it.

        Returns:
            The persisted ``Process``, carrying the new value and its index.

        Raises:
            ValueError: If the team is not found, has been deleted, or the card
                declares no ``metadata_type`` while a value is supplied.
            pydantic.ValidationError: If the value does not validate against the
                declared ``metadata_type``. Nothing is written in either case.
        """
        process = self._event_store.load_team(team_id)
        if process is None:
            logger.warning("Metadata update rejected: team %s not found", team_id)
            msg = f"Team {team_id} not found"
            raise ValueError(msg)
        if process.status == TeamStatus.DELETED:
            logger.warning("Metadata update rejected: team %s has been deleted", team_id)
            msg = f"Cannot update metadata for team {team_id}: team has been deleted"
            raise ValueError(msg)

        validated_metadata = self._validate_metadata(process.team_card, metadata)

        # Copy-with-override, as resume_team and stop_team already do: only the
        # three fields this path owns are named, so every other field — including
        # the next one added to Process — is carried forward by construction.
        updated_process = process.model_copy(
            update={
                "updated_at": datetime.now(UTC),
                "metadata": validated_metadata,
                "metadata_indexes": derive_metadata_indexes(validated_metadata),
            }
        )
        self._event_store.save_team(updated_process)

        # Database first, actor second — and only while the team is live
        runtime = self._runtimes.get(team_id)
        if process.status == TeamStatus.RUNNING and runtime is not None:
            self._push_metadata(team_id, runtime, validated_metadata)
        elif process.status == TeamStatus.RUNNING:
            # RUNNING but untracked here: another worker owns it, or this worker
            # restarted. Not an error — the write stands and the orchestrator
            # repopulates from the Process on the next resume. Logged so a stale
            # live value is explainable rather than mysterious.
            logger.debug(
                "Team %s is RUNNING but has no runtime tracked here — "
                "metadata written, orchestrator push skipped",
                team_id,
            )

        logger.info("Metadata updated for team %s", team_id)
        return updated_process
