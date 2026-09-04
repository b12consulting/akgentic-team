"""Domain models for team lifecycle management.

TeamCard, TeamCardMember, TeamRuntime, TeamStatus, AgentRef, AgentCardRef,
Process, PersistedEvent, AgentStateSnapshot.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, cast

from pydantic import Field, PrivateAttr, model_validator

from akgentic.core.actor_address import ActorAddress
from akgentic.core.actor_address_impl import ActorAddressProxy
from akgentic.core.actor_system_impl import ActorSystem
from akgentic.core.agent import Akgent
from akgentic.core.agent_card import AgentCard
from akgentic.core.agent_state import BaseState
from akgentic.core.messages.message import Message
from akgentic.core.orchestrator import Orchestrator
from akgentic.core.user_proxy import UserProxy
from akgentic.core.utils import hydrate_addresses
from akgentic.core.utils.serializer import SerializableBaseModel


class TeamCardMember(SerializableBaseModel):
    """A member slot in a team card, wrapping an AgentCard with multiplicity.

    TeamCardMember is self-referential: each member can contain subordinate
    members, forming a tree that describes hierarchical team structures.

    Attributes:
        card: The agent card describing this member's role and capabilities.
        headcount: Number of agent instances to create for this slot.
        members: Subordinate members managed by this member.
    """

    card: AgentCard = Field(description="Agent card describing this member's role and capabilities")
    headcount: int = Field(default=1, description="Number of agent instances for this slot")
    members: list[TeamCardMember] = Field(
        default_factory=list,
        description="Subordinate members managed by this member",
    )


class TeamCard(SerializableBaseModel):
    """Declarative definition of a team's structure, entry point, and routing.

    TeamCard describes the full hierarchy of agents in a team. The entry_point
    is the agent that receives external messages. Members form a tree that can
    be walked to discover all agent cards and supervisory relationships.

    Attributes:
        name: Unique name identifying this team definition.
        description: Human-readable summary of what the team does.
        entry_point: The member that serves as the team's external interface.
        members: Top-level members of the team (excluding the entry point).
        message_types: Message classes the team handles; first is the default.
        metadata_type: Model class describing this team's business metadata.
            ``None`` when the team carries none. Declared as a field rather than
            a type parameter on purpose -- a parameterised ``TeamCard[X]`` has no
            importable dotted path, which would break the ``__model__``
            round-trip ``Process.team_card`` exists to perform.
        welcome_message: Optional static greeting announced on the team's event
            stream when the team is first created. ``None`` disables it.
    """

    name: str | None = Field(
        default=None, description="Unique name identifying this team definition"
    )
    description: str | None = Field(
        default=None, description="Human-readable summary of what the team does"
    )
    entry_point: TeamCardMember = Field(
        description="The member that serves as the team's external interface",
    )
    members: list[TeamCardMember] = Field(
        default_factory=list,
        description="Top-level members of the team excluding the entry point",
    )
    message_types: list[type] = Field(
        default_factory=list,
        description="Message classes the team handles; first is the default",
    )
    metadata_type: type[SerializableBaseModel] | None = Field(
        default=None,
        description=(
            "Model class describing this team's business metadata, typically a "
            "TeamMetadata subclass. None when the team carries none."
        ),
    )
    agent_profiles: list[AgentCard] = Field(
        default_factory=list,
        description="AgentCards available for runtime hiring, not instantiated at startup",
    )
    welcome_message: str | None = Field(
        default=None,
        description=(
            "Optional static greeting announced on the team's event stream when "
            "the team is first created. None disables it."
        ),
    )

    @property
    def agent_cards(self) -> dict[str, AgentCard]:
        """Return a flat index of all AgentCards in the member tree.

        Walks the entry_point and all members recursively, collecting every
        AgentCard keyed by its ``config.name``.

        Returns:
            Dictionary mapping config name to AgentCard for every member.
        """
        result: dict[str, AgentCard] = {}
        self._collect_cards(self.entry_point, result)
        for member in self.members:
            self._collect_cards(member, result)
        return result

    @property
    def supervisors(self) -> list[AgentCard]:
        """Return AgentCards for the first layer of ``members`` only.

        Supervisors are the agents that receive external messages routed
        through the entry point.  The entry point itself is excluded --
        it is the *sender*, not a recipient.  Deeper members (children
        of first-layer members) are also excluded -- they are internal
        to their supervisor's subtree.

        Returns:
            List of AgentCards for each top-level member.
        """
        return [m.card for m in self.members]

    @staticmethod
    def _collect_cards(
        member: TeamCardMember,
        result: dict[str, AgentCard],
    ) -> None:
        """Recursively collect AgentCards from a member subtree.

        Args:
            member: The member node to start from.
            result: Accumulator dict to populate with discovered cards.

        Raises:
            ValueError: If a duplicate config name is detected in the tree.
        """
        name = member.card.config.name
        if name in result:
            msg = (
                f"Duplicate config name '{name}' in team member tree. "
                f"Each AgentCard must have a unique config.name."
            )
            raise ValueError(msg)
        result[name] = member.card
        for child in member.members:
            TeamCard._collect_cards(child, result)


def spawned_names(member: TeamCardMember) -> list[str]:
    """Return the names ``TeamFactory`` gives the actors it spawns for *member*.

    The single statement of the naming rule, mirroring
    ``TeamFactory._spawn_member``: a member with ``headcount == 1`` keeps the
    bare ``card.config.name``, and any higher headcount is expanded into
    ``"<name>_<i>"`` for each index. The bare name is therefore **never** a live
    agent name for a multi-instance member — which is precisely the trap
    ``supervisor_addrs`` falls into by matching on it (see ADR-26).

    Callers that record who was spawned must go through this function rather
    than reading ``card.config.name``, so the recorded name and the spawned
    actor can never disagree.

    Args:
        member: The member slot to expand.

    Returns:
        One name per instance, in spawn (index) order.
    """
    name = member.card.config.name
    if member.headcount == 1:
        return [name]
    return [f"{name}_{i}" for i in range(member.headcount)]


# --- TeamRuntime ---


class TeamRuntime(SerializableBaseModel):
    """Live handle to a running team that survives serialization for persistence.

    Stores persistent actor addresses and rebuilds ephemeral proxies on
    construction via ``model_post_init``. Persistent fields survive
    ``model_dump()`` / ``model_validate()`` round-trips while ephemeral
    proxies are excluded from serialization.

    Carries no ``TeamCard``. Everything it used to read off one is answered
    elsewhere, and by three different things: ``team_name`` and
    ``message_types`` are plain values it holds itself; the entry point's agent
    class comes from the orchestrator's role catalog, keyed by the entry
    address's own role; and whether a target is a ``UserProxy`` comes from that
    target's ``ActorAddress`` — the actor's actual type, never a card's
    declaration, so the catalog is not on that path at all. That is what lets
    the restore path build a runtime from the stored projection alone, once
    story 31-3 stops reading ``Process.team_card`` (ADR-26 §Decision 6).

    Attributes:
        id: Externally assigned unique identifier for this runtime instance.
        team_name: The team's name, used in error messages. ``None`` for an
            unnamed team.
        message_types: Message classes the team handles; the first is the type
            a ``str`` sent through ``send`` / ``send_to`` is wrapped in.
        actor_system: The actor system hosting this team's actors.
        orchestrator_addr: Persistent address of the orchestrator actor.
        entry_addr: Persistent address of the team's entry-point actor.
        supervisor_addrs: Persistent addresses of supervisor actors keyed by name.
        addrs: Persistent addresses of all actors keyed by name.
    """

    id: uuid.UUID = Field(description="Externally assigned unique identifier for this runtime")
    team_name: str | None = Field(
        default=None, description="The team's name, used in error messages"
    )
    message_types: list[type] = Field(
        default_factory=list,
        description="Message classes the team handles; first is the default",
    )
    actor_system: ActorSystem = Field(
        exclude=True,
        description="Actor system hosting this team's actors",
    )
    orchestrator_addr: ActorAddress = Field(
        description="Persistent address of the orchestrator actor",
    )
    entry_addr: ActorAddress = Field(
        description="Persistent address of the entry-point actor",
    )
    supervisor_addrs: dict[str, ActorAddress] = Field(
        default_factory=dict,
        description="Persistent addresses of supervisor actors keyed by name",
    )
    addrs: dict[str, ActorAddress] = Field(
        default_factory=dict,
        description="Persistent addresses of all actors keyed by name",
    )

    _orchestrator_proxy: Orchestrator = PrivateAttr()
    _orchestrator_proxy_tell: Orchestrator = PrivateAttr()
    _entry_proxy: Akgent[Any, Any] = PrivateAttr()
    _message_cls: type[Message] | None = PrivateAttr(default=None)
    _addr_map: dict[uuid.UUID, ActorAddress] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Rebuild all ephemeral proxies from persistent addresses.

        Always overwrites all ephemeral fields completely to ensure
        idempotency — safe to call multiple times.

        The entry point's agent class comes from the orchestrator's role
        catalog, keyed by the entry address's own ``role``. The catalog is
        therefore a construction-time dependency: both call sites register it
        before building a runtime (``TeamFactory.build`` step 4, and the
        restorer's rebuild phase 2f).

        Args:
            __context: Pydantic validation context (unused).

        Raises:
            ValueError: If the catalog holds no card for the entry point's role.
        """
        self._orchestrator_proxy = self.actor_system.proxy_ask(self.orchestrator_addr, Orchestrator)
        self._orchestrator_proxy_tell = self.actor_system.proxy_tell(
            self.orchestrator_addr, Orchestrator
        )
        entry_role = self.entry_addr.role
        entry_card = self._orchestrator_proxy.get_agent_profile(entry_role)
        if entry_card is None:
            msg = (
                f"No agent card registered for entry point role '{entry_role}' "
                f"in team '{self.team_name}'"
            )
            raise ValueError(msg)
        self._entry_proxy = self.actor_system.proxy_tell(
            self.entry_addr, entry_card.get_agent_class()
        )

        self._message_cls = self.message_types[0] if self.message_types else None

    def _make_message(self, content: str) -> Message:
        """Create a message from the team's declared message type.

        Args:
            content: The message content.

        Returns:
            A Message instance of the team's declared type.

        Raises:
            RuntimeError: If no message type is declared for this team.
        """
        if self._message_cls is None:
            msg = "No message type declared for this team"
            raise RuntimeError(msg)
        return self._message_cls(content=content)  # type: ignore[call-arg]

    def _coerce_message(self, content: str | Message) -> Message:
        """Coerce a routing payload to a Message.

        A ``str`` is wrapped in the team's default type via ``_make_message``;
        a ``Message`` is passed through untouched (the caller picked the type).
        """
        return content if isinstance(content, Message) else self._make_message(content)

    def emitMessage(self, message: Message) -> None:  # noqa: N802
        """Publish a pre-formed message into the team's event record.

        Fire-and-forget tell to the orchestrator, which stamps ``team_id`` and
        fans out to subscribers (persist + live stream) with no agent processing
        and no outbound dispatch. Rationale: ADR-22.
        """
        self._orchestrator_proxy_tell.emitMessage(message)

    def send(self, content: str | Message) -> None:
        """Send a message into the team through the entry-point agent.

        Routes through the entry proxy so that ``sender`` is set to the
        entry agent.  The entry point itself never receives a copy -- it is
        the sender, not a recipient.

        Args:
            content: The message content, or a pre-formed ``Message`` to route
                untouched (a passed ``Message`` is the SAME instance for every
                supervisor -- passthrough per ADR-22, not cloned per recipient).
        """
        for addr in self.supervisor_addrs.values():
            self._entry_proxy.send(addr, self._coerce_message(content))

    def _resolve_addr(self, agent_name: str, addr: ActorAddress) -> ActorAddress:
        """Resolve a potentially stale proxy address to a live address.

        If the address is an ``ActorAddressProxy`` (from deserialized data),
        looks up the live address in ``_addr_map``.  Returns the address
        unchanged if it is already live.

        Args:
            agent_name: Name of the agent (used in error messages).
            addr: The address to resolve.

        Returns:
            A live ``ActorAddress``.

        Raises:
            ValueError: If the address is a stale proxy with no live mapping.
        """
        if isinstance(addr, ActorAddressProxy):
            live = self._addr_map.get(addr.agent_id)
            if live is None:
                msg = (
                    f"Agent '{agent_name}' has stale proxy address"
                    " — no live mapping available"
                )
                raise ValueError(msg)
            return live
        return addr

    def _lookup_member(self, agent_name: str) -> ActorAddress:
        """Look up a team member by name and resolve stale proxies.

        Args:
            agent_name: Name of the agent to look up.

        Returns:
            A live ``ActorAddress`` for the agent.

        Raises:
            ValueError: If the agent is not found or has a stale proxy.
        """
        addr = self._orchestrator_proxy.get_team_member(agent_name)
        if addr is None:
            msg = f"Agent '{agent_name}' not found in team '{self.team_name}'"
            raise ValueError(msg)
        return self._resolve_addr(agent_name, addr)

    def send_to(self, agent_name: str, content: str | Message) -> None:
        """Send a directed message to a specific agent by name.

        Looks up the agent via the orchestrator proxy and sends a message
        via the entry proxy. Includes a safety net to resolve stale
        ``ActorAddressProxy`` refs that may leak through after restore.

        Args:
            agent_name: Name of the target agent.
            content: The message content, or a pre-formed ``Message`` to route
                untouched.

        Raises:
            ValueError: If the agent is not found or has a stale proxy address.
        """
        actor_addr = self._lookup_member(agent_name)
        message = self._coerce_message(content)
        self._entry_proxy.send(actor_addr, message)

    def send_from_to(
        self,
        sender_name: str,
        recipient_name: str,
        content: str | Message,
    ) -> None:
        """Send a message from a specified agent to another agent.

        Unlike ``send_to()`` which always sends via the entry proxy (so
        the sender is the entry agent), this method obtains a proxy for
        the *sender* agent and calls ``send()`` on it, so the message's
        ``sender`` field is set to the specified sender's address.

        Args:
            sender_name: Name of the agent to send from.
            recipient_name: Name of the agent to send to.
            content: The message content.

        Raises:
            ValueError: If sender or recipient is not found or has a stale proxy.
        """
        sender_addr = self._lookup_member(sender_name)
        recipient_addr = self._lookup_member(recipient_name)
        sender_proxy = self.actor_system.proxy_tell(sender_addr, Akgent)
        sender_proxy.send(recipient_addr, self._coerce_message(content))

    def process_human_input(self, content: str, message: Message) -> None:
        """Route human input to the correct UserProxy by recipient name.

        Validates that the message has a recipient, rehydrates any
        ``ActorAddressProxy`` references via the orchestrator, then
        routes by ``message.recipient.name`` to the correct ``UserProxy``.

        The target is qualified by ``ActorAddress.is_user_proxy`` — the actor's
        **actual** type, cached at address construction — rather than by the
        agent class a card declares. Two consequences worth stating: an agent
        hired at runtime routes like any other (it is in the live roster and in
        no card list on the ``Process``), and a "not found" recipient stays
        distinguishable from a found one that is not a ``UserProxy``. They are
        different bugs to whoever reads the log.

        Args:
            content: The human's text response.
            message: The original message from the requesting agent.

        Raises:
            ValueError: If the message has no recipient, the recipient cannot
                be rehydrated, the recipient name is absent from the team's
                live roster, or the target actor is not a ``UserProxy``. The
                last two carry distinct messages on purpose: before the check
                moved onto the address, an unknown name had no card and so
                reported the type error, hiding which of the two had happened.
        """
        if message.recipient is None:
            msg = "Cannot route human input: message has no recipient"
            raise ValueError(msg)

        def _resolve(proxy: ActorAddressProxy) -> ActorAddress:
            live = self._orchestrator_proxy.get_team_member(proxy.name)
            if live is None:
                err = (
                    f"Cannot rehydrate address for agent '{proxy.name}': "
                    "not found in team"
                )
                raise ValueError(err)
            return live

        live_message = cast(Message, hydrate_addresses(message, _resolve))

        target_name = live_message.recipient.name  # type: ignore[union-attr]
        target_addr = self._lookup_member(target_name)

        if not target_addr.is_user_proxy:
            msg = f"Agent '{target_name}' is not a UserProxy"
            raise ValueError(msg)

        proxy = self.actor_system.proxy_ask(target_addr, UserProxy)
        proxy.process_human_input(content, live_message)

    @property
    def orchestrator_proxy(self) -> Orchestrator:
        """Read-only access to the orchestrator proxy (ask)."""
        return self._orchestrator_proxy

    @property
    def entry_proxy(self) -> Akgent[Any, Any]:
        """Read-only access to the entry-point proxy (tell)."""
        return self._entry_proxy


# --- Persistence Models ---


class TeamStatus(StrEnum):
    """Lifecycle states for a team instance."""

    RUNNING = "running"
    STOPPED = "stopped"
    DELETED = "deleted"


class AgentRef(SerializableBaseModel):
    """One spawned agent identity, and the role it was spawned from.

    Attributes:
        name: The agent's **spawned** name — the key into ``TeamRuntime.addrs``
            and ``TeamRuntime.supervisor_addrs``. Already expanded for
            ``headcount`` by :func:`spawned_names`, so a member declared with
            ``headcount=3`` contributes three refs named ``"@Name_0"`` …
            ``"@Name_2"`` and the bare ``"@Name"`` appears nowhere. There is
            deliberately no expansion step left for a reader to forget.
        role: The agent's role — the key into ``Process.agent_cards``, where the
            card it was built from is referenced.
    """

    name: str = Field(description="Spawned agent name; the key into TeamRuntime.addrs")
    role: str = Field(description="Agent role; the key into Process.agent_cards")


class AgentCardRef(SerializableBaseModel):
    """A reference to the ``AgentCard`` behind one role.

    Attributes:
        role: The role this card defines — the key ``AgentRef.role`` resolves
            against, and the key of the role catalog.
        card_hash: Content hash of the card, produced by
            ``akgentic.team.projection.hash_agent_card``. It is the key into the
            content-addressed card store that story 31-7 adds; until then it
            resolves against nothing, which is an accepted intermediate state.
        can_be_hired: Whether an agent may hire this role at runtime. Excluded
            from ``card_hash`` on purpose: hireability is a property of the team
            that names the role, not of the card's content, so the same card
            reached through ``agent_profiles`` and through the member tree
            addresses one stored blob.
    """

    role: str = Field(description="Role this card defines; the catalog key")
    card_hash: str = Field(description="Content hash of the AgentCard this ref points at")
    can_be_hired: bool = Field(
        default=False,
        description="Whether an agent may hire this role at runtime",
    )


class Process(SerializableBaseModel):
    """Persisted team metadata for crash recovery.

    Carries a flat **structural projection** of the team — what was actually
    spawned — rather than only the declarative card it was asked for. The
    projection is derived by exactly one function,
    ``akgentic.team.projection.derive_team_projection``, so the stored record can
    never disagree with what a fresh team would produce (ADR-26). This is NOT the
    TeamRuntime -- addresses are stale after stop/crash.

    ``team_card`` is still stored alongside the projection while the restore path
    reads it; story 31-3 removes the field once it is the last reader.

    Attributes:
        team_id: Unique identifier for this team instance.
        team_card: Declarative team definition for rebuilding on resume.
        status: Current lifecycle state of the team.
        user_id: Identifier of the user who owns this team.
        user_email: Email of the user who owns this team.
        created_at: Timestamp when the team was created.
        updated_at: Timestamp of the last status change.
        catalog_namespace: Opaque catalog-namespace tag, or ``None``.
        metadata: The team's business metadata value, an instance of the card's
            declared ``metadata_type``, or ``None``.
        metadata_indexes: Flattened ``"key|value"`` entries derived from
            ``metadata``. Derived on write by ``derive_metadata_indexes``, never
            accepted from a caller, and never written apart from ``metadata``.
        team_name: The team's name, projected off the card at creation.
        team_description: A mutable human-readable description of *this team
            instance*. ``None`` at creation and deliberately **not** seeded from
            the card's description: the card describes a blueprint, this
            describes one running team, and conflating the two makes a
            user-edited description snap back to the blueprint's text.
        entry_point: Ref to the agent that receives external messages.
        supervisors: Refs to the first-layer members, in declaration order,
            already expanded for ``headcount`` — two instances of one role are
            two refs here, because both identities are addressable.
        agent_cards: One ref per **role** reachable from the team: the entry
            point, the whole member tree, and ``agent_profiles``. Keyed by role,
            unlike ``TeamCard.agent_cards`` which is keyed by ``config.name``.
        message_types: Message classes the team handles; first is the default.
        metadata_type: Model class describing this team's business metadata,
            ``None`` when the team carries none. The contract ``metadata`` is
            validated against, read here rather than off the nested card.
    """

    team_id: uuid.UUID = Field(description="Unique identifier for this team instance")
    team_card: TeamCard = Field(description="Declarative team definition for rebuilding on resume")
    status: TeamStatus = Field(description="Current lifecycle state of the team")
    user_id: str = Field(default="cli", description="Identifier of the user who owns this team")
    user_email: str = Field(default="", description="Email of the user who owns this team")
    created_at: datetime = Field(description="Timestamp when the team was created")
    updated_at: datetime = Field(description="Timestamp of the last status change")
    catalog_namespace: str | None = Field(
        default=None,
        description=(
            "Optional opaque catalog-namespace tag recorded when this team was "
            "instantiated from a catalog. Not interpreted by akgentic-team."
        ),
    )
    metadata: SerializableBaseModel | None = Field(
        default=None,
        description=(
            "The team's business metadata value, an instance of the card's "
            "declared metadata_type. None when the team carries none."
        ),
    )
    metadata_indexes: list[str] = Field(
        default_factory=list,
        description=(
            "Flattened 'key|value' index derived from metadata on every write. "
            "Never client-supplied and never derived on read -- a read-time "
            "derivation would mask a write path that forgot to update it."
        ),
    )
    team_name: str | None = Field(
        default=None, description="The team's name, projected off the card at creation"
    )
    team_description: str | None = Field(
        default=None,
        description=(
            "Mutable description of this team instance. None at creation and "
            "never seeded from the card's description."
        ),
    )
    entry_point: AgentRef = Field(
        description="Ref to the agent that receives external messages",
    )
    supervisors: list[AgentRef] = Field(
        default_factory=list,
        description="Refs to the first-layer members, headcount already expanded",
    )
    agent_cards: list[AgentCardRef] = Field(
        default_factory=list,
        description="One ref per role reachable from the team, including agent_profiles",
    )
    message_types: list[type] = Field(
        default_factory=list,
        description="Message classes the team handles; first is the default",
    )
    metadata_type: type[SerializableBaseModel] | None = Field(
        default=None,
        description="Model class describing this team's business metadata, or None",
    )

    @model_validator(mode="after")
    def reject_duplicate_card_roles(self) -> Process:
        """Reject a projection holding two ``AgentCardRef``s for one role.

        ``mode="after"`` for the same reason as
        :meth:`require_resolvable_agent_refs`: ``model_validate`` of a stored
        document is the path a hand-written or half-migrated record takes, and
        it is the only path that can produce a duplicate —
        ``derive_team_projection`` dedups by role, so this constrains nothing it
        writes.

        Tolerating a duplicate would not be neutral.
        ``Orchestrator.register_agent_profile`` keys the catalog by
        ``card.role``, so two refs sharing a role resolve to one catalog entry
        with no signal about which card won.

        Raises:
            ValueError: If any role appears twice in ``agent_cards``. Every
                duplicated role is named.
        """
        seen: set[str] = set()
        duplicates: set[str] = set()
        for ref in self.agent_cards:
            if ref.role in seen:
                duplicates.add(ref.role)
            seen.add(ref.role)
        if duplicates:
            msg = (
                f"Process agent_cards holds more than one ref for role(s) "
                f"{sorted(duplicates)}; roles are the catalog key and must be unique."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def require_resolvable_agent_refs(self) -> Process:
        """Reject a projection whose refs do not resolve against ``agent_cards``.

        ``mode="after"`` so the check fires on ``Process(...)`` **and** on
        ``Process.model_validate(...)``. Validating only on construction would
        leave the read path — the one a corrupted or half-migrated document
        actually takes — unguarded, and the record would fail much later, inside
        a resume, with an error naming neither the role nor the document.

        Raises:
            ValueError: If ``entry_point`` or any supervisor names a role that
                has no entry in ``agent_cards``. The offending role is named.
        """
        known = {ref.role for ref in self.agent_cards}
        checked: list[tuple[str, AgentRef]] = [("entry_point", self.entry_point)]
        checked += [("supervisor", ref) for ref in self.supervisors]
        for label, ref in checked:
            if ref.role not in known:
                msg = (
                    f"Process {label} ref '{ref.name}' names role '{ref.role}', "
                    f"which has no entry in agent_cards (roles: {sorted(known)})."
                )
                raise ValueError(msg)
        return self


class PersistedEvent(SerializableBaseModel):
    """Append-only event log entry for event-sourced persistence.

    Each entry captures a single event (Message subclass) with its
    sequence number for ordered replay during team restoration.
    """

    team_id: uuid.UUID = Field(description="Team instance this event belongs to")
    sequence: int = Field(description="Monotonically increasing event sequence number")
    event: Message = Field(description="Polymorphic event payload preserving concrete Message type")
    timestamp: datetime = Field(description="Timestamp when the event was persisted")


class AgentStateSnapshot(SerializableBaseModel):
    """Overwrite-strategy snapshot of an agent's state.

    Captures the latest state of a single agent for fast recovery
    without full event replay. Each snapshot overwrites the previous
    one for the same (team_id, agent_id) pair.
    """

    team_id: uuid.UUID = Field(description="Team instance this snapshot belongs to")
    agent_id: str = Field(
        description=(
            "Agent UUID in string form (str(uuid)). Legacy snapshots written "
            "before this field's semantics changed may hold the agent display "
            "name instead; such snapshots load with name=None and self-heal on "
            "the agent's next state change."
        )
    )
    name: str | None = Field(
        default=None,
        description="Agent display name; None for snapshots written before this field existed",
    )
    state: BaseState = Field(
        description="Polymorphic agent state preserving concrete BaseState type"
    )
    updated_at: datetime = Field(description="Timestamp when the snapshot was taken")
