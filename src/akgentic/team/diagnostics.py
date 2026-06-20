"""Memory-leak diagnostics for the team lifecycle.

``TeamMemoryProbe`` answers one question across a create → stop → delete cycle:
which framework objects are still resident after teardown, and who holds them?

It works by weak-referencing every live "team object" at peak, then — after the
caller has stopped the team and dropped its own handles — counting which
weakrefs are still alive before and after ``gc.collect()`` and naming the
referrers that pin any survivor. Imports stay core-only (the akgentic-team
boundary); higher-layer suspects (ContextManager, ReactAgent, BaseAgent) are
tracked by class name, not import.
"""

from __future__ import annotations

import gc
import weakref
from typing import Any

from pydantic import BaseModel, Field

from akgentic.core.agent import Akgent
from akgentic.core.agent_state import BaseState
from akgentic.team.models import TeamRuntime
from akgentic.team.subscriber import PersistenceSubscriber, TimerStopSubscriber

# Importable core/team bases whose instances are per-team framework objects.
# ``EventSubscriber`` itself is a non-runtime-checkable Protocol, so we track the
# concrete per-team subscribers by class instead. threading.Thread is excluded:
# it would match every Pykka actor thread and the system listener, drowning real
# leaks in runtime noise. Pass it in ``tracked_bases`` explicitly when chasing
# the TimerStopSubscriber daemon thread on the inactivity-timer stop path.
DEFAULT_TRACKED_BASES: tuple[type, ...] = (
    Akgent,  # orchestrator + every agent + UserProxy/HumanProxy
    BaseState,  # agent state (holds the agent<->state observer cycle)
    TeamRuntime,  # holds actor_system + addresses + rebuilt proxies
    PersistenceSubscriber,  # per-team event-store bridge
    TimerStopSubscriber,  # holds TeamManager -> _runtimes -> every team
)

# Higher-layer suspects we may not import (boundary) — matched by class name.
# NOTE: this curated set is only for the in-process TeamMemoryProbe. The Docker
# A/B uses ObjectCensus, which censuses EVERY class (no list), so it is the
# comprehensive tool — this list just mirrors what that census has surfaced so
# the probe-based test can assert on the same leak classes.
DEFAULT_TRACKED_NAMES: frozenset[str] = frozenset(
    {
        # llm / agent object graph
        "ContextManager",  # llm: holds agent in its observer list
        "ReactAgent",  # llm: holds ContextManager
        "MockReactAgent",  # llm loadtest: same shape as ReactAgent
        "BaseAgent",  # agent: holds ReactAgent
        # core actor infrastructure (pykka) — leak #1 signature
        "Timer",  # core: orchestrator inactivity timer (callback cycle)
        "ActorRef",  # pykka actor reference
        "ActorProxy",  # pykka proxy (carries AttrInfo)
        "ProxyWrapper",  # pykka proxy wrapper
        "ActorAddressImpl",  # core actor address
        "AttrInfo",  # pykka per-proxy method introspection (amplifier)
        # per-team async + state — leak #2 signature
        "_UnixSelectorEventLoop",  # one asyncio loop per team, retained
        "StructuredOutput",  # agent routing output, dragged along
        "TaskState",  # planning state
        "Request",  # routing request
        # OpenTelemetry instruments pinned by the global MeterProvider — leak #2
        "_ProxyHistogram",
        "_ProxyMeter",
        # dynamically-built pydantic model classes never freed — leak #2
        "ModelMetaclass",
        "SchemaValidator",
        "SchemaSerializer",
    }
)


class ReferrerInfo(BaseModel):
    """One object that still references a leaked instance."""

    referrer_type: str = Field(description="Class name of the referring object")
    detail: str = Field(description="Trimmed repr of the referrer for identification")


class RetainedType(BaseModel):
    """A tracked type with surviving instances after teardown."""

    type_name: str = Field(description="Class name of the surviving objects")
    count: int = Field(description="Number of live instances after gc.collect()")
    referrers: list[ReferrerInfo] = Field(
        default_factory=list,
        description="Sample of objects pinning the survivors (the leak holders)",
    )


class LeakReport(BaseModel):
    """Outcome of one probe cycle: what survived teardown and who held it."""

    label: str = Field(description="Caller-supplied label for this cycle")
    total_tracked: int = Field(description="Tracked instances captured at peak")
    alive_before_gc: int = Field(description="Survivors after teardown, before gc.collect()")
    alive_after_gc: int = Field(description="Survivors after gc.collect()")
    gc_collected: int = Field(description="Objects reclaimed by gc.collect()")
    uncollectable: int = Field(description="len(gc.garbage) — objects gc could not free")
    retained: list[RetainedType] = Field(
        default_factory=list, description="Per-type breakdown of survivors with holders"
    )

    @property
    def leaked(self) -> bool:
        """True if any tracked instance survived teardown + gc.collect()."""
        return self.alive_after_gc > 0

    @property
    def cycle_held(self) -> bool:
        """True if survivors were freed only by the cyclic collector, not refcount."""
        return self.alive_before_gc > self.alive_after_gc

    def format(self) -> str:
        """Render a human-readable summary for logs / test failure messages."""
        head = (
            f"[{self.label}] tracked={self.total_tracked} "
            f"alive(pre-gc)={self.alive_before_gc} gc_freed={self.gc_collected} "
            f"alive(post-gc)={self.alive_after_gc} uncollectable={self.uncollectable}"
        )
        lines = [head]
        for rt in self.retained:
            lines.append(f"  LEAK {rt.type_name} x{rt.count}")
            for ref in rt.referrers:
                lines.append(f"      held by {ref.referrer_type}: {ref.detail}")
        if not self.retained:
            lines.append("  no survivors — team fully released")
        return "\n".join(lines)


class TeamMemoryProbe:
    """Weak-reference probe that pinpoints retained team objects after teardown.

    Usage::

        probe = TeamMemoryProbe()
        probe.mark_baseline()             # ignore objects from other teams/tests
        runtime = manager.create_team(card)
        probe.mark_peak()                 # snapshot objects this cycle created
        del runtime                       # drop the caller's strong handle
        manager.stop_team(team_id)
        manager.delete_team(team_id)
        report = probe.report(label="cycle-1")
        assert not report.leaked, report.format()
    """

    def __init__(
        self,
        tracked_bases: tuple[type, ...] = DEFAULT_TRACKED_BASES,
        tracked_names: frozenset[str] = DEFAULT_TRACKED_NAMES,
    ) -> None:
        self._bases = tracked_bases
        self._names = tracked_names
        self._baseline_ids: frozenset[int] = frozenset()
        self._watched: list[weakref.ref[Any]] = []

    def _is_tracked(self, obj: object) -> bool:
        """True if ``obj`` is a per-team framework object we should follow."""
        try:
            if isinstance(obj, self._bases):
                return True
        except TypeError:
            pass  # a non-runtime-checkable Protocol slipped into tracked_bases
        return type(obj).__name__ in self._names

    def count_live(self) -> dict[str, int]:
        """Return {type_name: count} of tracked instances currently alive."""
        counts: dict[str, int] = {}
        for obj in gc.get_objects():
            if self._is_tracked(obj):
                name = type(obj).__name__
                counts[name] = counts.get(name, 0) + 1
        return counts

    def mark_baseline(self) -> int:
        """Record the ids of tracked objects that already exist.

        Call before creating the team. ``mark_peak`` then ignores everything
        captured here, so the probe follows only objects the cycle creates —
        making it robust to objects other tests / teams leave resident. Returns
        the baseline count.
        """
        ids = {id(obj) for obj in gc.get_objects() if self._is_tracked(obj)}
        self._baseline_ids = frozenset(ids)
        return len(ids)

    def mark_peak(self) -> int:
        """Weak-reference each tracked instance created since ``mark_baseline``.

        Runs the scan in this frame and keeps only weakrefs, so the probe never
        pins the objects it measures. Objects present at baseline are excluded by
        id (id reuse after a baseline object dies is the only blind spot, and it
        can only hide a leak, never invent one). Returns the number captured.
        """
        watched: list[weakref.ref[Any]] = []
        for obj in gc.get_objects():
            if id(obj) in self._baseline_ids or not self._is_tracked(obj):
                continue
            try:
                watched.append(weakref.ref(obj))
            except TypeError:
                pass  # not weak-referenceable
        self._watched = watched
        return len(watched)

    def report(self, label: str = "") -> LeakReport:
        """Stop the world, collect cycles, and report what survived and who holds it."""
        total = len(self._watched)
        alive_before = sum(1 for ref in self._watched if ref() is not None)

        collected = gc.collect()
        survivors = [obj for ref in self._watched if (obj := ref()) is not None]
        alive_after = len(survivors)

        retained = self._build_retained(survivors)
        report = LeakReport(
            label=label,
            total_tracked=total,
            alive_before_gc=alive_before,
            alive_after_gc=alive_after,
            gc_collected=collected,
            uncollectable=len(gc.garbage),
            retained=retained,
        )
        del survivors
        return report

    def _build_retained(self, survivors: list[Any]) -> list[RetainedType]:
        """Group survivors by type and attach a sample of their holders."""
        by_type: dict[str, list[Any]] = {}
        for obj in survivors:
            by_type.setdefault(type(obj).__name__, []).append(obj)

        retained: list[RetainedType] = []
        for name, objs in sorted(by_type.items()):
            referrers = self._describe_holders(objs[0])
            retained.append(RetainedType(type_name=name, count=len(objs), referrers=referrers))
        return retained

    def _describe_holders(self, obj: object, limit: int = 5) -> list[ReferrerInfo]:
        """Summarize the objects that still reference ``obj`` (the leak holders)."""
        skip = {id(self), id(self._watched), id(obj)}
        infos: list[ReferrerInfo] = []
        for r in gc.get_referrers(obj):
            if id(r) in skip or isinstance(r, weakref.ref) or _is_frame(r):
                continue
            rep = repr(r)
            if len(rep) > 100:
                rep = rep[:97] + "..."
            infos.append(ReferrerInfo(referrer_type=type(r).__name__, detail=rep))
            if len(infos) >= limit:
                break
        return infos


def _is_frame(obj: object) -> bool:
    """True for frame objects (the probe's own call stack), which we never report."""
    return type(obj).__name__ == "frame"
