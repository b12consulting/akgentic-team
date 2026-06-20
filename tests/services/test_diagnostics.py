"""Unit tests for TeamMemoryProbe and its report models.

Actor-free: the probe is exercised against synthetic tracked objects so the
detection logic (baseline exclusion, survivor counting, holder naming) is
verified in isolation from the actor runtime.
"""

from __future__ import annotations

import gc

from akgentic.team.diagnostics import (
    LeakReport,
    RetainedType,
    TeamMemoryProbe,
)


class Widget:
    """Stand-in tracked object with no framework coupling."""


def _probe() -> TeamMemoryProbe:
    """Probe that tracks only the local Widget class."""
    return TeamMemoryProbe(tracked_bases=(Widget,), tracked_names=frozenset())


class TestLeakReportModel:
    """The Pydantic report exposes the right derived flags and rendering."""

    def test_leaked_and_cycle_held_flags(self) -> None:
        report = LeakReport(
            label="x",
            total_tracked=5,
            alive_before_gc=3,
            alive_after_gc=1,
            gc_collected=2,
            uncollectable=0,
            retained=[RetainedType(type_name="Widget", count=1)],
        )
        assert report.leaked is True
        assert report.cycle_held is True  # before(3) > after(1): cycle reclaimed some

    def test_clean_report_is_not_leaked(self) -> None:
        report = LeakReport(
            label="x",
            total_tracked=4,
            alive_before_gc=0,
            alive_after_gc=0,
            gc_collected=0,
            uncollectable=0,
        )
        assert report.leaked is False
        assert report.cycle_held is False
        assert "no survivors" in report.format()

    def test_format_lists_holders(self) -> None:
        report = LeakReport(
            label="leak",
            total_tracked=1,
            alive_before_gc=1,
            alive_after_gc=1,
            gc_collected=0,
            uncollectable=0,
            retained=[
                RetainedType(
                    type_name="Widget",
                    count=2,
                    referrers=[],
                )
            ],
        )
        rendered = report.format()
        assert "LEAK Widget x2" in rendered
        assert "[leak]" in rendered


class TestTeamMemoryProbe:
    """Detection logic: baseline exclusion, survivor count, holder naming."""

    def test_count_live_sees_tracked_objects(self) -> None:
        probe = _probe()
        widgets = [Widget(), Widget()]  # noqa: F841 — kept alive for the count
        assert probe.count_live().get("Widget", 0) >= 2

    def test_no_leak_when_object_is_dropped(self) -> None:
        probe = _probe()
        probe.mark_baseline()
        w = Widget()
        probe.mark_peak()
        del w
        report = probe.report(label="clean")
        assert not report.leaked, report.format()

    def test_detects_retained_object_and_names_dict_holder(self) -> None:
        probe = _probe()
        probe.mark_baseline()
        w = Widget()
        probe.mark_peak()
        holder = {"w": w}  # the leak: a dict keeps the Widget alive
        del w
        report = probe.report(label="held")
        try:
            assert report.leaked
            widget = next(rt for rt in report.retained if rt.type_name == "Widget")
            assert widget.count == 1
            assert any("dict" in r.referrer_type for r in widget.referrers)
        finally:
            holder.clear()

    def test_baseline_excludes_pre_existing_objects(self) -> None:
        probe = _probe()
        pre_existing = Widget()  # exists before baseline
        probe.mark_baseline()
        probe.mark_peak()  # nothing new created
        # pre_existing is alive but was captured at baseline → not a survivor
        report = probe.report(label="baseline")
        assert report.total_tracked == 0
        assert not report.leaked
        del pre_existing
        gc.collect()
