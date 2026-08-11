"""Unit tests for MemorySampler trend classification.

Drives the sampler against a synthetic, controllable allocation pattern so the
heap-vs-RSS verdict logic is verified without an actor runtime.
"""

from __future__ import annotations

from pathlib import Path

from akgentic.team.sampler import (
    MemorySample,
    MemorySampler,
    MemoryTrend,
    ObjectCensus,
)


class Blob:
    """Allocation unit large enough to move the traced-heap needle."""

    def __init__(self) -> None:
        self.payload = bytearray(64 * 1024)  # 64 KiB


class TestMemoryTrendVerdict:
    """The trend's verdict classifies leak vs arena vs stable."""

    def _trend(self, heap_growth: int, obj_growth: int, rss_growth: int = 0) -> MemoryTrend:
        first = MemorySample(label="a", iteration=0, heap_bytes=0, rss_bytes=0, gc_objects=0)
        last = MemorySample(
            label="b",
            iteration=1,
            heap_bytes=heap_growth,
            rss_bytes=rss_growth,
            gc_objects=obj_growth,
        )
        return MemoryTrend(
            samples=[first, last],
            rss_available=rss_growth > 0,
            heap_growth_bytes=heap_growth,
            rss_growth_bytes=rss_growth,
            object_growth=obj_growth,
        )

    def test_growing_heap_is_real_leak(self) -> None:
        trend = self._trend(heap_growth=8 * 1024 * 1024, obj_growth=5000)
        assert trend.is_object_leak
        assert "REAL LEAK" in trend.verdict

    def test_rss_only_is_arena_retention(self) -> None:
        trend = self._trend(heap_growth=100_000, obj_growth=10, rss_growth=20 * 1024 * 1024)
        assert not trend.is_object_leak
        assert "ARENA RETENTION" in trend.verdict
        assert "ARENA RETENTION" in trend.format()  # render path

    def test_flat_is_stable(self) -> None:
        trend = self._trend(heap_growth=50_000, obj_growth=5)
        assert not trend.is_object_leak
        assert "STABLE" in trend.verdict


class TestObjectCensusDiff:
    """A/B differential census: capture, persist, diff, render."""

    def test_capture_counts_live_instances(self) -> None:
        widgets = [Blob() for _ in range(5)]  # noqa: F841 — keep alive for the count
        census = ObjectCensus.capture(label="with-blobs")
        assert census.label == "with-blobs"
        assert census.counts.get("Blob", 0) >= 5

    def test_diff_ranks_classes_the_candidate_added(self) -> None:
        baseline = ObjectCensus(label="a", counts={"Foo": 10, "Bar": 5})
        candidate = ObjectCensus(label="b", counts={"Foo": 14, "Bar": 5, "Baz": 3})
        rows = ObjectCensus.diff(baseline, candidate)
        # Only positive deltas, sorted by delta desc: Foo (+4), Baz (+3); Bar (0) dropped.
        assert [(r.type_name, r.delta) for r in rows] == [("Foo", 4), ("Baz", 3)]

    def test_diff_empty_when_nothing_grew(self) -> None:
        same = {"Foo": 7}
        rows = ObjectCensus.diff(ObjectCensus(counts=same), ObjectCensus(counts=same))
        assert rows == []
        assert "no object leak" in ObjectCensus.format_diff(rows)

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        census = ObjectCensus(label="run-a", counts={"Foo": 3, "Bar": 9})
        path = tmp_path / "census-a.json"
        census.save(path)
        loaded = ObjectCensus.load(path)
        assert loaded.label == "run-a"
        assert loaded.counts == {"Foo": 3, "Bar": 9}


class TestMemorySamplerDetectsAccumulation:
    """End-to-end: a deliberate accumulation is flagged with the culprit type."""

    def test_retained_blobs_show_as_heap_and_type_growth(self) -> None:
        sampler = MemorySampler()
        sampler.start()

        retained: list[Blob] = []
        for i in range(40):
            retained.extend(Blob() for _ in range(10))  # leak 10 blobs/iteration
            sampler.sample(label="leak", iteration=i)

        trend = sampler.report()
        sampler.stop()

        rendered = trend.format()
        assert trend.is_object_leak, rendered
        assert "Blob" in rendered  # the culprit type is named in the report
        blob_growth = next((t for t in trend.top_type_growth if t.type_name == "Blob"), None)
        assert blob_growth is not None, rendered
        assert blob_growth.delta >= 350  # ~400 retained, minus baseline noise
        assert len(retained) == 400  # keep them alive through the assertions

    def test_dropped_blobs_do_not_accumulate(self) -> None:
        sampler = MemorySampler()
        sampler.start()

        for i in range(20):
            transient = [Blob() for _ in range(10)]  # created then dropped
            del transient
            sampler.sample(label="clean", iteration=i)

        trend = sampler.report()
        sampler.stop()

        assert not trend.is_object_leak, trend.format()
