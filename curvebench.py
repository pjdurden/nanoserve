"""Day 45: Week 12's two sweeps on one pair of axes, and where the knee is.

Run from the repo root with the venv python:

    cd ~/nanoserve && .venv/bin/python curvebench.py

    cd ~/nanoserve && .venv/bin/python curvebench.py --ttft-slo 5 --e2e-slo 30 \
        --csv docs/daily/data/day-45-curve.csv

This measures nothing. Every number it prints was measured earlier in the week and
written to a CSV: Day 42's open-loop rate sweep over a real socket
(`day-42-servebench.csv`) and Day 44's offline slot sweep against HuggingFace
`generate` (`day-44-hfbench.csv`). What this file does is put them on the same two
axes, throughput across and latency up, and ask the four questions that only make
sense once they are on a curve rather than in a table: where is the peak, where is
the knee, which settings are dominated, and how much throughput survives a latency
budget.

Two knobs, one pair of axes, and that is the point. The rate sweep turns offered
load up on a server with a queue; the slot sweep turns residency up on an offline
burst. They are different experiments and neither is a substitute for the other,
but both produce output tokens per second against seconds of latency, so both can
be drawn together, and drawn together they run in *opposite directions* on the
TTFT axis: more offered load fills a queue, more slots drain one.

The knee moves depending on which latency is on y, which is the day's real finding
and the reason both plots are printed rather than a chosen one. On the end-to-end
axis the slot sweep's knee is 4 slots, agreeing with Day 44's throughput-only
reading; on the TTFT axis it is 8, because at 8 slots the burst never queues. Same
data, same definition of knee, different question.

`nanoserve.curve` holds the arithmetic and the gate and is unit-tested against
points placed by hand and against an M/M/1 queue whose knee is known in closed
form. This file only parses CSVs, builds curves and prints. The in-process path,
for a caller holding a live `LoadReport` or `SystemRun` rather than a CSV, is
`point_from_load_report` and `point_from_run` in the same module.

`--sanity` is on by default and stops the run if either sweep never bent backwards,
because a curve with no dominated setting has not found a capacity and its peak is
just the largest number that was tried.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from nanoserve.curve import (
    CurveUnsound,
    OperatingPoint,
    Segment,
    TradeoffCurve,
    capacity_under_slo,
    check_swept_to_saturation,
    dominates,
    render,
)

ROOT = Path(__file__).resolve().parent
RATE_CSV = ROOT / "docs" / "daily" / "data" / "day-42-servebench.csv"
SLOT_CSV = ROOT / "docs" / "daily" / "data" / "day-44-hfbench.csv"


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"no such file: {path} (run the earlier day's bench first)")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _price(seg: Segment | None) -> str:
    """What one step cost, in the same three words the summary uses.

    A step that gave latency back has no price in seconds per token per second, so
    it says so rather than printing the negative number the division produces: a
    reader sorting the column would otherwise put the best steps at the bottom.
    """
    if seg is None:
        return ""
    if seg.backward:
        return "inf"
    if seg.free:
        return "free"
    return f"{seg.latency_per_tps:.4f}"


def _slots(raw: str) -> str:
    n = int(float(raw))
    return f"{n} slot" if n == 1 else f"{n} slots"


def rate_curve(path: Path, *, latency: str) -> TradeoffCurve:
    """Day 42's open-loop sweep: offered requests per second as the knob.

    `achieved_rps` is in the CSV and is deliberately not the x axis: the axis is
    output tokens per second, so that the two sweeps share it. The offered rate
    stays on the knob, where a hidden parameter belongs.
    """
    column = {"ttft": ("ttft_p50_ms", 1e-3), "e2e": ("e2e_p50_s", 1.0)}[latency]
    points = [
        OperatingPoint(
            label=f"{float(row['offered_rps']):g} rps",
            knob="offered_rps",
            knob_value=float(row["offered_rps"]),
            throughput_tps=float(row["output_tps"]),
            latency_s=float(row[column[0]]) * column[1],
        )
        for row in _rows(path)
    ]
    name = {"ttft": "ttft_p50", "e2e": "e2e_p50"}[latency]
    return TradeoffCurve.from_points("rate sweep", points, latency_name=name, marker="o")


def slot_curve(path: Path, *, latency: str) -> TradeoffCurve:
    """Day 44's offline sweep: how many requests may be resident as the knob.

    The latencies here are means over eight requests rather than percentiles over a
    stream, which is what an offline burst can honestly report. They are drawn on
    the same axis as the rate sweep's percentiles, and the difference is a real
    caveat rather than a rounding one: with every request the same length, the mean
    and the p50 of this run are within a few percent of each other, but that is a
    property of this workload and not a general licence.
    """
    column = {"ttft": "engine_ttft_s", "e2e": "engine_latency_s"}[latency]
    points = [
        OperatingPoint(
            label=_slots(row["slots"]),
            knob="slots",
            knob_value=float(row["slots"]),
            throughput_tps=float(row["engine_tps"]),
            latency_s=float(row[column]),
        )
        for row in _rows(path)
    ]
    name = {"ttft": "ttft_p50", "e2e": "e2e_p50"}[latency]
    return TradeoffCurve.from_points("slot sweep", points, latency_name=name, marker="x")


def report(curves: list[TradeoffCurve], *, slo_s: float) -> str:
    """Both curves' summaries, the plot they share, and the SLO reading under it."""
    blocks = [curve.summary() for curve in curves]
    blocks.append(render(curves, width=62, height=17, slo_s=slo_s))
    lines = []
    for curve in curves:
        best = capacity_under_slo(curve, slo_s)
        if best is None:
            lines.append(
                f"    {curve.name}: no setting keeps {curve.latency_name} under "
                f"{slo_s:.2f}s, so this box cannot serve that budget at all"
            )
        else:
            share = best.throughput_tps / curve.max_throughput
            lines.append(
                f"    {curve.name}: {best.throughput_tps:.2f} tok/s at {best.label} "
                f"({best.latency_s:.2f}s), which is {share:.0%} of its peak"
            )
    blocks.append(f"  under a {slo_s:.2f}s {curves[0].latency_name} budget\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def write_csv(path: Path, curves: list[TradeoffCurve]) -> None:
    """One row per point, with the verdict each point earned.

    `dominated_by` is the interesting column: it names the setting that beat this
    one on both axes, which is the whole reason the row should never be quoted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "curve",
                "latency_name",
                "knob",
                "knob_value",
                "label",
                "throughput_tps",
                "latency_s",
                "power",
                "verdict",
                "dominated_by",
                "d_throughput",
                "d_latency",
                "latency_per_tps",
            ]
        )
        for curve in curves:
            steps = {seg.hi.label: seg for seg in curve.segments}
            for point in curve.points:
                beaten_by = next((o.label for o in curve.points if dominates(o, point)), "")
                verdict = "dominated" if beaten_by else "frontier"
                if point is curve.knee:
                    verdict += "+knee"
                if point is curve.peak:
                    verdict += "+peak"
                seg = steps.get(point.label)
                writer.writerow(
                    [
                        curve.name,
                        curve.latency_name,
                        curve.knob,
                        f"{point.knob_value:g}",
                        point.label,
                        f"{point.throughput_tps:.4f}",
                        f"{point.latency_s:.4f}",
                        f"{point.power:.4f}",
                        verdict,
                        beaten_by,
                        f"{seg.d_throughput:+.4f}" if seg else "",
                        f"{seg.d_latency:+.4f}" if seg else "",
                        _price(seg),
                    ]
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rate-csv", type=Path, default=RATE_CSV)
    parser.add_argument("--slot-csv", type=Path, default=SLOT_CSV)
    parser.add_argument("--ttft-slo", type=float, default=5.0, help="TTFT budget in seconds")
    parser.add_argument("--e2e-slo", type=float, default=30.0, help="end-to-end budget in seconds")
    parser.add_argument("--csv", type=Path, default=None, help="write every point here")
    parser.add_argument("--sanity", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    everything: list[TradeoffCurve] = []
    for latency, slo in (("ttft", args.ttft_slo), ("e2e", args.e2e_slo)):
        curves = [
            rate_curve(args.rate_csv, latency=latency),
            slot_curve(args.slot_csv, latency=latency),
        ]
        if args.sanity:
            try:
                for curve in curves:
                    check_swept_to_saturation(curve)
            except CurveUnsound as exc:
                print(f"SWEEP UNSOUND: {exc}", file=sys.stderr)
                return 2
        print(f"\n=== {latency} on the y axis ===\n")
        print(report(curves, slo_s=slo))
        everything.extend(curves)

    if args.csv is not None:
        write_csv(args.csv, everything)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
