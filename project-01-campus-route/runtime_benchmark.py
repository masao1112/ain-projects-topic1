"""
Project 01 — Midterm runtime benchmark.

Run from project-01-campus-route:
    python runtime_benchmark.py

This script measures runtime over many repetitions while keeping the
deterministic search metrics (path, cost, expanded nodes, frontier size)
from one representative run.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from starter.graph import load_graph
from starter.student_core import (
    astar,
    bfs,
    euclidean_heuristic,
    greedy,
    manhattan_heuristic,
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RUNS = 1000


def path_cost(graph, path, edge_weight="distance"):
    if path is None:
        return None

    total = 0.0
    for a, b in zip(path, path[1:]):
        for edge in graph.neighbors(a):
            if edge.target == b:
                total += getattr(edge, edge_weight)
                break
    return total


def timed_call(function, runs=RUNS):
    # Warm-up to reduce one-time import/cache effects.
    for _ in range(10):
        function()

    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)

    return {
        "avg_runtime": statistics.mean(samples),
        "min_runtime": min(samples),
        "max_runtime": max(samples),
        "std_runtime": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def main():
    graph = load_graph(DATA_DIR)

    with open(DATA_DIR / "scenarios.json", encoding="utf-8") as fh:
        scenarios = {s["id"]: s for s in json.load(fh)["scenarios"]}

    s1 = scenarios["S1_normal_route"]
    s2 = scenarios["S2_blocked_edge"]
    s3 = scenarios["S3_accessibility_mode"]
    s4 = scenarios["S4_heuristic_comparison"]
    s5 = scenarios["S5_accessibility_no_path"]

    jobs = [
        ("S1_normal_route", "BFS", "-", lambda: bfs(graph, s1["start"], s1["end"])),
        (
            "S1_normal_route",
            "Greedy",
            "Euclidean",
            lambda: greedy(
                graph, s1["start"], s1["end"], euclidean_heuristic
            ),
        ),
        (
            "S1_normal_route",
            "A*",
            "Euclidean",
            lambda: astar(
                graph, s1["start"], s1["end"], euclidean_heuristic
            ),
        ),
        (
            "S2_blocked_edge",
            "A*",
            "Euclidean",
            lambda: astar(
                graph,
                s2["start"],
                s2["end"],
                euclidean_heuristic,
                blocked_edges=s2["constraints"]["blocked_edges"],
            ),
        ),
        (
            "S3_accessibility_mode",
            "A*",
            "Euclidean",
            lambda: astar(
                graph,
                s3["start"],
                s3["end"],
                euclidean_heuristic,
                accessible_only=True,
            ),
        ),
        (
            "S4_heuristic_comparison",
            "A*",
            "Euclidean",
            lambda: astar(
                graph, s4["start"], s4["end"], euclidean_heuristic
            ),
        ),
        (
            "S4_heuristic_comparison",
            "A*",
            "Manhattan",
            lambda: astar(
                graph, s4["start"], s4["end"], manhattan_heuristic
            ),
        ),
        (
            "S5_accessibility_no_path",
            "A*",
            "Euclidean",
            lambda: astar(
                graph,
                s5["start"],
                s5["end"],
                euclidean_heuristic,
                accessible_only=True,
            ),
        ),
    ]

    rows = []

    for scenario, algorithm, heuristic, function in jobs:
        path, expanded, peak_frontier = function()
        timing = timed_call(function)

        rows.append(
            {
                "scenario": scenario,
                "algorithm": algorithm,
                "heuristic": heuristic,
                "cost": path_cost(graph, path),
                "expanded": expanded,
                "peak_frontier": peak_frontier,
                **timing,
            }
        )

    output = PROJECT_DIR / "runtime_results.json"
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print("=" * 120)
    print(f"RUNTIME BENCHMARK — {RUNS} runs per configuration")
    print("=" * 120)
    print(
        f"{'Scenario':24} {'Algorithm':10} {'Heuristic':11} "
        f"{'Cost':>9} {'Expanded':>9} {'Frontier':>9} "
        f"{'Avg(s)':>12} {'Std(s)':>12}"
    )
    print("-" * 120)

    for r in rows:
        cost = "None" if r["cost"] is None else f"{r['cost']:.2f}"
        print(
            f"{r['scenario']:24} {r['algorithm']:10} {r['heuristic']:11} "
            f"{cost:>9} {r['expanded']:>9} {r['peak_frontier']:>9} "
            f"{r['avg_runtime']:>12.9f} {r['std_runtime']:>12.9f}"
        )

    print("-" * 120)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
