"""
Project 01 — Midterm experiment runner.

This version fixes the heuristic metadata in the summary and records
deterministic search metrics. Runtime is benchmarked separately by
runtime_benchmark.py.
"""
from __future__ import annotations

import json
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


def run_and_record(
    scenario_id,
    algorithm,
    heuristic_name,
    function,
    graph,
    edge_weight="distance",
):
    start = time.perf_counter()
    path, expanded, peak_frontier = function()
    runtime = time.perf_counter() - start

    return {
        "scenario": scenario_id,
        "algorithm": algorithm,
        "heuristic": heuristic_name,
        "path": path,
        "cost": path_cost(graph, path, edge_weight),
        "expanded": expanded,
        "peak_frontier": peak_frontier,
        "runtime": runtime,
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

    rows = []

    rows.append(
        run_and_record(
            "S1_normal_route", "BFS", "-",
            lambda: bfs(graph, s1["start"], s1["end"]), graph
        )
    )
    rows.append(
        run_and_record(
            "S1_normal_route", "Greedy", "Euclidean",
            lambda: greedy(
                graph, s1["start"], s1["end"], euclidean_heuristic
            ), graph
        )
    )
    rows.append(
        run_and_record(
            "S1_normal_route", "A*", "Euclidean",
            lambda: astar(
                graph, s1["start"], s1["end"], euclidean_heuristic
            ), graph
        )
    )
    rows.append(
        run_and_record(
            "S2_blocked_edge", "A*", "Euclidean",
            lambda: astar(
                graph,
                s2["start"],
                s2["end"],
                euclidean_heuristic,
                blocked_edges=s2["constraints"]["blocked_edges"],
            ), graph
        )
    )
    rows.append(
        run_and_record(
            "S3_accessibility_mode", "A*", "Euclidean",
            lambda: astar(
                graph,
                s3["start"],
                s3["end"],
                euclidean_heuristic,
                accessible_only=True,
            ), graph
        )
    )
    rows.append(
        run_and_record(
            "S4_heuristic_comparison", "A*", "Euclidean",
            lambda: astar(
                graph, s4["start"], s4["end"], euclidean_heuristic
            ), graph
        )
    )
    rows.append(
        run_and_record(
            "S4_heuristic_comparison", "A*", "Manhattan",
            lambda: astar(
                graph, s4["start"], s4["end"], manhattan_heuristic
            ), graph
        )
    )
    rows.append(
        run_and_record(
            "S5_accessibility_no_path", "A*", "Euclidean",
            lambda: astar(
                graph,
                s5["start"],
                s5["end"],
                euclidean_heuristic,
                accessible_only=True,
            ), graph
        )
    )

    print("=" * 120)
    print("MIDTERM EXPERIMENT SUMMARY")
    print("=" * 120)
    print(
        f"{'Scenario':24} {'Algorithm':10} {'Heuristic':11} "
        f"{'Cost':>9} {'Expanded':>9} {'Frontier':>9} {'Runtime(s)':>12}"
    )
    print("-" * 120)

    for r in rows:
        cost = "None" if r["cost"] is None else f"{r['cost']:.2f}"
        print(
            f"{r['scenario']:24} {r['algorithm']:10} {r['heuristic']:11} "
            f"{cost:>9} {r['expanded']:>9} {r['peak_frontier']:>9} "
            f"{r['runtime']:>12.9f}"
        )

    print("=" * 120)


if __name__ == "__main__":
    main()
