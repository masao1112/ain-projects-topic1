"""Empty AI core stubs for Project 01 (Smart Campus Route Planner).

Implement BFS, Greedy Best-First Search, and A* here, plus at least one
heuristic function. Every public function below raises `NotImplementedError`
until you replace its body with your own implementation. Do not call a
hosted LLM to produce a route: see the auto-zero note in the instructor
rubric.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph import Graph  # noqa: E402

# (path of node ids from start to goal inclusive, or None) and the number of
# nodes popped from the frontier while searching (used by the required
# experiment to compare algorithms/heuristics).
PathResult = Tuple[Optional[List[str]], int, int]

Heuristic = Callable[[Graph, str, str], float]


def bfs(graph: Graph, start_id: str, goal_id: str) -> PathResult:
    queue = [start_id]
    visited = {start_id}
    parent = {start_id: None}

    expanded = 0
    peak_frontier = 1

    while queue:
        current = queue.pop(0)
        expanded += 1

        if current == goal_id:
            break

        for edge in graph.neighbors(current):
            neighbor = edge.target

            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
                peak_frontier = max(peak_frontier, len(queue))

    if goal_id not in visited:
        return None, expanded, peak_frontier

    path = []
    current = goal_id

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path, expanded, peak_frontier      


def euclidean_heuristic(graph: Graph, node_id: str, goal_id: str) -> float:
    node = graph.get_node(node_id)
    goal = graph.get_node(goal_id)

    dx = node.x - goal.x
    dy = node.y - goal.y

    return (dx ** 2 + dy ** 2) ** 0.5


def manhattan_heuristic(graph: Graph, node_id: str, goal_id: str) -> float:
    node = graph.get_node(node_id)
    goal = graph.get_node(goal_id)

    dx = abs(node.x - goal.x)
    dy = abs(node.y - goal.y)

    return dx + dy

def greedy(graph: Graph, start_id: str, goal_id: str, heuristic: Heuristic) -> PathResult:
    frontier = [(heuristic(graph, start_id, goal_id), start_id)]

    visited = set()
    parent = {start_id: None}

    expanded = 0
    peak_frontier = 1

    while frontier:
        frontier.sort(key=lambda item: item[0])
        _, current = frontier.pop(0)

        if current in visited:
            continue

        visited.add(current)
        expanded += 1

        if current == goal_id:
            break

        for edge in graph.neighbors(current):
            neighbor = edge.target

            if neighbor not in visited:
                if neighbor not in parent:
                    parent[neighbor] = current

                h = heuristic(
                    graph,
                    neighbor,
                    goal_id
                )

                frontier.append((h, neighbor))
                peak_frontier = max(peak_frontier, len(frontier))

    if goal_id not in visited:
        return None, expanded, peak_frontier

    path = []
    current = goal_id

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()

    return path, expanded, peak_frontier


def astar(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Heuristic,
    edge_weight: str = "distance",
    blocked_edges: Optional[List[List[str]]] = None,
    accessible_only: bool = False,
) -> PathResult:

    if edge_weight not in {"distance", "travel_time"}:
        raise ValueError(
            "edge_weight must be 'distance' or 'travel_time'"
        )

    frontier = [
        (heuristic(graph, start, goal), start)
    ]

    parent = {start: None}
    g_score = {start: 0.0}
    visited = set()

    expanded = 0
    peak_frontier = 1

    while frontier:
        frontier.sort(key=lambda item: item[0])
        _, current = frontier.pop(0)

        if current in visited:
            continue

        visited.add(current)
        expanded += 1

        if current == goal:
            path = []
            node = goal

            while node is not None:
                path.append(node)
                node = parent[node]

            path.reverse()

            return path, expanded, peak_frontier

        for edge in graph.neighbors(current):
            neighbor = edge.target

            if blocked_edges:
                blocked = False

                for blocked_edge in blocked_edges:
                    a, b = blocked_edge

                    if (
                        (current == a and neighbor == b)
                        or
                        (current == b and neighbor == a)
                    ):
                        blocked = True
                        break

                if blocked:
                    continue

            if neighbor in visited:
                continue

            if accessible_only and not edge.accessible:
                continue

            step_cost = getattr(edge, edge_weight)

            tentative_g = g_score[current] + step_cost

            if (
                neighbor not in g_score
                or tentative_g < g_score[neighbor]
            ):
                g_score[neighbor] = tentative_g
                parent[neighbor] = current

                h = heuristic(
                    graph,
                    neighbor,
                    goal
                )

                f = tentative_g + h

                frontier.append((f, neighbor))
                peak_frontier = max(peak_frontier, len(frontier))

    return None, expanded, peak_frontier

def benchmark_astar(
        graph,
        start,
        goal,
        heuristic,
        edge_weight="distance",
        blocked_edges=None,
        accessible_only=False
):
    start_time = time.perf_counter()

    path, expanded, peak_frontier = astar(
        graph,
        start,
        goal,
        heuristic,
        edge_weight,
        blocked_edges,
        accessible_only
    )

    runtime = time.perf_counter() - start_time

    if path is None:
        cost = None
    else:
        cost = 0
        for a, b in zip(path, path[1:]):
            for edge in graph.neighbors(a):
                if edge.target == b:
                    cost += getattr(edge, edge_weight)
                    break

    return {
        "path": path,
        "expanded": expanded,
        "peak_frontier": peak_frontier,
        "cost": cost,
        "runtime": runtime
    }

def benchmark_search(
    search_function,
    graph,
    start,
    goal,
    heuristic=None,
):
    import time

    start_time = time.perf_counter()

    if heuristic is None:
        path, expanded, peak_frontier = search_function(
            graph,
            start,
            goal
        )
    else:
        path, expanded, peak_frontier = search_function(
            graph,
            start,
            goal,
            heuristic
        )

    runtime = time.perf_counter() - start_time

    if path is None:
        cost = None
    else:
        cost = 0

        for a, b in zip(path, path[1:]):
            for edge in graph.neighbors(a):
                if edge.target == b:
                    cost += edge.distance
                    break

    return {
        "path": path,
        "expanded": expanded,
        "cost": cost,
        "peak_frontier": peak_frontier,
        "runtime": runtime,
    }

def check_euclidean_consistency(graph, goal_id):
    violations = []

    for node_id in graph._nodes:
        h_node = euclidean_heuristic(
            graph,
            node_id,
            goal_id
        )

        for edge in graph.neighbors(node_id):
            neighbor_id = edge.target

            h_neighbor = euclidean_heuristic(
                graph,
                neighbor_id,
                goal_id
            )

            edge_cost = edge.distance

            if h_node > edge_cost + h_neighbor + 1e-9:
                violations.append({
                    "node": node_id,
                    "neighbor": neighbor_id,
                    "h(node)": h_node,
                    "edge_cost": edge_cost,
                    "h(neighbor)": h_neighbor,
                    "difference":
                        h_node - (edge_cost + h_neighbor)
                })

    return violations