"""Pathfinding algorithms and evaluation infrastructure.

Provides the shared SearchResult data container, path validation and
reconstruction utilities, heuristic-guided and unweighted graph search algorithms
(Dijkstra, A*, BFS, Greedy Best-First), and a unified algorithm dispatcher.
"""

from collections import deque
from dataclasses import dataclass
import heapq
import itertools
import math
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import warnings

from src.grid import Grid, MUD, Pos, SQRT2, WALL
from src.grid import (
    add_mud_patches,
    generate_mud_barrier_grid,
    generate_random_grid,
    generate_trap_grid,
    generate_unsolvable_grid,
)
from src.heuristics import default_heuristic_name, get_heuristic


@dataclass
class SearchResult:
    """Standardized result container for graph search algorithms.

    Attributes:
        algorithm: Name of the executed algorithm (e.g. 'Dijkstra', 'A*').
        found: True if a path from start to goal was found.
        path: Ordered list of coordinates from start to goal. Empty if not found.
        path_steps: Number of transition steps (len(path) - 1). 0 if not found.
        path_cost: Total accumulated movement cost. inf if not found.
        nodes_expanded: Number of unique nodes popped and processed.
        nodes_generated: Total nodes pushed onto the priority queue / frontier.
        max_frontier_size: Peak size of the open set / priority queue.
        runtime_ms: Core search duration in milliseconds.
        expansion_order: Sequence of coordinates expanded during search.
        heuristic: Name of heuristic used, if applicable.
        allow_diagonal: Whether diagonal moves were enabled.
    """

    algorithm: str
    found: bool
    path: List[Pos]
    path_steps: int
    path_cost: float
    nodes_expanded: int
    nodes_generated: int
    max_frontier_size: int
    runtime_ms: float
    expansion_order: List[Pos]
    heuristic: Optional[str] = None
    allow_diagonal: bool = False

    def expanded_density_pct(self, grid: Grid) -> float:
        """Calculate percentage of walkable grid cells expanded during search.

        Formula: 100 * nodes_expanded / grid.free_cell_count().

        Args:
            grid: Grid instance search was performed on.

        Returns:
            Expansion density as a percentage in [0.0, 100.0].
        """
        free = grid.free_cell_count()
        if free == 0:
            return 0.0
        return 100.0 * float(self.nodes_expanded) / float(free)

    def to_dict(self, grid: Optional[Grid] = None) -> Dict[str, Any]:
        """Convert result into a flat dictionary suitable for DataFrame row creation.

        Excludes large array fields (path, expansion_order). Sets path_steps and
        path_cost to float('nan') when found is False.

        Args:
            grid: Optional Grid to compute expanded_density_pct.

        Returns:
            Flat dictionary of search metrics.
        """
        data: Dict[str, Any] = {
            "algorithm": self.algorithm,
            "heuristic": self.heuristic,
            "allow_diagonal": self.allow_diagonal,
            "found": self.found,
            "path_steps": self.path_steps if self.found else float("nan"),
            "path_cost": self.path_cost if self.found else float("nan"),
            "nodes_expanded": self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "max_frontier_size": self.max_frontier_size,
            "runtime_ms": self.runtime_ms,
        }
        if grid is not None:
            data["expanded_density_pct"] = self.expanded_density_pct(grid)
        return data


def _path_cost(grid: Grid, path: List[Pos]) -> float:
    """Calculate the total terrain-weighted cost along a reconstructed path.

    Args:
        grid: Grid on which path transitions take place.
        path: Ordered list of coordinates from start to goal.

    Returns:
        Summed movement cost along consecutive path coordinates.
    """
    if len(path) <= 1:
        return 0.0
    cost = 0.0
    for i in range(1, len(path)):
        cost += grid.move_cost(path[i - 1], path[i])
    return cost


def _build_result(
    grid: Grid,
    algorithm: str,
    found: bool,
    came_from: Dict[Pos, Pos],
    nodes_expanded: int,
    nodes_generated: int,
    max_frontier_size: int,
    runtime_ms: float,
    expansion_order: List[Pos],
    heuristic: Optional[str] = None,
    allow_diagonal: bool = False,
    known_cost: Optional[float] = None,
) -> SearchResult:
    """Assemble a standardized SearchResult container.

    Args:
        grid: Grid searched upon.
        algorithm: Algorithm name.
        found: Search success status.
        came_from: Predecessor links for path reconstruction.
        nodes_expanded: Nodes popped from frontier.
        nodes_generated: Nodes enqueued/pushed to frontier.
        max_frontier_size: Peak frontier capacity.
        runtime_ms: Core search duration.
        expansion_order: Ordered list of expanded positions.
        heuristic: Heuristic name if applied.
        allow_diagonal: Diagonal movement flag.
        known_cost: Pre-tracked optimal cost (e.g. g_score[goal]) or None to compute via _path_cost.

    Returns:
        Constructed SearchResult.
    """
    if found:
        path = reconstruct_path(came_from, grid.start, grid.goal)
        path_steps = len(path) - 1
        path_cost = known_cost if known_cost is not None else _path_cost(grid, path)
    else:
        path = []
        path_steps = 0
        path_cost = float("inf")

    return SearchResult(
        algorithm=algorithm,
        found=found,
        path=path,
        path_steps=path_steps,
        path_cost=path_cost,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
        runtime_ms=runtime_ms,
        expansion_order=expansion_order,
        heuristic=heuristic,
        allow_diagonal=allow_diagonal,
    )


def reconstruct_path(
    came_from: Dict[Pos, Pos],
    start: Pos,
    goal: Pos,
) -> List[Pos]:
    """Iteratively reconstruct path from start to goal walking backwards from goal.

    Uses an explicit loop without recursion to support large grid depths.

    Args:
        came_from: Mapping from child node to its predecessor parent node.
        start: Origin coordinate.
        goal: Destination coordinate.

    Returns:
        List of coordinates forming the path from start to goal (inclusive).

    Raises:
        ValueError: If a parent link is missing (broken chain) or a loop is encountered.
    """
    if start == goal:
        return [start]

    path: List[Pos] = [goal]
    curr: Pos = goal
    visited: set[Pos] = {goal}

    while curr != start:
        if curr not in came_from:
            raise ValueError(
                f"Broken path chain: node {curr} has no parent in came_from"
            )
        curr = came_from[curr]
        if curr in visited:
            raise ValueError(f"Loop detected in came_from chain at node {curr}")
        visited.add(curr)
        path.append(curr)

    path.reverse()
    return path


def validate_path(
    grid: Grid,
    path: List[Pos],
    allow_diagonal: bool = False,
) -> float:
    """Validate path feasibility and recompute total transition cost independently.

    Verifies that:
      1. path is non-empty.
      2. path[0] == grid.start and path[-1] == grid.goal.
      3. Each consecutive pair (path[i-1], path[i]) is a valid neighbor under
         grid.neighbors(prev, allow_diagonal) (no walls, no teleportation, no corner-cutting).

    Args:
        grid: Grid instance path was searched on.
        path: Candidate coordinate sequence.
        allow_diagonal: Whether diagonal moves are permitted.

    Returns:
        Total path traversal cost recomputed via grid.move_cost.

    Raises:
        ValueError: On empty path, mismatched endpoints, or illegal transitions.
    """
    if not path:
        raise ValueError("Cannot validate an empty path")

    if path[0] != grid.start:
        raise ValueError(
            f"Path start {path[0]} does not match grid start {grid.start}"
        )

    if path[-1] != grid.goal:
        raise ValueError(
            f"Path goal {path[-1]} does not match grid goal {grid.goal}"
        )

    if len(path) == 1:
        return 0.0

    total_cost = 0.0
    for i in range(1, len(path)):
        prev = path[i - 1]
        curr = path[i]
        valid_neighbors = grid.neighbors(prev, allow_diagonal=allow_diagonal)
        if curr not in valid_neighbors:
            raise ValueError(
                f"Invalid transition at step {i}: {curr} is not a valid neighbor of "
                f"{prev} (allow_diagonal={allow_diagonal})"
            )
        total_cost += grid.move_cost(prev, curr)

    return total_cost


def dijkstra(
    grid: Grid,
    allow_diagonal: bool = False,
    record_order: bool = True,
) -> SearchResult:
    """Find the lowest-cost path from start to goal using Dijkstra's algorithm.

    Implementation details:
      - Uses heapq priority queue with entries (g_cost, tie_counter, pos).
      - tie_counter from itertools.count() ensures deterministic FIFO tie-breaking
        without coordinate tuple comparisons.
      - Lazy deletion: stale frontier entries for already-closed nodes are discarded on pop.
      - Goal check strictly on node pop: with non-uniform terrain costs, the first time
        a goal node is pushed to the frontier is not guaranteed to be the lowest-cost path.
      - Tracks nodes_generated, nodes_expanded, and peak frontier size.
      - Only the core search loop is measured by time.perf_counter() (in ms);
        path reconstruction overhead is excluded to isolate search algorithm performance.

    Args:
        grid: Grid on which to execute search.
        allow_diagonal: If True, allows 8-direction movement.
        record_order: If True, logs node expansion order for animation.

    Returns:
        A SearchResult instance capturing search metrics and found path.
    """
    tie_counter = itertools.count()
    heap: List[Tuple[float, int, Pos]] = [(0.0, next(tie_counter), grid.start)]
    g_score: Dict[Pos, float] = {grid.start: 0.0}
    came_from: Dict[Pos, Pos] = {}
    closed: set[Pos] = set()

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1
    expansion_order: List[Pos] = []

    t0 = time.perf_counter()
    found = False

    while heap:
        g, _, pos = heapq.heappop(heap)

        # Lazy deletion: ignore stale entries already closed
        if pos in closed:
            continue

        closed.add(pos)
        nodes_expanded += 1
        if record_order:
            expansion_order.append(pos)

        # The goal test is performed when the goal is TAKEN OFF the frontier (popped/dequeued),
        # never when it is generated. This keeps nodes_expanded semantically identical across all algorithms.
        if pos == grid.goal:
            found = True
            break

        for nb in grid.neighbors(pos, allow_diagonal=allow_diagonal):
            if nb in closed:
                continue

            new_g = g + grid.move_cost(pos, nb)
            if new_g < g_score.get(nb, float("inf")):
                g_score[nb] = new_g
                came_from[nb] = pos
                heapq.heappush(heap, (new_g, next(tie_counter), nb))
                nodes_generated += 1
                if len(heap) > max_frontier_size:
                    max_frontier_size = len(heap)

    t1 = time.perf_counter()
    runtime_ms = (t1 - t0) * 1000.0

    return _build_result(
        grid=grid,
        algorithm="Dijkstra",
        found=found,
        came_from=came_from,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
        runtime_ms=runtime_ms,
        expansion_order=expansion_order,
        heuristic=None,
        allow_diagonal=allow_diagonal,
        known_cost=g_score[grid.goal] if found else None,
    )


def astar(
    grid: Grid,
    allow_diagonal: bool = False,
    heuristic: Optional[str] = None,
    record_order: bool = True,
) -> SearchResult:
    """Find the lowest-cost path from start to goal using A* search.

    Combines true path cost g with heuristic estimate h (f = g + h).
    Uses lazy deletion without a closed set (skipping popped entries when g > g_score[pos]),
    which preserves correctness even with admissible-but-inconsistent heuristics.

    Args:
        grid: Grid on which to execute search.
        allow_diagonal: If True, allows 8-direction movement.
        heuristic: Name of heuristic function to use. Defaults to default_heuristic_name(allow_diagonal).
        record_order: If True, logs node expansion sequence.

    Returns:
        A SearchResult instance capturing search metrics and found path.
    """
    h_name = heuristic if heuristic is not None else default_heuristic_name(allow_diagonal)
    h_func = get_heuristic(h_name)

    if h_name == "manhattan" and allow_diagonal:
        warnings.warn(
            "Manhattan distance is not admissible with diagonal movement and may yield sub-optimal paths.",
            UserWarning,
            stacklevel=2,
        )

    tie_counter = itertools.count()
    h0 = h_func(grid.start, grid.goal)
    # Heap entries: (f, tie_counter, g, pos)
    heap: List[Tuple[float, int, float, Pos]] = [(h0, next(tie_counter), 0.0, grid.start)]
    g_score: Dict[Pos, float] = {grid.start: 0.0}
    came_from: Dict[Pos, Pos] = {}

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1
    expansion_order: List[Pos] = []

    t0 = time.perf_counter()
    found = False

    while heap:
        f, _, g, pos = heapq.heappop(heap)

        # Stale entry check: ignore if a cheaper path to pos has already been processed
        if g > g_score[pos]:
            continue

        nodes_expanded += 1
        if record_order:
            expansion_order.append(pos)

        # The goal test is performed when the goal is TAKEN OFF the frontier (popped/dequeued),
        # never when it is generated. This keeps nodes_expanded semantically identical across all algorithms.
        if pos == grid.goal:
            found = True
            break

        for nb in grid.neighbors(pos, allow_diagonal=allow_diagonal):
            new_g = g + grid.move_cost(pos, nb)
            if new_g < g_score.get(nb, float("inf")):
                g_score[nb] = new_g
                came_from[nb] = pos
                h_nb = h_func(nb, grid.goal)
                f_nb = new_g + h_nb
                # Ties are broken only by tie_counter; preferring lower h on ties would expand fewer nodes,
                # left as a later experiment.
                heapq.heappush(heap, (f_nb, next(tie_counter), new_g, nb))
                nodes_generated += 1
                if len(heap) > max_frontier_size:
                    max_frontier_size = len(heap)

    t1 = time.perf_counter()
    runtime_ms = (t1 - t0) * 1000.0

    return _build_result(
        grid=grid,
        algorithm="A*",
        found=found,
        came_from=came_from,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
        runtime_ms=runtime_ms,
        expansion_order=expansion_order,
        heuristic=h_name,
        allow_diagonal=allow_diagonal,
        known_cost=g_score[grid.goal] if found else None,
    )


def bfs(
    grid: Grid,
    allow_diagonal: bool = False,
    record_order: bool = True,
) -> SearchResult:
    """Find the path minimizing the number of steps using Breadth-First Search (BFS).

    BFS minimizes the NUMBER OF MOVES and ignores terrain cost, so on weighted grids
    its path_cost can be higher than optimal; with allow_diagonal every move still counts
    as one step.

    Args:
        grid: Grid on which to execute search.
        allow_diagonal: If True, allows 8-direction movement.
        record_order: If True, logs node expansion sequence.

    Returns:
        A SearchResult instance capturing search metrics and found path.
    """
    queue: deque[Pos] = deque([grid.start])
    discovered: Set[Pos] = {grid.start}
    came_from: Dict[Pos, Pos] = {}

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1
    expansion_order: List[Pos] = []

    t0 = time.perf_counter()
    found = False

    while queue:
        pos = queue.popleft()
        nodes_expanded += 1
        if record_order:
            expansion_order.append(pos)

        # The goal test is performed when the goal is TAKEN OFF the frontier (popped/dequeued),
        # never when it is generated. This keeps nodes_expanded semantically identical across all algorithms.
        if pos == grid.goal:
            found = True
            break

        for nb in grid.neighbors(pos, allow_diagonal=allow_diagonal):
            if nb not in discovered:
                discovered.add(nb)
                came_from[nb] = pos
                queue.append(nb)
                nodes_generated += 1
                if len(queue) > max_frontier_size:
                    max_frontier_size = len(queue)

    t1 = time.perf_counter()
    runtime_ms = (t1 - t0) * 1000.0

    return _build_result(
        grid=grid,
        algorithm="BFS",
        found=found,
        came_from=came_from,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
        runtime_ms=runtime_ms,
        expansion_order=expansion_order,
        heuristic=None,
        allow_diagonal=allow_diagonal,
        known_cost=None,
    )


def greedy_best_first(
    grid: Grid,
    allow_diagonal: bool = False,
    heuristic: Optional[str] = None,
    record_order: bool = True,
) -> SearchResult:
    """Find a path by greedily expanding nodes closest to the goal by heuristic estimate.

    Complete on finite grids because of the discovered set, but neither cost-optimal
    nor step-optimal; no admissibility warning is needed here.

    Args:
        grid: Grid on which to execute search.
        allow_diagonal: If True, allows 8-direction movement.
        heuristic: Name of heuristic function to use. Defaults to default_heuristic_name(allow_diagonal).
        record_order: If True, logs node expansion sequence.

    Returns:
        A SearchResult instance capturing search metrics and found path.
    """
    h_name = heuristic if heuristic is not None else default_heuristic_name(allow_diagonal)
    h_func = get_heuristic(h_name)

    tie_counter = itertools.count()
    h0 = h_func(grid.start, grid.goal)
    heap: List[Tuple[float, int, Pos]] = [(h0, next(tie_counter), grid.start)]
    discovered: Set[Pos] = {grid.start}
    came_from: Dict[Pos, Pos] = {}

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1
    expansion_order: List[Pos] = []

    t0 = time.perf_counter()
    found = False

    while heap:
        _, _, pos = heapq.heappop(heap)
        nodes_expanded += 1
        if record_order:
            expansion_order.append(pos)

        # The goal test is performed when the goal is TAKEN OFF the frontier (popped/dequeued),
        # never when it is generated. This keeps nodes_expanded semantically identical across all algorithms.
        if pos == grid.goal:
            found = True
            break

        for nb in grid.neighbors(pos, allow_diagonal=allow_diagonal):
            if nb not in discovered:
                discovered.add(nb)
                came_from[nb] = pos
                h_nb = h_func(nb, grid.goal)
                heapq.heappush(heap, (h_nb, next(tie_counter), nb))
                nodes_generated += 1
                if len(heap) > max_frontier_size:
                    max_frontier_size = len(heap)

    t1 = time.perf_counter()
    runtime_ms = (t1 - t0) * 1000.0

    return _build_result(
        grid=grid,
        algorithm="Greedy Best-First",
        found=found,
        came_from=came_from,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
        runtime_ms=runtime_ms,
        expansion_order=expansion_order,
        heuristic=h_name,
        allow_diagonal=allow_diagonal,
        known_cost=None,
    )


# Registry of available pathfinding algorithms
ALGORITHMS: Dict[str, Callable[..., SearchResult]] = {
    "Dijkstra": dijkstra,
    "A*": astar,
    "BFS": bfs,
    "Greedy Best-First": greedy_best_first,
}

HEURISTIC_ALGORITHMS: Set[str] = {"A*", "Greedy Best-First"}

ALGORITHM_INFO: Dict[str, Dict[str, Any]] = {
    "Dijkstra": {
        "description": "Explores nodes in order of accumulated path cost from the start.",
        "uses_heuristic": False,
        "optimal_cost": "yes",
        "optimal_steps": "no",
    },
    "A*": {
        "description": "Informed search balancing accumulated cost and heuristic estimate to goal.",
        "uses_heuristic": True,
        "optimal_cost": "yes if heuristic admissible",
        "optimal_steps": "no",
    },
    "BFS": {
        "description": "Unweighted level-by-level exploration minimizing number of moves.",
        "uses_heuristic": False,
        "optimal_cost": "no",
        "optimal_steps": "yes",
    },
    "Greedy Best-First": {
        "description": "Greedy search expanding nodes strictly closest to the goal by heuristic.",
        "uses_heuristic": True,
        "optimal_cost": "no",
        "optimal_steps": "no",
    },
}


def run_algorithm(
    name: str,
    grid: Grid,
    allow_diagonal: bool = False,
    heuristic: Optional[str] = None,
    record_order: bool = True,
) -> SearchResult:
    """Execute a registered pathfinding algorithm by name.

    Args:
        name: Name of algorithm ('Dijkstra', 'A*', 'BFS', 'Greedy Best-First').
        grid: Grid on which to execute search.
        allow_diagonal: If True, allows 8-direction movement.
        heuristic: Heuristic name (forwarded only to heuristic-based algorithms).
        record_order: If True, logs node expansion sequence.

    Returns:
        SearchResult instance.

    Raises:
        KeyError: If algorithm name is not recognized.
    """
    if name not in ALGORITHMS:
        valid_keys = ", ".join(repr(k) for k in ALGORITHMS.keys())
        raise KeyError(
            f"Unknown algorithm {name!r}. Available algorithms are: {valid_keys}"
        )

    algo_fn = ALGORITHMS[name]
    if name in HEURISTIC_ALGORITHMS:
        return algo_fn(
            grid=grid,
            allow_diagonal=allow_diagonal,
            heuristic=heuristic,
            record_order=record_order,
        )
    return algo_fn(
        grid=grid,
        allow_diagonal=allow_diagonal,
        record_order=record_order,
    )


# =====================================================================
# Self-Test Block
# =====================================================================

if __name__ == "__main__":
    import pandas as pd

    def _reference_bfs_steps(g: Grid) -> Optional[int]:
        """Test oracle: simple BFS returning minimal steps on unit-cost 4-dir grid."""
        if g.start == g.goal:
            return 0
        queue: deque[Tuple[Pos, int]] = deque([(g.start, 0)])
        visited: set[Pos] = {g.start}
        while queue:
            curr, steps = queue.popleft()
            if curr == g.goal:
                return steps
            for nb in g.neighbors(curr, allow_diagonal=False):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, steps + 1))
        return None

    def _render_search_ascii(g: Grid, res: SearchResult) -> str:
        """Render grid with search solution and expansions."""
        path_set = set(res.path)
        expanded_set = set(res.expansion_order)
        lines: List[str] = []
        for r in range(g.rows):
            row_chars: List[str] = []
            for c in range(g.cols):
                pos = (r, c)
                if pos == g.start:
                    row_chars.append("S")
                elif pos == g.goal:
                    row_chars.append("G")
                elif pos in path_set:
                    row_chars.append("*")
                elif pos in expanded_set:
                    row_chars.append("o")
                elif g.cells[r, c] == WALL:
                    row_chars.append("#")
                elif g.cells[r, c] == MUD:
                    row_chars.append("~")
                else:
                    row_chars.append(".")
            lines.append("".join(row_chars))
        return "\n".join(lines)

    print("Running Phase 2 regression tests...")

    # 1. Empty 5x5 grid: 4-dir -> steps 8, cost 8.0; 8-dir -> steps 4, cost 4*SQRT2
    g_empty = Grid(5, 5)
    res_empty_4 = dijkstra(g_empty, allow_diagonal=False)
    assert res_empty_4.found, "Test 1 failed: path not found on empty 5x5 (4-dir)"
    assert res_empty_4.path_steps == 8, f"Test 1 failed: expected 8 steps, got {res_empty_4.path_steps}"
    assert math.isclose(res_empty_4.path_cost, 8.0), f"Test 1 failed: expected cost 8.0, got {res_empty_4.path_cost}"

    res_empty_8 = dijkstra(g_empty, allow_diagonal=True)
    assert res_empty_8.found, "Test 1 failed: path not found on empty 5x5 (8-dir)"
    assert res_empty_8.path_steps == 4, f"Test 1 failed: expected 4 steps, got {res_empty_8.path_steps}"
    assert math.isclose(res_empty_8.path_cost, 4.0 * SQRT2), (
        f"Test 1 failed: expected cost {4.0 * SQRT2}, got {res_empty_8.path_cost}"
    )

    # 2. Trap grid 12x20 (4-dir): found, path_steps == 27
    g_trap = generate_trap_grid(12, 20)
    res_trap = dijkstra(g_trap, allow_diagonal=False)
    assert res_trap.found, "Test 2 failed: path not found on trap grid"
    assert res_trap.path_steps == 27, f"Test 2 failed: expected 27 steps, got {res_trap.path_steps}"

    # 3. Mud barrier grid 15x25 (4-dir): found, path_cost == 36.0, no MUD in path
    g_mud = generate_mud_barrier_grid(15, 25)
    res_mud = dijkstra(g_mud, allow_diagonal=False)
    assert res_mud.found, "Test 3 failed: path not found on mud barrier grid"
    assert math.isclose(res_mud.path_cost, 36.0), f"Test 3 failed: expected cost 36.0, got {res_mud.path_cost}"
    assert all(g_mud.cells[r, c] != MUD for r, c in res_mud.path), (
        "Test 3 failed: optimal route must avoid mud by taking free top corridor"
    )

    # 4. Unsolvable grid: found False, path == [], path_cost inf, nodes_expanded > 0
    g_unsolv = generate_unsolvable_grid(12, 20, seed=42)
    res_unsolv = dijkstra(g_unsolv)
    assert not res_unsolv.found, "Test 4 failed: unsolvable grid reported path found"
    assert res_unsolv.path == [], f"Test 4 failed: expected empty path, got {res_unsolv.path}"
    assert math.isinf(res_unsolv.path_cost), f"Test 4 failed: expected inf cost, got {res_unsolv.path_cost}"
    assert res_unsolv.nodes_expanded > 0, "Test 4 failed: expected nodes_expanded > 0"

    # 5. start == goal on 3x3 grid: steps 0, cost 0.0
    g_same = Grid(3, 3, start=(1, 1), goal=(1, 1))
    res_same = dijkstra(g_same)
    assert res_same.found, "Test 5 failed: start==goal reported not found"
    assert res_same.path_steps == 0, f"Test 5 failed: expected 0 steps, got {res_same.path_steps}"
    assert math.isclose(res_same.path_cost, 0.0), f"Test 5 failed: expected cost 0.0, got {res_same.path_cost}"
    assert res_same.path == [(1, 1)], f"Test 5 failed: expected [(1, 1)], got {res_same.path}"

    # 6. Oracle comparison with reference BFS for seeds 0..19
    for seed in range(20):
        density = 0.2 + 0.1 * (seed % 2)
        g_rand = generate_random_grid(20, 20, density, seed=seed)
        res_rand = dijkstra(g_rand, allow_diagonal=False)
        oracle_steps = _reference_bfs_steps(g_rand)
        assert oracle_steps is not None, f"Test 6 failed: oracle BFS found no path on seed {seed}"
        assert res_rand.path_steps == oracle_steps, (
            f"Test 6 failed on seed {seed}: Dijkstra steps ({res_rand.path_steps}) "
            f"!= Oracle BFS steps ({oracle_steps})"
        )

    # 7. validate_path succeeds for every found result above and cost matches
    found_results = [
        (g_empty, res_empty_4, False),
        (g_empty, res_empty_8, True),
        (g_trap, res_trap, False),
        (g_mud, res_mud, False),
        (g_same, res_same, False),
    ]
    for g_test, r_test, diag in found_results:
        recomputed_cost = validate_path(g_test, r_test.path, allow_diagonal=diag)
        assert math.isclose(recomputed_cost, r_test.path_cost), (
            f"Test 7 failed: recomputed cost {recomputed_cost} != result cost {r_test.path_cost}"
        )

    # 8. Determinism: two identical runs give identical path and expansion_order
    res_trap_run1 = dijkstra(g_trap, allow_diagonal=False, record_order=True)
    res_trap_run2 = dijkstra(g_trap, allow_diagonal=False, record_order=True)
    assert res_trap_run1.path == res_trap_run2.path, "Test 8 failed: non-deterministic paths"
    assert res_trap_run1.expansion_order == res_trap_run2.expansion_order, (
        "Test 8 failed: non-deterministic expansion orders"
    )

    # 9. record_order=False gives expansion_order == [] while every counter matches
    res_trap_no_order = dijkstra(g_trap, allow_diagonal=False, record_order=False)
    assert res_trap_no_order.expansion_order == [], (
        "Test 9 failed: expansion_order not empty when record_order=False"
    )
    assert res_trap_no_order.nodes_expanded == res_trap_run1.nodes_expanded, (
        "Test 9 failed: nodes_expanded mismatch"
    )
    assert res_trap_no_order.nodes_generated == res_trap_run1.nodes_generated, (
        "Test 9 failed: nodes_generated mismatch"
    )
    assert res_trap_no_order.max_frontier_size == res_trap_run1.max_frontier_size, (
        "Test 9 failed: max_frontier_size mismatch"
    )
    assert res_trap_no_order.path_steps == res_trap_run1.path_steps, (
        "Test 9 failed: path_steps mismatch"
    )
    assert math.isclose(res_trap_no_order.path_cost, res_trap_run1.path_cost), (
        "Test 9 failed: path_cost mismatch"
    )
    assert res_trap_no_order.path == res_trap_run1.path, (
        "Test 9 failed: path mismatch with record_order=False"
    )

    # 10. ASCII rendering of the trap-grid result + to_dict output
    print("\n--- Trap Grid Solution Rendering (Phase 2 Dijkstra) ---")
    print(_render_search_ascii(g_trap, res_trap))
    print("\n--- Trap Grid Search Metrics (to_dict) ---")
    metrics_dict = res_trap.to_dict(g_trap)
    for k, v in metrics_dict.items():
        print(f"  {k}: {v}")

    print("\nPhase 2 regression tests passed! Running Phase 3 extended tests...\n")

    # =================================================================
    # Phase 3 Extended Tests
    # =================================================================

    # Test A: Optimality agreement, 4-direction (seeds 0..29)
    test_a_grids: List[Grid] = []
    densities = [0.10, 0.20, 0.30, 0.40]
    for seed in range(30):
        dens = densities[seed % 4]
        grid_a = generate_random_grid(25, 25, dens, seed=seed)
        if seed % 2 == 1:
            grid_a = add_mud_patches(grid_a, 4, 4, seed=seed)
        test_a_grids.append(grid_a)

        r_dijk = dijkstra(grid_a, allow_diagonal=False)
        r_a_manh = astar(grid_a, allow_diagonal=False, heuristic="manhattan")
        r_a_eucl = astar(grid_a, allow_diagonal=False, heuristic="euclidean")
        r_a_zero = astar(grid_a, allow_diagonal=False, heuristic="zero")

        assert r_dijk.found == r_a_manh.found == r_a_eucl.found == r_a_zero.found, (
            f"Test A failed on seed {seed}: reachability disagreement"
        )
        if r_dijk.found:
            assert math.isclose(r_dijk.path_cost, r_a_manh.path_cost), (
                f"Test A failed on seed {seed}: Dijkstra cost ({r_dijk.path_cost}) "
                f"!= A*(manhattan) cost ({r_a_manh.path_cost})"
            )
            assert math.isclose(r_dijk.path_cost, r_a_eucl.path_cost), (
                f"Test A failed on seed {seed}: Dijkstra cost ({r_dijk.path_cost}) "
                f"!= A*(euclidean) cost ({r_a_eucl.path_cost})"
            )
            assert math.isclose(r_dijk.path_cost, r_a_zero.path_cost), (
                f"Test A failed on seed {seed}: Dijkstra cost ({r_dijk.path_cost}) "
                f"!= A*(zero) cost ({r_a_zero.path_cost})"
            )

    # Test B: A*(zero) reproduces Dijkstra (same cost AND nodes_expanded on Test A grids)
    for idx, grid_b in enumerate(test_a_grids):
        r_dijk_b = dijkstra(grid_b, allow_diagonal=False)
        r_zero_b = astar(grid_b, allow_diagonal=False, heuristic="zero")
        assert math.isclose(r_dijk_b.path_cost, r_zero_b.path_cost), (
            f"Test B failed on grid {idx}: cost mismatch"
        )
        assert r_dijk_b.nodes_expanded == r_zero_b.nodes_expanded, (
            f"Test B failed on grid {idx}: Dijkstra expanded ({r_dijk_b.nodes_expanded}) "
            f"!= A*(zero) expanded ({r_zero_b.nodes_expanded})"
        )

    # Test C: Optimality agreement, 8-direction (seeds 0..14)
    test_c_grids: List[Grid] = []
    for seed in range(15):
        grid_c = generate_random_grid(25, 25, 0.25, seed=seed)
        test_c_grids.append(grid_c)

        r_dijk_c = dijkstra(grid_c, allow_diagonal=True)
        r_a_oct = astar(grid_c, allow_diagonal=True, heuristic="octile")
        r_a_eucl_c = astar(grid_c, allow_diagonal=True, heuristic="euclidean")

        assert math.isclose(r_dijk_c.path_cost, r_a_oct.path_cost), (
            f"Test C failed on seed {seed}: Dijkstra ({r_dijk_c.path_cost}) "
            f"!= A*(octile) ({r_a_oct.path_cost})"
        )
        assert math.isclose(r_dijk_c.path_cost, r_a_eucl_c.path_cost), (
            f"Test C failed on seed {seed}: Dijkstra ({r_dijk_c.path_cost}) "
            f"!= A*(euclidean) ({r_a_eucl_c.path_cost})"
        )

    # Test D: The Manhattan + diagonal warning
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        astar(g_empty, allow_diagonal=True, heuristic="manhattan")
        assert len(caught_warnings) == 1, (
            f"Test D failed: expected 1 warning for Manhattan+diagonal, got {len(caught_warnings)}"
        )
        assert issubclass(caught_warnings[-1].category, UserWarning), (
            f"Test D failed: warning is not UserWarning: {caught_warnings[-1].category}"
        )

    with warnings.catch_warnings(record=True) as caught_warnings_octile:
        warnings.simplefilter("always")
        astar(g_empty, allow_diagonal=True, heuristic="octile")
        assert len(caught_warnings_octile) == 0, (
            f"Test D failed: unexpected warning emitted for Octile+diagonal: {caught_warnings_octile}"
        )

    # Test E: BFS on mud-free 4-direction grids (seeds 0..29)
    for seed in range(30):
        grid_e = generate_random_grid(20, 20, 0.2, seed=seed)
        r_bfs_e = bfs(grid_e, allow_diagonal=False)
        r_dijk_e = dijkstra(grid_e, allow_diagonal=False)
        assert r_bfs_e.path_steps == r_dijk_e.path_steps, (
            f"Test E failed on seed {seed}: BFS steps ({r_bfs_e.path_steps}) "
            f"!= Dijkstra steps ({r_dijk_e.path_steps})"
        )
        assert math.isclose(r_bfs_e.path_cost, r_dijk_e.path_cost), (
            f"Test E failed on seed {seed}: BFS cost ({r_bfs_e.path_cost}) "
            f"!= Dijkstra cost ({r_dijk_e.path_cost})"
        )

    # Test F: Greedy sanity on every grid from tests A and C
    for idx, grid_fa in enumerate(test_a_grids):
        r_greedy_fa = greedy_best_first(grid_fa, allow_diagonal=False, heuristic="manhattan")
        r_dijk_fa = dijkstra(grid_fa, allow_diagonal=False)
        if r_greedy_fa.found:
            val_cost = validate_path(grid_fa, r_greedy_fa.path, allow_diagonal=False)
            assert math.isclose(val_cost, r_greedy_fa.path_cost), (
                f"Test F failed on Test-A grid {idx}: validate_path cost mismatch"
            )
            assert r_greedy_fa.path_cost >= r_dijk_fa.path_cost - 1e-9, (
                f"Test F failed on Test-A grid {idx}: Greedy path cheaper than optimal Dijkstra"
            )

    for idx, grid_fc in enumerate(test_c_grids):
        r_greedy_fc = greedy_best_first(grid_fc, allow_diagonal=True, heuristic="octile")
        r_dijk_fc = dijkstra(grid_fc, allow_diagonal=True)
        if r_greedy_fc.found:
            val_cost_c = validate_path(grid_fc, r_greedy_fc.path, allow_diagonal=True)
            assert math.isclose(val_cost_c, r_greedy_fc.path_cost), (
                f"Test F failed on Test-C grid {idx}: validate_path cost mismatch"
            )
            assert r_greedy_fc.path_cost >= r_dijk_fc.path_cost - 1e-9, (
                f"Test F failed on Test-C grid {idx}: Greedy path cheaper than optimal Dijkstra"
            )

    # Test G: No re-expansion with consistent heuristics (Manhattan & Euclidean on Test A)
    for idx, grid_g in enumerate(test_a_grids):
        r_a_m = astar(grid_g, allow_diagonal=False, heuristic="manhattan")
        r_a_e = astar(grid_g, allow_diagonal=False, heuristic="euclidean")
        assert r_a_m.nodes_expanded == len(set(r_a_m.expansion_order)), (
            f"Test G failed on grid {idx}: A*(manhattan) re-expanded nodes"
        )
        assert r_a_e.nodes_expanded == len(set(r_a_e.expansion_order)), (
            f"Test G failed on grid {idx}: A*(euclidean) re-expanded nodes"
        )

    # Test H: Mud barrier 15x25 (4-direction)
    g_mb = generate_mud_barrier_grid(15, 25)
    r_h_dijk = dijkstra(g_mb, allow_diagonal=False)
    r_h_astar = astar(g_mb, allow_diagonal=False, heuristic="manhattan")
    r_h_bfs = bfs(g_mb, allow_diagonal=False)
    r_h_greedy = greedy_best_first(g_mb, allow_diagonal=False, heuristic="manhattan")

    assert math.isclose(r_h_dijk.path_cost, 36.0) and r_h_dijk.path_steps == 36, (
        f"Test H failed: Dijkstra cost={r_h_dijk.path_cost}, steps={r_h_dijk.path_steps}"
    )
    assert math.isclose(r_h_astar.path_cost, 36.0), (
        f"Test H failed: A*(manhattan) cost={r_h_astar.path_cost}"
    )
    assert r_h_bfs.path_steps == 22 and math.isclose(r_h_bfs.path_cost, 50.0), (
        f"Test H failed: BFS steps={r_h_bfs.path_steps}, cost={r_h_bfs.path_cost}"
    )
    assert r_h_greedy.path_steps == 22 and math.isclose(r_h_greedy.path_cost, 50.0), (
        f"Test H failed: Greedy steps={r_h_greedy.path_steps}, cost={r_h_greedy.path_cost}"
    )
    assert r_h_greedy.nodes_expanded < r_h_astar.nodes_expanded < r_h_dijk.nodes_expanded, (
        f"Test H failed: expansion order violation: greedy ({r_h_greedy.nodes_expanded}) "
        f"< astar ({r_h_astar.nodes_expanded}) < dijkstra ({r_h_dijk.nodes_expanded})"
    )

    # Test I: Trap grid 12x20 (4-direction)
    g_tr = generate_trap_grid(12, 20)
    r_i_dijk = dijkstra(g_tr, allow_diagonal=False)
    r_i_astar = astar(g_tr, allow_diagonal=False, heuristic="manhattan")
    r_i_bfs = bfs(g_tr, allow_diagonal=False)
    r_i_greedy = greedy_best_first(g_tr, allow_diagonal=False, heuristic="manhattan")

    assert r_i_dijk.path_steps == 27, f"Test I failed: Dijkstra steps {r_i_dijk.path_steps} != 27"
    assert r_i_astar.path_steps == 27, f"Test I failed: A* steps {r_i_astar.path_steps} != 27"
    assert r_i_bfs.path_steps == 27, f"Test I failed: BFS steps {r_i_bfs.path_steps} != 27"
    assert r_i_greedy.path_steps == 27, f"Test I failed: Greedy steps {r_i_greedy.path_steps} != 27"

    assert r_i_greedy.nodes_expanded < r_i_astar.nodes_expanded < r_i_dijk.nodes_expanded, (
        f"Test I failed: expansion order violation: greedy ({r_i_greedy.nodes_expanded}) "
        f"< astar ({r_i_astar.nodes_expanded}) < dijkstra ({r_i_dijk.nodes_expanded})"
    )

    rm = 12 // 2
    half = max(2, 12 // 2 - 2)
    top_i, bot_i = rm - half, rm + half
    a_i, b_i = 20 // 2 - 4, 20 // 2 + 2
    wasted_inside = sum(
        1 for (r, c) in r_i_greedy.expansion_order
        if (top_i < r < bot_i and a_i <= c < b_i)
    )
    assert wasted_inside > 0, f"Test I failed: expected Greedy expansions inside cup > 0, got {wasted_inside}"
    print(f"Greedy wasted expansions inside the cup: {wasted_inside}")

    # Test J: Unsolvable grid
    g_unsolv_j = generate_unsolvable_grid(12, 20, seed=42)
    for algo_name in ["Dijkstra", "A*", "BFS", "Greedy Best-First"]:
        r_j = run_algorithm(algo_name, g_unsolv_j, allow_diagonal=False)
        assert not r_j.found, f"Test J failed: {algo_name} reported path found"
        assert r_j.path == [], f"Test J failed: {algo_name} returned non-empty path"
        assert math.isinf(r_j.path_cost), f"Test J failed: {algo_name} cost is not inf"

    # Test K: Determinism and record_order=False for all 4 algorithms
    for algo_name in ["Dijkstra", "A*", "BFS", "Greedy Best-First"]:
        r_k1 = run_algorithm(algo_name, g_tr, allow_diagonal=False, record_order=True)
        r_k2 = run_algorithm(algo_name, g_tr, allow_diagonal=False, record_order=True)
        assert r_k1.path == r_k2.path, f"Test K failed: non-deterministic path for {algo_name}"
        assert r_k1.expansion_order == r_k2.expansion_order, (
            f"Test K failed: non-deterministic expansion_order for {algo_name}"
        )

        r_k_no = run_algorithm(algo_name, g_tr, allow_diagonal=False, record_order=False)
        assert r_k_no.expansion_order == [], (
            f"Test K failed: expansion_order not empty for {algo_name} with record_order=False"
        )
        assert r_k_no.nodes_expanded == r_k1.nodes_expanded, f"Test K failed: nodes_expanded mismatch for {algo_name}"
        assert r_k_no.nodes_generated == r_k1.nodes_generated, f"Test K failed: nodes_generated mismatch for {algo_name}"
        assert r_k_no.max_frontier_size == r_k1.max_frontier_size, f"Test K failed: max_frontier_size mismatch for {algo_name}"
        assert r_k_no.path_steps == r_k1.path_steps, f"Test K failed: path_steps mismatch for {algo_name}"
        assert math.isclose(r_k_no.path_cost, r_k1.path_cost), f"Test K failed: path_cost mismatch for {algo_name}"
        assert r_k_no.path == r_k1.path, f"Test K failed: path mismatch for {algo_name}"

    # Test L: run_algorithm dispatcher
    for name in ["Dijkstra", "A*", "BFS", "Greedy Best-First"]:
        res_l = run_algorithm(name, g_empty, allow_diagonal=False, heuristic="manhattan")
        assert res_l.found, f"Test L failed: {name} did not find path on empty grid"

    try:
        run_algorithm("NonExistentAlgorithm", g_empty)
        assert False, "Test L failed: expected KeyError for unknown algorithm"
    except KeyError as exc:
        assert "Available algorithms are" in str(exc)

    # Test M: Presentation (printed output)
    print("\n" + "=" * 80)
    print(" COMPARISON TABLE: Trap Grid & Mud Barrier Grid (Phase 3)")
    print("=" * 80)

    rows_data: List[Dict[str, Any]] = []
    benchmark_instances = [
        ("trap", g_tr),
        ("mud_barrier", g_mb),
    ]
    algo_runs = [
        ("BFS", None),
        ("Dijkstra", None),
        ("A*", "manhattan"),
        ("A*", "euclidean"),
        ("A*", "zero"),
        ("Greedy Best-First", "manhattan"),
    ]

    for grid_label, grid_obj in benchmark_instances:
        for algo_label, heur in algo_runs:
            res_m = run_algorithm(
                algo_label,
                grid_obj,
                allow_diagonal=False,
                heuristic=heur,
                record_order=True,
            )
            d = res_m.to_dict(grid_obj)
            d["grid"] = grid_label
            # Reorder dictionary to match requested columns
            row = {
                "grid": d["grid"],
                "algorithm": d["algorithm"],
                "heuristic": d["heuristic"] if d["heuristic"] is not None else "-",
                "found": d["found"],
                "path_steps": int(d["path_steps"]) if not math.isnan(d["path_steps"]) else "-",
                "path_cost": d["path_cost"],
                "nodes_expanded": d["nodes_expanded"],
                "nodes_generated": d["nodes_generated"],
                "max_frontier_size": d["max_frontier_size"],
                "runtime_ms": round(d["runtime_ms"], 3),
                "expanded_density_pct": round(d["expanded_density_pct"], 1),
            }
            rows_data.append(row)

    df = pd.DataFrame(rows_data)
    print(df.to_string(index=False))

    def _render_mud_ascii(g: Grid, res: SearchResult) -> str:
        """Helper to render mud barrier results according to Test M specification."""
        path_set = set(res.path)
        expanded_set = set(res.expansion_order)
        lines: List[str] = []
        for r in range(g.rows):
            chars: List[str] = []
            for c in range(g.cols):
                pos = (r, c)
                if pos == g.start:
                    chars.append("S")
                elif pos == g.goal:
                    chars.append("G")
                elif pos in path_set:
                    chars.append("*")
                elif pos in expanded_set:
                    chars.append("o")
                elif g.cells[r, c] == WALL:
                    chars.append("#")
                elif g.cells[r, c] == MUD:
                    chars.append("~")
                else:
                    chars.append(".")
            lines.append("".join(chars))
        return "\n".join(lines)

    res_greedy_mb = run_algorithm(
        "Greedy Best-First", g_mb, allow_diagonal=False, heuristic="manhattan"
    )
    res_astar_mb = run_algorithm(
        "A*", g_mb, allow_diagonal=False, heuristic="manhattan"
    )

    print("\n--- Mud Barrier: Greedy Best-First (cuts through mud) ---")
    print(_render_mud_ascii(g_mb, res_greedy_mb))

    print("\n--- Mud Barrier: A* (detours around mud via top corridor) ---")
    print(_render_mud_ascii(g_mb, res_astar_mb))

    print("\nAll Phase 3 tests passed!")
