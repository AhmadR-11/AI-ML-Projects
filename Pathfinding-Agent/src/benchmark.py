"""Benchmark experiment runner and metrics evaluation engine.

Executes graph pathfinding algorithms across procedural scenarios, measures
runtime metrics with GC control and warmups, validates path correctness,
aggregates summary statistics and winner tallies, and exports benchmark logs.
"""

from dataclasses import dataclass
import datetime
import gc
import json
import math
import pathlib
import platform
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import warnings

import numpy as np
import pandas as pd

from src.algorithms import (
    SearchResult,
    run_algorithm,
    validate_path,
)
from src.grid import (
    Grid,
    MUD,
    add_mud_patches,
    generate_mud_barrier_grid,
    generate_perfect_maze,
    generate_random_grid,
    generate_trap_grid,
    generate_unsolvable_grid,
)


# =====================================================================
# PART A — Data Classes
# =====================================================================

@dataclass(frozen=True)
class AlgoConfig:
    """Immutable configuration descriptor for a benchmarked search algorithm.

    Attributes:
        name: Base algorithm name ('Dijkstra', 'A*', 'BFS', 'Greedy Best-First').
        heuristic: Heuristic name if applicable ('manhattan', 'euclidean', 'octile', etc.).
    """

    name: str
    heuristic: Optional[str] = None

    @property
    def label(self) -> str:
        """Formatted human-readable label e.g. 'A* (manhattan)' or 'Dijkstra'."""
        return f"{self.name} ({self.heuristic})" if self.heuristic else self.name

    def expects_optimal_cost(self, allow_diagonal: bool) -> bool:
        """Return whether this configuration is guaranteed to return the optimal cost.

        Dijkstra is always cost-optimal. A* is cost-optimal with admissible heuristics,
        which holds unless Manhattan heuristic is used on an 8-direction grid.
        BFS and Greedy Best-First are not cost-optimal on weighted grids.
        """
        if self.name == "Dijkstra":
            return True
        if self.name == "A*":
            return not (self.heuristic == "manhattan" and allow_diagonal)
        return False

    def admissible(self, allow_diagonal: bool) -> bool:
        """Return whether the heuristic configuration is strictly admissible.

        Returns False only for A* paired with Manhattan distance on an 8-direction grid.
        """
        if self.name == "A*" and self.heuristic == "manhattan" and allow_diagonal:
            return False
        return True


@dataclass
class Scenario:
    """Benchmark test instance representing a grid problem configuration.

    Attributes:
        name: Unique identifier string ('category|param_label|seed=...').
        category: Test category ('density_sweep', 'trap', 'size_sweep', etc.).
        param_label: Parameter descriptor string ('density=0.20', 'trap=12x20', etc.).
        grid: Instantiated Grid environment.
        allow_diagonal: Whether 8-direction transitions are permitted.
        seed: Random seed used to construct the grid (None for deterministic grids).
    """

    name: str
    category: str
    param_label: str
    grid: Grid
    allow_diagonal: bool
    seed: Optional[int]


@dataclass
class BenchmarkResult:
    """Encapsulates the complete outputs from a benchmark suite execution.

    Attributes:
        results: Raw DataFrame containing all trial runs.
        summary: Aggregated summary statistics grouped by scenario and algorithm.
        winners: Best-performing algorithm counts per scenario instance.
        meta: Execution metadata (timestamps, versions, platform, protocol).
    """

    results: pd.DataFrame
    summary: pd.DataFrame
    winners: pd.DataFrame
    meta: Dict[str, Any]


# =====================================================================
# PART B — Algorithm Config Lists
# =====================================================================

def default_algorithm_configs(
    allow_diagonal: bool = False,
    include_inadmissible_demo: bool = False,
) -> List[AlgoConfig]:
    """Return standard algorithm configurations for benchmarking.

    Dijkstra is always placed first to serve as the baseline optimality reference.

    Args:
        allow_diagonal: Whether 8-direction movement is allowed.
        include_inadmissible_demo: If True and allow_diagonal is True, appends
            A*(manhattan) as the last configuration (deliberately inadmissible).

    Returns:
        List of AlgoConfig instances.
    """
    if not allow_diagonal:
        return [
            AlgoConfig("Dijkstra"),
            AlgoConfig("BFS"),
            AlgoConfig("A*", "manhattan"),
            AlgoConfig("A*", "euclidean"),
            AlgoConfig("Greedy Best-First", "manhattan"),
        ]

    configs = [
        AlgoConfig("Dijkstra"),
        AlgoConfig("BFS"),
        AlgoConfig("A*", "octile"),
        AlgoConfig("A*", "euclidean"),
        AlgoConfig("Greedy Best-First", "octile"),
    ]
    if include_inadmissible_demo:
        configs.append(AlgoConfig("A*", "manhattan"))
    return configs


# =====================================================================
# PART C — Scenario Builders
# =====================================================================

SUITES: Dict[str, Dict[str, Any]] = {
    "quick": {
        "trials": 3,
        "density_grid": 30,
        "densities": (0.10, 0.30),
        "sizes": (20, 40),
        "maze_sizes": (21,),
        "weighted_grid": 30,
        "trap_sizes": ((12, 20),),
        "mud_barrier_sizes": ((15, 25),),
        "unsolvable_size": 20,
        "diagonal_grid": 30,
        "repeats": 3,
    },
    "standard": {
        "trials": 10,
        "density_grid": 50,
        "densities": (0.10, 0.20, 0.30, 0.35),
        "sizes": (20, 50, 100),
        "maze_sizes": (21, 51, 101),
        "weighted_grid": 50,
        "trap_sizes": ((12, 20), (24, 40), (48, 80)),
        "mud_barrier_sizes": ((15, 25), (31, 51), (61, 101)),
        "unsolvable_size": 30,
        "diagonal_grid": 50,
        "repeats": 7,
    },
}

SIZE_SWEEP_DENSITY: float = 0.25
WEIGHTED_DENSITY: float = 0.20
WEIGHTED_PATCHES: int = 6
WEIGHTED_PATCH_SIZE: int = 5
DIAGONAL_DENSITY: float = 0.25
# Note: densities above 0.35 are avoided because 4-direction random grids near ~40% obstacles
# are almost never solvable (percolation threshold), so generation would mostly fail.


def build_scenarios(
    suite: str = "standard",
    trials: Optional[int] = None,
    base_seed: int = 0,
) -> Tuple[List[Scenario], List[str]]:
    """Construct procedural benchmark scenarios for the requested suite.

    Args:
        suite: Name of suite ('quick' or 'standard').
        trials: Number of random trials per category. Overrides suite default if given.
        base_seed: Starting integer seed for reproducibility.

    Returns:
        Tuple of (list of generated Scenario instances, list of skipped scenario descriptions).

    Raises:
        KeyError: If suite name is not recognized.
    """
    if suite not in SUITES:
        valid_suites = ", ".join(repr(k) for k in SUITES.keys())
        raise KeyError(f"Unknown suite {suite!r}. Valid suites are: {valid_suites}")

    suite_cfg = SUITES[suite]
    num_trials = trials if trials is not None else suite_cfg["trials"]
    scenarios: List[Scenario] = []
    skipped: List[str] = []

    # 1. density_sweep
    for d in suite_cfg["densities"]:
        p_label = f"density={d:.2f}"
        for i in range(num_trials):
            seed = base_seed + i
            try:
                grid = generate_random_grid(
                    suite_cfg["density_grid"],
                    suite_cfg["density_grid"],
                    obstacle_density=d,
                    seed=seed,
                    ensure_solvable=True,
                )
                scenarios.append(
                    Scenario(
                        f"density_sweep|{p_label}|seed={seed}",
                        "density_sweep",
                        p_label,
                        grid,
                        False,
                        seed,
                    )
                )
            except RuntimeError as err:
                skipped.append(f"density_sweep|{p_label}|seed={seed}: {err}")

    # 2. size_sweep
    for n in suite_cfg["sizes"]:
        p_label = f"size={n}x{n}"
        for i in range(num_trials):
            seed = base_seed + i
            try:
                grid = generate_random_grid(
                    n,
                    n,
                    obstacle_density=SIZE_SWEEP_DENSITY,
                    seed=seed,
                    ensure_solvable=True,
                )
                scenarios.append(
                    Scenario(
                        f"size_sweep|{p_label}|seed={seed}",
                        "size_sweep",
                        p_label,
                        grid,
                        False,
                        seed,
                    )
                )
            except RuntimeError as err:
                skipped.append(f"size_sweep|{p_label}|seed={seed}: {err}")

    # 3. perfect_maze
    for n in suite_cfg["maze_sizes"]:
        p_label = f"maze={n}x{n}"
        for i in range(num_trials):
            seed = base_seed + i
            grid = generate_perfect_maze(n, n, seed=seed)
            scenarios.append(
                Scenario(
                    f"perfect_maze|{p_label}|seed={seed}",
                    "perfect_maze",
                    p_label,
                    grid,
                    False,
                    seed,
                )
            )

    # 4. weighted_random
    p_label_w = "weighted d=0.20 + mud"
    for i in range(num_trials):
        seed = base_seed + i
        try:
            base_grid = generate_random_grid(
                suite_cfg["weighted_grid"],
                suite_cfg["weighted_grid"],
                obstacle_density=WEIGHTED_DENSITY,
                seed=seed,
                ensure_solvable=True,
            )
            mud_grid = add_mud_patches(
                base_grid,
                num_patches=WEIGHTED_PATCHES,
                patch_size=WEIGHTED_PATCH_SIZE,
                seed=seed,
            )
            scenarios.append(
                Scenario(
                    f"weighted_random|{p_label_w}|seed={seed}",
                    "weighted_random",
                    p_label_w,
                    mud_grid,
                    False,
                    seed,
                )
            )
        except RuntimeError as err:
            skipped.append(f"weighted_random|{p_label_w}|seed={seed}: {err}")

    # 5. trap
    for r, c in suite_cfg["trap_sizes"]:
        p_label = f"trap={r}x{c}"
        grid = generate_trap_grid(r, c)
        scenarios.append(
            Scenario(
                f"trap|{p_label}|seed=None",
                "trap",
                p_label,
                grid,
                False,
                None,
            )
        )

    # 6. mud_barrier
    for r, c in suite_cfg["mud_barrier_sizes"]:
        p_label = f"mud_barrier={r}x{c}"
        grid = generate_mud_barrier_grid(r, c)
        scenarios.append(
            Scenario(
                f"mud_barrier|{p_label}|seed=None",
                "mud_barrier",
                p_label,
                grid,
                False,
                None,
            )
        )

    # 7. unsolvable
    n_unsolv = suite_cfg["unsolvable_size"]
    p_label_u = f"unsolvable={n_unsolv}x{n_unsolv}"
    unsolv_trials = min(num_trials, 3)
    for i in range(unsolv_trials):
        seed = base_seed + i
        grid = generate_unsolvable_grid(n_unsolv, n_unsolv, seed=seed)
        scenarios.append(
            Scenario(
                f"unsolvable|{p_label_u}|seed={seed}",
                "unsolvable",
                p_label_u,
                grid,
                False,
                seed,
            )
        )

    # 8. diagonal
    p_label_d = "diagonal d=0.25"
    for i in range(num_trials):
        seed = base_seed + i
        try:
            grid = generate_random_grid(
                suite_cfg["diagonal_grid"],
                suite_cfg["diagonal_grid"],
                obstacle_density=DIAGONAL_DENSITY,
                seed=seed,
                ensure_solvable=True,
            )
            scenarios.append(
                Scenario(
                    f"diagonal|{p_label_d}|seed={seed}",
                    "diagonal",
                    p_label_d,
                    grid,
                    True,
                    seed,
                )
            )
        except RuntimeError as err:
            skipped.append(f"diagonal|{p_label_d}|seed={seed}: {err}")

    return scenarios, skipped


# =====================================================================
# PART D — Measurement Engine
# =====================================================================

def measure_config(
    grid: Grid,
    config: AlgoConfig,
    allow_diagonal: bool,
    repeats: int,
    warmup: int = 1,
) -> Tuple[SearchResult, Dict[str, float]]:
    """Measure search execution runtime with warmups, GC suppression, and determinism checks.

    Single-shot timings in Python can fluctuate by 2x to 3x due to garbage collection
    pauses, CPU cache state, and OS scheduling jitter. This protocol performs untimed
    warmup passes, disables Python GC during timed loops (restoring it reliably in
    a finally block), records multiple executions, and computes the median runtime.

    Args:
        grid: Grid on which to execute search.
        config: Algorithm configuration descriptor.
        allow_diagonal: Whether 8-direction movement is allowed.
        repeats: Number of measured repetitions.
        warmup: Number of untimed warmup iterations before measurement.

    Returns:
        Tuple of (representative SearchResult from first run, stats dictionary).

    Raises:
        RuntimeError: If nondeterministic search behavior is detected across repeats.
    """
    # Warmup runs (suppress intentional UserWarnings from inadmissible configs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(warmup):
            run_algorithm(
                config.name,
                grid,
                allow_diagonal=allow_diagonal,
                heuristic=config.heuristic,
                record_order=False,
            )

    # Timed runs with GC suspended
    gc.collect()
    was_gc_enabled = gc.isenabled()
    gc.disable()
    runtimes: List[float] = []
    results: List[SearchResult] = []

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(repeats):
                res = run_algorithm(
                    config.name,
                    grid,
                    allow_diagonal=allow_diagonal,
                    heuristic=config.heuristic,
                    record_order=False,
                )
                results.append(res)
                runtimes.append(res.runtime_ms)
    finally:
        if was_gc_enabled:
            gc.enable()

    # Determinism assertion across all repeats
    first = results[0]
    for other in results[1:]:
        if other.found != first.found:
            raise RuntimeError(
                f"Non-deterministic search detected for {config.label}: found status mismatch"
            )
        if other.path_steps != first.path_steps:
            raise RuntimeError(
                f"Non-deterministic search detected for {config.label}: path_steps mismatch "
                f"({other.path_steps} != {first.path_steps})"
            )
        if not (
            (math.isnan(first.path_cost) and math.isnan(other.path_cost))
            or (math.isinf(first.path_cost) and math.isinf(other.path_cost))
            or math.isclose(first.path_cost, other.path_cost, rel_tol=1e-9, abs_tol=1e-9)
        ):
            raise RuntimeError(
                f"Non-deterministic search detected for {config.label}: path_cost mismatch "
                f"({other.path_cost} != {first.path_cost})"
            )
        if other.nodes_expanded != first.nodes_expanded:
            raise RuntimeError(
                f"Non-deterministic search detected for {config.label}: nodes_expanded mismatch "
                f"({other.nodes_expanded} != {first.nodes_expanded})"
            )
        if other.nodes_generated != first.nodes_generated:
            raise RuntimeError(
                f"Non-deterministic search detected for {config.label}: nodes_generated mismatch "
                f"({other.nodes_generated} != {first.nodes_generated})"
            )
        if other.max_frontier_size != first.max_frontier_size:
            raise RuntimeError(
                f"Non-deterministic search detected for {config.label}: max_frontier_size mismatch "
                f"({other.max_frontier_size} != {first.max_frontier_size})"
            )

    stats = {
        "runtime_median_ms": float(statistics.median(runtimes)),
        "runtime_min_ms": float(min(runtimes)),
        "runtime_std_ms": (
            float(statistics.pstdev(runtimes)) if len(runtimes) > 1 else 0.0
        ),
        "repeats": float(len(runtimes)),
    }
    return first, stats


def run_scenario(
    scenario: Scenario,
    configs: List[AlgoConfig],
    repeats: int,
    warmup: int = 1,
    strict: bool = True,
) -> List[Dict[str, Any]]:
    """Execute all algorithm configurations on a single scenario and compile metrics.

    Args:
        scenario: Scenario under evaluation.
        configs: List of AlgoConfig descriptors to run (first must be Dijkstra).
        repeats: Number of timing repetitions.
        warmup: Number of warmup runs.
        strict: If True, raises RuntimeError on correctness guard violations.

    Returns:
        List of result dictionaries, one per configuration.

    Raises:
        ValueError: If first config is not Dijkstra.
        RuntimeError: If correctness guards fail (path validity, optimality, reachability).
    """
    if not configs or configs[0].name != "Dijkstra":
        raise ValueError("The first algorithm config must be Dijkstra (optimality baseline)")

    measured: List[Tuple[AlgoConfig, SearchResult, Dict[str, float]]] = []
    for cfg in configs:
        res, stats = measure_config(
            scenario.grid,
            cfg,
            allow_diagonal=scenario.allow_diagonal,
            repeats=repeats,
            warmup=warmup,
        )
        measured.append((cfg, res, stats))

    # Identify reference values from Dijkstra and BFS
    _, dijkstra_res, _ = measured[0]
    dijkstra_found = dijkstra_res.found
    dijkstra_cost = dijkstra_res.path_cost
    dijkstra_expanded = dijkstra_res.nodes_expanded

    bfs_steps: Optional[int] = None
    for cfg, res, _ in measured:
        if cfg.name == "BFS" and res.found:
            bfs_steps = res.path_steps
            break

    rows: List[Dict[str, Any]] = []
    total_cells = scenario.grid.rows * scenario.grid.cols
    mud_density = (
        float(np.count_nonzero(scenario.grid.cells == MUD)) / float(total_cells)
    )

    for cfg, res, stats in measured:
        # Validate path feasibility if found
        if res.found:
            recomputed_cost = validate_path(
                scenario.grid,
                res.path,
                allow_diagonal=scenario.allow_diagonal,
            )
            if not math.isclose(recomputed_cost, res.path_cost, rel_tol=1e-9, abs_tol=1e-9):
                raise RuntimeError(
                    f"Path validation cost mismatch in {scenario.name} for {cfg.label}: "
                    f"recomputed {recomputed_cost} != reported {res.path_cost}"
                )

        # Correctness guards
        if strict:
            if res.found != dijkstra_found:
                raise RuntimeError(
                    f"Correctness guard failure in {scenario.name} for {cfg.label}: "
                    f"found={res.found} != Dijkstra found={dijkstra_found}"
                )
            if res.found and res.path_cost < dijkstra_cost - 1e-9:
                raise RuntimeError(
                    f"Correctness guard failure in {scenario.name} for {cfg.label}: "
                    f"path_cost {res.path_cost} < Dijkstra optimal cost {dijkstra_cost}"
                )
            if cfg.expects_optimal_cost(scenario.allow_diagonal) and res.found:
                if not math.isclose(
                    res.path_cost, dijkstra_cost, rel_tol=1e-9, abs_tol=1e-9
                ):
                    raise RuntimeError(
                        f"Correctness guard failure in {scenario.name} for {cfg.label}: "
                        f"expected optimal cost {dijkstra_cost}, got {res.path_cost}"
                    )

        # Derived ratios
        if not res.found or not dijkstra_found:
            cost_ratio = float("nan")
        elif math.isclose(res.path_cost, 0.0) and math.isclose(dijkstra_cost, 0.0):
            cost_ratio = 1.0
        else:
            cost_ratio = res.path_cost / dijkstra_cost

        is_cost_optimal = (
            bool(cost_ratio <= 1.0 + 1e-9) if not math.isnan(cost_ratio) else False
        )

        if bfs_steps is not None and res.found and bfs_steps > 0:
            step_ratio = float(res.path_steps) / float(bfs_steps)
        else:
            step_ratio = float("nan")

        nodes_vs_dijkstra_pct = (
            (100.0 * float(res.nodes_expanded) / float(dijkstra_expanded))
            if dijkstra_expanded > 0
            else 100.0
        )

        row: Dict[str, Any] = {
            "scenario": scenario.name,
            "category": scenario.category,
            "param_label": scenario.param_label,
            "seed": scenario.seed,
            "rows": scenario.grid.rows,
            "cols": scenario.grid.cols,
            "allow_diagonal": scenario.allow_diagonal,
            "wall_density": scenario.grid.wall_density(),
            "mud_density": mud_density,
            "free_cells": scenario.grid.free_cell_count(),
            "algorithm": cfg.label,
            "base_algorithm": cfg.name,
            "heuristic": cfg.heuristic,
            "admissible": cfg.admissible(scenario.allow_diagonal),
            "expects_optimal_cost": cfg.expects_optimal_cost(scenario.allow_diagonal),
            "found": res.found,
            "path_steps": res.path_steps if res.found else float("nan"),
            "path_cost": res.path_cost if res.found else float("nan"),
            "optimal_cost": dijkstra_cost if dijkstra_found else float("nan"),
            "cost_ratio": cost_ratio,
            "is_cost_optimal": is_cost_optimal,
            "step_ratio": step_ratio,
            "nodes_expanded": res.nodes_expanded,
            "nodes_generated": res.nodes_generated,
            "max_frontier_size": res.max_frontier_size,
            "expanded_density_pct": res.expanded_density_pct(scenario.grid),
            "nodes_vs_dijkstra_pct": nodes_vs_dijkstra_pct,
            "runtime_median_ms": stats["runtime_median_ms"],
            "runtime_min_ms": stats["runtime_min_ms"],
            "runtime_std_ms": stats["runtime_std_ms"],
            "repeats": int(stats["repeats"]),
        }
        rows.append(row)

    return rows


# =====================================================================
# PART E — The Benchmark Driver
# =====================================================================

def run_benchmark(
    suite: str = "standard",
    repeats: Optional[int] = None,
    warmup: int = 1,
    trials: Optional[int] = None,
    base_seed: int = 0,
    include_inadmissible_demo: bool = True,
    strict: bool = True,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> BenchmarkResult:
    """Run an entire suite of benchmark experiments and summarize results.

    Args:
        suite: Name of predefined suite ('quick' or 'standard').
        repeats: Repetitions per timing measurement (defaults to suite setting).
        warmup: Number of untimed warmup runs before measurement.
        trials: Number of random trials per category (defaults to suite setting).
        base_seed: Base seed for procedural generators.
        include_inadmissible_demo: Whether to include inadmissible A*(manhattan) in 8-dir.
        strict: If True, enforces strict optimality and reachability assertions.
        progress: Optional progress callback progress(done: int, total: int, label: str).

    Returns:
        BenchmarkResult container with raw results, summary, winners, and metadata.
    """
    if suite not in SUITES:
        valid_suites = ", ".join(repr(k) for k in SUITES.keys())
        raise KeyError(f"Unknown suite {suite!r}. Valid suites are: {valid_suites}")

    start_time = time.perf_counter()
    suite_cfg = SUITES[suite]
    act_repeats = repeats if repeats is not None else suite_cfg["repeats"]

    scenarios, skipped = build_scenarios(
        suite=suite,
        trials=trials,
        base_seed=base_seed,
    )

    all_rows: List[Dict[str, Any]] = []
    total_scenarios = len(scenarios)

    for idx, scenario in enumerate(scenarios):
        configs = default_algorithm_configs(
            allow_diagonal=scenario.allow_diagonal,
            include_inadmissible_demo=(
                include_inadmissible_demo if scenario.allow_diagonal else False
            ),
        )
        scenario_rows = run_scenario(
            scenario=scenario,
            configs=configs,
            repeats=act_repeats,
            warmup=warmup,
            strict=strict,
        )
        all_rows.extend(scenario_rows)

        if progress is not None:
            progress(idx + 1, total_scenarios, scenario.name)

    columns = [
        "scenario",
        "category",
        "param_label",
        "seed",
        "rows",
        "cols",
        "allow_diagonal",
        "wall_density",
        "mud_density",
        "free_cells",
        "algorithm",
        "base_algorithm",
        "heuristic",
        "admissible",
        "expects_optimal_cost",
        "found",
        "path_steps",
        "path_cost",
        "optimal_cost",
        "cost_ratio",
        "is_cost_optimal",
        "step_ratio",
        "nodes_expanded",
        "nodes_generated",
        "max_frontier_size",
        "expanded_density_pct",
        "nodes_vs_dijkstra_pct",
        "runtime_median_ms",
        "runtime_min_ms",
        "runtime_std_ms",
        "repeats",
    ]

    results_df = pd.DataFrame(all_rows, columns=columns)
    summary_df = summarize(results_df)
    winners_df = winner_counts(results_df)

    meta = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "suite": suite,
        "repeats": act_repeats,
        "warmup": warmup,
        "trials": trials if trials is not None else suite_cfg["trials"],
        "base_seed": base_seed,
        "n_scenarios": total_scenarios,
        "n_rows": len(results_df),
        "skipped": skipped,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "total_seconds": round(time.perf_counter() - start_time, 2),
        "timing_protocol": (
            f"Timing measured across {act_repeats} repeats after {warmup} warmup run "
            "with garbage collection disabled; reported as median ms."
        ),
    }

    return BenchmarkResult(
        results=results_df,
        summary=summary_df,
        winners=winners_df,
        meta=meta,
    )


# =====================================================================
# PART F — Summaries
# =====================================================================

def summarize(
    df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Aggregate benchmark results across scenario and algorithm groupings.

    Args:
        df: Raw results DataFrame from run_benchmark.
        group_cols: Grouping columns. Defaults to ['category', 'param_label', 'algorithm'].

    Returns:
        Aggregated summary DataFrame with metrics rounded to 3 decimals.
    """
    if group_cols is None:
        group_cols = ["category", "param_label", "algorithm"]

    records: List[Dict[str, Any]] = []

    for keys, sub_df in df.groupby(group_cols, sort=False, as_index=False):
        # Normalize keys as list if single grouping column
        if not isinstance(keys, (list, tuple)):
            keys = (keys,)

        row_dict = dict(zip(group_cols, keys))
        n_runs = len(sub_df)
        success_rate_pct = float(sub_df["found"].astype(float).mean() * 100.0)

        # Path metrics computed across valid / found paths
        found_sub = sub_df[sub_df["found"] == True]
        pct_cost_optimal = (
            float(found_sub["is_cost_optimal"].astype(float).mean() * 100.0)
            if len(found_sub) > 0
            else 0.0
        )

        row_dict["n_runs"] = n_runs
        row_dict["success_rate_pct"] = round(success_rate_pct, 3)
        row_dict["mean_path_steps"] = (
            round(float(sub_df["path_steps"].dropna().mean()), 3)
            if not sub_df["path_steps"].dropna().empty
            else float("nan")
        )
        row_dict["mean_path_cost"] = (
            round(float(sub_df["path_cost"].dropna().mean()), 3)
            if not sub_df["path_cost"].dropna().empty
            else float("nan")
        )
        row_dict["mean_cost_ratio"] = (
            round(float(sub_df["cost_ratio"].dropna().mean()), 3)
            if not sub_df["cost_ratio"].dropna().empty
            else float("nan")
        )
        row_dict["pct_cost_optimal"] = round(pct_cost_optimal, 3)
        row_dict["mean_nodes_expanded"] = round(float(sub_df["nodes_expanded"].mean()), 3)
        row_dict["mean_expanded_density_pct"] = round(
            float(sub_df["expanded_density_pct"].mean()), 3
        )
        row_dict["mean_nodes_vs_dijkstra_pct"] = round(
            float(sub_df["nodes_vs_dijkstra_pct"].mean()), 3
        )
        row_dict["mean_max_frontier"] = round(float(sub_df["max_frontier_size"].mean()), 3)
        row_dict["median_runtime_ms"] = round(
            float(sub_df["runtime_median_ms"].median()), 3
        )
        row_dict["mean_runtime_ms"] = round(float(sub_df["runtime_median_ms"].mean()), 3)

        records.append(row_dict)

    summary_cols = group_cols + [
        "n_runs",
        "success_rate_pct",
        "mean_path_steps",
        "mean_path_cost",
        "mean_cost_ratio",
        "pct_cost_optimal",
        "mean_nodes_expanded",
        "mean_expanded_density_pct",
        "mean_nodes_vs_dijkstra_pct",
        "mean_max_frontier",
        "median_runtime_ms",
        "mean_runtime_ms",
    ]
    return pd.DataFrame(records, columns=summary_cols)


def winner_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Tally winning algorithms per scenario instance by expanded nodes and runtime.

    Only solutions that are found, cost-optimal, and admissible are eligible. Ties
    are broken deterministically by the first algorithm appearing in config order.

    Args:
        df: Raw benchmark results DataFrame.

    Returns:
        DataFrame with columns [metric, algorithm, wins, win_pct].
    """
    eligible = df[
        (df["found"] == True)
        & (df["is_cost_optimal"] == True)
        & (df["admissible"] == True)
    ]

    metrics = ["nodes_expanded", "runtime_median_ms"]
    records: List[Dict[str, Any]] = []

    for metric in metrics:
        wins_by_algo: Dict[str, int] = {}
        total_scenarios_evaluated = 0

        for _, scenario_df in eligible.groupby("scenario", sort=False):
            if scenario_df.empty:
                continue

            # Find best value; in case of tie, idxmin returns the first occurrence
            best_idx = scenario_df[metric].idxmin()
            winner_algo = scenario_df.loc[best_idx, "algorithm"]
            wins_by_algo[winner_algo] = wins_by_algo.get(winner_algo, 0) + 1
            total_scenarios_evaluated += 1

        if total_scenarios_evaluated > 0:
            for algo, wins in wins_by_algo.items():
                win_pct = round(100.0 * float(wins) / float(total_scenarios_evaluated), 1)
                records.append(
                    {
                        "metric": metric,
                        "algorithm": algo,
                        "wins": wins,
                        "win_pct": win_pct,
                    }
                )

    winners_df = pd.DataFrame(
        records, columns=["metric", "algorithm", "wins", "win_pct"]
    )
    if not winners_df.empty:
        winners_df = winners_df.sort_values(
            by=["metric", "wins"], ascending=[True, False]
        ).reset_index(drop=True)
    return winners_df


# =====================================================================
# PART G — Export
# =====================================================================

def export_results(
    bench: BenchmarkResult,
    out_dir: Union[str, pathlib.Path] = "outputs/logs",
) -> Dict[str, str]:
    """Export benchmark DataFrame tables and execution metadata to disk.

    Writes:
      - results.csv: Full raw benchmark trial records.
      - summary.csv: Aggregated summary statistics table.
      - winners.csv: Algorithm win counts and win percentages.
      - run_log.json: JSON metadata dictionary capturing environment & parameters.

    Args:
        bench: BenchmarkResult to serialize.
        out_dir: Target output directory path.

    Returns:
        Dictionary mapping filename to absolute file path string.
    """
    target_path = pathlib.Path(out_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    results_file = target_path / "results.csv"
    summary_file = target_path / "summary.csv"
    winners_file = target_path / "winners.csv"
    log_file = target_path / "run_log.json"

    bench.results.to_csv(results_file, index=False)
    bench.summary.to_csv(summary_file, index=False)
    bench.winners.to_csv(winners_file, index=False)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(bench.meta, f, indent=2)

    return {
        "results.csv": str(results_file.resolve()),
        "summary.csv": str(summary_file.resolve()),
        "winners.csv": str(winners_file.resolve()),
        "run_log.json": str(log_file.resolve()),
    }


# =====================================================================
# PART H — CLI and Self-Test
# =====================================================================

if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(
        description="Pathfinding benchmark experiment runner and evaluation suite."
    )
    parser.add_argument(
        "--run",
        choices=["quick", "standard"],
        help="Execute specified benchmark suite and export logs.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help="Number of timing repetitions per config.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of random scenario trials per category.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/logs",
        help="Output directory for CSV and JSON benchmark exports.",
    )

    args = parser.parse_args()

    if args.run:
        print(f"Starting {args.run.upper()} benchmark suite...")

        def _cli_progress(done: int, total: int, label: str) -> None:
            sys.stdout.write(f"\r[{done}/{total}] {label[:60]:<60}")
            sys.stdout.flush()

        res = run_benchmark(
            suite=args.run,
            repeats=args.repeats,
            trials=args.trials,
            progress=_cli_progress,
        )
        print("\n\nExecution finished. Compiling summaries...\n")

        compact_cols = [
            "category",
            "param_label",
            "algorithm",
            "mean_nodes_expanded",
            "mean_cost_ratio",
            "median_runtime_ms",
        ]
        print("=" * 80)
        print(" BENCHMARK SUMMARY")
        print("=" * 80)
        print(res.summary[compact_cols].to_string(index=False))

        print("\n" + "=" * 80)
        print(" WINNER TALLIES")
        print("=" * 80)
        print(res.winners.to_string(index=False))

        exported = export_results(res, out_dir=args.out)
        print(f"\nBenchmark exported to {args.out}:")
        for fname, fpath in exported.items():
            print(f"  {fname}: {fpath}")
        print(f"Total benchmark time: {res.meta['total_seconds']} seconds.")

    else:
        # =============================================================
        # Self-Test Mode (Run on "quick" suite)
        # =============================================================
        print("Executing Phase 4 Benchmark self-test suite...")

        # Assertion 1: build_scenarios("quick") validity and error handling
        scenarios_q, skipped_q = build_scenarios("quick")
        categories = {s.category for s in scenarios_q}
        assert len(categories) == 8, (
            f"Assertion 1 failed: expected 8 categories, got {len(categories)}"
        )
        scenario_names = [s.name for s in scenarios_q]
        assert len(scenario_names) == len(set(scenario_names)), (
            "Assertion 1 failed: duplicate scenario names detected"
        )
        assert isinstance(skipped_q, list), "Assertion 1 failed: skipped is not a list"

        try:
            build_scenarios("invalid_suite_name")
            assert False, "Assertion 1 failed: expected KeyError for unknown suite"
        except KeyError as exc:
            assert "Valid suites are" in str(exc)

        # Assertion 2: run_benchmark("quick", repeats=3) execution & columns
        b_res = run_benchmark("quick", repeats=3)
        assert not b_res.results.empty, "Assertion 2 failed: results DataFrame is empty"
        assert not b_res.summary.empty, "Assertion 2 failed: summary DataFrame is empty"

        expected_cols = [
            "scenario",
            "category",
            "param_label",
            "seed",
            "rows",
            "cols",
            "allow_diagonal",
            "wall_density",
            "mud_density",
            "free_cells",
            "algorithm",
            "base_algorithm",
            "heuristic",
            "admissible",
            "expects_optimal_cost",
            "found",
            "path_steps",
            "path_cost",
            "optimal_cost",
            "cost_ratio",
            "is_cost_optimal",
            "step_ratio",
            "nodes_expanded",
            "nodes_generated",
            "max_frontier_size",
            "expanded_density_pct",
            "nodes_vs_dijkstra_pct",
            "runtime_median_ms",
            "runtime_min_ms",
            "runtime_std_ms",
            "repeats",
        ]
        assert list(b_res.results.columns) == expected_cols, (
            "Assertion 2 failed: results DataFrame column schema mismatch"
        )

        # Assertion 3: Optimality bound & expects_optimal_cost validation
        valid_cost_rows = b_res.results[b_res.results["found"] == True]
        for _, row in valid_cost_rows.iterrows():
            assert row["cost_ratio"] >= 1.0 - 1e-9, (
                f"Assertion 3 failed: cost_ratio < 1 in {row['scenario']} for {row['algorithm']}"
            )
            if row["expects_optimal_cost"]:
                assert math.isclose(row["cost_ratio"], 1.0, rel_tol=1e-9, abs_tol=1e-9), (
                    f"Assertion 3 failed: expected optimal cost in {row['scenario']} "
                    f"for {row['algorithm']}, got cost_ratio={row['cost_ratio']}"
                )

        # Assertion 4: Consistent reachability & unsolvable success rate 0%
        for _, scen_df in b_res.results.groupby("scenario"):
            unique_found = scen_df["found"].unique()
            assert len(unique_found) == 1, (
                f"Assertion 4 failed: inconsistent found states within scenario {scen_df['scenario'].iloc[0]}"
            )

        unsolv_df = b_res.results[b_res.results["category"] == "unsolvable"]
        assert not unsolv_df["found"].any(), (
            "Assertion 4 failed: found True in unsolvable category"
        )
        unsolv_summary = b_res.summary[b_res.summary["category"] == "unsolvable"]
        assert (unsolv_summary["success_rate_pct"] == 0.0).all(), (
            "Assertion 4 failed: success_rate_pct != 0 for unsolvable summary"
        )

        # Assertion 5: Mud barrier 15x25 cost ratios
        mb_rows = b_res.results[b_res.results["param_label"] == "mud_barrier=15x25"]
        for _, row in mb_rows.iterrows():
            algo = row["algorithm"]
            if algo in ["Dijkstra", "A* (manhattan)", "A* (euclidean)"]:
                assert math.isclose(row["cost_ratio"], 1.0, rel_tol=1e-9, abs_tol=1e-9), (
                    f"Assertion 5 failed: {algo} cost_ratio != 1.0 on mud barrier"
                )
            elif algo in ["BFS", "Greedy Best-First (manhattan)"]:
                expected_ratio = 50.0 / 36.0
                assert math.isclose(row["cost_ratio"], expected_ratio, rel_tol=1e-3), (
                    f"Assertion 5 failed: {algo} cost_ratio ({row['cost_ratio']}) "
                    f"!= {expected_ratio}"
                )

        # Assertion 6: Trap 12x20 steps and expansion order
        trap_rows = b_res.results[b_res.results["param_label"] == "trap=12x20"].set_index(
            "algorithm"
        )
        assert (trap_rows["path_steps"] == 27).all(), (
            "Assertion 6 failed: path_steps != 27 on trap grid"
        )
        greedy_exp = trap_rows.loc["Greedy Best-First (manhattan)", "nodes_expanded"]
        astar_exp = trap_rows.loc["A* (manhattan)", "nodes_expanded"]
        dijk_exp = trap_rows.loc["Dijkstra", "nodes_expanded"]
        assert greedy_exp < astar_exp < dijk_exp, (
            f"Assertion 6 failed: expansion order violation on trap grid "
            f"(Greedy={greedy_exp}, A*={astar_exp}, Dijkstra={dijk_exp})"
        )

        # Assertion 7: Runtime stats validity and repeats check
        for _, row in b_res.results.iterrows():
            assert row["runtime_min_ms"] <= row["runtime_median_ms"], (
                f"Assertion 7 failed: min > median for {row['algorithm']} in {row['scenario']}"
            )
            assert row["runtime_std_ms"] >= 0.0, (
                f"Assertion 7 failed: std < 0 for {row['algorithm']} in {row['scenario']}"
            )
            assert row["repeats"] == 3, (
                f"Assertion 7 failed: repeats ({row['repeats']}) != 3"
            )

        # Assertion 8: Determinism across runs
        b_res2 = run_benchmark("quick", repeats=2)
        r1_indexed = b_res.results.set_index(["scenario", "algorithm"])
        r2_indexed = b_res2.results.set_index(["scenario", "algorithm"])
        common_keys = r1_indexed.index.intersection(r2_indexed.index)

        for key in common_keys:
            r1_item = r1_indexed.loc[key]
            r2_item = r2_indexed.loc[key]
            assert r1_item["found"] == r2_item["found"], (
                f"Assertion 8 failed: found mismatch across benchmark runs for {key}"
            )
            assert (
                (math.isnan(r1_item["path_steps"]) and math.isnan(r2_item["path_steps"]))
                or r1_item["path_steps"] == r2_item["path_steps"]
            ), f"Assertion 8 failed: path_steps mismatch for {key}"
            assert (
                (math.isnan(r1_item["path_cost"]) and math.isnan(r2_item["path_cost"]))
                or (math.isinf(r1_item["path_cost"]) and math.isinf(r2_item["path_cost"]))
                or math.isclose(r1_item["path_cost"], r2_item["path_cost"])
            ), f"Assertion 8 failed: path_cost mismatch for {key}"
            assert r1_item["nodes_expanded"] == r2_item["nodes_expanded"], (
                f"Assertion 8 failed: nodes_expanded mismatch for {key}"
            )
            assert r1_item["nodes_generated"] == r2_item["nodes_generated"], (
                f"Assertion 8 failed: nodes_generated mismatch for {key}"
            )

        # Assertion 9: Summary & winners schema and percentage integrity
        assert (b_res.summary["success_rate_pct"] >= 0.0).all() and (
            b_res.summary["success_rate_pct"] <= 100.0
        ).all(), "Assertion 9 failed: success_rate_pct out of range [0, 100]"

        for metric in ["nodes_expanded", "runtime_median_ms"]:
            metric_wins = b_res.winners[b_res.winners["metric"] == metric]
            if not metric_wins.empty:
                sum_win_pct = metric_wins["win_pct"].sum()
                assert math.isclose(sum_win_pct, 100.0, abs_tol=0.5), (
                    f"Assertion 9 failed: win_pct sum for {metric} ({sum_win_pct}) != 100"
                )

        # Assertion 10: Export to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            exported_files = export_results(b_res, out_dir=temp_dir)
            assert len(exported_files) == 4, "Assertion 10 failed: expected 4 exported files"

            for fname, fpath in exported_files.items():
                p = pathlib.Path(fpath)
                assert p.exists() and p.is_file(), (
                    f"Assertion 10 failed: file {fname} does not exist at {fpath}"
                )

            reloaded_results = pd.read_csv(exported_files["results.csv"])
            assert len(reloaded_results) == len(b_res.results), (
                f"Assertion 10 failed: reloaded row count ({len(reloaded_results)}) "
                f"!= original ({len(b_res.results)})"
            )

            with open(exported_files["run_log.json"], "r", encoding="utf-8") as f:
                reloaded_meta = json.load(f)
            for required_key in ["suite", "repeats", "n_scenarios", "n_rows", "skipped"]:
                assert required_key in reloaded_meta, (
                    f"Assertion 10 failed: missing key {required_key} in run_log.json"
                )

        # Print quick-suite summary for trap, mud_barrier, and diagonal categories
        sub_summary = b_res.summary[
            b_res.summary["category"].isin(["trap", "mud_barrier", "diagonal"])
        ]
        display_cols = [
            "param_label",
            "algorithm",
            "mean_nodes_expanded",
            "mean_cost_ratio",
            "median_runtime_ms",
        ]
        print("\n" + "=" * 80)
        print(" QUICK SUITE SAMPLE SUMMARY (trap, mud_barrier, diagonal)")
        print("=" * 80)
        print(sub_summary[display_cols].to_string(index=False))

        print("\nAll Phase 4 tests passed!")
