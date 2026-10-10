"""Pure application logic for the Pathfinding Agent Streamlit application.

Decoupled from Streamlit and Matplotlib.pyplot for headless testability.
Provides grid specification data models, generator dispatchers, ASCII grid
parsing, algorithm execution, comparison metric tables, natural-language
insights, and byte serialization helpers.
"""

from dataclasses import dataclass
import io
import math
import pathlib
import sys
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import warnings

from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from src.algorithms import SearchResult, run_algorithm
from src.animator import (
    animate_race,
    animate_search,
)
from src.benchmark import (
    AlgoConfig,
    default_algorithm_configs,
    measure_config,
)
from src.grid import (
    FREE,
    MUD,
    WALL,
    Grid,
    Pos,
    add_mud_patches,
    generate_mud_barrier_grid,
    generate_perfect_maze,
    generate_random_grid,
    generate_trap_grid,
    generate_unsolvable_grid,
    is_solvable,
)
from src.visualizer import (
    slugify,
    sort_algorithm_labels,
)


# =====================================================================
# PART A1 & A2 — Data Models and Preset Constants
# =====================================================================

@dataclass(frozen=True)
class GridSpec:
    """Descriptor specifying environment generation parameters.

    Attributes:
        preset: Name of preset configuration.
        rows: Number of grid rows.
        cols: Number of grid columns.
        density: Obstacle percolation density (for random presets).
        seed: Random generator seed.
        band_width: Width of mud barrier strip (for mud barrier presets).
        add_mud: Whether to deposit additional mud patches.
        mud_patches: Number of mud patches to scatter.
    """

    preset: str
    rows: int
    cols: int
    density: float = 0.25
    seed: int = 3
    band_width: int = 7
    add_mud: bool = False
    mud_patches: int = 4


MIN_ROWS: int = 9
MIN_COLS: int = 15
MAX_DIM: int = 120
MAX_ANIM_CELLS: int = 5000
DEFAULT_PRESET: str = "Mud barrier"

PRESETS: Dict[str, Dict[str, Any]] = {
    "Mud barrier": {
        "rows": 15,
        "cols": 25,
        "uses_density": False,
        "uses_seed": False,
        "uses_band": True,
        "supports_mud": False,
        "description": "Horizontal mud band testing cost vs distance tradeoff.",
    },
    "Trap (cup)": {
        "rows": 12,
        "cols": 20,
        "uses_density": False,
        "uses_seed": False,
        "uses_band": False,
        "supports_mud": False,
        "description": "Concave obstacle pocket penalizing pure greedy heuristics.",
    },
    "Random obstacles": {
        "rows": 30,
        "cols": 30,
        "uses_density": True,
        "uses_seed": True,
        "uses_band": False,
        "supports_mud": True,
        "description": "Percolation maze with configurable obstacle density.",
    },
    "Perfect maze": {
        "rows": 21,
        "cols": 41,
        "uses_density": False,
        "uses_seed": True,
        "uses_band": False,
        "supports_mud": True,
        "description": "Spanning tree maze with exactly one path between any two points.",
    },
    "Unsolvable": {
        "rows": 20,
        "cols": 30,
        "uses_density": False,
        "uses_seed": True,
        "uses_band": False,
        "supports_mud": False,
        "description": "Enclosed goal proving exhaustive frontier termination.",
    },
}


# =====================================================================
# PART A3 & A4 — Cache Keys and Grid Builders
# =====================================================================

def spec_key(spec: GridSpec) -> str:
    """Generate a stable cache key incorporating only relevant preset parameters.

    Args:
        spec: GridSpec instance.

    Returns:
        Formatted descriptor string.

    Raises:
        KeyError: If preset name is unrecognized.
    """
    if spec.preset not in PRESETS:
        valid_presets = list(PRESETS.keys())
        raise KeyError(f"Unknown preset {spec.preset!r}. Valid presets are: {valid_presets}")

    cfg = PRESETS[spec.preset]
    parts: List[str] = [
        f"preset={spec.preset}",
        f"rows={spec.rows}",
        f"cols={spec.cols}",
    ]

    if cfg["uses_band"]:
        parts.append(f"band={spec.band_width}")
    if cfg["uses_density"]:
        parts.append(f"density={spec.density:.4f}")
    if cfg["uses_seed"]:
        parts.append(f"seed={spec.seed}")
    if cfg["supports_mud"]:
        parts.append(f"add_mud={spec.add_mud}")
        if spec.add_mud:
            parts.append(f"mud_patches={spec.mud_patches}")

    return "|".join(parts)


def build_grid(spec: GridSpec) -> Grid:
    """Instantiate a Grid from a GridSpec descriptor.

    For 'Perfect maze', even row or column counts are automatically rounded up
    to odd integers to ensure maze wall consistency.

    Args:
        spec: GridSpec configuration.

    Returns:
        Grid instance.

    Raises:
        ValueError: If dimensions fall outside [MIN_ROWS, MAX_DIM] or [MIN_COLS, MAX_DIM].
        KeyError: If preset is unknown.
    """
    if spec.preset not in PRESETS:
        valid_presets = list(PRESETS.keys())
        raise KeyError(f"Unknown preset {spec.preset!r}. Valid presets are: {valid_presets}")

    if spec.rows < MIN_ROWS or spec.rows > MAX_DIM or spec.cols < MIN_COLS or spec.cols > MAX_DIM:
        raise ValueError(
            f"Grid dimensions must satisfy {MIN_ROWS} <= rows <= {MAX_DIM} and "
            f"{MIN_COLS} <= cols <= {MAX_DIM}, got rows={spec.rows}, cols={spec.cols}"
        )

    p = spec.preset
    if p == "Mud barrier":
        grid = generate_mud_barrier_grid(rows=spec.rows, cols=spec.cols, band_width=spec.band_width)
    elif p == "Trap (cup)":
        grid = generate_trap_grid(rows=spec.rows, cols=spec.cols)
    elif p == "Random obstacles":
        grid = generate_random_grid(
            rows=spec.rows,
            cols=spec.cols,
            obstacle_density=spec.density,
            seed=spec.seed,
        )
    elif p == "Perfect maze":
        grid = generate_perfect_maze(rows=spec.rows, cols=spec.cols, seed=spec.seed)
    elif p == "Unsolvable":
        grid = generate_unsolvable_grid(rows=spec.rows, cols=spec.cols, seed=spec.seed)
    else:
        raise KeyError(f"Unhandled preset {p!r}")

    cfg = PRESETS[p]
    if cfg["supports_mud"] and spec.add_mud:
        patch_size = max(3, min(grid.rows, grid.cols) // 6)
        grid = add_mud_patches(
            grid=grid,
            num_patches=spec.mud_patches,
            patch_size=patch_size,
            seed=spec.seed,
        )

    return grid


# =====================================================================
# PART A5 & A6 — ASCII Parser and Info
# =====================================================================

def parse_ascii_grid(text: str) -> Grid:
    """Parse a multi-line ASCII maze representation into a Grid instance.

    Args:
        text: Multi-line string with 'S', 'G', '#', '~', '.' characters.

    Returns:
        Grid instance.

    Raises:
        ValueError: On empty inputs, ragged line lengths, illegal characters,
            or invalid start/goal marker counts.
    """
    raw_lines = [line.rstrip() for line in text.splitlines()]
    while raw_lines and not raw_lines[0]:
        raw_lines.pop(0)
    while raw_lines and not raw_lines[-1]:
        raw_lines.pop()

    if not raw_lines:
        raise ValueError("The maze is empty")

    expected_len = len(raw_lines[0])
    for idx, line in enumerate(raw_lines, start=1):
        if len(line) != expected_len:
            raise ValueError(
                f"Row {idx} has {len(line)} characters but row 1 has {expected_len}"
            )

    allowed = {"S", "G", "#", "~", "."}
    start_pos: Optional[Pos] = None
    goal_pos: Optional[Pos] = None
    s_count = 0
    g_count = 0

    rows = len(raw_lines)
    cols = expected_len
    cells = np.full((rows, cols), FREE, dtype=np.int8)

    for r, line in enumerate(raw_lines):
        for c, ch in enumerate(line):
            if ch not in allowed:
                raise ValueError(
                    f"Invalid character '{ch}' at row {r + 1}, column {c + 1}"
                )
            if ch == "#":
                cells[r, c] = WALL
            elif ch == "~":
                cells[r, c] = MUD
            elif ch == "S":
                s_count += 1
                start_pos = (r, c)
            elif ch == "G":
                g_count += 1
                goal_pos = (r, c)

    if s_count != 1:
        raise ValueError(f"Expected exactly one 'S' marker, but found {s_count}")
    if g_count != 1:
        raise ValueError(f"Expected exactly one 'G' marker, but found {g_count}")

    assert start_pos is not None and goal_pos is not None
    return Grid(rows=rows, cols=cols, cells=cells, start=start_pos, goal=goal_pos)


def grid_info(grid: Grid, allow_diagonal: bool) -> Dict[str, Any]:
    """Calculate summary topology metrics and reachability for a Grid."""
    total = grid.rows * grid.cols
    wall_count = int(np.count_nonzero(grid.cells == WALL))
    mud_count = int(np.count_nonzero(grid.cells == MUD))
    free_cells = grid.free_cell_count()

    wall_pct = round((wall_count / float(total)) * 100.0, 1)
    mud_pct = round((mud_count / float(total)) * 100.0, 1)
    solvable = is_solvable(grid, allow_diagonal=allow_diagonal)

    return {
        "rows": grid.rows,
        "cols": grid.cols,
        "wall_pct": wall_pct,
        "mud_pct": mud_pct,
        "free_cells": free_cells,
        "solvable": solvable,
    }


# =====================================================================
# PART A7 — Config Helpers
# =====================================================================

def config_key(config: AlgoConfig) -> Tuple[str, Optional[str]]:
    """Extract a hashable key tuple (name, heuristic) from an AlgoConfig."""
    return (config.name, config.heuristic)


def available_configs(allow_diagonal: bool) -> Dict[str, AlgoConfig]:
    """Return map of label -> AlgoConfig for standard benchmark algorithms."""
    cfgs = default_algorithm_configs(
        allow_diagonal=allow_diagonal,
        include_inadmissible_demo=True,
    )
    return {c.label: c for c in cfgs}


def configs_from_keys(keys: Sequence[str], allow_diagonal: bool) -> List[AlgoConfig]:
    """Resolve a sequence of algorithm labels into AlgoConfig instances."""
    avail = available_configs(allow_diagonal)
    res: List[AlgoConfig] = []
    for k in keys:
        if k not in avail:
            valid_keys = list(avail.keys())
            raise KeyError(f"Unknown config key {k!r}. Valid keys are: {valid_keys}")
        res.append(avail[k])
    return res


# =====================================================================
# PART A8, A9, A10 — Execution and Timing Engines
# =====================================================================

def solve_configs(
    grid: Grid,
    configs: Sequence[AlgoConfig],
    allow_diagonal: bool,
) -> Dict[str, SearchResult]:
    """Execute given algorithms on a Grid, guaranteeing Dijkstra runs first.

    Suppresses intentional UserWarnings emitted by demo inadmissible configurations.
    """
    dijkstra_cfg = AlgoConfig("Dijkstra")
    seen_labels = set()
    ordered_cfgs: List[AlgoConfig] = [dijkstra_cfg]
    seen_labels.add(dijkstra_cfg.label)

    for c in configs:
        if c.label not in seen_labels:
            ordered_cfgs.append(c)
            seen_labels.add(c.label)

    results: Dict[str, SearchResult] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for c in ordered_cfgs:
            results[c.label] = run_algorithm(
                name=c.name,
                grid=grid,
                allow_diagonal=allow_diagonal,
                heuristic=c.heuristic,
                record_order=True,
            )

    return results


def precise_runtimes(
    grid: Grid,
    configs: Sequence[AlgoConfig],
    allow_diagonal: bool,
    repeats: int = 5,
) -> Dict[str, float]:
    """Evaluate median search runtime per algorithm via repeated benchmarking."""
    runtimes: Dict[str, float] = {}
    for c in configs:
        _, stats = measure_config(
            grid=grid,
            config=c,
            allow_diagonal=allow_diagonal,
            repeats=repeats,
            warmup=1,
        )
        runtimes[c.label] = float(stats["runtime_median_ms"])
    return runtimes


def cost_verdict(cost: float, best_cost: Optional[float], found: bool = True) -> str:
    """Format an optimality badge relative to the best found path cost."""
    if not found:
        return "no path"
    if best_cost is None or abs(cost - best_cost) <= 1e-9:
        return "optimal"
    diff_pct = (cost / best_cost - 1.0) * 100.0
    return f"+{diff_pct:.1f}%"


# =====================================================================
# PART A11 & A12 — Comparison Table & Natural Language Insights
# =====================================================================

def comparison_table(
    grid: Grid,
    results: Mapping[str, SearchResult],
    runtimes: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Construct standard multi-algorithm comparison metrics table.

    Columns:
        Algorithm, Found, Steps, Path cost, Cost vs optimal, Nodes expanded,
        % of free cells, % of Dijkstra's expansions, Peak frontier,
        Search efficiency %, Runtime (ms).
    """
    ordered_labels = sort_algorithm_labels(results.keys())
    found_costs = [res.path_cost for res in results.values() if res.found]
    best_cost = min(found_costs) if found_costs else None

    dijk_exp: Optional[int] = None
    if "Dijkstra" in results:
        dijk_exp = results["Dijkstra"].nodes_expanded

    free_cells = grid.free_cell_count()
    rows: List[Dict[str, Any]] = []

    for lbl in ordered_labels:
        res = results[lbl]
        found = res.found

        steps_val = res.path_steps if found else None
        cost_val = round(res.path_cost, 1) if found else None
        cost_verdict_str = cost_verdict(res.path_cost, best_cost, found=found)

        free_cell_pct = (
            round((res.nodes_expanded / float(free_cells)) * 100.0, 1)
            if free_cells > 0
            else 0.0
        )
        dijk_pct = (
            round((res.nodes_expanded / float(dijk_exp)) * 100.0, 1)
            if (dijk_exp is not None and dijk_exp > 0)
            else 100.0
        )
        eff_pct = (
            round(100.0 * (res.path_steps + 1) / float(res.nodes_expanded), 1)
            if (found and res.nodes_expanded > 0)
            else 0.0
        )

        if runtimes is not None and lbl in runtimes:
            rt_val = round(runtimes[lbl], 3)
        else:
            rt_val = round(res.runtime_ms, 3)

        rows.append(
            {
                "Algorithm": lbl,
                "Found": found,
                "Steps": steps_val,
                "Path cost": cost_val,
                "Cost vs optimal": cost_verdict_str,
                "Nodes expanded": res.nodes_expanded,
                "% of free cells": free_cell_pct,
                "% of Dijkstra's expansions": dijk_pct,
                "Peak frontier": res.max_frontier_size,
                "Search efficiency %": eff_pct,
                "Runtime (ms)": rt_val,
            }
        )

    cols = [
        "Algorithm",
        "Found",
        "Steps",
        "Path cost",
        "Cost vs optimal",
        "Nodes expanded",
        "% of free cells",
        "% of Dijkstra's expansions",
        "Peak frontier",
        "Search efficiency %",
        "Runtime (ms)",
    ]
    return pd.DataFrame(rows, columns=cols)


def build_insights(
    grid: Grid,
    results: Mapping[str, SearchResult],
) -> List[str]:
    """Generate at most 5 ordered human-readable insights from search results."""
    found_results = {lbl: res for lbl, res in results.items() if res.found}

    # 1. No path found
    if not found_results:
        return [f"No path exists between S and G ({grid.rows}x{grid.cols} grid)."]

    insights: List[str] = []
    dijk_res = results.get("Dijkstra")
    dijk_nodes = dijk_res.nodes_expanded if (dijk_res and dijk_res.found) else None

    # 2. Fewest expansions
    sorted_found_labels = sort_algorithm_labels(found_results.keys())
    best_exp_lbl = min(sorted_found_labels, key=lambda l: found_results[l].nodes_expanded)
    min_exp_nodes = found_results[best_exp_lbl].nodes_expanded

    if best_exp_lbl == "Dijkstra" or dijk_nodes is None or dijk_nodes == 0:
        insights.append(f"Fewest expansions: {best_exp_lbl} expanded {min_exp_nodes} nodes")
    else:
        pct_fewer = ((dijk_nodes - min_exp_nodes) / float(dijk_nodes)) * 100.0
        insights.append(
            f"Fewest expansions: {best_exp_lbl} expanded {min_exp_nodes} nodes "
            f"({pct_fewer:.0f}% fewer than Dijkstra)"
        )

    # 3. Cost-optimal vs suboptimal
    best_cost = min(res.path_cost for res in found_results.values())
    optimal_labels = [
        lbl for lbl, res in found_results.items() if abs(res.path_cost - best_cost) <= 1e-9
    ]
    suboptimal_labels = [
        lbl for lbl, res in found_results.items() if abs(res.path_cost - best_cost) > 1e-9
    ]

    opt_str = ", ".join(sort_algorithm_labels(optimal_labels))
    opt_line = f"Cost-optimal: {opt_str}"
    if suboptimal_labels:
        sub_items = [
            f"{lbl} +{(found_results[lbl].path_cost / best_cost - 1.0) * 100.0:.1f}%"
            for lbl in sort_algorithm_labels(suboptimal_labels)
        ]
        opt_line += "; Not optimal: " + "; ".join(sub_items)
    insights.append(opt_line)

    # 4. Fewer steps != lower cost
    if dijk_res and dijk_res.found:
        dijk_steps = dijk_res.path_steps
        dijk_cost = dijk_res.path_cost
        divergent_labels = [
            lbl
            for lbl, res in found_results.items()
            if res.path_steps < dijk_steps and res.path_cost > dijk_cost
        ]
        if divergent_labels:
            sorted_div = sort_algorithm_labels(divergent_labels)
            first_div = found_results[sorted_div[0]]
            insights.append(
                f"Fewer steps != lower cost: {', '.join(sorted_div)} take {first_div.path_steps} steps vs "
                f"{dijk_steps} for the optimal path, but cost {first_div.path_cost:.1f} vs {dijk_cost:.1f}"
            )

    # 5. Most focused search
    def _eff_key(lbl: str) -> float:
        res = found_results[lbl]
        return (res.path_steps + 1) / float(res.nodes_expanded)

    most_focused_lbl = max(sorted_found_labels, key=_eff_key)
    max_eff = _eff_key(most_focused_lbl) * 100.0
    insights.append(
        f"Most focused search: {most_focused_lbl} - {max_eff:.0f}% of its expansions lie on the final path"
    )

    return insights[:5]


# =====================================================================
# PART A13, A14, A15 — Byte Serialization & Name Formatting
# =====================================================================

def fig_to_png_bytes(fig: Figure, dpi: int = 150) -> bytes:
    """Serialize a Matplotlib Figure into PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    buf.seek(0)
    return buf.getvalue()


def gif_bytes_for_search(
    grid: Grid,
    result: SearchResult,
    theme: str = "dark",
    max_frames: int = 60,
) -> bytes:
    """Render a single search GIF into raw in-memory bytes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_p = pathlib.Path(tmp_dir) / "search.gif"
        animate_search(
            grid=grid,
            result=result,
            out_path=out_p,
            theme=theme,
            max_frames=max_frames,
        )
        with open(out_p, "rb") as f:
            return f.read()


def gif_bytes_for_race(
    grid: Grid,
    results: Mapping[str, SearchResult],
    theme: str = "dark",
    max_frames: int = 60,
    ncols: int = 3,
) -> bytes:
    """Render a multi-algorithm race GIF into raw in-memory bytes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        out_p = pathlib.Path(tmp_dir) / "race.gif"
        animate_race(
            grid=grid,
            results=results,
            out_path=out_p,
            theme=theme,
            max_frames=max_frames,
            ncols=ncols,
        )
        with open(out_p, "rb") as f:
            return f.read()


def display_name(spec: GridSpec, grid: Grid) -> str:
    """Format canonical scenario identifier string (e.g. 'mud_barrier_15x25')."""
    return f"{slugify(spec.preset)}_{grid.rows}x{grid.cols}"


# =====================================================================
# PART A16 — Self-Test Block
# =====================================================================

if __name__ == "__main__":
    print("Executing Phase 7 app_logic self-test suite...")

    # 1. Round trip for every preset at default size
    for p_name, p_cfg in PRESETS.items():
        s = GridSpec(preset=p_name, rows=p_cfg["rows"], cols=p_cfg["cols"])
        g_orig = build_grid(s)
        ascii_text = g_orig.to_ascii()
        g_parsed = parse_ascii_grid(ascii_text)
        assert g_parsed.shape == g_orig.shape, f"Shape mismatch for {p_name}"
        assert np.array_equal(g_parsed.cells, g_orig.cells), f"Cells mismatch for {p_name}"
        assert g_parsed.start == g_orig.start, f"Start mismatch for {p_name}"
        assert g_parsed.goal == g_orig.goal, f"Goal mismatch for {p_name}"

    # 2. parse_ascii_grid errors
    try:
        parse_ascii_grid("")
        assert False, "Empty should fail"
    except ValueError as e:
        assert "empty" in str(e).lower()

    try:
        parse_ascii_grid("..#..\n..#..\n..#..")
        assert False, "Missing S and G should fail"
    except ValueError as e:
        assert "'S'" in str(e) or "s marker" in str(e).lower()

    try:
        parse_ascii_grid("S.G..\n..G..")
        assert False, "Two Gs should fail"
    except ValueError as e:
        assert "'G'" in str(e) or "g marker" in str(e).lower()

    try:
        parse_ascii_grid("S....\n#...\n....G")
        assert False, "Ragged rows should fail"
    except ValueError as e:
        assert "row" in str(e).lower()

    try:
        parse_ascii_grid("S..X.\n....G")
        assert False, "Invalid character X should fail"
    except ValueError as e:
        assert "X" in str(e)

    # 3. build_grid checks
    for p_name, p_cfg in PRESETS.items():
        s = GridSpec(preset=p_name, rows=p_cfg["rows"], cols=p_cfg["cols"])
        g = build_grid(s)
        if p_name == "Perfect maze":
            assert g.rows % 2 == 1 and g.cols % 2 == 1
        elif p_name == "Unsolvable":
            assert not is_solvable(g, allow_diagonal=False)

    # Seed reproducibility
    g_rnd1 = build_grid(GridSpec("Random obstacles", 30, 30, seed=1))
    g_rnd2 = build_grid(GridSpec("Random obstacles", 30, 30, seed=1))
    g_rnd3 = build_grid(GridSpec("Random obstacles", 30, 30, seed=2))
    assert np.array_equal(g_rnd1.cells, g_rnd2.cells)
    assert not np.array_equal(g_rnd1.cells, g_rnd3.cells)

    # Generator error propagation
    try:
        build_grid(GridSpec("Mud barrier", 15, 25, band_width=30))
        assert False, "Oversized band_width must raise ValueError"
    except ValueError:
        pass

    try:
        build_grid(GridSpec("Mud barrier", 5, 25))
        assert False, "Dimensions below minimum must raise ValueError"
    except ValueError:
        pass

    # 4. spec_key checks
    key_trap1 = spec_key(GridSpec("Trap (cup)", 12, 20, density=0.10))
    key_trap2 = spec_key(GridSpec("Trap (cup)", 12, 20, density=0.30))
    assert key_trap1 == key_trap2, "Trap spec_key must ignore density"

    key_rnd1 = spec_key(GridSpec("Random obstacles", 30, 30, density=0.10))
    key_rnd2 = spec_key(GridSpec("Random obstacles", 30, 30, density=0.30))
    assert key_rnd1 != key_rnd2, "Random obstacles spec_key must include density"

    key_mud1 = spec_key(GridSpec("Mud barrier", 15, 25, seed=1))
    key_mud2 = spec_key(GridSpec("Mud barrier", 15, 25, seed=2))
    assert key_mud1 == key_mud2, "Mud barrier spec_key must ignore seed"

    key_patch1 = spec_key(GridSpec("Random obstacles", 30, 30, add_mud=False, mud_patches=2))
    key_patch2 = spec_key(GridSpec("Random obstacles", 30, 30, add_mud=False, mud_patches=8))
    assert key_patch1 == key_patch2, "mud_patches must be ignored when add_mud is False"

    key_patch3 = spec_key(GridSpec("Random obstacles", 30, 30, add_mud=True, mud_patches=2))
    key_patch4 = spec_key(GridSpec("Random obstacles", 30, 30, add_mud=True, mud_patches=8))
    assert key_patch3 != key_patch4, "mud_patches must be counted when add_mud is True"

    try:
        spec_key(GridSpec("Unknown Preset", 10, 20))
        assert False, "Unknown preset must raise KeyError"
    except KeyError:
        pass

    # 5. grid_info on mud barrier 15x25
    g_mb = build_grid(GridSpec("Mud barrier", 15, 25))
    info_mb = grid_info(g_mb, allow_diagonal=False)
    assert info_mb["wall_pct"] == 0.0
    assert info_mb["free_cells"] == 375
    assert info_mb["solvable"] is True
    assert math.isclose(info_mb["mud_pct"], 98.0 / 375.0 * 100.0, abs_tol=0.05)

    # 6. solve_configs on mud barrier (4-dir)
    cfgs_4 = list(available_configs(False).values())
    res_mb = solve_configs(g_mb, cfgs_4, allow_diagonal=False)
    assert set(res_mb.keys()) == set(available_configs(False).keys())
    assert res_mb["Dijkstra"].path_cost == 36.0
    assert res_mb["A* (manhattan)"].path_cost == 36.0
    assert res_mb["A* (euclidean)"].path_cost == 36.0
    assert res_mb["BFS"].path_cost == 50.0
    assert res_mb["Greedy Best-First (manhattan)"].path_cost == 50.0
    assert res_mb["Dijkstra"].path_steps == 36
    assert res_mb["BFS"].path_steps == 22
    assert res_mb["Greedy Best-First (manhattan)"].path_steps == 22

    # Passing only Greedy config still returns Dijkstra first
    res_only_greedy = solve_configs(
        g_mb,
        [AlgoConfig("Greedy Best-First", "manhattan")],
        allow_diagonal=False,
    )
    assert list(res_only_greedy.keys())[0] == "Dijkstra"
    assert "Greedy Best-First (manhattan)" in res_only_greedy

    # 7. comparison_table checks
    tbl = comparison_table(g_mb, res_mb)
    expected_cols = [
        "Algorithm",
        "Found",
        "Steps",
        "Path cost",
        "Cost vs optimal",
        "Nodes expanded",
        "% of free cells",
        "% of Dijkstra's expansions",
        "Peak frontier",
        "Search efficiency %",
        "Runtime (ms)",
    ]
    assert list(tbl.columns) == expected_cols
    tbl_indexed = tbl.set_index("Algorithm")
    assert tbl_indexed.loc["Dijkstra", "Cost vs optimal"] == "optimal"
    assert tbl_indexed.loc["Greedy Best-First (manhattan)", "Cost vs optimal"] == "+38.9%"
    assert tbl_indexed.loc["Greedy Best-First (manhattan)", "Search efficiency %"] == 100.0
    assert tbl_indexed.loc["Dijkstra", "Search efficiency %"] == 11.6

    # Unsolvable comparison table
    g_unsolv = build_grid(GridSpec("Unsolvable", 20, 30))
    res_unsolv = solve_configs(g_unsolv, cfgs_4, allow_diagonal=False)
    tbl_unsolv = comparison_table(g_unsolv, res_unsolv)
    assert not tbl_unsolv["Found"].any()
    assert (tbl_unsolv["Cost vs optimal"] == "no path").all()

    # 8. build_insights checks
    ins_mud = build_insights(g_mb, res_mb)
    assert any(
        line.startswith("Fewer steps")
        and "Greedy Best-First (manhattan)" in line
        and "50.0" in line
        for line in ins_mud
    ), f"Mud barrier insights missing step tradeoff line: {ins_mud}"

    g_trap = build_grid(GridSpec("Trap (cup)", 12, 20))
    res_trap = solve_configs(g_trap, cfgs_4, allow_diagonal=False)
    ins_trap = build_insights(g_trap, res_trap)
    assert ins_trap[0].startswith("Fewest expansions: Greedy Best-First (manhattan)")

    ins_unsolv = build_insights(g_unsolv, res_unsolv)
    assert len(ins_unsolv) == 1
    assert ins_unsolv[0].startswith("No path exists")
    assert len(ins_mud) <= 5 and len(ins_trap) <= 5

    # 9. available_configs
    ac_4 = available_configs(False)
    ac_8 = available_configs(True)
    assert "A* (manhattan)" in ac_4 and "A* (octile)" not in ac_4
    assert "A* (octile)" in ac_8
    assert list(ac_8.keys())[-1] == "A* (manhattan)"

    # 10. precise_runtimes
    rts = precise_runtimes(g_mb, cfgs_4[:2], allow_diagonal=False, repeats=2)
    for lbl, val in rts.items():
        assert val > 0.0

    # 11. fig_to_png_bytes, gif_bytes_for_search, gif_bytes_for_race
    from src.visualizer import plot_result
    test_fig = plot_result(g_trap, res_trap["Dijkstra"], theme="dark")
    png_b = fig_to_png_bytes(test_fig)
    assert png_b.startswith(b"\x89PNG")
    del test_fig

    s_gif = gif_bytes_for_search(g_trap, res_trap["Dijkstra"], theme="dark", max_frames=20)
    assert s_gif.startswith(b"GIF89a")
    r_gif = gif_bytes_for_race(g_trap, res_trap, theme="dark", max_frames=20)
    assert r_gif.startswith(b"GIF89a")

    # 12. Rule enforcement
    assert "streamlit" not in sys.modules, "Rule violation: streamlit must not be imported in app_logic"
    assert "matplotlib.pyplot" not in sys.modules, "Rule violation: pyplot must not be imported in app_logic"

    print("All app_logic tests passed!")
