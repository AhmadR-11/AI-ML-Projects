"""Static visualizations and low-level grid rendering engine.

Provides reusable RGB buffer generation, object-oriented matplotlib grid and
path plotting, multi-algorithm side-by-side comparisons, benchmark performance
charts, and figure gallery exporters.
"""

from dataclasses import dataclass
import math
import pathlib
import re
import sys
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import matplotlib
from matplotlib.figure import Figure
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from src.algorithms import SearchResult, run_algorithm
from src.benchmark import (
    BenchmarkResult,
    run_benchmark,
    summarize,
    winner_counts,
)
from src.grid import (
    FREE,
    MUD,
    WALL,
    Grid,
    generate_mud_barrier_grid,
    generate_perfect_maze,
    generate_random_grid,
    generate_trap_grid,
    generate_unsolvable_grid,
)


# =====================================================================
# PART A — Themes, Palettes, and Helpers
# =====================================================================

THEMES: Dict[str, Dict[str, Any]] = {
    "dark": {
        "bg": "#0d1117",
        "panel": "#161b22",
        "text": "#e6edf3",
        "muted": "#8b949e",
        "grid_line": "#21262d",
        "free": "#1b222c",
        "wall": "#5c6370",
        "mud": "#8a6a3b",
        "path": "#f2cc60",
        "start": "#3fb950",
        "goal": "#ff7b72",
        "explored_cmap": "cool",
        "explored_flat": "#1f6feb",
        "explored_alpha": 0.55,
    },
    "light": {
        "bg": "#ffffff",
        "panel": "#f6f8fa",
        "text": "#1f2328",
        "muted": "#656d76",
        "grid_line": "#d0d7de",
        "free": "#ffffff",
        "wall": "#24292f",
        "mud": "#d4a373",
        "path": "#cf222e",
        "start": "#1a7f37",
        "goal": "#8250df",
        "explored_cmap": "Blues",
        "explored_flat": "#54aeff",
        "explored_alpha": 0.60,
    },
}

ALGO_ORDER: List[str] = [
    "Dijkstra",
    "BFS",
    "A* (manhattan)",
    "A* (euclidean)",
    "A* (octile)",
    "A* (zero)",
    "Greedy Best-First (manhattan)",
    "Greedy Best-First (octile)",
]

ALGO_COLORS: Dict[str, str] = {
    "Dijkstra": "#58a6ff",
    "BFS": "#8b949e",
    "A* (manhattan)": "#3fb950",
    "A* (euclidean)": "#bc8cff",
    "A* (octile)": "#56d364",
    "A* (zero)": "#d29922",
    "Greedy Best-First (manhattan)": "#ff7b72",
    "Greedy Best-First (octile)": "#ffa198",
}


def get_theme(theme: str) -> Dict[str, Any]:
    """Retrieve theme settings dictionary or raise KeyError listing valid options."""
    if theme not in THEMES:
        valid_keys = ", ".join(repr(k) for k in THEMES.keys())
        raise KeyError(f"Unknown theme {theme!r}. Available themes are: {valid_keys}")
    return THEMES[theme]


def _hex_to_rgb255(hex_str: str) -> np.ndarray:
    """Convert hex color string to a 3-element uint8 numpy RGB array."""
    rgb_float = mcolors.to_rgb(hex_str)
    return np.array([int(round(c * 255.0)) for c in rgb_float], dtype=np.uint8)


def result_label(result: SearchResult) -> str:
    """Format a descriptive label for a search result e.g. 'A* (manhattan)'."""
    if result.heuristic:
        return f"{result.algorithm} ({result.heuristic})"
    return result.algorithm


def algo_color(label: str) -> str:
    """Return theme color for an algorithm, with deterministic fallback for unknown labels."""
    if label in ALGO_COLORS:
        return ALGO_COLORS[label]

    # Deterministic fallback based on stable character code sum (not salted hash)
    char_sum = sum(ord(c) for c in label)
    cmap = matplotlib.colormaps["tab10"]
    rgba = cmap(char_sum % 10)
    return mcolors.to_hex(rgba)


def sort_algorithm_labels(labels: Iterable[str]) -> List[str]:
    """Sort algorithm label strings matching ALGO_ORDER, with unknown labels sorted last."""
    def _sort_key(item: str) -> Tuple[int, Any]:
        if item in ALGO_ORDER:
            return (0, ALGO_ORDER.index(item))
        return (1, item)

    return sorted(labels, key=_sort_key)


def natural_key(label: str) -> float:
    """Extract the first numerical value in a label for natural numeric sorting."""
    match = re.search(r"[-+]?\d*\.?\d+", label)
    return float(match.group()) if match else float("inf")


def slugify(text: str) -> str:
    """Convert string to lowercase alphanumeric tokens separated by single underscores."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower())
    return cleaned.strip("_")


def _save_fig(fig: Figure, save_path: Union[str, pathlib.Path], dpi: int = 150) -> None:
    """Save matplotlib Figure ensuring parent directories exist."""
    p = pathlib.Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        p,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )


# =====================================================================
# PART B — Low-Level Renderer
# =====================================================================

def render_rgb(
    grid: Grid,
    result: Optional[SearchResult] = None,
    theme: str = "dark",
    expanded_upto: Optional[int] = None,
    show_explored: bool = True,
    explored_style: str = "gradient",
    path_as_cells: bool = False,
) -> np.ndarray:
    """Render a 2D grid into an RGB uint8 array suitable for image display or animation.

    Args:
        grid: Grid instance to render.
        result: Optional SearchResult containing path and expansion data.
        theme: Theme name ('dark' or 'light').
        expanded_upto: Truncate expansion sequence to this count (all if None).
        show_explored: If True, overlay explored cell visitation history.
        explored_style: Explored coloring style ('gradient' or 'flat').
        path_as_cells: If True, paints path cells and start/goal directly onto buffer.

    Returns:
        np.ndarray of shape (rows, cols, 3) and dtype uint8.
    """
    t_cfg = get_theme(theme)

    free_rgb = _hex_to_rgb255(t_cfg["free"])
    wall_rgb = _hex_to_rgb255(t_cfg["wall"])
    mud_rgb = _hex_to_rgb255(t_cfg["mud"])

    # Base terrain layer
    rgb = np.empty((grid.rows, grid.cols, 3), dtype=np.uint8)
    rgb[grid.cells == FREE] = free_rgb
    rgb[grid.cells == WALL] = wall_rgb
    rgb[grid.cells == MUD] = mud_rgb

    # Overlay explored cells
    if result is not None and show_explored and result.expansion_order:
        order = (
            result.expansion_order[:expanded_upto]
            if expanded_upto is not None
            else result.expansion_order
        )
        if order:
            n_full = len(result.expansion_order)
            # Track latest occurrence of each cell in order
            last_idx = np.full((grid.rows, grid.cols), -1, dtype=np.int32)
            rows_arr = np.array([p[0] for p in order], dtype=np.int32)
            cols_arr = np.array([p[1] for p in order], dtype=np.int32)
            idx_arr = np.arange(len(order), dtype=np.int32)
            last_idx[rows_arr, cols_arr] = idx_arr

            mask = last_idx >= 0
            alpha = float(t_cfg["explored_alpha"])

            if explored_style == "gradient":
                cmap = matplotlib.colormaps[t_cfg["explored_cmap"]]
                t_vals = last_idx[mask] / float(max(n_full - 1, 1))
                overlay_rgb = cmap(t_vals)[:, :3] * 255.0
            else:
                flat_rgb = np.array(mcolors.to_rgb(t_cfg["explored_flat"])) * 255.0
                overlay_rgb = np.broadcast_to(flat_rgb, (np.count_nonzero(mask), 3))

            base_float = rgb[mask].astype(np.float64)
            blended = (1.0 - alpha) * base_float + alpha * overlay_rgb
            rgb[mask] = np.clip(np.round(blended), 0, 255).astype(np.uint8)

    # Optional rasterization of path cells
    if path_as_cells and result is not None and result.found and result.path:
        path_rgb = _hex_to_rgb255(t_cfg["path"])
        start_rgb = _hex_to_rgb255(t_cfg["start"])
        goal_rgb = _hex_to_rgb255(t_cfg["goal"])

        for p in result.path:
            rgb[p[0], p[1]] = path_rgb

        rgb[grid.start[0], grid.start[1]] = start_rgb
        rgb[grid.goal[0], grid.goal[1]] = goal_rgb

    return rgb


# =====================================================================
# PART C — Drawing on an Axes
# =====================================================================

def draw_grid(
    ax: Any,
    grid: Grid,
    result: Optional[SearchResult] = None,
    theme: str = "dark",
    show_explored: bool = True,
    explored_style: str = "gradient",
    show_path: bool = True,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    expanded_upto: Optional[int] = None,
) -> None:
    """Draw a grid environment and search trajectory onto a matplotlib Axes."""
    t_cfg = get_theme(theme)

    rgb = render_rgb(
        grid=grid,
        result=result,
        theme=theme,
        expanded_upto=expanded_upto,
        show_explored=show_explored,
        explored_style=explored_style,
        path_as_cells=False,
    )
    ax.imshow(rgb, interpolation="nearest")
    ax.axis("off")
    ax.set_facecolor(t_cfg["panel"])

    # Cell grid lines for small/medium grids
    max_dim = max(grid.rows, grid.cols)
    if max_dim <= 30:
        for r in range(grid.rows + 1):
            ax.axhline(r - 0.5, color=t_cfg["grid_line"], linewidth=0.5, zorder=2)
        for c in range(grid.cols + 1):
            ax.axvline(c - 0.5, color=t_cfg["grid_line"], linewidth=0.5, zorder=2)
        ax.set_xlim(-0.5, grid.cols - 0.5)
        ax.set_ylim(grid.rows - 0.5, -0.5)

    # Path vector line
    if show_path and result is not None and result.found and result.path:
        path_cols = [p[1] for p in result.path]
        path_rows = [p[0] for p in result.path]
        lw = max(1.0, min(3.0, 60.0 / float(max_dim)))
        ax.plot(
            path_cols,
            path_rows,
            color=t_cfg["path"],
            linewidth=lw,
            solid_capstyle="round",
            zorder=3,
        )

    # Start and goal markers
    marker_size = max(4.0, min(14.0, 180.0 / float(max_dim)))
    ax.plot(
        grid.start[1],
        grid.start[0],
        marker="o",
        markersize=marker_size,
        color=t_cfg["start"],
        markeredgecolor=t_cfg["bg"],
        markeredgewidth=1.0,
        zorder=4,
    )
    ax.plot(
        grid.goal[1],
        grid.goal[0],
        marker="s",
        markersize=marker_size,
        color=t_cfg["goal"],
        markeredgecolor=t_cfg["bg"],
        markeredgewidth=1.0,
        zorder=4,
    )

    if max_dim <= 30:
        fs = max(5, int(marker_size * 0.7))
        ax.text(
            grid.start[1],
            grid.start[0],
            "S",
            color=t_cfg["text"],
            fontsize=fs,
            ha="center",
            va="center",
            weight="bold",
            zorder=5,
        )
        ax.text(
            grid.goal[1],
            grid.goal[0],
            "G",
            color=t_cfg["text"],
            fontsize=fs,
            ha="center",
            va="center",
            weight="bold",
            zorder=5,
        )

    # Title assembly
    full_title: Optional[str] = None
    if title and subtitle:
        full_title = f"{title}\n{subtitle}"
    elif title:
        full_title = title
    elif subtitle:
        full_title = subtitle

    if full_title:
        ax.set_title(full_title, color=t_cfg["text"], fontsize=9, pad=6)


# =====================================================================
# PART D — Figures
# =====================================================================

def _add_legend(
    fig: Figure,
    theme: str,
    grid: Grid,
    show_explored: bool,
    explored_style: str = "gradient",
) -> None:
    """Attach a standardized lower legend to the Figure."""
    t_cfg = get_theme(theme)
    handles: List[Any] = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            color="none",
            markeredgecolor=t_cfg["bg"],
            markerfacecolor=t_cfg["start"],
            markersize=8,
            label="Start",
        ),
        mlines.Line2D(
            [],
            [],
            marker="s",
            color="none",
            markeredgecolor=t_cfg["bg"],
            markerfacecolor=t_cfg["goal"],
            markersize=8,
            label="Goal",
        ),
        mlines.Line2D(
            [],
            [],
            color=t_cfg["path"],
            linewidth=2.5,
            label="Path",
        ),
    ]

    if show_explored:
        lbl = (
            "Explored (earlier -> later)"
            if explored_style == "gradient"
            else "Explored"
        )
        handles.append(mpatches.Patch(facecolor=t_cfg["explored_flat"], label=lbl))

    handles.append(mpatches.Patch(facecolor=t_cfg["wall"], label="Wall"))

    if np.any(grid.cells == MUD):
        handles.append(mpatches.Patch(facecolor=t_cfg["mud"], label="Mud"))

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        labelcolor=t_cfg["text"],
        fontsize=8,
        bbox_to_anchor=(0.5, 0.01),
    )


def result_subtitle(
    grid: Grid,
    result: SearchResult,
    best_cost: Optional[float] = None,
) -> str:
    """Generate a subtitle string summarizing search metrics and relative optimality."""
    n = result.nodes_expanded
    pct = result.expanded_density_pct(grid)

    if not result.found:
        return f"NO PATH FOUND · expanded {n} ({pct:.1f}%)"

    s = result.path_steps
    c = result.path_cost
    ms = result.runtime_ms
    base = f"steps {s} · cost {c:.1f} · expanded {n} ({pct:.1f}%) · {ms:.2f} ms"

    if best_cost is not None and best_cost > 0.0:
        if abs(c - best_cost) <= 1e-9:
            base += " · optimal"
        else:
            pct_diff = (c / best_cost - 1.0) * 100.0
            base += f" · +{pct_diff:.1f}% cost"

    return base


def plot_result(
    grid: Grid,
    result: SearchResult,
    theme: str = "dark",
    show_explored: bool = True,
    explored_style: str = "gradient",
    title: Optional[str] = None,
    save_path: Optional[Union[str, pathlib.Path]] = None,
    dpi: int = 150,
) -> Figure:
    """Generate a standalone plot figure for a single search result."""
    t_cfg = get_theme(theme)

    aspect = grid.rows / float(grid.cols)
    width = max(4.0, min(9.0, 7.0))
    height = max(4.0, min(9.0, width * aspect + 1.2))

    fig = Figure(figsize=(width, height))
    fig.set_facecolor(t_cfg["bg"])
    ax = fig.add_subplot(1, 1, 1)

    disp_title = title if title is not None else result_label(result)
    disp_sub = result_subtitle(grid, result)

    draw_grid(
        ax=ax,
        grid=grid,
        result=result,
        theme=theme,
        show_explored=show_explored,
        explored_style=explored_style,
        title=disp_title,
        subtitle=disp_sub,
    )
    _add_legend(fig, theme, grid, show_explored, explored_style)
    fig.subplots_adjust(bottom=0.14, top=0.88)

    if save_path:
        _save_fig(fig, save_path, dpi=dpi)

    return fig


def plot_comparison(
    grid: Grid,
    results: Mapping[str, SearchResult],
    theme: str = "dark",
    ncols: int = 3,
    show_explored: bool = True,
    suptitle: Optional[str] = None,
    save_path: Optional[Union[str, pathlib.Path]] = None,
    dpi: int = 150,
) -> Figure:
    """Generate side-by-side subplot panels comparing multiple search algorithms on a grid."""
    if not results:
        raise ValueError("results mapping cannot be empty")

    t_cfg = get_theme(theme)
    labels = sort_algorithm_labels(results.keys())

    found_costs = [res.path_cost for res in results.values() if res.found]
    best_cost = min(found_costs) if found_costs else None

    n_plots = len(labels)
    nrows = math.ceil(n_plots / float(ncols))
    aspect = grid.rows / float(grid.cols)

    fig_w = ncols * 3.5 + 1.0
    fig_h = nrows * max(2.5, 3.5 * aspect) + (1.5 if suptitle else 1.2)

    fig = Figure(figsize=(fig_w, fig_h))
    fig.set_facecolor(t_cfg["bg"])

    if suptitle:
        fig.suptitle(suptitle, color=t_cfg["text"], fontsize=13, y=0.98)

    for i, lbl in enumerate(labels):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        res = results[lbl]
        sub = result_subtitle(grid, res, best_cost=best_cost)
        draw_grid(
            ax=ax,
            grid=grid,
            result=res,
            theme=theme,
            show_explored=show_explored,
            title=lbl,
            subtitle=sub,
        )

    for j in range(n_plots, nrows * ncols):
        ax_empty = fig.add_subplot(nrows, ncols, j + 1)
        ax_empty.set_visible(False)

    _add_legend(fig, theme, grid, show_explored)
    fig.subplots_adjust(
        bottom=0.12,
        top=0.90 if suptitle else 0.94,
        wspace=0.25,
        hspace=0.35,
    )

    if save_path:
        _save_fig(fig, save_path, dpi=dpi)

    return fig


# =====================================================================
# PART E — Benchmark Charts
# =====================================================================

def plot_metric_by_param(
    summary: pd.DataFrame,
    category: str,
    metric: str,
    ylabel: str,
    title: str,
    theme: str = "dark",
    log_y: bool = False,
    save_path: Optional[Union[str, pathlib.Path]] = None,
    dpi: int = 150,
) -> Figure:
    """Generate a parameter sweep line chart for a specific category in summary data."""
    t_cfg = get_theme(theme)
    cat_df = summary[summary["category"] == category]

    if cat_df.empty:
        raise ValueError(f"Category {category!r} has no rows in summary")

    params = sorted(cat_df["param_label"].unique(), key=natural_key)
    algos = sort_algorithm_labels(cat_df["algorithm"].unique())

    fig = Figure(figsize=(7.5, 4.8))
    fig.set_facecolor(t_cfg["bg"])
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(t_cfg["panel"])

    for algo in algos:
        sub = cat_df[cat_df["algorithm"] == algo].set_index("param_label")
        y_vals = [sub.loc[p, metric] if p in sub.index else np.nan for p in params]
        ax.plot(
            params,
            y_vals,
            marker="o",
            color=algo_color(algo),
            label=algo,
            linewidth=2.0,
            markersize=6.0,
        )

    if log_y:
        ax.set_yscale("log")

    ax.set_title(title, color=t_cfg["text"], fontsize=11, pad=10)
    ax.set_ylabel(ylabel, color=t_cfg["text"], fontsize=10)
    ax.set_xlabel("Parameter", color=t_cfg["text"], fontsize=10)
    ax.tick_params(colors=t_cfg["muted"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(t_cfg["grid_line"])
    ax.grid(True, color=t_cfg["grid_line"], linestyle="--", alpha=0.6)
    ax.legend(
        frameon=False,
        labelcolor=t_cfg["text"],
        fontsize=8.5,
        loc="best",
    )

    if save_path:
        _save_fig(fig, save_path, dpi=dpi)

    return fig


def plot_expansion_efficiency(
    summary: pd.DataFrame,
    theme: str = "dark",
    save_path: Optional[Union[str, pathlib.Path]] = None,
    dpi: int = 150,
) -> Figure:
    """Generate grouped bar chart comparing node expansion efficiency against Dijkstra."""
    t_cfg = get_theme(theme)
    valid_categories = [c for c in summary["category"].unique() if c != "unsolvable"]
    algos = sort_algorithm_labels(
        summary[summary["category"].isin(valid_categories)]["algorithm"].unique()
    )

    fig = Figure(figsize=(9.0, 5.0))
    fig.set_facecolor(t_cfg["bg"])
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(t_cfg["panel"])

    x = np.arange(len(valid_categories))
    width = 0.8 / float(max(len(algos), 1))

    for i, algo in enumerate(algos):
        vals: List[float] = []
        for cat in valid_categories:
            sub = summary[(summary["category"] == cat) & (summary["algorithm"] == algo)]
            vals.append(float(sub["mean_nodes_vs_dijkstra_pct"].mean()) if not sub.empty else 0.0)
        offset = (i - len(algos) / 2.0 + 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=algo,
            color=algo_color(algo),
            edgecolor=t_cfg["panel"],
        )

    ax.axhline(
        100.0,
        color=t_cfg["muted"],
        linestyle="--",
        linewidth=1.5,
        label="Dijkstra baseline",
    )

    ax.set_title(
        "Search Efficiency Relative to Dijkstra (Nodes Expanded %)",
        color=t_cfg["text"],
        fontsize=11,
        pad=10,
    )
    ax.set_ylabel("Nodes Expanded (% of Dijkstra)", color=t_cfg["text"], fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_categories, color=t_cfg["muted"], rotation=15, ha="right", fontsize=9)
    ax.tick_params(colors=t_cfg["muted"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(t_cfg["grid_line"])
    ax.grid(True, color=t_cfg["grid_line"], linestyle="--", alpha=0.6, axis="y")
    ax.legend(frameon=False, labelcolor=t_cfg["text"], fontsize=8, loc="best")

    if save_path:
        _save_fig(fig, save_path, dpi=dpi)

    return fig


def plot_cost_quality(
    summary: pd.DataFrame,
    theme: str = "dark",
    save_path: Optional[Union[str, pathlib.Path]] = None,
    dpi: int = 150,
) -> Figure:
    """Generate grouped bar chart comparing suboptimality cost percentages across categories."""
    t_cfg = get_theme(theme)
    valid_categories = [c for c in summary["category"].unique() if c != "unsolvable"]
    algos = sort_algorithm_labels(
        summary[summary["category"].isin(valid_categories)]["algorithm"].unique()
    )

    fig = Figure(figsize=(9.0, 5.0))
    fig.set_facecolor(t_cfg["bg"])
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(t_cfg["panel"])

    x = np.arange(len(valid_categories))
    width = 0.8 / float(max(len(algos), 1))

    for i, algo in enumerate(algos):
        vals: List[float] = []
        for cat in valid_categories:
            sub = summary[(summary["category"] == cat) & (summary["algorithm"] == algo)]
            if not sub.empty and not math.isnan(sub["mean_cost_ratio"].mean()):
                ratio = float(sub["mean_cost_ratio"].mean())
                vals.append(max(0.0, (ratio - 1.0) * 100.0))
            else:
                vals.append(0.0)

        offset = (i - len(algos) / 2.0 + 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=algo,
            color=algo_color(algo),
            edgecolor=t_cfg["panel"],
        )

        for bar_patch, val in zip(bars, vals):
            if val > 0.05:
                ax.text(
                    bar_patch.get_x() + bar_patch.get_width() / 2.0,
                    val + 0.5,
                    f"+{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color=t_cfg["text"],
                )

    ax.axhline(0.0, color=t_cfg["muted"], linestyle="-", linewidth=1.0)

    ax.set_title(
        "Solution Quality Suboptimality (% Cost Above Optimal)",
        color=t_cfg["text"],
        fontsize=11,
        pad=10,
    )
    ax.set_ylabel("% Above Optimal Cost", color=t_cfg["text"], fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_categories, color=t_cfg["muted"], rotation=15, ha="right", fontsize=9)
    ax.tick_params(colors=t_cfg["muted"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(t_cfg["grid_line"])
    ax.grid(True, color=t_cfg["grid_line"], linestyle="--", alpha=0.6, axis="y")
    ax.legend(frameon=False, labelcolor=t_cfg["text"], fontsize=8, loc="best")

    if save_path:
        _save_fig(fig, save_path, dpi=dpi)

    return fig


def per_node_cost_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate linear least-squares microseconds per expanded node per algorithm."""
    found_df = results_df[results_df["found"] == True]
    records: List[Dict[str, Any]] = []

    for algo in sort_algorithm_labels(found_df["algorithm"].unique()):
        sub = found_df[found_df["algorithm"] == algo]
        x = sub["nodes_expanded"].to_numpy(dtype=float)
        y = sub["runtime_median_ms"].to_numpy(dtype=float)

        denom = float(np.sum(x * x))
        us_per_node = float(1000.0 * np.sum(x * y) / denom) if denom > 0.0 else 0.0

        records.append(
            {
                "algorithm": algo,
                "n_runs": len(sub),
                "us_per_node": round(us_per_node, 3),
            }
        )

    return pd.DataFrame(records)


def plot_runtime_vs_expansions(
    results_df: pd.DataFrame,
    theme: str = "dark",
    save_path: Optional[Union[str, pathlib.Path]] = None,
    dpi: int = 150,
) -> Figure:
    """Generate log-log scatter plot of nodes expanded vs runtime with per-node slopes."""
    t_cfg = get_theme(theme)
    found_df = results_df[results_df["found"] == True]
    pn_df = per_node_cost_table(results_df).set_index("algorithm")

    fig = Figure(figsize=(8.0, 5.0))
    fig.set_facecolor(t_cfg["bg"])
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(t_cfg["panel"])

    for algo in sort_algorithm_labels(found_df["algorithm"].unique()):
        sub = found_df[found_df["algorithm"] == algo]
        us_val = pn_df.loc[algo, "us_per_node"] if algo in pn_df.index else 0.0
        label_text = f"{algo} ({us_val:.1f} µs/node)"

        ax.scatter(
            sub["nodes_expanded"],
            sub["runtime_median_ms"],
            color=algo_color(algo),
            label=label_text,
            alpha=0.75,
            edgecolors="none",
            s=28,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title("Runtime vs. Nodes Expanded (Log-Log Scale)", color=t_cfg["text"], fontsize=11, pad=10)
    ax.set_xlabel("Nodes Expanded", color=t_cfg["text"], fontsize=10)
    ax.set_ylabel("Runtime (ms)", color=t_cfg["text"], fontsize=10)
    ax.tick_params(colors=t_cfg["muted"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(t_cfg["grid_line"])
    ax.grid(True, color=t_cfg["grid_line"], linestyle="--", alpha=0.6)
    ax.legend(frameon=False, labelcolor=t_cfg["text"], fontsize=8.5, loc="upper left")

    if save_path:
        _save_fig(fig, save_path, dpi=dpi)

    return fig


def plot_winner_counts(
    winners: pd.DataFrame,
    theme: str = "dark",
    save_path: Optional[Union[str, pathlib.Path]] = None,
    dpi: int = 150,
) -> Figure:
    """Generate dual horizontal-bar panels tallying algorithm win shares."""
    t_cfg = get_theme(theme)
    fig = Figure(figsize=(9.0, 4.5))
    fig.set_facecolor(t_cfg["bg"])

    metrics_cfg = [
        ("nodes_expanded", "Fewest nodes expanded"),
        ("runtime_median_ms", "Fastest runtime"),
    ]

    all_algos = sort_algorithm_labels(winners["algorithm"].unique())

    for idx, (metric_key, panel_title) in enumerate(metrics_cfg):
        ax = fig.add_subplot(1, 2, idx + 1)
        ax.set_facecolor(t_cfg["panel"])

        sub = winners[winners["metric"] == metric_key].set_index("algorithm")
        y_pos = np.arange(len(all_algos))
        win_pcts = [float(sub.loc[a, "win_pct"]) if a in sub.index else 0.0 for a in all_algos]

        bars = ax.barh(
            y_pos,
            win_pcts,
            color=[algo_color(a) for a in all_algos],
            edgecolor=t_cfg["panel"],
            height=0.6,
        )

        for bar_patch, pct in zip(bars, win_pcts):
            ax.text(
                pct + 1.5,
                bar_patch.get_y() + bar_patch.get_height() / 2.0,
                f"{pct:.1f}%",
                va="center",
                fontsize=8,
                color=t_cfg["text"],
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_algos if idx == 0 else [], color=t_cfg["text"], fontsize=8.5)
        ax.set_xlim(0, 110)
        ax.set_title(panel_title, color=t_cfg["text"], fontsize=10.5, pad=8)
        ax.set_xlabel("Win %", color=t_cfg["text"], fontsize=9.5)
        ax.tick_params(colors=t_cfg["muted"], labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color(t_cfg["grid_line"])
        ax.grid(True, color=t_cfg["grid_line"], linestyle="--", alpha=0.6, axis="x")

    fig.subplots_adjust(wspace=0.25, bottom=0.15, top=0.88)

    if save_path:
        _save_fig(fig, save_path, dpi=dpi)

    return fig


# =====================================================================
# PART F — Export Helpers
# =====================================================================

def export_gallery(
    name: str,
    grid: Grid,
    results: Mapping[str, SearchResult],
    out_dir: Union[str, pathlib.Path] = "outputs/paths",
    theme: str = "light",
) -> List[str]:
    """Export single-run result PNGs and comparison PNG for a scenario instance.

    Args:
        name: Scenario descriptor name.
        grid: Grid instance.
        results: Dictionary mapping algorithm label to SearchResult.
        out_dir: Output folder.
        theme: Theme name ('light' or 'dark').

    Returns:
        List of generated image file paths.
    """
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    slug_name = slugify(name)
    saved_files: List[str] = []

    for label, res in results.items():
        slug_label = slugify(label)
        file_path = out_path / f"{slug_name}__{slug_label}.png"
        fig = plot_result(
            grid=grid,
            result=res,
            theme=theme,
            title=label,
            save_path=file_path,
        )
        saved_files.append(str(file_path.resolve()))
        del fig

    comp_path = out_path / f"{slug_name}__comparison.png"
    fig_comp = plot_comparison(
        grid=grid,
        results=results,
        theme=theme,
        suptitle=name,
        save_path=comp_path,
    )
    saved_files.append(str(comp_path.resolve()))
    del fig_comp

    return saved_files


def export_benchmark_charts(
    bench: Any,
    out_dir: Union[str, pathlib.Path] = "outputs/charts",
    theme: str = "dark",
) -> Dict[str, str]:
    """Generate and write standard benchmark analysis charts (01 to 07).

    Args:
        bench: BenchmarkResult object (or any object with results, summary, winners).
        out_dir: Target output directory.
        theme: Matplotlib theme palette ('dark' or 'light').

    Returns:
        Mapping of chart filename to absolute output file path string.
    """
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    chart_paths: Dict[str, str] = {}

    summary = bench.summary
    results = bench.results
    winners = bench.winners

    # 01_nodes_vs_density.png
    if "density_sweep" in summary["category"].values:
        p01 = out_path / "01_nodes_vs_density.png"
        fig01 = plot_metric_by_param(
            summary=summary,
            category="density_sweep",
            metric="mean_nodes_expanded",
            ylabel="Nodes Expanded",
            title="Nodes Expanded vs. Obstacle Density",
            theme=theme,
            save_path=p01,
        )
        chart_paths["01_nodes_vs_density.png"] = str(p01.resolve())
        del fig01

    # 02_nodes_vs_size.png
    if "size_sweep" in summary["category"].values:
        p02 = out_path / "02_nodes_vs_size.png"
        fig02 = plot_metric_by_param(
            summary=summary,
            category="size_sweep",
            metric="mean_nodes_expanded",
            ylabel="Nodes Expanded (Log Scale)",
            title="Nodes Expanded vs. Grid Size",
            theme=theme,
            log_y=True,
            save_path=p02,
        )
        chart_paths["02_nodes_vs_size.png"] = str(p02.resolve())
        del fig02

    # 03_runtime_vs_size.png
    if "size_sweep" in summary["category"].values:
        p03 = out_path / "03_runtime_vs_size.png"
        fig03 = plot_metric_by_param(
            summary=summary,
            category="size_sweep",
            metric="median_runtime_ms",
            ylabel="Median Runtime (ms, Log Scale)",
            title="Runtime vs. Grid Size",
            theme=theme,
            log_y=True,
            save_path=p03,
        )
        chart_paths["03_runtime_vs_size.png"] = str(p03.resolve())
        del fig03

    # 04_expansion_efficiency.png
    p04 = out_path / "04_expansion_efficiency.png"
    fig04 = plot_expansion_efficiency(summary=summary, theme=theme, save_path=p04)
    chart_paths["04_expansion_efficiency.png"] = str(p04.resolve())
    del fig04

    # 05_cost_quality.png
    p05 = out_path / "05_cost_quality.png"
    fig05 = plot_cost_quality(summary=summary, theme=theme, save_path=p05)
    chart_paths["05_cost_quality.png"] = str(p05.resolve())
    del fig05

    # 06_runtime_vs_expansions.png
    p06 = out_path / "06_runtime_vs_expansions.png"
    fig06 = plot_runtime_vs_expansions(results_df=results, theme=theme, save_path=p06)
    chart_paths["06_runtime_vs_expansions.png"] = str(p06.resolve())
    del fig06

    # 07_winners.png
    p07 = out_path / "07_winners.png"
    fig07 = plot_winner_counts(winners=winners, theme=theme, save_path=p07)
    chart_paths["07_winners.png"] = str(p07.resolve())
    del fig07

    return chart_paths


def charts_from_csv(
    csv_path: Union[str, pathlib.Path] = "outputs/logs/results.csv",
    out_dir: Union[str, pathlib.Path] = "outputs/charts",
    theme: str = "dark",
) -> Dict[str, str]:
    """Load benchmark results from CSV, regenerate summary and winner tables, and export charts."""
    path = pathlib.Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Results CSV not found at {csv_path}. Please run: python -m src.benchmark --run standard"
        )

    results_df = pd.read_csv(path)
    summary_df = summarize(results_df)
    winners_df = winner_counts(results_df)

    bench = BenchmarkResult(
        results=results_df,
        summary=summary_df,
        winners=winners_df,
        meta={"source_csv": str(path.resolve())},
    )
    return export_benchmark_charts(bench=bench, out_dir=out_dir, theme=theme)


# =====================================================================
# PART G — CLI and Self-Test
# =====================================================================

if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(
        description="Pathfinding static visualizer and benchmark chart exporter."
    )
    parser.add_argument(
        "--charts",
        action="store_true",
        help="Generate standard benchmark charts from CSV.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="outputs/logs/results.csv",
        help="Path to results CSV.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/charts",
        help="Output folder for charts.",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="dark",
        choices=["dark", "light"],
        help="Theme palette.",
    )

    args = parser.parse_args()

    if args.charts:
        charts = charts_from_csv(csv_path=args.csv, out_dir=args.out, theme=args.theme)
        print("Generated benchmark charts:")
        for fname, fpath in charts.items():
            print(f"  {fname}: {fpath}")
        sys.exit(0)

    # =================================================================
    # Self-Test Execution
    # =================================================================
    print("Executing Phase 5 Visualizer self-test suite...")

    # 1. Gallery export across 4 sample scenarios
    sample_scenarios: List[Tuple[str, Grid]] = [
        ("trap 12x20", generate_trap_grid(12, 20)),
        ("mud barrier 15x25", generate_mud_barrier_grid(15, 25)),
        ("random 40x40", generate_random_grid(40, 40, 0.25, seed=3)),
        ("maze 21x21", generate_perfect_maze(21, 21, seed=7)),
    ]
    algo_selection = [
        ("Dijkstra", None),
        ("BFS", None),
        ("A* (manhattan)", "manhattan"),
        ("A* (euclidean)", "euclidean"),
        ("Greedy Best-First (manhattan)", "manhattan"),
    ]

    png_signature = b"\x89PNG\r\n\x1a\n"
    all_exported_files: List[str] = []

    for scen_name, scen_grid in sample_scenarios:
        scen_results: Dict[str, SearchResult] = {}
        for base_algo, heur in algo_selection:
            # Map algorithm name for run_algorithm
            name_key = "A*" if "A*" in base_algo else ("Greedy Best-First" if "Greedy" in base_algo else base_algo)
            label_key = f"{name_key} ({heur})" if heur else name_key
            res = run_algorithm(
                name=name_key,
                grid=scen_grid,
                allow_diagonal=False,
                heuristic=heur,
                record_order=True,
            )
            scen_results[label_key] = res

        exported_paths = export_gallery(
            name=scen_name,
            grid=scen_grid,
            results=scen_results,
            out_dir="outputs/paths",
            theme="light",
        )
        assert len(exported_paths) == 6, (
            f"Test 1 failed: expected 6 files for {scen_name}, got {len(exported_paths)}"
        )
        for ep in exported_paths:
            p_file = pathlib.Path(ep)
            assert p_file.exists() and p_file.stat().st_size > 0, (
                f"Test 1 failed: file missing or empty at {ep}"
            )
            with open(p_file, "rb") as pf:
                sig = pf.read(8)
                assert sig == png_signature[:8], f"Test 1 failed: invalid PNG signature in {ep}"
        all_exported_files.extend(exported_paths)

    # 2. Low-level render_rgb verification
    grid_test = generate_random_grid(20, 20, 0.2, seed=42)
    res_test = run_algorithm("Dijkstra", grid_test, allow_diagonal=False, record_order=True)

    rgb_base = render_rgb(grid_test, result=None, theme="dark")
    assert rgb_base.dtype == np.uint8, "Test 2 failed: rgb dtype != uint8"
    assert rgb_base.shape == (20, 20, 3), f"Test 2 failed: rgb shape {rgb_base.shape} != (20, 20, 3)"

    theme_dark = THEMES["dark"]
    wall_rgb_exp = _hex_to_rgb255(theme_dark["wall"])
    free_rgb_exp = _hex_to_rgb255(theme_dark["free"])
    # Find a wall and free coordinate
    wall_pos = tuple(np.argwhere(grid_test.cells == WALL)[0])
    free_pos = tuple(np.argwhere(grid_test.cells == FREE)[0])
    assert np.array_equal(rgb_base[wall_pos[0], wall_pos[1]], wall_rgb_exp), (
        "Test 2 failed: wall cell color mismatch"
    )
    assert np.array_equal(rgb_base[free_pos[0], free_pos[1]], free_rgb_exp), (
        "Test 2 failed: free cell color mismatch"
    )

    rgb_zero = render_rgb(grid_test, result=res_test, theme="dark", expanded_upto=0)
    assert np.array_equal(rgb_zero, rgb_base), (
        "Test 2 failed: expanded_upto=0 does not equal result=None array"
    )

    rgb_five = render_rgb(grid_test, result=res_test, theme="dark", expanded_upto=5)
    diff_mask = np.any(rgb_five != rgb_base, axis=-1)
    unique_first_5 = len(set(res_test.expansion_order[:5]))
    assert np.count_nonzero(diff_mask) == unique_first_5, (
        f"Test 2 failed: diff cells ({np.count_nonzero(diff_mask)}) != unique first 5 ({unique_first_5})"
    )

    rgb_path = render_rgb(grid_test, result=res_test, theme="dark", path_as_cells=True)
    path_rgb_exp = _hex_to_rgb255(theme_dark["path"])
    start_rgb_exp = _hex_to_rgb255(theme_dark["start"])
    goal_rgb_exp = _hex_to_rgb255(theme_dark["goal"])

    assert np.array_equal(rgb_path[grid_test.start[0], grid_test.start[1]], start_rgb_exp), (
        "Test 2 failed: start cell color mismatch in path_as_cells mode"
    )
    assert np.array_equal(rgb_path[grid_test.goal[0], grid_test.goal[1]], goal_rgb_exp), (
        "Test 2 failed: goal cell color mismatch in path_as_cells mode"
    )
    for p in res_test.path[1:-1]:
        assert np.array_equal(rgb_path[p[0], p[1]], path_rgb_exp), (
            f"Test 2 failed: interior path cell at {p} color mismatch"
        )

    # 3. Helpers verification
    density_labels = ["density=0.30", "density=0.10", "density=0.35", "density=0.20"]
    sorted_densities = sorted(density_labels, key=natural_key)
    assert sorted_densities == ["density=0.10", "density=0.20", "density=0.30", "density=0.35"], (
        f"Test 3 failed: natural_key density sort mismatch: {sorted_densities}"
    )

    size_labels = ["size=100x100", "size=20x20", "size=50x50"]
    sorted_sizes = sorted(size_labels, key=natural_key)
    assert sorted_sizes == ["size=20x20", "size=50x50", "size=100x100"], (
        f"Test 3 failed: natural_key size sort mismatch: {sorted_sizes}"
    )

    test_algo_list = ["Greedy Best-First (octile)", "Unknown Z", "Dijkstra", "A* (manhattan)", "Unknown A"]
    sorted_algos = sort_algorithm_labels(test_algo_list)
    assert sorted_algos == [
        "Dijkstra",
        "A* (manhattan)",
        "Greedy Best-First (octile)",
        "Unknown A",
        "Unknown Z",
    ], f"Test 3 failed: sort_algorithm_labels mismatch: {sorted_algos}"

    assert slugify("A* (manhattan)") == "a_manhattan", "Test 3 failed: slugify mismatch"
    known_colors = [ALGO_COLORS[k] for k in ALGO_ORDER]
    assert len(set(known_colors)) == 8, "Test 3 failed: duplicate colors among 8 known algorithms"

    c1 = algo_color("Unknown X")
    c2 = algo_color("Unknown X")
    assert c1 == c2, "Test 3 failed: non-deterministic algo_color fallback"

    # 4. plot_result returns Figure across variants
    fig_found = plot_result(grid_test, res_test)
    assert isinstance(fig_found, Figure), "Test 4 failed: plot_result did not return Figure"

    grid_unsolv = generate_unsolvable_grid(10, 10, seed=0)
    res_unsolv = run_algorithm("Dijkstra", grid_unsolv, allow_diagonal=False)
    sub_unsolv = result_subtitle(grid_unsolv, res_unsolv)
    assert sub_unsolv.startswith("NO PATH FOUND"), (
        f"Test 4 failed: unsolvable subtitle '{sub_unsolv}' does not start with 'NO PATH FOUND'"
    )
    fig_unsolv = plot_result(grid_unsolv, res_unsolv)
    assert isinstance(fig_unsolv, Figure), "Test 4 failed: plot_result failed on unsolvable grid"

    res_no_order = run_algorithm("Dijkstra", grid_test, allow_diagonal=False, record_order=False)
    fig_no_order = plot_result(grid_test, res_no_order)
    assert isinstance(fig_no_order, Figure), "Test 4 failed: plot_result failed on record_order=False"

    # 5. plot_comparison on mud barrier verification
    grid_mb = generate_mud_barrier_grid(15, 25)
    mb_results: Dict[str, SearchResult] = {}
    for base_a, h in algo_selection:
        nk = "A*" if "A*" in base_a else ("Greedy Best-First" if "Greedy" in base_a else base_a)
        lk = f"{nk} ({h})" if h else nk
        mb_results[lk] = run_algorithm(nk, grid_mb, allow_diagonal=False, heuristic=h, record_order=True)

    fig_comp_mb = plot_comparison(grid_mb, mb_results, ncols=3)
    visible_axes = [ax for ax in fig_comp_mb.axes if ax.get_visible()]
    assert len(visible_axes) == 5, f"Test 5 failed: expected 5 visible axes, got {len(visible_axes)}"

    # Check axes titles
    titles = [ax.get_title() for ax in visible_axes]
    greedy_title = next(t for t in titles if "Greedy Best-First" in t)
    dijkstra_title = next(t for t in titles if "Dijkstra" in t)
    assert "+38.9% cost" in greedy_title, (
        f"Test 5 failed: Greedy title missing '+38.9% cost', got '{greedy_title}'"
    )
    assert "optimal" in dijkstra_title, (
        f"Test 5 failed: Dijkstra title missing 'optimal', got '{dijkstra_title}'"
    )

    try:
        plot_comparison(grid_mb, {})
        assert False, "Test 5 failed: expected ValueError for empty results mapping"
    except ValueError:
        pass

    # 6. run_benchmark("quick") and export_benchmark_charts
    b_res_q = run_benchmark("quick", repeats=3)
    pn_table = per_node_cost_table(b_res_q.results)
    assert (pn_table["us_per_node"] > 0).all(), (
        f"Test 6 failed: us_per_node <= 0 detected:\n{pn_table}"
    )

    try:
        plot_metric_by_param(b_res_q.summary, "non_existent_category", "mean_nodes_expanded", "Y", "Title")
        assert False, "Test 6 failed: expected ValueError for nonexistent category"
    except ValueError:
        pass

    exported_charts = export_benchmark_charts(b_res_q, out_dir="outputs/charts/quick")
    assert len(exported_charts) == 7, (
        f"Test 6 failed: expected 7 exported charts, got {len(exported_charts)}"
    )
    for cname, cpath in exported_charts.items():
        cp = pathlib.Path(cpath)
        assert cp.exists() and cp.stat().st_size > 0, f"Test 6 failed: chart missing or empty at {cpath}"
        with open(cp, "rb") as cf:
            assert cf.read(8) == png_signature[:8], f"Test 6 failed: chart {cname} not valid PNG"
        all_exported_files.append(str(cp))

    # 7. Rule enforcement: matplotlib.pyplot must NOT be in sys.modules
    assert "matplotlib.pyplot" not in sys.modules, (
        "Test 7 failed: 'matplotlib.pyplot' was imported, violating OO API isolation rule!"
    )

    print("\n" + "=" * 80)
    print(" LINEAR REGRESSION PER-NODE RUNTIME COST (µs / node)")
    print("=" * 80)
    print(pn_table.to_string(index=False))

    print("\n" + "=" * 80)
    print(f" GENERATED FILES ({len(all_exported_files)} total)")
    print("=" * 80)
    for f in all_exported_files:
        print(f"  {f}")

    print("\nAll Phase 5 tests passed!")
