"""GIF animations, search progress logs, and scenario bundle export engine.

Provides deterministic single-search and multi-algorithm race GIF animations,
detailed step-by-step search logging with search efficiency metrics, Matplotlib
progress charts (using the object-oriented API without pyplot), and unified
scenario bundle exporters.
"""

from dataclasses import replace
import json
import math
import pathlib
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from PIL import Image, ImageDraw, ImageFont
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from src.algorithms import SearchResult, run_algorithm
from src.benchmark import default_algorithm_configs
from src.grid import (
    Grid,
    MUD,
    Pos,
    generate_mud_barrier_grid,
    generate_perfect_maze,
    generate_random_grid,
    generate_trap_grid,
    generate_unsolvable_grid,
)
from src.visualizer import (
    THEMES,
    algo_color,
    export_gallery,
    get_theme,
    render_rgb,
    slugify,
    sort_algorithm_labels,
)


# =====================================================================
# PART A — Constants and Small Helpers
# =====================================================================

DEFAULT_MAX_FRAMES: int = 80
DEFAULT_FRAME_MS: int = 60
DEFAULT_PATH_MS: int = 45
DEFAULT_HOLD_MS: int = 2000
MAX_PATH_FRAMES: int = 24
MIN_CANVAS_W: int = 320


def build_frame_steps(n_total: int, max_frames: int) -> List[int]:
    """Generate strictly increasing, unique integer step indices between 0 and n_total.

    Always includes 0 and n_total, with at most max_frames entries evenly distributed.
    For n_total == 0 returns [0]. For small n_total (<= max_frames - 1) returns every
    integer from 0 to n_total.
    """
    if n_total <= 0:
        return [0]
    if n_total <= max_frames - 1:
        return list(range(n_total + 1))

    # Evenly spread linspace, rounded and deduplicated
    raw_steps = np.round(np.linspace(0, n_total, max_frames)).astype(int)
    unique_steps = np.unique(raw_steps)
    unique_steps[0] = 0
    unique_steps[-1] = n_total
    clean = sorted(set(int(x) for x in unique_steps))
    return clean


def choose_cell_px(
    rows: int,
    cols: int,
    max_panel_px: int,
    max_cell_px: int = 24,
) -> int:
    """Calculate integer pixel width per grid cell constrained by panel dimensions."""
    return max(1, min(max_cell_px, max_panel_px // max(rows, cols)))


def race_ranking(results: Mapping[str, SearchResult]) -> Dict[str, Optional[int]]:
    """Compute competition ranks (1, 2, 3, 4, 4, 6...) based on ascending nodes_expanded.

    Ties share the best rank with standard skipping. Only searches that found a path
    receive integer ranks. Unsuccessful searches (found == False) map to None.
    """
    ranks: Dict[str, Optional[int]] = {}
    found_items = [(lbl, res) for lbl, res in results.items() if res.found]
    found_items.sort(key=lambda item: item[1].nodes_expanded)

    prev_nodes: Optional[int] = None
    prev_rank: Optional[int] = None

    for i, (lbl, res) in enumerate(found_items):
        if prev_nodes is not None and res.nodes_expanded == prev_nodes:
            ranks[lbl] = prev_rank
        else:
            rank = i + 1
            ranks[lbl] = rank
            prev_rank = rank
            prev_nodes = res.nodes_expanded

    for lbl, res in results.items():
        if not res.found:
            ranks[lbl] = None

    return ranks


def _hex_to_rgb(hex_str: str) -> Tuple[int, int, int]:
    """Convert hex color string (e.g. '#121212' or '121212') to (R, G, B) integer tuple."""
    clean = hex_str.lstrip("#")
    return (int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16))


def _load_font(size: int) -> ImageFont.ImageFont:
    """Load default Pillow font at given point size, falling back if size is unsupported."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: Any,
    max_w: float,
) -> str:
    """Truncate text with an ellipsis ('…') until its rendered width fits within max_w."""
    if draw.textlength(text, font=font) <= max_w:
        return text
    curr = text
    while curr and draw.textlength(curr + "…", font=font) > max_w:
        curr = curr[:-1]
    return curr + "…" if curr else "…"


def _format_ordinal(n: int) -> str:
    """Format an integer into an ordinal string e.g. 1st, 2nd, 3rd, 4th."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _upscale(
    img_rgb: np.ndarray,
    cell_px: int,
    lines: bool,
    line_rgb: Tuple[int, int, int],
) -> np.ndarray:
    """Nearest-neighbour integer upscale with optional 50% cell boundary blending."""
    upscaled = np.repeat(np.repeat(img_rgb, cell_px, axis=0), cell_px, axis=1)
    if lines and cell_px >= 10:
        h, w, _ = upscaled.shape
        line_arr = np.array(line_rgb, dtype=np.float32)
        r_lines = np.unique(np.append(np.arange(0, h, cell_px), h - 1))
        c_lines = np.unique(np.append(np.arange(0, w, cell_px), w - 1))

        upscaled[r_lines, :, :] = np.round(
            upscaled[r_lines, :, :].astype(np.float32) * 0.5 + line_arr * 0.5
        ).astype(np.uint8)
        upscaled[:, c_lines, :] = np.round(
            upscaled[:, c_lines, :].astype(np.float32) * 0.5 + line_arr * 0.5
        ).astype(np.uint8)
    return upscaled


# =====================================================================
# PART B — GIF Writer
# =====================================================================

def save_gif(
    frames: List[Image.Image],
    durations_ms: List[int],
    path: Union[str, pathlib.Path],
    loop: int = 0,
) -> int:
    """Write an animated GIF with stable global palette quantization and frame deduplication.

    Args:
        frames: Sequence of Pillow Image instances.
        durations_ms: Sequence of frame display durations in milliseconds.
        path: Destination file path.
        loop: Number of animation loops (0 = infinite loop).

    Returns:
        Number of frames actually saved after merging consecutive identical frames.

    Raises:
        ValueError: If frames is empty, durations count mismatches, or frame sizes differ.
    """
    if not frames:
        raise ValueError("frames list cannot be empty")
    if len(frames) != len(durations_ms):
        raise ValueError(
            f"Frame count ({len(frames)}) does not match durations count ({len(durations_ms)})"
        )

    size0 = frames[0].size
    if any(f.size != size0 for f in frames):
        raise ValueError("All animation frames must share identical dimensions")

    # Merge consecutive identical frames by accumulating their display durations
    merged_frames: List[Image.Image] = [frames[0]]
    merged_durations: List[int] = [durations_ms[0]]
    for f, d in zip(frames[1:], durations_ms[1:]):
        if f.tobytes() == merged_frames[-1].tobytes():
            merged_durations[-1] += d
        else:
            merged_frames.append(f)
            merged_durations.append(d)

    # Build one global palette across sampled frames to prevent color flickering
    n_merged = len(merged_frames)
    n_samples = min(8, n_merged)
    sample_indices = np.round(np.linspace(0, n_merged - 1, n_samples)).astype(int)
    sample_indices = sorted(set(int(idx) for idx in sample_indices))

    w, h = size0
    stacked = Image.new("RGB", (w, h * len(sample_indices)))
    for pos_idx, s_idx in enumerate(sample_indices):
        f_rgb = merged_frames[s_idx].convert("RGB")
        stacked.paste(f_rgb, (0, pos_idx * h))

    palette_img = stacked.quantize(
        colors=255,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )

    # Quantize every frame using the unified global palette
    quantized_frames = [
        f.convert("RGB").quantize(palette=palette_img, dither=Image.Dither.NONE)
        for f in merged_frames
    ]

    out_p = pathlib.Path(path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    quantized_frames[0].save(
        out_p,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=merged_durations,
        loop=loop,
        optimize=False,
        disposal=1,
    )

    return len(quantized_frames)


# =====================================================================
# PART C — Single-Search Animation
# =====================================================================

def search_frames(
    grid: Grid,
    result: SearchResult,
    theme: str = "dark",
    max_frames: int = DEFAULT_MAX_FRAMES,
    frame_ms: int = DEFAULT_FRAME_MS,
    path_ms: int = DEFAULT_PATH_MS,
    hold_ms: int = DEFAULT_HOLD_MS,
    max_panel_px: int = 420,
    title: Optional[str] = None,
    explored_style: str = "gradient",
) -> Tuple[List[Image.Image], List[int]]:
    """Construct sequential animation frames and durations for a single search result.

    Args:
        grid: Environment grid.
        result: SearchResult with recorded expansion history.
        theme: Theme name ('dark' or 'light').
        max_frames: Upper bound on exploration animation frames.
        frame_ms: Exploration frame duration in milliseconds.
        path_ms: Path reveal frame duration in milliseconds.
        hold_ms: Final summary frame display hold duration in milliseconds.
        max_panel_px: Maximum bounding dimension for grid pixels.
        title: Optional custom header title.
        explored_style: Exploration color style ('gradient' or 'flat').

    Returns:
        Tuple of (list of PIL Images, list of integer durations in ms).

    Raises:
        ValueError: If result was run with record_order=False.
    """
    if not result.expansion_order:
        raise ValueError("record_order=True is required to animate")

    t_cfg = get_theme(theme)
    bg_rgb = _hex_to_rgb(t_cfg["bg"])
    text_rgb = _hex_to_rgb(t_cfg["text"])
    muted_rgb = _hex_to_rgb(t_cfg["muted"])
    grid_line_rgb = _hex_to_rgb(t_cfg["grid_line"])

    cell_px = choose_cell_px(grid.rows, grid.cols, max_panel_px, max_cell_px=24)
    lines = (cell_px >= 10)
    grid_px_w = grid.cols * cell_px
    grid_px_h = grid.rows * cell_px

    pad = 12
    header_h = 40
    canvas_w = max(MIN_CANVAS_W, grid_px_w + 2 * pad)
    canvas_h = 3 * pad + header_h + grid_px_h
    grid_x = (canvas_w - grid_px_w) // 2
    grid_y = 2 * pad + header_h

    font15 = _load_font(15)
    font12 = _load_font(12)

    def _compose_frame(img_rgb: np.ndarray, line1: str, line2: str) -> Image.Image:
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg_rgb)
        draw = ImageDraw.Draw(canvas)
        l1_fit = _fit_text(draw, line1, font15, canvas_w - 2 * pad)
        l2_fit = _fit_text(draw, line2, font12, canvas_w - 2 * pad)
        draw.text((pad, pad), l1_fit, fill=text_rgb, font=font15)
        draw.text((pad, pad + 20), l2_fit, fill=muted_rgb, font=font12)

        upscaled = _upscale(img_rgb, cell_px, lines, grid_line_rgb)
        grid_img = Image.fromarray(upscaled, mode="RGB")
        canvas.paste(grid_img, (grid_x, grid_y))
        return canvas

    header_line1 = (
        title
        if title is not None
        else (f"{result.algorithm} ({result.heuristic})" if result.heuristic else result.algorithm)
    )
    n_exp = len(result.expansion_order)
    exp_steps = build_frame_steps(n_exp, max_frames)

    frames: List[Image.Image] = []
    durations: List[int] = []

    # 1. Expansion frames
    for s in exp_steps:
        img_rgb = render_rgb(
            grid=grid,
            result=result,
            theme=theme,
            expanded_upto=s,
            explored_style=explored_style,
            path_as_cells=False,
        )
        l2 = f"expanded {s}/{n_exp}"
        frames.append(_compose_frame(img_rgb, header_line1, l2))
        durations.append(frame_ms)

    # 2. Path reveal frames (when found with >= 2 path points)
    if result.found and len(result.path) >= 2:
        path_steps_total = len(result.path) - 1
        k_steps = [k for k in build_frame_steps(path_steps_total, MAX_PATH_FRAMES) if k != 0]
        for k in k_steps:
            res_trunc = replace(result, path=result.path[: k + 1])
            img_rgb = render_rgb(
                grid=grid,
                result=res_trunc,
                theme=theme,
                expanded_upto=None,
                explored_style=explored_style,
                path_as_cells=True,
            )
            l2 = f"path {k}/{path_steps_total} steps"
            frames.append(_compose_frame(img_rgb, header_line1, l2))
            durations.append(path_ms)

    # 3. Final summary frame hold
    summary_l2 = (
        f"{result.path_steps} steps · cost {result.path_cost:.1f}"
        if result.found
        else "NO PATH FOUND"
    )
    final_img_rgb = render_rgb(
        grid=grid,
        result=result,
        theme=theme,
        expanded_upto=None,
        explored_style=explored_style,
        path_as_cells=bool(result.found and result.path),
    )
    final_frame = _compose_frame(final_img_rgb, header_line1, summary_l2)
    frames.append(final_frame)
    durations.append(hold_ms)

    return frames, durations


def animate_search(
    grid: Grid,
    result: SearchResult,
    out_path: Union[str, pathlib.Path],
    **kwargs: Any,
) -> str:
    """Render and save an animated GIF for a single search execution.

    Args:
        grid: Environment grid.
        result: SearchResult instance.
        out_path: Destination path for GIF file.
        **kwargs: Options forwarded to search_frames.

    Returns:
        Resolved output file path string.
    """
    frames, durations = search_frames(grid=grid, result=result, **kwargs)
    save_gif(frames=frames, durations_ms=durations, path=out_path, loop=0)
    return str(pathlib.Path(out_path).resolve())


# =====================================================================
# PART D — Race Animation
# =====================================================================

def race_frames(
    grid: Grid,
    results: Mapping[str, SearchResult],
    theme: str = "dark",
    ncols: int = 3,
    max_frames: int = DEFAULT_MAX_FRAMES,
    frame_ms: int = DEFAULT_FRAME_MS,
    hold_ms: int = DEFAULT_HOLD_MS,
    max_panel_px: int = 300,
    explored_style: str = "gradient",
    title: Optional[str] = None,
) -> Tuple[List[Image.Image], List[int]]:
    """Construct multi-algorithm comparison animation frames synchronized on expansion steps.

    Synchronized by expansion steps rather than execution time: runtime measurements
    are hardware-dependent and noisy, whereas node expansion counts are strictly deterministic.

    Args:
        grid: Environment grid.
        results: Mapping of algorithm labels to SearchResult instances.
        theme: Theme name ('dark' or 'light').
        ncols: Number of panel columns in comparison grid.
        max_frames: Upper bound on total animation frames.
        frame_ms: Frame display duration in milliseconds.
        hold_ms: Final hold duration in milliseconds.
        max_panel_px: Panel dimension constraint for cell sizing.
        explored_style: Exploration overlay style ('gradient' or 'flat').
        title: Optional canvas super-title.

    Returns:
        Tuple of (list of PIL Images, list of integer durations in ms).

    Raises:
        ValueError: If results mapping is empty or any result has empty expansion_order.
    """
    if not results:
        raise ValueError("results mapping cannot be empty")

    for lbl, res in results.items():
        if not res.expansion_order:
            raise ValueError(
                f"record_order=True is required for all results to animate (missing for {lbl!r})"
            )

    t_cfg = get_theme(theme)
    bg_rgb = _hex_to_rgb(t_cfg["bg"])
    panel_bg_rgb = _hex_to_rgb(t_cfg["panel"])
    text_rgb = _hex_to_rgb(t_cfg["text"])
    muted_rgb = _hex_to_rgb(t_cfg["muted"])
    grid_line_rgb = _hex_to_rgb(t_cfg["grid_line"])

    ordered_labels = sort_algorithm_labels(results.keys())
    rankings = race_ranking(results)

    n_results = len(ordered_labels)
    actual_ncols = min(ncols, n_results)
    nrows = math.ceil(n_results / float(actual_ncols))

    cell_px = choose_cell_px(grid.rows, grid.cols, max_panel_px, max_cell_px=24)
    lines = (cell_px >= 10)
    grid_w = grid.cols * cell_px
    grid_h = grid.rows * cell_px

    header_h = 36
    panel_pad = 6
    panel_w = max(grid_w + 2 * panel_pad, 160)
    panel_h = header_h + grid_h + 2 * panel_pad

    pad_outer = 10
    gap = 8
    title_h = 32 if title is not None else 0

    canvas_w = pad_outer * 2 + actual_ncols * panel_w + (actual_ncols - 1) * gap
    canvas_h = pad_outer * 2 + title_h + nrows * panel_h + (nrows - 1) * gap

    font_title = _load_font(16)
    font13 = _load_font(13)
    font11 = _load_font(11)

    max_exp = max(len(results[lbl].expansion_order) for lbl in ordered_labels)
    steps = build_frame_steps(max_exp, max_frames)

    frames: List[Image.Image] = []

    for step_val in steps:
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg_rgb)
        draw = ImageDraw.Draw(canvas)

        if title is not None:
            t_fit = _fit_text(draw, title, font_title, canvas_w - 2 * pad_outer)
            draw.text((pad_outer, pad_outer + 4), t_fit, fill=text_rgb, font=font_title)

        for idx, lbl in enumerate(ordered_labels):
            res = results[lbl]
            n_i = len(res.expansion_order)
            is_done = (step_val >= n_i)
            shown_exp = min(step_val, n_i)

            r = idx // actual_ncols
            c = idx % actual_ncols
            px = pad_outer + c * (panel_w + gap)
            py = pad_outer + title_h + r * (panel_h + gap)

            # Panel background
            draw.rectangle(
                [px, py, px + panel_w - 1, py + panel_h - 1],
                fill=panel_bg_rgb,
            )

            # Panel border: 2px algo_color when done, 1px grid_line when running
            if is_done:
                b_color = _hex_to_rgb(algo_color(lbl))
                draw.rectangle([px, py, px + panel_w - 1, py + panel_h - 1], outline=b_color, width=2)
            else:
                draw.rectangle([px, py, px + panel_w - 1, py + panel_h - 1], outline=grid_line_rgb, width=1)

            # Header text lines
            l1_fit = _fit_text(draw, lbl, font13, panel_w - 2 * panel_pad)
            draw.text((px + panel_pad, py + 4), l1_fit, fill=text_rgb, font=font13)

            if is_done:
                if res.found:
                    r_val = rankings[lbl]
                    r_txt = _format_ordinal(r_val) if r_val is not None else ""
                    l2 = f"✓ {r_txt} · {n_i} nodes · cost {res.path_cost:.1f}"
                else:
                    l2 = f"no path · {n_i} nodes"
            else:
                l2 = f"expanded {shown_exp}/{n_i}"

            l2_fit = _fit_text(draw, l2, font11, panel_w - 2 * panel_pad)
            draw.text((px + panel_pad, py + 20), l2_fit, fill=muted_rgb, font=font11)

            # Grid image
            img_rgb = render_rgb(
                grid=grid,
                result=res,
                theme=theme,
                expanded_upto=n_i if is_done else shown_exp,
                show_explored=True,
                explored_style=explored_style,
                path_as_cells=is_done,
            )
            upscaled = _upscale(img_rgb, cell_px, lines, grid_line_rgb)
            grid_img = Image.fromarray(upscaled, mode="RGB")
            gx = px + (panel_w - grid_w) // 2
            gy = py + header_h + panel_pad
            canvas.paste(grid_img, (gx, gy))

        frames.append(canvas)

    durations = [frame_ms] * len(frames)
    durations[-1] = hold_ms

    return frames, durations


def animate_race(
    grid: Grid,
    results: Mapping[str, SearchResult],
    out_path: Union[str, pathlib.Path],
    **kwargs: Any,
) -> str:
    """Render and save an animated multi-algorithm race GIF.

    Args:
        grid: Environment grid.
        results: Mapping of algorithm labels to SearchResult instances.
        out_path: Destination path for GIF file.
        **kwargs: Options forwarded to race_frames.

    Returns:
        Resolved output file path string.
    """
    frames, durations = race_frames(grid=grid, results=results, **kwargs)
    save_gif(frames=frames, durations_ms=durations, path=out_path, loop=0)
    return str(pathlib.Path(out_path).resolve())


# =====================================================================
# PART E — Search Log, Efficiency & Progress Chart
# =====================================================================

def build_search_log(
    grid: Grid,
    results: Mapping[str, SearchResult],
) -> pd.DataFrame:
    """Construct step-by-step expansion progress logs for all algorithms.

    Args:
        grid: Environment grid.
        results: Mapping of algorithm labels to SearchResult instances.

    Returns:
        DataFrame containing one row per expansion step per algorithm.
    """
    rows: List[Dict[str, Any]] = []
    ordered_labels = sort_algorithm_labels(results.keys())

    for lbl in ordered_labels:
        res = results[lbl]
        if not res.expansion_order:
            continue

        path_set = set(res.path) if res.found else set()
        best_dist = float("inf")
        cum_on_path = 0

        for step_idx, pos in enumerate(res.expansion_order, start=1):
            r, c = pos
            dist = math.hypot(r - grid.goal[0], c - grid.goal[1])
            if dist < best_dist:
                best_dist = dist

            on_path = (pos in path_set)
            if on_path:
                cum_on_path += 1

            eff_pct = (100.0 * cum_on_path) / float(step_idx)

            rows.append(
                {
                    "algorithm": lbl,
                    "step": step_idx,
                    "row": r,
                    "col": c,
                    "dist_to_goal": dist,
                    "best_dist_so_far": best_dist,
                    "on_final_path": on_path,
                    "cumulative_on_path": cum_on_path,
                    "cumulative_efficiency_pct": eff_pct,
                }
            )

    cols = [
        "algorithm",
        "step",
        "row",
        "col",
        "dist_to_goal",
        "best_dist_so_far",
        "on_final_path",
        "cumulative_on_path",
        "cumulative_efficiency_pct",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)

    return pd.DataFrame(rows)


def summarize_search_log(
    log_df: pd.DataFrame,
    results: Mapping[str, SearchResult],
) -> pd.DataFrame:
    """Summarize search efficiency and convergence metrics per algorithm.

    Every node on the final path was expanded during search, so path_nodes
    is exactly the count of useful expansions.

    Args:
        log_df: Step-level expansion log from build_search_log.
        results: Mapping of algorithm labels to SearchResult instances.

    Returns:
        DataFrame summarizing search efficiency, wasted expansions, and cost ratios.
    """
    ordered_labels = sort_algorithm_labels(results.keys())
    found_costs = [res.path_cost for res in results.values() if res.found]
    cheapest_cost = min(found_costs) if found_costs else None

    summary_rows: List[Dict[str, Any]] = []

    for lbl in ordered_labels:
        res = results[lbl]
        found = res.found
        n_exp = res.nodes_expanded
        path_nodes = (res.path_steps + 1) if found else 0
        eff_pct = (100.0 * path_nodes / n_exp) if (found and n_exp > 0) else 0.0
        wasted = (n_exp - path_nodes) if found else n_exp

        # Determine step at which Euclidean distance to goal was halved
        step_half: float = np.nan
        if not log_df.empty:
            algo_log = log_df[log_df["algorithm"] == lbl]
            if not algo_log.empty and found:
                start_dist = float(algo_log.iloc[0]["dist_to_goal"])
                half_dist = start_dist * 0.5
                match_half = algo_log[algo_log["best_dist_so_far"] <= half_dist]
                if not match_half.empty:
                    step_half = float(match_half.iloc[0]["step"])

        if found and cheapest_cost is not None and cheapest_cost > 0:
            c_ratio = float(res.path_cost / cheapest_cost)
        else:
            c_ratio = np.nan

        summary_rows.append(
            {
                "algorithm": lbl,
                "found": found,
                "nodes_expanded": n_exp,
                "path_nodes": path_nodes,
                "search_efficiency_pct": round(eff_pct, 3),
                "wasted_expansions": wasted,
                "step_half_distance": round(step_half, 3) if not np.isnan(step_half) else np.nan,
                "cost_ratio": round(c_ratio, 3) if not np.isnan(c_ratio) else np.nan,
            }
        )

    return pd.DataFrame(summary_rows)


def plot_search_progress(
    log_df: pd.DataFrame,
    grid: Grid,
    theme: str = "dark",
    save_path: Optional[Union[str, pathlib.Path]] = None,
    dpi: int = 150,
) -> Figure:
    """Generate dual-axis search convergence and efficiency comparison plots.

    Args:
        log_df: Step-by-step search log DataFrame from build_search_log.
        grid: Environment grid.
        theme: Theme name ('dark' or 'light').
        save_path: Optional file path to save figure.
        dpi: Export DPI.

    Returns:
        Matplotlib Figure containing 2 subplot axes.

    Raises:
        ValueError: If log_df is empty.
    """
    if log_df.empty:
        raise ValueError("log_df cannot be empty")

    t_cfg = get_theme(theme)
    fig = Figure(figsize=(9.0, 7.5))
    fig.set_facecolor(t_cfg["bg"])

    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    unique_algos = sort_algorithm_labels(log_df["algorithm"].unique())

    for lbl in unique_algos:
        sub = log_df[log_df["algorithm"] == lbl]
        if sub.empty:
            continue
        c = algo_color(lbl)
        steps = sub["step"].to_numpy()
        dists = sub["best_dist_so_far"].to_numpy()
        effs = sub["cumulative_efficiency_pct"].to_numpy()

        ax1.plot(steps, dists, color=c, label=lbl, linewidth=2.0)
        ax1.plot(steps[-1], dists[-1], marker="o", markersize=5, color=c)

        ax2.plot(steps, effs, color=c, label=lbl, linewidth=2.0)
        ax2.plot(steps[-1], effs[-1], marker="o", markersize=5, color=c)

    for ax in (ax1, ax2):
        ax.set_facecolor(t_cfg["panel"])
        ax.tick_params(colors=t_cfg["muted"], labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(t_cfg["grid_line"])
        ax.grid(True, color=t_cfg["grid_line"], linestyle="--", alpha=0.6)

    ax1.set_title("Closest approach to the goal", color=t_cfg["text"], fontsize=12, pad=10)
    ax1.set_ylabel("distance (cells)", color=t_cfg["text"], fontsize=10)
    ax1.legend(
        loc="upper right",
        frameon=True,
        facecolor=t_cfg["panel"],
        edgecolor=t_cfg["grid_line"],
        labelcolor=t_cfg["text"],
        fontsize=8.5,
    )

    ax2.set_title("Share of expansions that lie on the final path", color=t_cfg["text"], fontsize=12, pad=10)
    ax2.set_ylabel("%", color=t_cfg["text"], fontsize=10)
    ax2.set_xlabel("Expansion step", color=t_cfg["text"], fontsize=10)

    fig.subplots_adjust(hspace=0.32, left=0.10, right=0.96, top=0.92, bottom=0.08)

    if save_path:
        p = pathlib.Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig


def export_run_logs(
    name: str,
    grid: Grid,
    results: Mapping[str, SearchResult],
    out_dir: Union[str, pathlib.Path] = "outputs/logs",
) -> List[str]:
    """Serialize step-by-step CSV search logs and scenario JSON optimization metrics.

    Args:
        name: Scenario descriptor string.
        grid: Environment grid.
        results: Mapping of algorithm labels to SearchResult instances.
        out_dir: Output folder.

    Returns:
        List containing CSV and JSON file paths.
    """
    out_p = pathlib.Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    slug = slugify(name)

    log_df = build_search_log(grid=grid, results=results)
    csv_file = out_p / f"{slug}__search_log.csv"
    log_df.to_csv(csv_file, index=False)

    summary_df = summarize_search_log(log_df=log_df, results=results)
    summary_map = {row["algorithm"]: row for _, row in summary_df.iterrows()}

    found_costs = [res.path_cost for res in results.values() if res.found]
    cheapest_cost = min(found_costs) if found_costs else None
    first_res = next(iter(results.values()))

    algo_data: Dict[str, Any] = {}
    for lbl in sort_algorithm_labels(results.keys()):
        res = results[lbl]
        s_row = summary_map.get(lbl, {})
        c_ratio = s_row.get("cost_ratio")
        c_ratio_val = None if (c_ratio is None or pd.isna(c_ratio)) else float(c_ratio)
        ovh_val = None if c_ratio_val is None else round((c_ratio_val - 1.0) * 100.0, 3)

        algo_data[lbl] = {
            "found": res.found,
            "path_steps": res.path_steps,
            "path_cost": None if not res.found else float(res.path_cost),
            "cost_ratio": c_ratio_val,
            "overhead_pct": ovh_val,
            "nodes_expanded": res.nodes_expanded,
            "search_efficiency_pct": float(s_row.get("search_efficiency_pct", 0.0)),
            "runtime_ms": float(res.runtime_ms),
        }

    payload = {
        "scenario": name,
        "grid": {
            "rows": grid.rows,
            "cols": grid.cols,
            "wall_density": round(grid.wall_density(), 4),
            "mud_cells": int(np.count_nonzero(grid.cells == MUD)),
            "allow_diagonal": bool(first_res.allow_diagonal),
        },
        "best_cost": None if cheapest_cost is None else float(cheapest_cost),
        "algorithms": algo_data,
    }

    json_file = out_p / f"{slug}__optimization.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return [str(csv_file.resolve()), str(json_file.resolve())]


# =====================================================================
# PART F — Export Helpers
# =====================================================================

def export_animations(
    name: str,
    grid: Grid,
    results: Mapping[str, SearchResult],
    out_dir: Union[str, pathlib.Path] = "outputs/animations",
    theme: str = "dark",
    individual: bool = True,
) -> List[str]:
    """Export animated GIFs for race comparison and optional individual searches.

    Args:
        name: Scenario descriptor string.
        grid: Environment grid.
        results: Mapping of algorithm labels to SearchResult instances.
        out_dir: Output folder for GIFs.
        theme: Theme name ('dark' or 'light').
        individual: If True, writes individual GIFs for every algorithm.

    Returns:
        List of generated GIF file paths.
    """
    out_p = pathlib.Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    slug = slugify(name)
    written: List[str] = []

    race_path = out_p / f"{slug}__race.gif"
    animate_race(grid=grid, results=results, out_path=race_path, theme=theme, title=name)
    written.append(str(race_path.resolve()))

    if individual:
        for lbl, res in results.items():
            indiv_path = out_p / f"{slug}__{slugify(lbl)}.gif"
            animate_search(
                grid=grid,
                result=res,
                out_path=indiv_path,
                theme=theme,
                title=f"{name} — {lbl}",
            )
            written.append(str(indiv_path.resolve()))

    return written


def export_scenario_bundle(
    name: str,
    grid: Grid,
    results: Mapping[str, SearchResult],
    out_root: Union[str, pathlib.Path] = "outputs",
    theme: str = "dark",
) -> Dict[str, List[str]]:
    """Export full benchmark scenario artifacts (gallery, animations, progress chart, logs).

    Args:
        name: Scenario descriptor string.
        grid: Environment grid.
        results: Mapping of algorithm labels to SearchResult instances.
        out_root: Root outputs directory.
        theme: Theme name for animations and logs.

    Returns:
        Dictionary mapping artifact categories to generated file paths.
    """
    root_p = pathlib.Path(out_root)
    slug = slugify(name)

    # 1. Static gallery plots (light theme)
    gallery_files = export_gallery(
        name=name,
        grid=grid,
        results=results,
        out_dir=root_p / "paths",
        theme="light",
    )

    # 2. Animated GIFs (race and individual)
    animation_files = export_animations(
        name=name,
        grid=grid,
        results=results,
        out_dir=root_p / "animations",
        theme=theme,
        individual=True,
    )

    # 3. Dual-axis convergence progress chart
    log_df = build_search_log(grid=grid, results=results)
    chart_path = root_p / "charts" / f"{slug}__search_progress.png"
    fig = plot_search_progress(
        log_df=log_df,
        grid=grid,
        theme=theme,
        save_path=chart_path,
    )
    del fig
    chart_files = [str(chart_path.resolve())]

    # 4. Step-by-step CSV and optimization JSON logs
    log_files = export_run_logs(
        name=name,
        grid=grid,
        results=results,
        out_dir=root_p / "logs",
    )

    return {
        "gallery": gallery_files,
        "animations": animation_files,
        "progress_chart": chart_files,
        "logs": log_files,
    }


# =====================================================================
# PART G — CLI and Self-Test
# =====================================================================

if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(
        description="Pathfinding GIF animator, search efficiency logger, and scenario bundler."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["trap", "mud_barrier", "random", "maze"],
        help="Target scenario category to animate and export.",
    )
    parser.add_argument("--rows", type=int, default=None, help="Grid rows.")
    parser.add_argument("--cols", type=int, default=None, help="Grid columns.")
    parser.add_argument("--seed", type=int, default=None, help="Random generator seed.")
    parser.add_argument("--density", type=float, default=None, help="Obstacle density.")
    parser.add_argument("--diagonal", action="store_true", help="Allow 8-direction movement.")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory root.")
    parser.add_argument("--theme", type=str, default="dark", choices=["dark", "light"], help="Theme palette.")

    args = parser.parse_args()

    if args.scenario:
        scen_type = args.scenario
        diag = bool(args.diagonal)
        if scen_type == "trap":
            r = args.rows or 12
            c = args.cols or 20
            g = generate_trap_grid(r, c)
            s_name = f"trap {r}x{c}"
        elif scen_type == "mud_barrier":
            r = args.rows or 15
            c = args.cols or 25
            g = generate_mud_barrier_grid(r, c)
            s_name = f"mud barrier {r}x{c}"
        elif scen_type == "random":
            r = args.rows or 40
            c = args.cols or 40
            d = args.density if args.density is not None else 0.25
            s = args.seed if args.seed is not None else 3
            g = generate_random_grid(r, c, obstacle_density=d, seed=s)
            s_name = f"random {r}x{c}"
        else:
            r = args.rows or 21
            c = args.cols or 21
            s = args.seed if args.seed is not None else 7
            g = generate_perfect_maze(r, c, seed=s)
            s_name = f"maze {r}x{c}"

        configs = default_algorithm_configs(allow_diagonal=diag)
        run_res: Dict[str, SearchResult] = {}
        for cfg in configs:
            run_res[cfg.label] = run_algorithm(
                name=cfg.name,
                grid=g,
                allow_diagonal=diag,
                heuristic=cfg.heuristic,
                record_order=True,
            )

        bundle = export_scenario_bundle(
            name=s_name,
            grid=g,
            results=run_res,
            out_root=args.out,
            theme=args.theme,
        )

        print(f"Generated scenario bundle for {s_name}:")
        for category, paths in bundle.items():
            print(f"  [{category}] ({len(paths)} files):")
            for p in paths:
                print(f"    {p}")
        sys.exit(0)

    # =================================================================
    # Self-Test Execution
    # =================================================================
    print("Executing Phase 6 Animator self-test suite...")

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = pathlib.Path(tmp_dir_str)

        # 1. build_frame_steps & choose_cell_px
        steps_216 = build_frame_steps(216, 80)
        assert steps_216[0] == 0, f"Expected start at 0, got {steps_216[0]}"
        assert steps_216[-1] == 216, f"Expected end at 216, got {steps_216[-1]}"
        assert all(steps_216[i] < steps_216[i + 1] for i in range(len(steps_216) - 1)), "Steps must strictly increase"
        assert len(steps_216) <= 80, f"Expected <= 80 items, got {len(steps_216)}"

        steps_5 = build_frame_steps(5, 80)
        assert steps_5 == [0, 1, 2, 3, 4, 5], f"Expected 0..5, got {steps_5}"

        steps_0 = build_frame_steps(0, 80)
        assert steps_0 == [0], f"Expected [0], got {steps_0}"

        assert choose_cell_px(12, 20, 420) == 21, "choose_cell_px(12, 20, 420) must equal 21"
        assert choose_cell_px(100, 100, 300) == 3, "choose_cell_px(100, 100, 300) must equal 3"

        # 2. race_ranking: ties share ranks and not-found maps to None
        class _MockSearchResult:
            def __init__(self, nodes: int, found: bool) -> None:
                self.nodes_expanded = nodes
                self.found = found

        mock_results: Dict[str, Any] = {
            "A": _MockSearchResult(95, True),
            "B": _MockSearchResult(186, True),
            "C": _MockSearchResult(204, True),
            "D": _MockSearchResult(216, True),
            "E": _MockSearchResult(216, True),
            "F": _MockSearchResult(50, False),
        }
        ranks = race_ranking(mock_results)
        assert ranks["A"] == 1
        assert ranks["B"] == 2
        assert ranks["C"] == 3
        assert ranks["D"] == 4
        assert ranks["E"] == 4
        assert ranks["F"] is None

        # 3. Solve trap 12x20 and mud barrier 15x25 with 5 four-direction algorithms
        trap_grid = generate_trap_grid(12, 20)
        mud_grid = generate_mud_barrier_grid(15, 25)

        configs_4dir = default_algorithm_configs(allow_diagonal=False)
        trap_res: Dict[str, SearchResult] = {}
        mud_res: Dict[str, SearchResult] = {}

        for cfg in configs_4dir:
            trap_res[cfg.label] = run_algorithm(
                name=cfg.name,
                grid=trap_grid,
                allow_diagonal=False,
                heuristic=cfg.heuristic,
                record_order=True,
            )
            mud_res[cfg.label] = run_algorithm(
                name=cfg.name,
                grid=mud_grid,
                allow_diagonal=False,
                heuristic=cfg.heuristic,
                record_order=True,
            )

        # animate_search for Dijkstra on trap grid
        dijkstra_gif_path = tmp_dir / "trap__dijkstra.gif"
        written_frames = animate_search(
            grid=trap_grid,
            result=trap_res["Dijkstra"],
            out_path=dijkstra_gif_path,
            theme="dark",
        )
        assert pathlib.Path(dijkstra_gif_path).exists()

        with open(dijkstra_gif_path, "rb") as gf:
            header_bytes = gf.read(6)
            assert header_bytes == b"GIF89a", f"Expected GIF89a header, got {header_bytes!r}"

        with Image.open(dijkstra_gif_path) as im:
            n_frames = im.n_frames
            # save_gif merges consecutive identical frames
            assert n_frames > 0
            size_0 = im.size
            im.seek(0)
            f0_bytes = im.tobytes()
            p00_color = im.convert("RGB").getpixel((0, 0))

            im.seek(n_frames - 1)
            last_dur = im.info.get("duration")
            assert last_dur == DEFAULT_HOLD_MS, f"Expected last duration {DEFAULT_HOLD_MS}, got {last_dur}"
            assert im.info.get("loop") == 0, f"Expected loop=0, got {im.info.get('loop')}"
            fn_bytes = im.tobytes()
            assert f0_bytes != fn_bytes, "First and last frames must differ"

            # Check that pixel at (0, 0) has identical RGB in EVERY frame
            for f_i in range(n_frames):
                im.seek(f_i)
                assert im.size == size_0, "All frames must have identical dimensions"
                assert im.convert("RGB").getpixel((0, 0)) == p00_color, f"Frame {f_i} background pixel differed"

        # 4. animate_search raises ValueError on record_order=False
        no_order_res = run_algorithm(
            name="Dijkstra",
            grid=trap_grid,
            allow_diagonal=False,
            record_order=False,
        )
        try:
            animate_search(trap_grid, no_order_res, tmp_dir / "fail.gif")
            assert False, "Should have raised ValueError on record_order=False"
        except ValueError as err:
            assert "record_order=True is required" in str(err)

        # 5. animate_race on mud barrier with 5 results
        race_gif_path = tmp_dir / "mud__race.gif"
        written_race = animate_race(
            grid=mud_grid,
            results=mud_res,
            out_path=race_gif_path,
            theme="dark",
        )
        assert pathlib.Path(race_gif_path).exists()

        with Image.open(race_gif_path) as im_race:
            assert im_race.n_frames > 0
            im_race.seek(im_race.n_frames - 1)
            assert im_race.info.get("duration") == DEFAULT_HOLD_MS

        mud_ranks = race_ranking(mud_res)
        assert mud_ranks["Greedy Best-First (manhattan)"] == 1
        assert mud_ranks["A* (manhattan)"] == 2
        assert mud_ranks["A* (euclidean)"] == 3
        assert mud_ranks["BFS"] == 4
        assert mud_ranks["Dijkstra"] == 5

        try:
            animate_race(mud_grid, {}, tmp_dir / "empty_race.gif")
            assert False, "Should have raised ValueError on empty results"
        except ValueError:
            pass

        # 6. Search log invariants for every found result of both scenarios
        for scen_grid, scen_res in [(trap_grid, trap_res), (mud_grid, mud_res)]:
            log_df = build_search_log(scen_grid, scen_res)
            for lbl, res in scen_res.items():
                if not res.found:
                    continue
                sub_log = log_df[log_df["algorithm"] == lbl]
                n = len(sub_log)
                assert n == res.nodes_expanded
                assert list(sub_log["step"]) == list(range(1, n + 1))

                # best_dist_so_far non-increasing
                b_dists = sub_log["best_dist_so_far"].to_numpy()
                assert np.all(np.diff(b_dists) <= 1e-9), "best_dist_so_far must be non-increasing"

                # first row is start
                row0 = sub_log.iloc[0]
                assert (int(row0["row"]), int(row0["col"])) == scen_grid.start
                expected_start_dist = math.hypot(
                    scen_grid.start[0] - scen_grid.goal[0],
                    scen_grid.start[1] - scen_grid.goal[1],
                )
                assert math.isclose(row0["dist_to_goal"], expected_start_dist, rel_tol=1e-5)

                # last row is goal with dist == 0
                row_last = sub_log.iloc[-1]
                assert (int(row_last["row"]), int(row_last["col"])) == scen_grid.goal
                assert math.isclose(row_last["dist_to_goal"], 0.0, abs_tol=1e-9)

                # cumulative_on_path at last step == path_steps + 1
                assert int(row_last["cumulative_on_path"]) == res.path_steps + 1

                # 0 <= cumulative_efficiency_pct <= 100
                effs = sub_log["cumulative_efficiency_pct"].to_numpy()
                assert np.all(effs >= 0.0) and np.all(effs <= 100.0)

        # 7. summarize_search_log checks
        mud_log = build_search_log(mud_grid, mud_res)
        mud_summary = summarize_search_log(mud_log, mud_res).set_index("algorithm")
        greedy_eff = mud_summary.loc["Greedy Best-First (manhattan)", "search_efficiency_pct"]
        dijk_eff = mud_summary.loc["Dijkstra", "search_efficiency_pct"]
        assert math.isclose(greedy_eff, 100.0, abs_tol=0.01)
        assert math.isclose(dijk_eff, 100.0 * 37 / 320, abs_tol=0.01)

        trap_log = build_search_log(trap_grid, trap_res)
        trap_summary = summarize_search_log(trap_log, trap_res).set_index("algorithm")
        trap_dijk_eff = trap_summary.loc["Dijkstra", "search_efficiency_pct"]
        trap_greedy_eff = trap_summary.loc["Greedy Best-First (manhattan)", "search_efficiency_pct"]
        assert math.isclose(trap_dijk_eff, 100.0 * 28 / 216, abs_tol=0.01)
        assert trap_greedy_eff > trap_dijk_eff

        # Unsolvable grid
        unsolv_grid = generate_unsolvable_grid(12, 20, seed=1)
        unsolv_res = {
            "Dijkstra": run_algorithm("Dijkstra", unsolv_grid, allow_diagonal=False, record_order=True)
        }
        unsolv_log = build_search_log(unsolv_grid, unsolv_res)
        unsolv_summary = summarize_search_log(unsolv_log, unsolv_res).iloc[0]
        assert unsolv_summary["found"] is False or unsolv_summary["found"] == False
        assert unsolv_summary["path_nodes"] == 0
        assert unsolv_summary["search_efficiency_pct"] == 0.0

        # 8. plot_search_progress
        prog_fig = plot_search_progress(
            log_df=trap_log,
            grid=trap_grid,
            theme="dark",
            save_path=tmp_dir / "prog.png",
        )
        assert len(prog_fig.axes) == 2, f"Expected 2 axes, got {len(prog_fig.axes)}"
        assert (tmp_dir / "prog.png").exists()
        del prog_fig

        try:
            plot_search_progress(pd.DataFrame(), trap_grid)
            assert False, "Should have raised ValueError on empty log_df"
        except ValueError:
            pass

        # 9. export_run_logs
        written_logs = export_run_logs("mud barrier 15x25", mud_grid, mud_res, out_dir=tmp_dir / "logs")
        assert len(written_logs) == 2
        csv_p, json_p = written_logs
        assert pathlib.Path(csv_p).exists()
        assert pathlib.Path(json_p).exists()

        with open(json_p, "r", encoding="utf-8") as jf:
            parsed_json = json.load(jf)
            assert "algorithms" in parsed_json
            for cfg in configs_4dir:
                assert cfg.label in parsed_json["algorithms"]
            greedy_json_ratio = parsed_json["algorithms"]["Greedy Best-First (manhattan)"]["cost_ratio"]
            assert math.isclose(greedy_json_ratio, 50.0 / 36.0, rel_tol=1e-3)

        # 10. export_scenario_bundle into temp dir
        bundle_temp = export_scenario_bundle(
            name="mud barrier 15x25",
            grid=mud_grid,
            results=mud_res,
            out_root=tmp_dir / "bundle_test",
            theme="dark",
        )
        assert len(bundle_temp["gallery"]) == 6, f"Expected 6 gallery files, got {len(bundle_temp['gallery'])}"
        assert len(bundle_temp["animations"]) == 6, f"Expected 6 animation GIFs, got {len(bundle_temp['animations'])}"
        assert len(bundle_temp["progress_chart"]) == 1, f"Expected 1 chart, got {len(bundle_temp['progress_chart'])}"
        assert len(bundle_temp["logs"]) == 2, f"Expected 2 logs, got {len(bundle_temp['logs'])}"
        all_temp_files = (
            bundle_temp["gallery"]
            + bundle_temp["animations"]
            + bundle_temp["progress_chart"]
            + bundle_temp["logs"]
        )
        assert len(all_temp_files) == 15
        for tf in all_temp_files:
            assert pathlib.Path(tf).stat().st_size > 0, f"File {tf} was empty"

        # 11. Rule enforcement
        assert "matplotlib.pyplot" not in sys.modules, "Rule violation: matplotlib.pyplot must never be imported"
        assert "imageio" not in sys.modules, "Rule violation: imageio must never be imported"

    # 12. FINAL REAL EXPORT into outputs folder
    print("\nGenerating real scenario bundle exports for trap 12x20 and mud barrier 15x25...")
    real_trap_bundle = export_scenario_bundle(
        name="trap 12x20",
        grid=trap_grid,
        results=trap_res,
        out_root="outputs",
        theme="dark",
    )
    real_mud_bundle = export_scenario_bundle(
        name="mud barrier 15x25",
        grid=mud_grid,
        results=mud_res,
        out_root="outputs",
        theme="dark",
    )

    print("\n" + "=" * 80)
    print(" SEARCH EFFICIENCY SUMMARY — TRAP 12x20")
    print("=" * 80)
    trap_df = summarize_search_log(build_search_log(trap_grid, trap_res), trap_res)
    print(trap_df.to_string(index=False))

    print("\n" + "=" * 80)
    print(" SEARCH EFFICIENCY SUMMARY — MUD BARRIER 15x25")
    print("=" * 80)
    mud_df = summarize_search_log(build_search_log(mud_grid, mud_res), mud_res)
    print(mud_df.to_string(index=False))

    print("\n" + "=" * 80)
    print(" GENERATED ANIMATED GIFS")
    print("=" * 80)
    all_real_gifs = real_trap_bundle["animations"] + real_mud_bundle["animations"]
    for g_path in all_real_gifs:
        print(f"  {g_path}")

    print("\nAll Phase 6 tests passed!")
