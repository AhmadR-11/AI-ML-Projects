"""Streamlit Web Application for Heuristic Graph Pathfinding Agent.

Interactive exploration of pathfinding algorithms (Dijkstra, A*, BFS,
Greedy Best-First) across diverse environments (percolation mazes, spanning tree
mazes, concave trap pockets, and multi-cost terrain).
"""

import base64
import hashlib
import pathlib
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from matplotlib.figure import Figure
import pandas as pd
import streamlit as st

from src.algorithms import ALGORITHM_INFO, SearchResult
from src.animator import (
    build_search_log,
    export_scenario_bundle,
    plot_search_progress,
    race_ranking,
)
from src.app_logic import (
    DEFAULT_PRESET,
    MAX_ANIM_CELLS,
    MAX_DIM,
    MIN_COLS,
    MIN_ROWS,
    PRESETS,
    GridSpec,
    available_configs,
    build_grid,
    build_insights,
    comparison_table,
    config_key,
    display_name,
    fig_to_png_bytes,
    gif_bytes_for_race,
    gif_bytes_for_search,
    grid_info,
    parse_ascii_grid,
    precise_runtimes,
    solve_configs,
    spec_key,
)
from src.benchmark import AlgoConfig, default_algorithm_configs
from src.grid import Grid
from src.visualizer import (
    plot_comparison,
    plot_result,
    slugify,
)


# =====================================================================
# PART B1 — Page Initialization and Styles
# =====================================================================

st.set_page_config(
    page_title="Pathfinding Agent",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_file = pathlib.Path("assets/style.css")
if css_file.exists():
    css_text = css_file.read_text(encoding="utf-8").strip()
    if css_text:
        st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)


# =====================================================================
# PART B2 — Compatibility & Display Helpers
# =====================================================================

def stretch(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Invoke Streamlit element layout with stretch width falling back gracefully."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except (TypeError, ValueError, st.errors.StreamlitAPIException):
        return fn(*args, use_container_width=True, **kwargs)


def show_gif(gif_bytes: bytes, max_width: int = 900) -> None:
    """Render in-memory animated GIF using an HTML base64 data URI."""
    b64 = base64.b64encode(gif_bytes).decode("ascii")
    st.markdown(
        f'<img src="data:image/gif;base64,{b64}" style="width: 100%; max-width: {max_width}px; border-radius: 6px;">',
        unsafe_allow_html=True,
    )


# =====================================================================
# PART B3 — Cached Execution Wrappers
# =====================================================================

@st.cache_data(show_spinner=False, max_entries=128)
def cached_solve(
    ascii_text: str,
    allow_diagonal: bool,
    config_keys: Tuple[Tuple[str, Optional[str]], ...],
) -> Dict[str, SearchResult]:
    """Execute search algorithms on a serialized ASCII grid environment."""
    grid = parse_ascii_grid(ascii_text)
    configs = [AlgoConfig(name=k[0], heuristic=k[1]) for k in config_keys]
    return solve_configs(grid=grid, configs=configs, allow_diagonal=allow_diagonal)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_runtimes(
    ascii_text: str,
    allow_diagonal: bool,
    config_keys: Tuple[Tuple[str, Optional[str]], ...],
    repeats: int = 5,
) -> Dict[str, float]:
    """Compute median repeated execution timings for specified algorithm configurations."""
    grid = parse_ascii_grid(ascii_text)
    configs = [AlgoConfig(name=k[0], heuristic=k[1]) for k in config_keys]
    return precise_runtimes(
        grid=grid,
        configs=configs,
        allow_diagonal=allow_diagonal,
        repeats=repeats,
    )


@st.cache_data(show_spinner=False, max_entries=32)
def cached_search_gif(
    ascii_text: str,
    allow_diagonal: bool,
    cfg_key: Tuple[str, Optional[str]],
    theme: str = "dark",
) -> bytes:
    """Generate single-search exploration and path reveal animation GIF."""
    grid = parse_ascii_grid(ascii_text)
    cfg = AlgoConfig(name=cfg_key[0], heuristic=cfg_key[1])
    res_map = solve_configs(grid=grid, configs=[cfg], allow_diagonal=allow_diagonal)
    res = res_map[cfg.label]
    return gif_bytes_for_search(grid=grid, result=res, theme=theme, max_frames=60)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_race_gif(
    ascii_text: str,
    allow_diagonal: bool,
    config_keys: Tuple[Tuple[str, Optional[str]], ...],
    theme: str = "dark",
) -> bytes:
    """Generate multi-algorithm synchronized race animation GIF."""
    grid = parse_ascii_grid(ascii_text)
    configs = [AlgoConfig(name=k[0], heuristic=k[1]) for k in config_keys]
    results = solve_configs(grid=grid, configs=configs, allow_diagonal=allow_diagonal)
    return gif_bytes_for_race(grid=grid, results=results, theme=theme, max_frames=60, ncols=3)


# =====================================================================
# PART B4 — Sidebar Controls
# =====================================================================

def _on_preset_change() -> None:
    """Callback to synchronize dimensions when preset selection changes."""
    selected_p = st.session_state.get("preset", DEFAULT_PRESET)
    if selected_p in PRESETS:
        st.session_state["rows"] = PRESETS[selected_p]["rows"]
        st.session_state["cols"] = PRESETS[selected_p]["cols"]


def render_sidebar() -> Tuple[GridSpec, bool]:
    """Render sidebar inputs and return configured GridSpec and movement flag."""
    st.sidebar.radio(
        label="Navigation Page",
        options=["🧪 Playground", "⚖️ Compare", "📊 Benchmark", "💡 Insights"],
        key="page",
        label_visibility="collapsed",
    )

    st.sidebar.markdown("### 🗺️ Maze Setup")

    st.sidebar.selectbox(
        label="Preset",
        options=list(PRESETS.keys()),
        index=list(PRESETS.keys()).index(DEFAULT_PRESET),
        key="preset",
        on_change=_on_preset_change,
    )

    cur_preset = st.session_state.get("preset", DEFAULT_PRESET)
    p_meta = PRESETS.get(cur_preset, PRESETS[DEFAULT_PRESET])

    if "rows" not in st.session_state:
        st.session_state["rows"] = p_meta["rows"]
    if "cols" not in st.session_state:
        st.session_state["cols"] = p_meta["cols"]

    st.sidebar.slider(
        label="Rows",
        min_value=MIN_ROWS,
        max_value=MAX_DIM,
        step=1,
        key="rows",
    )
    st.sidebar.slider(
        label="Columns",
        min_value=MIN_COLS,
        max_value=MAX_DIM,
        step=1,
        key="cols",
    )

    density_val = 0.25
    if p_meta["uses_density"]:
        density_val = st.sidebar.slider(
            label="Obstacle density",
            min_value=0.0,
            max_value=0.35,
            value=0.25,
            step=0.05,
            key="density",
            help="Above ~35% random 4-direction grids are rarely solvable.",
        )
        st.sidebar.caption("Above ~35% random 4-direction grids are rarely solvable.")

    seed_val = 3
    if p_meta["uses_seed"]:
        c_s1, c_s2 = st.sidebar.columns([3, 2])
        seed_val = int(
            c_s1.number_input(
                label="Seed",
                min_value=0,
                max_value=999999,
                value=3,
                step=1,
                key="seed",
            )
        )
        c_s2.button(
            label="🎲 New seed",
            key="btn_new_seed",
            on_click=lambda: st.session_state.update(seed=st.session_state.get("seed", 3) + 1),
        )

    band_val = 7
    if p_meta["uses_band"]:
        band_val = st.sidebar.slider(
            label="Mud band width",
            min_value=3,
            max_value=15,
            value=7,
            step=1,
            key="band_width",
        )

    add_mud_val = False
    mud_patches_val = 4
    if p_meta["supports_mud"]:
        add_mud_val = st.sidebar.toggle("Add mud patches", value=False, key="add_mud")
        if add_mud_val:
            mud_patches_val = st.sidebar.slider(
                label="Mud patches",
                min_value=1,
                max_value=12,
                value=4,
                step=1,
                key="mud_patches",
            )

    st.sidebar.caption(p_meta["description"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Movement Rules")
    movement_mode = st.sidebar.radio(
        label="Allowed transitions",
        options=["4-direction", "8-direction (diagonal)"],
        key="movement",
    )
    allow_diagonal = movement_mode.startswith("8")

    st.sidebar.markdown("---")
    st.sidebar.caption("🧭 Heuristic Graph Pathfinding Agent")

    spec = GridSpec(
        preset=cur_preset,
        rows=int(st.session_state["rows"]),
        cols=int(st.session_state["cols"]),
        density=float(density_val),
        seed=int(seed_val),
        band_width=int(band_val),
        add_mud=bool(add_mud_val),
        mud_patches=int(mud_patches_val),
    )
    return spec, allow_diagonal


# =====================================================================
# PART B5 — Maze State Engine
# =====================================================================

def ensure_maze(spec: GridSpec) -> Grid:
    """Retrieve or regenerate the active Grid, maintaining synchronization with session state."""
    current_key = spec_key(spec)
    last_key = st.session_state.get("spec_key_last")

    if "grid_ascii" not in st.session_state or current_key != last_key:
        try:
            grid = build_grid(spec)
            st.session_state["grid_ascii"] = grid.to_ascii()
            st.session_state["spec_key_last"] = current_key
        except (ValueError, RuntimeError) as err:
            st.sidebar.error(f"Generation error: {err}")
            if "grid_ascii" not in st.session_state:
                fb_spec = GridSpec(
                    preset=DEFAULT_PRESET,
                    rows=PRESETS[DEFAULT_PRESET]["rows"],
                    cols=PRESETS[DEFAULT_PRESET]["cols"],
                )
                fb_grid = build_grid(fb_spec)
                st.session_state["grid_ascii"] = fb_grid.to_ascii()
                st.session_state["spec_key_last"] = spec_key(fb_spec)

    return parse_ascii_grid(st.session_state["grid_ascii"])


# =====================================================================
# PART B6 — Playground Page
# =====================================================================

def render_playground(grid: Grid, spec: GridSpec, allow_diagonal: bool) -> None:
    """Render the single-search Playground page."""
    st.header("🧪 Playground")

    info = grid_info(grid, allow_diagonal)
    status_str = "solvable" if info["solvable"] else "unsolvable"
    st.caption(
        f"{info['rows']}x{info['cols']} · walls {info['wall_pct']}% · mud {info['mud_pct']}% · "
        f"{info['free_cells']} free cells · {status_str}"
    )

    if not info["solvable"]:
        st.warning("No path exists between Start (S) and Goal (G) under current obstacle bounds.")

    avail_map = available_configs(allow_diagonal)
    algo_options = list(avail_map.keys())

    algo_key = f"pg_algo_{'8' if allow_diagonal else '4'}"
    default_algo = "A* (octile)" if allow_diagonal and "A* (octile)" in algo_options else "A* (manhattan)"
    default_idx = algo_options.index(default_algo) if default_algo in algo_options else 0

    selected_label = st.selectbox(
        label="Algorithm",
        options=algo_options,
        index=default_idx,
        key=algo_key,
    )
    selected_cfg = avail_map[selected_label]

    if allow_diagonal and selected_cfg.name == "A*" and selected_cfg.heuristic == "manhattan":
        st.warning(
            "A* with Manhattan distance is not admissible on 8-direction grids: "
            "it overestimates diagonal distances and may yield suboptimal paths."
        )

    # Solve with Dijkstra and the selected algorithm
    query_keys = [("Dijkstra", None), config_key(selected_cfg)]
    unique_keys = tuple(dict.fromkeys(query_keys))
    results = cached_solve(st.session_state["grid_ascii"], allow_diagonal, unique_keys)

    res = results[selected_label]
    dijk_res = results["Dijkstra"]

    col_left, col_right = st.columns([3, 2])

    with col_right:
        st.markdown("#### Performance Metrics")
        m_c1, m_c2, m_c3 = st.columns(3)
        m_c1.metric("Steps", res.path_steps if res.found else "—")
        m_c2.metric("Path cost", f"{res.path_cost:.1f}" if res.found else "—")
        m_c3.metric("Nodes expanded", res.nodes_expanded)

        m_c4, m_c5, m_c6 = st.columns(3)
        m_c4.metric("Explored % of free", f"{res.expanded_density_pct(grid):.1f}%")
        m_c5.metric("Peak frontier", res.max_frontier_size)
        m_c6.metric("Runtime (1 run)", f"{res.runtime_ms:.2f} ms")

        # Optimality verdict banner
        if not res.found:
            st.error("No path found")
        elif abs(res.path_cost - dijk_res.path_cost) <= 1e-9:
            st.success("Optimal (matches Dijkstra)")
        else:
            diff_pct = (res.path_cost / float(dijk_res.path_cost) - 1.0) * 100.0
            st.warning(f"+{diff_pct:.1f}% above the optimal cost")

        st.caption("Single-run timings are noisy; the Compare page can show median-of-5 timings.")

        with st.expander("About this algorithm"):
            info_dict = ALGORITHM_INFO.get(selected_cfg.name, {})
            st.write(info_dict.get("description", ""))
            st.markdown(f"- **Guaranteed optimal cost**: `{info_dict.get('optimal_cost', '—')}`")
            st.markdown(f"- **Guaranteed optimal steps**: `{info_dict.get('optimal_steps', '—')}`")
            st.markdown(f"- **Uses heuristic**: `{info_dict.get('uses_heuristic', False)}`")

    with col_left:
        tab_result, tab_anim, tab_log = st.tabs(["🗺️ Result", "🎞️ Animation", "📈 Search log"])

        with tab_result:
            show_exp = st.toggle("Show explored cells", value=True, key="pg_show_explored")
            fig_res = plot_result(
                grid=grid,
                result=res,
                theme="dark",
                show_explored=show_exp,
            )
            stretch(st.pyplot, fig_res)
            st.download_button(
                label="⬇️ Download PNG",
                data=fig_to_png_bytes(fig_res),
                file_name=f"{slugify(display_name(spec, grid))}__{slugify(selected_label)}.png",
                mime="image/png",
            )
            del fig_res

        with tab_anim:
            total_cells = grid.rows * grid.cols
            if total_cells > MAX_ANIM_CELLS:
                st.info(
                    f"Grid size ({total_cells} cells) exceeds animation threshold "
                    f"({MAX_ANIM_CELLS} cells). Please reduce dimensions to animate."
                )
            else:
                do_anim = st.toggle(
                    "Generate animation — takes a few seconds",
                    value=False,
                    key="pg_animate",
                )
                if do_anim:
                    with st.spinner("Generating animation..."):
                        gif_bytes = cached_search_gif(
                            ascii_text=st.session_state["grid_ascii"],
                            allow_diagonal=allow_diagonal,
                            cfg_key=config_key(selected_cfg),
                            theme="dark",
                        )
                    show_gif(gif_bytes)
                    st.download_button(
                        label="⬇️ Download GIF",
                        data=gif_bytes,
                        file_name=f"{slugify(display_name(spec, grid))}__{slugify(selected_label)}.gif",
                        mime="image/gif",
                    )

        with tab_log:
            comp_results = {
                lbl: results[lbl]
                for lbl in (["Dijkstra", selected_label] if selected_label != "Dijkstra" else ["Dijkstra"])
            }
            log_df = build_search_log(grid=grid, results=comp_results)
            if not log_df.empty:
                fig_prog = plot_search_progress(log_df=log_df, grid=grid, theme="dark")
                stretch(st.pyplot, fig_prog)
                del fig_prog
                st.dataframe(log_df, height=280)
                st.download_button(
                    label="⬇️ Download Search Log CSV",
                    data=log_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{slugify(display_name(spec, grid))}__{slugify(selected_label)}__search_log.csv",
                    mime="text/csv",
                )

    # ASCII Editor Expander
    with st.expander("✏️ Edit maze (ASCII)"):
        st.caption("Legend: `S` = start, `G` = goal, `#` = wall, `~` = mud, `.` = free")
        cur_ascii = st.session_state["grid_ascii"]
        fingerprint = hashlib.md5(cur_ascii.encode("utf-8")).hexdigest()[:8]
        nonce = st.session_state.get("editor_nonce", 0)
        ed_key = f"editor_{fingerprint}_{nonce}"

        ed_text = st.text_area(
            label="ASCII Maze",
            value=cur_ascii,
            height=min(600, 24 * grid.rows + 40),
            key=ed_key,
            label_visibility="collapsed",
        )

        c_a1, c_a2 = st.columns(2)
        if c_a1.button("Apply edits", key="apply_edit"):
            try:
                parsed_g = parse_ascii_grid(ed_text)
                st.session_state["grid_ascii"] = parsed_g.to_ascii()
                st.rerun()
            except ValueError as err:
                st.error(f"Invalid maze: {err}")

        if c_a2.button("Reset editor", key="reset_edit"):
            st.session_state["editor_nonce"] = nonce + 1
            st.rerun()


# =====================================================================
# PART B7 — Compare Page
# =====================================================================

def render_compare(grid: Grid, spec: GridSpec, allow_diagonal: bool) -> None:
    """Render the multi-algorithm Compare page."""
    st.header("⚖️ Compare Algorithms")

    avail_map = available_configs(allow_diagonal)
    standard_labels = [
        c.label for c in default_algorithm_configs(allow_diagonal, include_inadmissible_demo=False)
    ]

    cmp_key = f"cmp_algos_{'8' if allow_diagonal else '4'}"
    selected_labels = st.multiselect(
        label="Select algorithms to compare",
        options=list(avail_map.keys()),
        default=standard_labels,
        key=cmp_key,
    )

    if len(selected_labels) < 2:
        st.warning("Please select at least 2 algorithms to compare.")
        return

    do_precise = st.toggle(
        "Precise timing: median of 5 runs",
        value=False,
        key="cmp_precise",
    )
    if do_precise:
        st.caption("Timing mode: median of 5 repeated runs with Python garbage collection suppressed.")
    else:
        st.caption("Timing mode: single-run execution timing.")

    sel_configs = [avail_map[lbl] for lbl in selected_labels]
    cfg_keys = tuple(config_key(c) for c in sel_configs)

    with st.spinner("Solving ..."):
        results_map = cached_solve(
            ascii_text=st.session_state["grid_ascii"],
            allow_diagonal=allow_diagonal,
            config_keys=cfg_keys,
        )
        runtimes = (
            cached_runtimes(
                ascii_text=st.session_state["grid_ascii"],
                allow_diagonal=allow_diagonal,
                config_keys=cfg_keys,
                repeats=5,
            )
            if do_precise
            else None
        )

    # Filter to specifically chosen algorithms
    chosen_results = {lbl: results_map[lbl] for lbl in selected_labels if lbl in results_map}

    # Summary Insights Box
    insights = build_insights(grid=grid, results=chosen_results)
    if insights:
        st.info(insights[0])
        if len(insights) > 1:
            with st.container():
                for line in insights[1:]:
                    st.markdown(f"- {line}")

    # Metrics DataFrame Table
    tbl = comparison_table(grid=grid, results=chosen_results, runtimes=runtimes)
    col_config = {
        "% of free cells": st.column_config.ProgressColumn(
            min_value=0.0,
            max_value=100.0,
            format="%.1f%%",
        ),
        "Search efficiency %": st.column_config.ProgressColumn(
            min_value=0.0,
            max_value=100.0,
            format="%.1f%%",
        ),
    }
    stretch(st.dataframe, tbl, column_config=col_config, hide_index=True)

    tab_side, tab_race, tab_prog, tab_exp = st.tabs(
        ["🖼️ Side by side", "🏁 Race (GIF)", "📈 Search progress", "📦 Export"]
    )

    with tab_side:
        fig_comp = plot_comparison(
            grid=grid,
            results=chosen_results,
            theme="dark",
            ncols=3,
        )
        stretch(st.pyplot, fig_comp)
        st.download_button(
            label="⬇️ Download Comparison PNG",
            data=fig_to_png_bytes(fig_comp),
            file_name=f"{slugify(display_name(spec, grid))}__comparison.png",
            mime="image/png",
        )
        del fig_comp

    with tab_race:
        total_cells = grid.rows * grid.cols
        if total_cells > MAX_ANIM_CELLS:
            st.info(
                f"Grid size ({total_cells} cells) exceeds animation threshold "
                f"({MAX_ANIM_CELLS} cells). Please reduce dimensions to animate."
            )
        else:
            do_race = st.toggle("Generate race animation", value=False, key="cmp_race")
            if do_race:
                with st.spinner("Rendering multi-algorithm race GIF..."):
                    race_bytes = cached_race_gif(
                        ascii_text=st.session_state["grid_ascii"],
                        allow_diagonal=allow_diagonal,
                        config_keys=cfg_keys,
                        theme="dark",
                    )
                show_gif(race_bytes)

                ranks = race_ranking(chosen_results)
                found_entries = [
                    (lbl, ranks[lbl], chosen_results[lbl].nodes_expanded)
                    for lbl in ranks
                    if ranks[lbl] is not None
                ]
                found_entries.sort(key=lambda item: (item[1], item[2]))
                finish_parts = [f"{r}. {lbl} ({n})" for lbl, r, n in found_entries]
                not_found_parts = [f"{lbl} (no path)" for lbl, r in ranks.items() if r is None]
                caption_str = "Finish order (by nodes expanded): " + ", ".join(finish_parts + not_found_parts)
                st.caption(caption_str)

                st.download_button(
                    label="⬇️ Download Race GIF",
                    data=race_bytes,
                    file_name=f"{slugify(display_name(spec, grid))}__race.gif",
                    mime="image/gif",
                )

    with tab_prog:
        log_df = build_search_log(grid=grid, results=chosen_results)
        if not log_df.empty:
            fig_prog = plot_search_progress(log_df=log_df, grid=grid, theme="dark")
            stretch(st.pyplot, fig_prog)
            del fig_prog

    with tab_exp:
        if st.button("Export bundle to outputs/", key="cmp_export"):
            with st.spinner("Exporting scenario bundle..."):
                bundle = export_scenario_bundle(
                    name=display_name(spec, grid),
                    grid=grid,
                    results=chosen_results,
                    out_root="outputs",
                    theme="dark",
                )
                total_files = sum(len(p) for p in bundle.values())
                st.success(f"Exported {total_files} files to outputs/!")
                with st.expander("View exported file paths"):
                    for cat_name, paths in bundle.items():
                        st.markdown(f"**{cat_name.title()}**:")
                        for p in paths:
                            st.code(p)

        st.download_button(
            label="⬇️ Download Comparison Table CSV",
            data=tbl.to_csv(index=False).encode("utf-8"),
            file_name=f"{slugify(display_name(spec, grid))}__comparison.csv",
            mime="text/csv",
        )


# =====================================================================
# PART B8 — Benchmark and Insights Placeholder Pages
# =====================================================================

def render_benchmark() -> None:
    """Render placeholder for the upcoming Benchmark suite page."""
    st.header("📊 Benchmark Suite")
    st.info(
        "The automated benchmark runner and multi-scenario performance charts arrive in the next phase. "
        "This page will display sweeping experiments across obstacle densities, grid sizes, and maze topologies "
        "with statistical runtimes and winner distributions."
    )


def render_insights() -> None:
    """Render placeholder for the upcoming Insights deep-dive page."""
    st.header("💡 Algorithmic Insights")
    st.info(
        "The interactive insights and trade-off exploration engine arrives in the next phase. "
        "This page will provide deep-dive analysis on heuristic admissibility, step vs. cost divergence, "
        "and frontier memory efficiency."
    )


# =====================================================================
# PART B9 — Main Application Router
# =====================================================================

def main() -> None:
    """Application entrypoint."""
    spec, allow_diagonal = render_sidebar()
    grid = ensure_maze(spec)

    active_page = st.session_state.get("page", "🧪 Playground")
    if active_page == "🧪 Playground":
        render_playground(grid=grid, spec=spec, allow_diagonal=allow_diagonal)
    elif active_page == "⚖️ Compare":
        render_compare(grid=grid, spec=spec, allow_diagonal=allow_diagonal)
    elif active_page == "📊 Benchmark":
        render_benchmark()
    elif active_page == "💡 Insights":
        render_insights()
    else:
        render_playground(grid=grid, spec=spec, allow_diagonal=allow_diagonal)


if __name__ == "__main__":
    main()
