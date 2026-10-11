"""Benchmark and Insights UI pages for Streamlit application.

Interactive execution, visualization, analysis, and reporting of pathfinding
benchmark experiments.
"""

import math
import os
import pathlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import benchmark, report, visualizer
from src.app_logic import fig_to_png_bytes
from src.benchmark import BenchmarkResult
from src.ui_common import stretch


SAVED_CSV: str = report.DEFAULT_CSV
SOURCES: List[str] = [
    "Saved results (outputs/logs)",
    "Run quick suite",
    "Run standard suite",
    "Upload results.csv",
]

CHART_OPTIONS: List[str] = [
    "Nodes vs obstacle density",
    "Nodes vs grid size",
    "Runtime vs grid size",
    "Expansion efficiency",
    "Cost quality",
    "Runtime vs expansions",
]


@st.cache_data(show_spinner=False)
def _load_saved(csv_path: str, mtime: float) -> BenchmarkResult:
    """Load benchmark results from disk cache invalidated by file mtime."""
    return report.load_benchmark(csv_path)


def get_active_bench() -> Optional[BenchmarkResult]:
    """Retrieve currently active BenchmarkResult or fallback to saved disk results."""
    active = st.session_state.get("bench_active")
    if active is not None:
        return active
    saved_path = pathlib.Path(SAVED_CSV)
    if saved_path.exists():
        try:
            mtime = os.path.getmtime(saved_path)
            loaded = _load_saved(str(saved_path), mtime)
            st.session_state["bench_active"] = loaded
            return loaded
        except Exception:
            return None
    return None


# =====================================================================
# PART B — Benchmark Page
# =====================================================================

def render_benchmark_page() -> None:
    """Render interactive benchmark execution, chart exploration, and data export page."""
    st.title("Benchmark Laboratory")
    st.write("Run empirical search algorithm benchmarks, explore scaling metrics, and export data.")

    saved_exists = pathlib.Path(SAVED_CSV).exists()
    default_idx = 0 if saved_exists else 1

    source = st.selectbox(
        "Data source",
        SOURCES,
        index=default_idx,
        key="bm_source",
    )

    bench: Optional[BenchmarkResult] = None

    if source == SOURCES[0]:  # "Saved results (outputs/logs)"
        saved_path = pathlib.Path(SAVED_CSV)
        if not saved_path.exists():
            st.info(
                f"No saved results found at `{SAVED_CSV}`. "
                "Select 'Run quick suite' above or run `python -m src.benchmark --run standard` from your terminal."
            )
        else:
            try:
                mtime = os.path.getmtime(saved_path)
                bench = _load_saved(str(saved_path), mtime)
            except Exception as err:
                st.error(f"Error loading saved benchmark: {err}")

    elif source in (SOURCES[1], SOURCES[2]):  # "Run quick suite" or "Run standard suite"
        suite = "quick" if source == SOURCES[1] else "standard"
        default_reps = 3 if suite == "quick" else 7
        repeats = st.slider(
            "Repeats per scenario (median-of-N)",
            min_value=1,
            max_value=9,
            value=default_reps,
            key=f"bm_repeats_{suite}",
        )
        dur_msg = (
            "Expected duration: ~2–5 seconds (5 scenarios × 6 algorithms)"
            if suite == "quick"
            else "Expected duration: roughly 1–3+ minutes (42 scenarios × 6 algorithms)"
        )
        st.caption(dur_msg)

        can_run = True
        if suite == "standard":
            confirmed = st.checkbox(
                "I understand this takes a few minutes",
                key="bm_confirm_standard",
            )
            can_run = confirmed

        if st.button("Run Benchmark Suite", key="bm_run", disabled=not can_run):
            prog_bar = st.progress(0.0, text="Initializing benchmark suite...")
            t0 = time.time()

            def _progress_cb(done: int, total: int, sc_name: str) -> None:
                frac = float(done) / float(total) if total > 0 else 0.0
                prog_bar.progress(frac, text=f"Scenario {done}/{total}: {sc_name}")

            try:
                b_res = benchmark.run_benchmark(
                    suite=suite,
                    repeats=repeats,
                    progress=_progress_cb,
                    strict=True,
                )
                prog_bar.empty()
                elapsed = time.time() - t0
                st.success(f"Benchmark suite '{suite}' finished in {elapsed:.1f}s!")
                st.session_state[f"bench_{suite}"] = b_res
                st.session_state["bench_active"] = b_res
            except RuntimeError as err:
                prog_bar.empty()
                st.error(f"Correctness guard failure: {err}")
            except Exception as err:
                prog_bar.empty()
                st.error(f"Benchmark error: {err}")

        bench = st.session_state.get(f"bench_{suite}")
        if bench is None:
            st.info(f"Click 'Run Benchmark Suite' above to execute the {suite} suite.")

    elif source == SOURCES[3]:  # "Upload results.csv"
        uploaded_file = st.file_uploader("Upload results.csv", type=["csv"], key="bm_upload")
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                bench = report.benchmark_from_frame(df)
                st.session_state["bench_upload"] = bench
            except (ValueError, Exception) as err:
                st.error(f"Failed to parse uploaded CSV: {err}")
        else:
            bench = st.session_state.get("bench_upload")
            if bench is None:
                st.info("Upload a results CSV file to inspect benchmark metrics.")

    if bench is None:
        return

    # Update active benchmark session state
    st.session_state["bench_active"] = bench
    results_df = bench.results

    # Metadata strip
    m1, m2, m3, m4 = st.columns(4)
    suite_lbl = str(bench.meta.get("suite", "custom")).capitalize()
    n_scenarios = results_df["scenario"].nunique() if not results_df.empty else 0
    n_runs = len(results_df)
    n_algos = results_df["algorithm"].nunique() if not results_df.empty else 0

    m1.metric("Suite", suite_lbl)
    m2.metric("Scenarios", f"{n_scenarios}")
    m3.metric("Runs", f"{n_runs}")
    m4.metric("Algorithms", f"{n_algos}")

    timing_proto = bench.meta.get("timing_protocol", "median-of-N with GC disabled")
    rep_meta = bench.meta.get("repeats", "-")
    skipped_scenarios = bench.meta.get("skipped", [])
    st.caption(
        f"Timing: {timing_proto} | Repeats: {rep_meta} | Skipped scenarios: {len(skipped_scenarios)}"
    )
    if skipped_scenarios:
        with st.expander("Skipped scenarios"):
            for sk in skipped_scenarios:
                st.write(f"- {sk}")

    # Filters
    all_cats = sorted(results_df["category"].unique().tolist())
    all_algos = visualizer.sort_algorithm_labels(results_df["algorithm"].unique().tolist())

    col_cat, col_algo = st.columns(2)
    sel_cats = col_cat.multiselect(
        "Filter categories",
        options=all_cats,
        default=all_cats,
        key="bm_cats",
    )
    sel_algos = col_algo.multiselect(
        "Filter algorithms",
        options=all_algos,
        default=all_algos,
        key="bm_algos",
    )

    if not sel_cats or not sel_algos:
        st.info("No data to display. Please select at least one category and one algorithm.")
        return

    filtered_df = results_df[
        results_df["category"].isin(sel_cats) & results_df["algorithm"].isin(sel_algos)
    ].copy()

    if filtered_df.empty:
        st.info("No rows match the specified filter criteria.")
        return

    summary_df = benchmark.summarize(filtered_df)
    winners_df = benchmark.winner_counts(filtered_df)

    tab_charts, tab_summary, tab_winners, tab_raw = st.tabs(
        ["📈 Charts", "📋 Summary", "🏆 Winners", "🧾 Raw data"]
    )

    with tab_charts:
        chart_sel = st.selectbox(
            "Select benchmark chart",
            CHART_OPTIONS,
            key="bm_chart",
        )
        fig: Optional[plt.Figure] = None
        try:
            if chart_sel == "Nodes vs obstacle density":
                if "density_sweep" not in summary_df["category"].values:
                    st.info("Not enough data for this chart with the current filters")
                else:
                    fig = visualizer.plot_metric_by_param(
                        summary=summary_df,
                        category="density_sweep",
                        metric="mean_nodes_expanded",
                        ylabel="Nodes Expanded",
                        title="Nodes Expanded vs. Obstacle Density",
                        theme="dark",
                    )
            elif chart_sel == "Nodes vs grid size":
                if "size_sweep" not in summary_df["category"].values:
                    st.info("Not enough data for this chart with the current filters")
                else:
                    fig = visualizer.plot_metric_by_param(
                        summary=summary_df,
                        category="size_sweep",
                        metric="mean_nodes_expanded",
                        ylabel="Nodes Expanded (Log Scale)",
                        title="Nodes Expanded vs. Grid Size",
                        theme="dark",
                        log_y=True,
                    )
            elif chart_sel == "Runtime vs grid size":
                if "size_sweep" not in summary_df["category"].values:
                    st.info("Not enough data for this chart with the current filters")
                else:
                    fig = visualizer.plot_metric_by_param(
                        summary=summary_df,
                        category="size_sweep",
                        metric="median_runtime_ms",
                        ylabel="Median Runtime (ms, Log Scale)",
                        title="Runtime vs. Grid Size",
                        theme="dark",
                        log_y=True,
                    )
            elif chart_sel == "Expansion efficiency":
                fig = visualizer.plot_expansion_efficiency(summary=summary_df, theme="dark")
            elif chart_sel == "Cost quality":
                fig = visualizer.plot_cost_quality(summary=summary_df, theme="dark")
            elif chart_sel == "Runtime vs expansions":
                fig = visualizer.plot_runtime_vs_expansions(results_df=filtered_df, theme="dark")

            if fig is not None:
                stretch(st.pyplot, fig)
                png_bytes = fig_to_png_bytes(fig)
                st.download_button(
                    "Download Chart PNG",
                    data=png_bytes,
                    file_name=f"{visualizer.slugify(chart_sel)}.png",
                    mime="image/png",
                    key=f"dl_chart_{visualizer.slugify(chart_sel)}",
                )
                plt.close(fig)
        except (ValueError, KeyError) as err:
            st.info("Not enough data for this chart with the current filters")
        except Exception as err:
            st.error(f"Error rendering chart: {err}")

    with tab_summary:
        group_mode = st.selectbox(
            "Grouping level",
            ["By scenario parameter", "By category"],
            key="bm_group",
        )
        if group_mode == "By category":
            disp_sum = benchmark.summarize(filtered_df, group_cols=["category", "algorithm"])
        else:
            disp_sum = summary_df

        stretch(st.dataframe, disp_sum, hide_index=True)
        csv_sum = disp_sum.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Summary CSV",
            data=csv_sum,
            file_name="benchmark_summary.csv",
            mime="text/csv",
            key="bm_dl_summary",
        )

    with tab_winners:
        if winners_df.empty:
            st.info("No winner counts available for the selected data.")
        else:
            try:
                fig_win = visualizer.plot_winner_counts(winners=winners_df, theme="dark")
                stretch(st.pyplot, fig_win)
                plt.close(fig_win)
            except Exception:
                pass
            stretch(st.dataframe, winners_df, hide_index=True)

    with tab_raw:
        st.subheader("Per-Node Expansion Cost")
        try:
            cost_tbl = visualizer.per_node_cost_table(filtered_df)
            stretch(st.dataframe, cost_tbl, hide_index=True)
            st.caption(
                "µs/node is the least-squares empirical cost of one node expansion across all scenarios."
            )
        except Exception:
            pass

        st.subheader("Raw Benchmark Results")
        stretch(st.dataframe, filtered_df, height=420, hide_index=True)
        raw_csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Raw Results CSV",
            data=raw_csv,
            file_name="benchmark_results.csv",
            mime="text/csv",
            key="bm_dl_raw",
        )

    # Footer actions
    st.divider()
    foot_c1, foot_c2 = st.columns(2)

    logs_exist = pathlib.Path(SAVED_CSV).exists()
    can_save = True
    if logs_exist:
        overwrite = foot_c1.checkbox("Overwrite existing logs in outputs/logs", key="bm_overwrite")
        can_save = overwrite

    if foot_c1.button("Save to outputs/logs", key="bm_save", disabled=not can_save):
        paths_out = benchmark.export_results(bench, "outputs/logs")
        foot_c1.success(f"Saved results and metadata to `{paths_out['results']}`")

    if foot_c2.button("Generate report", key="bm_report"):
        with st.spinner("Generating publication report and charts..."):
            rep_res = report.export_report(bench, "outputs/report")
            foot_c2.success(f"Generated report at `{rep_res['report']}`")


# =====================================================================
# PART C — Insights Page
# =====================================================================

def render_insights_page() -> None:
    """Render analytical insights, findings cards, cheatsheets, and limitations."""
    st.title("Key Findings & Insights")
    st.write("Empirical conclusions, algorithm cheat sheets, and correctness verification principles.")

    bench = get_active_bench()
    if bench is not None and not bench.results.empty:
        h_nums = report.headline_numbers(bench.results)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Scenarios Tested", f"{int(h_nums['scenarios'])}")
        c2.metric("Algorithm Runs", f"{int(h_nums['runs'])}")
        opt_str = f"{h_nums['optimal_pct']:.1f}%" if not math.isnan(h_nums["optimal_pct"]) else "n/a"
        c3.metric("Path Optimality", opt_str)
        save_str = (
            f"{h_nums['best_saving_pct']:.0f}%"
            if not math.isnan(h_nums["best_saving_pct"])
            else "n/a"
        )
        c4.metric("Max A* Node Savings", save_str)

        st.subheader("Empirical Findings")
        findings = report.build_findings(bench.results)
        for f in findings:
            msg = f"**{f.title}.** {f.text}"
            if f.tone == "success":
                st.success(msg)
            elif f.tone == "warning":
                st.warning(msg)
            else:
                st.info(msg)
            if f.evidence:
                with st.expander(f"Evidence: {f.title}"):
                    ev_rows = [
                        {
                            "Metric": k,
                            "Value": f"{v:.4g}" if isinstance(v, float) else str(v),
                        }
                        for k, v in f.evidence.items()
                    ]
                    ev_df = pd.DataFrame(ev_rows)
                    stretch(st.dataframe, ev_df, hide_index=True)
    else:
        st.info("No active benchmark results loaded. Visit the Benchmark page to run or load benchmark results.")

    # Static reference sections
    st.divider()
    st.subheader("When to Use Which Algorithm")
    when_df = pd.DataFrame(report.WHEN_TO_USE, columns=["Situation", "Recommendation"])
    stretch(st.dataframe, when_df, hide_index=True)

    st.subheader("Algorithm Cheat Sheet")
    stretch(st.dataframe, report.algorithm_cheatsheet(), hide_index=True)

    st.subheader("Heuristics and Admissibility")
    stretch(st.dataframe, report.heuristic_table(), hide_index=True)

    st.subheader("How the Results Are Verified")
    st.markdown(
        """
- **Cell-by-cell path validation**: Every single path is checked against the grid to guarantee start-to-goal continuity, walkability, boundary adherence, and that diagonal transitions never cut corners.
- **Independent baseline check**: Dijkstra is checked against an independent BFS on 20 random grids to verify step and cost optimality.
- **A* zero-heuristic identity**: A* with the zero heuristic ($h=0$) must explore identical states and return identical costs to Dijkstra.
- **Consistency invariant**: Consistent heuristics ($h(u) \\le c(u, v) + h(v)$) never re-expand any node; maximum expansion count per node is exactly 1.
- **Strict optimality guard**: Any algorithm claiming an optimal path with a cost strictly less than Dijkstra's immediately aborts the benchmark with a RuntimeError.
        """
    )

    st.subheader("Limitations")
    for lim in report.LIMITATIONS:
        st.markdown(f"- {lim}")
