"""Automated benchmark report generator and analytical findings engine.

Parses benchmark experiment runs, evaluates statistical findings across search
algorithms and heuristics, generates publication-ready Markdown reports with
embedded figures, and provides CLI export tools without depending on Streamlit
or matplotlib.pyplot.
"""

from dataclasses import dataclass
import json
import math
import pathlib
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.algorithms import ALGORITHM_INFO, ALGORITHMS
from src.benchmark import (
    BenchmarkResult,
    run_benchmark,
    summarize,
    winner_counts,
)
from src.visualizer import (
    export_benchmark_charts,
    per_node_cost_table,
    sort_algorithm_labels,
)


# =====================================================================
# D1.1 — Constants
# =====================================================================

REQUIRED_RESULT_COLUMNS: List[str] = [
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

DEFAULT_CSV: str = "outputs/logs/results.csv"

LIMITATIONS: List[str] = [
    "Timings come from one machine and pure Python; median-of-N with GC disabled reduces but does not remove noise, so compare expansions first and runtimes second.",
    "Expansion counts depend on tie-breaking (insertion order); another tie-break rule changes counts but never path cost for admissible A*.",
    "Grids are small (up to ~100x100 in the standard suite) and obstacle density stops at 35% because 4-direction random grids near 40% are almost never solvable (percolation threshold).",
    "Only uniform random obstacles, perfect mazes, one trap and one mud barrier are tested; real maps have different structure.",
    "Terrain costs are limited to 1 (free) and 5 (mud); diagonal moves cost sqrt(2) times the terrain cost and never cut corners.",
    "Results describe the seeds and parameters in run_log.json, not all possible grids.",
]

WHEN_TO_USE: List[Tuple[str, str]] = [
    ("Uniform costs, fewest moves wanted", "BFS - cheapest per node, optimal in steps"),
    ("Weighted terrain, cheapest path wanted, no goal heuristic", "Dijkstra - optimal cost, explores in all directions"),
    ("Cheapest path wanted and the goal location is known", "A* with an admissible heuristic (Manhattan for 4-direction, Octile for 8-direction)"),
    ("Speed matters more than path quality", "Greedy Best-First - very few expansions, no optimality guarantee"),
]


# =====================================================================
# D1.2 — Loading Helpers
# =====================================================================

def benchmark_from_frame(
    df: pd.DataFrame,
    meta: Optional[Dict[str, Any]] = None,
) -> BenchmarkResult:
    """Validate DataFrame schema, cast flags, and reconstruct a BenchmarkResult.

    Args:
        df: Raw results DataFrame.
        meta: Optional execution metadata dictionary.

    Returns:
        BenchmarkResult instance containing results, summary, winners, and meta.

    Raises:
        ValueError: If required metric columns are missing from the DataFrame.
    """
    missing = [c for c in REQUIRED_RESULT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in benchmark DataFrame: {missing}")

    clean_df = df.copy()
    bool_cols = ["found", "admissible", "expects_optimal_cost", "is_cost_optimal", "allow_diagonal"]
    for col in bool_cols:
        clean_df[col] = clean_df[col].astype(bool)

    summary_df = summarize(clean_df)
    winners_df = winner_counts(clean_df)

    if meta is None:
        rep_val = int(clean_df["repeats"].max()) if not clean_df.empty and "repeats" in clean_df else 1
        meta = {
            "suite": "unknown",
            "n_rows": len(clean_df),
            "n_scenarios": int(clean_df["scenario"].nunique()),
            "skipped": [],
            "repeats": rep_val,
        }

    return BenchmarkResult(
        results=clean_df,
        summary=summary_df,
        winners=winners_df,
        meta=meta,
    )


def load_benchmark(
    csv_path: Union[str, pathlib.Path] = DEFAULT_CSV,
    meta_path: Optional[Union[str, pathlib.Path]] = None,
) -> BenchmarkResult:
    """Load benchmark results and optional run_log.json metadata from disk.

    Args:
        csv_path: Path to results.csv.
        meta_path: Optional explicit path to run_log.json.

    Returns:
        BenchmarkResult instance.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
    """
    p = pathlib.Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Benchmark results not found at {csv_path}. Hint: run: python -m src.benchmark --run standard"
        )

    df = pd.read_csv(p)

    meta_dict: Optional[Dict[str, Any]] = None
    if meta_path is not None:
        mp = pathlib.Path(meta_path)
        if mp.exists():
            meta_dict = json.loads(mp.read_text(encoding="utf-8"))
    else:
        mp = p.parent / "run_log.json"
        if mp.exists():
            meta_dict = json.loads(mp.read_text(encoding="utf-8"))

    return benchmark_from_frame(df, meta=meta_dict)


# =====================================================================
# D1.3 — Analytical Findings
# =====================================================================

@dataclass
class Finding:
    """Quantitative insight derived dynamically from benchmark experiment results.

    Attributes:
        key: Short alphanumeric identifier.
        title: Section title for report display.
        text: Descriptive narrative sentence.
        tone: Visual alert tone ('success', 'info', or 'warning').
        evidence: Dictionary of strictly finite floating-point metrics supporting the text.
    """

    key: str
    title: str
    text: str
    tone: str
    evidence: Dict[str, float]


def build_findings(results_df: pd.DataFrame) -> List[Finding]:
    """Extract prioritized, data-driven analytical takeaways from benchmark results.

    All conclusions, percentages, and trade-off statements are computed from
    the data. Missing metrics or scenarios are gracefully omitted.

    Args:
        results_df: Benchmark results DataFrame.

    Returns:
        List of Finding instances.
    """
    findings: List[Finding] = []

    # Helper filter for default A* configurations
    def_a_star = results_df[
        (results_df["base_algorithm"] == "A*")
        & (results_df["admissible"] == True)
        & (results_df["heuristic"].isin(["manhattan", "octile"]))
        & (results_df["found"] == True)
    ]

    # 1. Correctness
    corr_sub = results_df[(results_df["expects_optimal_cost"] == True) & (results_df["found"] == True)]
    if not corr_sub.empty:
        total_runs = len(corr_sub)
        ok_runs = int((corr_sub["cost_ratio"] <= 1.0 + 1e-9).sum())
        pct_ok = (ok_runs / float(total_runs)) * 100.0
        tone = "success" if ok_runs == total_runs else "warning"
        findings.append(
            Finding(
                key="correctness",
                title="Correctness",
                text=(
                    f"{ok_runs} of {total_runs} runs by Dijkstra and admissible A* returned "
                    f"a path whose cost equals the optimum ({pct_ok:.1f}%). "
                    "Every path was also re-validated against the grid."
                ),
                tone=tone,
                evidence={"total": float(total_runs), "ok": float(ok_runs), "pct": float(pct_ok)},
            )
        )

    # 2. Search Effort / Expansion Savings
    if not def_a_star.empty:
        overall_pct = float(def_a_star["nodes_vs_dijkstra_pct"].mean())
        cat_means = def_a_star.groupby("category")["nodes_vs_dijkstra_pct"].mean()
        if not cat_means.empty:
            cat_best = str(cat_means.idxmin())
            v_best = float(cat_means.min())
            cat_worst = str(cat_means.idxmax())
            v_worst = float(cat_means.max())
            findings.append(
                Finding(
                    key="expansion_savings",
                    title="Search effort",
                    text=(
                        f"On average A* expanded {overall_pct:.0f}% of the nodes Dijkstra expanded "
                        f"(best: {cat_best} at {v_best:.0f}%, worst: {cat_worst} at {v_worst:.0f}%)."
                    ),
                    tone="info",
                    evidence={
                        "overall_pct": overall_pct,
                        "best_pct": v_best,
                        "worst_pct": v_worst,
                    },
                )
            )

    # 3. Wallclock vs Expansions
    dijkstra_found = results_df[
        (results_df["base_algorithm"] == "Dijkstra") & (results_df["found"] == True)
    ].set_index("scenario")

    if not def_a_star.empty and not dijkstra_found.empty:
        common_scenarios = sorted(set(def_a_star["scenario"]).intersection(dijkstra_found.index))
        if common_scenarios:
            fewer_nodes_count = 0
            faster_time_count = 0
            n_pairs = 0

            for sc in common_scenarios:
                dijk_row = dijkstra_found.loc[sc]
                a_rows = def_a_star[def_a_star["scenario"] == sc]
                dijk_n = float(dijk_row["nodes_expanded"])
                dijk_rt = float(dijk_row["runtime_median_ms"])

                for _, a_r in a_rows.iterrows():
                    n_pairs += 1
                    if float(a_r["nodes_expanded"]) < dijk_n:
                        fewer_nodes_count += 1
                    if float(a_r["runtime_median_ms"]) < dijk_rt:
                        faster_time_count += 1

            if n_pairs > 0:
                fewer_pct = (fewer_nodes_count / float(n_pairs)) * 100.0
                faster_pct = (faster_time_count / float(n_pairs)) * 100.0

                # Per-node slope ratio
                ratio_val: Optional[float] = None
                ratio_sentence = ""
                try:
                    pn_df = per_node_cost_table(results_df).set_index("algorithm")
                    dijk_slope = float(pn_df.loc["Dijkstra", "us_per_node"]) if "Dijkstra" in pn_df.index else 0.0
                    a_slopes = [
                        float(pn_df.loc[lbl, "us_per_node"])
                        for lbl in ["A* (manhattan)", "A* (octile)"]
                        if lbl in pn_df.index
                    ]
                    if dijk_slope > 0.0 and a_slopes:
                        ratio_val = float(np.mean(a_slopes) / dijk_slope)
                        ratio_sentence = (
                            f" In pure Python each A* expansion costs about {ratio_val:.1f}x "
                            "a Dijkstra expansion (heuristic call plus a larger heap entry), "
                            "so the saving in nodes must be large before it pays off."
                        )
                except Exception:
                    ratio_val = None

                tone = "warning" if faster_pct < 50.0 else "info"
                ev_dict: Dict[str, float] = {
                    "fewer_nodes_pct": float(fewer_pct),
                    "faster_pct": float(faster_pct),
                }
                if ratio_val is not None:
                    ev_dict["per_node_ratio"] = float(ratio_val)

                findings.append(
                    Finding(
                        key="wallclock",
                        title="Expansions versus wall-clock time",
                        text=(
                            f"A* expanded fewer nodes than Dijkstra in {fewer_pct:.0f}% of scenarios "
                            f"but was faster in wall-clock time in {faster_pct:.0f}%.{ratio_sentence}"
                        ),
                        tone=tone,
                        evidence=ev_dict,
                    )
                )

    # 4. Greedy Best-First Tradeoff
    greedy_sub = results_df[
        (results_df["base_algorithm"] == "Greedy Best-First") & (results_df["found"] == True)
    ]
    if not greedy_sub.empty:
        greedy_nodes_pct = float(greedy_sub["nodes_vs_dijkstra_pct"].mean())
        sub_optimal_pct = float((greedy_sub["is_cost_optimal"] == False).sum() / float(len(greedy_sub))) * 100.0
        mean_cost_ratio = float(greedy_sub["cost_ratio"].mean())

        cat_ratios = greedy_sub.groupby("category")["cost_ratio"].mean()
        if not cat_ratios.empty:
            worst_cat = str(cat_ratios.idxmax())
            worst_cat_ratio = float(cat_ratios.max())
            findings.append(
                Finding(
                    key="greedy_tradeoff",
                    title="Greedy Best-First: speed versus quality",
                    text=(
                        f"Greedy Best-First expanded only {greedy_nodes_pct:.0f}% of Dijkstra's nodes "
                        f"but returned a sub-optimal path in {sub_optimal_pct:.0f}% of the scenarios "
                        f"(mean cost ratio {mean_cost_ratio:.3f}; worst category {worst_cat}: {worst_cat_ratio:.3f}x the optimal cost)."
                    ),
                    tone="warning",
                    evidence={
                        "nodes_pct": greedy_nodes_pct,
                        "suboptimal_pct": sub_optimal_pct,
                        "mean_cost_ratio": mean_cost_ratio,
                        "worst_cost_ratio": worst_cat_ratio,
                    },
                )
            )

    # 5. BFS Steps vs Cost
    bfs_sub = results_df[(results_df["base_algorithm"] == "BFS") & (results_df["found"] == True)]
    if not bfs_sub.empty:
        bfs_optimal_pct = float(bfs_sub["is_cost_optimal"].sum() / float(len(bfs_sub))) * 100.0
        cat_bfs_ratios = bfs_sub.groupby("category")["cost_ratio"].mean()
        losing_cats = [c for c, m in cat_bfs_ratios.items() if m > 1.001]
        cat_str = ", ".join(losing_cats) if losing_cats else "no category in this run"

        findings.append(
            Finding(
                key="bfs_steps_vs_cost",
                title="BFS: steps versus cost",
                text=(
                    f"BFS minimises the number of moves, not the cost: it was cost-optimal in "
                    f"{bfs_optimal_pct:.0f}% of scenarios and loses on weighted terrain ({cat_str})."
                ),
                tone="info",
                evidence={
                    "optimal_pct": bfs_optimal_pct,
                    "losing_categories_count": float(len(losing_cats)),
                },
            )
        )

    # 6. Heuristic Strength (4-direction Manhattan vs Euclidean)
    diag4_sub = results_df[(results_df["allow_diagonal"] == False) & (results_df["found"] == True)]
    a_manh = diag4_sub[diag4_sub["algorithm"] == "A* (manhattan)"].set_index("scenario")
    a_eucl = diag4_sub[diag4_sub["algorithm"] == "A* (euclidean)"].set_index("scenario")
    common_scen = sorted(set(a_manh.index).intersection(a_eucl.index))

    if common_scen:
        ratios = [
            float(a_eucl.loc[s, "nodes_expanded"]) / float(a_manh.loc[s, "nodes_expanded"])
            for s in common_scen
            if float(a_manh.loc[s, "nodes_expanded"]) > 0
        ]
        if ratios:
            pct_diff = (float(np.mean(ratios)) - 1.0) * 100.0
            word = "more" if pct_diff >= 0.0 else "fewer"
            findings.append(
                Finding(
                    key="heuristic_strength",
                    title="Heuristic strength",
                    text=(
                        f"With 4-direction movement the weaker Euclidean heuristic expanded "
                        f"{abs(pct_diff):.0f}% {word} nodes than Manhattan on average, with identical path cost."
                    ),
                    tone="info",
                    evidence={"diff_pct": float(abs(pct_diff))},
                )
            )

    # 7. Inadmissible Heuristic Demo (A* Manhattan on 8-dir grids)
    inad_sub = results_df[
        (results_df["base_algorithm"] == "A*")
        & (results_df["heuristic"] == "manhattan")
        & (results_df["allow_diagonal"] == True)
        & (results_df["found"] == True)
    ]
    if not inad_sub.empty:
        n_inad = float(inad_sub["nodes_vs_dijkstra_pct"].mean())
        p_inad = float(inad_sub["is_cost_optimal"].sum() / float(len(inad_sub))) * 100.0
        m_inad = float(inad_sub["cost_ratio"].mean())
        mx_inad = float(inad_sub["cost_ratio"].max())

        inad_scen = set(inad_sub["scenario"].unique())
        oct_sub = results_df[
            (results_df["scenario"].isin(inad_scen))
            & (results_df["base_algorithm"] == "A*")
            & (results_df["heuristic"] == "octile")
            & (results_df["found"] == True)
        ]
        o_oct = float(oct_sub["nodes_vs_dijkstra_pct"].mean()) if not oct_sub.empty else 0.0

        tone = "warning" if p_inad < 100.0 else "info"
        findings.append(
            Finding(
                key="inadmissible_demo",
                title="Inadmissible heuristic demo",
                text=(
                    f"Using Manhattan with diagonal moves (inadmissible) cut expansions to {n_inad:.0f}% of "
                    f"Dijkstra's (admissible Octile: {o_oct:.0f}%) but was cost-optimal in only {p_inad:.0f}% "
                    f"of scenarios (mean cost ratio {m_inad:.4f}, worst {mx_inad:.4f})."
                ),
                tone=tone,
                evidence={
                    "nodes_pct": n_inad,
                    "optimal_pct": p_inad,
                    "mean_cost_ratio": m_inad,
                    "max_cost_ratio": mx_inad,
                    "octile_nodes_pct": o_oct,
                },
            )
        )

    # 8. Scaling with Grid Size
    size_sub = results_df[
        (results_df["category"] == "size_sweep")
        & (results_df["found"] == True)
        & (
            (results_df["base_algorithm"] == "Dijkstra")
            | (
                (results_df["base_algorithm"] == "A*")
                & (results_df["admissible"] == True)
                & (results_df["heuristic"].isin(["manhattan", "octile"]))
            )
        )
    ]
    if not size_sub.empty:
        unique_sizes = sorted(size_sub["rows"].unique())
        if len(unique_sizes) >= 2:
            a_sz = int(unique_sizes[0])
            b_sz = int(unique_sizes[-1])
            area_ratio = float((b_sz * b_sz) / float(a_sz * a_sz))

            dijk_a = size_sub[(size_sub["rows"] == a_sz) & (size_sub["base_algorithm"] == "Dijkstra")]
            dijk_b = size_sub[(size_sub["rows"] == b_sz) & (size_sub["base_algorithm"] == "Dijkstra")]
            astar_a = size_sub[(size_sub["rows"] == a_sz) & (size_sub["base_algorithm"] == "A*")]
            astar_b = size_sub[(size_sub["rows"] == b_sz) & (size_sub["base_algorithm"] == "A*")]

            if not dijk_a.empty and not dijk_b.empty and not astar_a.empty and not astar_b.empty:
                d_growth = float(dijk_b["nodes_expanded"].mean() / float(dijk_a["nodes_expanded"].mean()))
                s_growth = float(astar_b["nodes_expanded"].mean() / float(astar_a["nodes_expanded"].mean()))
                findings.append(
                    Finding(
                        key="scaling",
                        title="Scaling with grid size",
                        text=(
                            f"From {a_sz}x{a_sz} to {b_sz}x{b_sz} the grid area grew {area_ratio:.0f}x; "
                            f"Dijkstra's expansions grew {d_growth:.1f}x and A*'s {s_growth:.1f}x."
                        ),
                        tone="info",
                        evidence={
                            "area_ratio": area_ratio,
                            "dijkstra_growth": d_growth,
                            "astar_growth": s_growth,
                        },
                    )
                )

    # 9. Unsolvable Mazes
    unsolv_sub = results_df[results_df["category"] == "unsolvable"]
    if not unsolv_sub.empty and (unsolv_sub["found"] == False).all():
        densities = unsolv_sub.groupby("algorithm")["expanded_density_pct"].mean()
        if not densities.empty:
            lo_den = float(densities.min())
            hi_den = float(densities.max())
            findings.append(
                Finding(
                    key="unsolvable",
                    title="Unsolvable mazes",
                    text=(
                        f"On unsolvable mazes every algorithm had to exhaust the reachable region: "
                        f"expansions ranged from {lo_den:.0f}% to {hi_den:.0f}% of the free cells "
                        "and all algorithms agreed that no path exists."
                    ),
                    tone="info",
                    evidence={"min_density_pct": lo_den, "max_density_pct": hi_den},
                )
            )

    # 10. Perfect Mazes (Single Route)
    if not def_a_star.empty:
        maze_a = def_a_star[def_a_star["category"] == "perfect_maze"]
        other_a = def_a_star[def_a_star["category"] != "perfect_maze"]
        if not maze_a.empty and not other_a.empty:
            m_pct = float(maze_a["nodes_vs_dijkstra_pct"].mean())
            o_pct = float(other_a["nodes_vs_dijkstra_pct"].mean())
            word_comp = "less" if m_pct > o_pct else "more"
            findings.append(
                Finding(
                    key="single_route_mazes",
                    title="Perfect mazes",
                    text=(
                        f"In perfect mazes (a single route) A* expanded {m_pct:.0f}% of Dijkstra's nodes "
                        f"versus {o_pct:.0f}% elsewhere - a heuristic helps {word_comp} when walls force the route."
                    ),
                    tone="info",
                    evidence={"maze_pct": m_pct, "other_pct": o_pct},
                )
            )

    return findings


def headline_numbers(results_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate executive high-level metrics across a benchmark callset."""
    findings = build_findings(results_df)
    f_map = {f.key: f for f in findings}

    opt_pct = float("nan")
    if "correctness" in f_map:
        opt_pct = f_map["correctness"].evidence.get("pct", float("nan"))

    best_saving = float("nan")
    if "expansion_savings" in f_map:
        best_val = f_map["expansion_savings"].evidence.get("best_pct", float("nan"))
        if not math.isnan(best_val):
            best_saving = 100.0 - best_val

    return {
        "scenarios": float(results_df["scenario"].nunique()),
        "runs": float(len(results_df)),
        "algorithms": float(results_df["algorithm"].nunique()),
        "categories": float(results_df["category"].nunique()),
        "optimal_pct": opt_pct,
        "best_saving_pct": best_saving,
    }


# =====================================================================
# D1.4 — Static Reference Tables
# =====================================================================

def algorithm_cheatsheet() -> pd.DataFrame:
    """Generate reference summary table of implemented pathfinding algorithms."""
    records: List[Dict[str, str]] = []
    for algo in ALGORITHMS.keys():
        info = ALGORITHM_INFO[algo]
        records.append(
            {
                "Algorithm": algo,
                "Uses heuristic": "Yes" if info["uses_heuristic"] else "No",
                "Cost-optimal": "Yes (if admissible)" if algo == "A*" else ("Yes" if info["optimal_cost"] == "yes" else "No"),
                "Step-optimal": "Yes" if info["optimal_steps"] == "yes" else "No",
                "Idea": info["description"],
            }
        )
    return pd.DataFrame(records)


def heuristic_table() -> pd.DataFrame:
    """Generate mathematical and admissibility reference table for heuristics."""
    records = [
        {
            "Heuristic": "manhattan",
            "Formula": "|dx| + |dy|",
            "Admissible (4-dir)": "Yes",
            "Admissible (8-dir)": "NO (overestimates diagonals)",
            "Note": "Standard L1 distance; optimal for 4-direction movement.",
        },
        {
            "Heuristic": "euclidean",
            "Formula": "sqrt(dx² + dy²)",
            "Admissible (4-dir)": "Yes",
            "Admissible (8-dir)": "Yes",
            "Note": "Straight-line L2 distance; admissible for any metric space.",
        },
        {
            "Heuristic": "octile",
            "Formula": "max(|dx|, |dy|) + (sqrt(2)-1)*min(|dx|, |dy|)",
            "Admissible (4-dir)": "Yes",
            "Admissible (8-dir)": "Yes",
            "Note": "Standard metric for 8-direction grid movement.",
        },
        {
            "Heuristic": "chebyshev",
            "Formula": "max(|dx|, |dy|)",
            "Admissible (4-dir)": "Yes",
            "Admissible (8-dir)": "Yes",
            "Note": "King-move metric where diagonal moves cost 1.0.",
        },
        {
            "Heuristic": "zero",
            "Formula": "0",
            "Admissible (4-dir)": "Yes",
            "Admissible (8-dir)": "Yes",
            "Note": "Degenerates A* into Dijkstra's algorithm.",
        },
    ]
    return pd.DataFrame(records)


# =====================================================================
# D1.5 — Markdown Report Builder
# =====================================================================

def df_to_markdown(
    df: pd.DataFrame,
    float_fmt: str = "{:.2f}",
    na: str = "-",
) -> str:
    """Format a pandas DataFrame as a GitHub-flavored Markdown pipe table.

    Operates without third-party dependencies (e.g. tabulate). Escapes pipe
    characters and handles missing or NaN values cleanly.
    """
    if df.empty:
        return ""

    headers = [str(c) for c in df.columns]
    lines: List[str] = [
        "| " + " | ".join(h.replace("|", "\\|") for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for _, row in df.iterrows():
        formatted_cells: List[str] = []
        for val in row:
            if val is None or (isinstance(val, (float, np.floating)) and np.isnan(val)):
                formatted_cells.append(na)
            elif isinstance(val, (float, np.floating)):
                formatted_cells.append(float_fmt.format(val))
            else:
                formatted_cells.append(str(val).replace("|", "\\|"))
        lines.append("| " + " | ".join(formatted_cells) + " |")

    return "\n".join(lines)


def build_report_markdown(
    bench: BenchmarkResult,
    findings: List[Finding],
    charts: List[Tuple[str, str]],
    figures: List[Tuple[str, str]],
) -> str:
    """Assemble complete benchmark Markdown report according to design specifications.

    Args:
        bench: BenchmarkResult instance.
        findings: Evaluated Finding instances.
        charts: Sequence of (caption, relative_path) tuples for benchmark charts.
        figures: Sequence of (caption, relative_path) tuples for scenario figures.

    Returns:
        Structured Markdown string.
    """
    df = bench.results
    hd = headline_numbers(df)

    # Scenarios summary table
    scen_records: List[Dict[str, Any]] = []
    for cat in sorted(df["category"].unique()):
        cat_df = df[df["category"] == cat]
        ex_label = cat_df["param_label"].iloc[0] if not cat_df.empty else "-"
        scen_records.append(
            {
                "Category": cat,
                "Instances": cat_df["scenario"].nunique(),
                "Example Configuration": ex_label,
            }
        )
    scen_table = pd.DataFrame(scen_records)

    # Summary table from benchmark.summarize
    sum_df = summarize(df, ["category", "algorithm"])
    sum_cols = [
        "category",
        "algorithm",
        "success_rate_pct",
        "mean_path_cost",
        "mean_cost_ratio",
        "pct_cost_optimal",
        "mean_nodes_expanded",
        "mean_expanded_density_pct",
        "mean_nodes_vs_dijkstra_pct",
        "median_runtime_ms",
    ]
    sum_display = sum_df[[c for c in sum_cols if c in sum_df.columns]]

    sections: List[str] = [
        "# Heuristic Graph Pathfinding Agent - Benchmark Report",
        "",
        "## Objective",
        (
            "This report evaluates informed and uninformed graph pathfinding algorithms "
            "across synthetic grid environments. We assess search effort (nodes expanded), "
            "path cost optimality, wall-clock latency, and frontier efficiency across "
            "diverse obstacle topologies and multi-cost terrains."
        ),
        "",
        "## Method",
        "### Algorithm Reference",
        df_to_markdown(algorithm_cheatsheet()),
        "",
        "### Heuristic Reference",
        df_to_markdown(heuristic_table()),
        "",
        "### Test Scenarios",
        df_to_markdown(scen_table),
        "",
        "### Metric Definitions",
        "- **Steps**: Total transitions along the returned path (`len(path) - 1`).",
        "- **Path cost**: Accumulated traversal cost taking terrain weights into account.",
        "- **Nodes expanded**: Total unique nodes popped and processed from the frontier.",
        "- **Expanded density %**: Ratio of expanded nodes to all non-wall grid cells.",
        "- **Cost ratio**: Returned path cost divided by Dijkstra's optimal reference cost.",
        "- **Search efficiency %**: Percentage of total expansions that lie on the final path.",
    ]

    if "timing_protocol" in bench.meta:
        sections.extend(
            [
                "",
                "### Timing Protocol",
                bench.meta["timing_protocol"],
            ]
        )

    sections.extend(
        [
            "",
            "## Results",
            f"- **Scenarios Evaluated**: {int(hd['scenarios'])}",
            f"- **Total Experimental Runs**: {int(hd['runs'])}",
            f"- **Algorithms Tested**: {int(hd['algorithms'])}",
            f"- **Topologies Covered**: {int(hd['categories'])}",
            f"- **Admissible Optimality Adherence**: {hd['optimal_pct']:.1f}%" if not math.isnan(hd["optimal_pct"]) else "- **Admissible Optimality Adherence**: -",
            f"- **Peak Node Savings vs Dijkstra**: {hd['best_saving_pct']:.1f}%" if not math.isnan(hd["best_saving_pct"]) else "- **Peak Node Savings vs Dijkstra**: -",
            "",
            "### Summary Statistics",
            df_to_markdown(sum_display),
            "",
            "### Scenario Winners",
            df_to_markdown(bench.winners),
            "",
            "### Analysis Charts",
        ]
    )

    for caption, path in charts:
        sections.append(f"![{caption}]({path})\n")

    if figures:
        sections.extend(
            [
                "",
                "## Scenario figures",
            ]
        )
        for caption, path in figures:
            sections.append(f"![{caption}]({path})\n")

    sections.extend(
        [
            "",
            "## Key findings",
        ]
    )
    for idx, f in enumerate(findings, start=1):
        sections.append(f"{idx}. **{f.title}.** {f.text}")

    sections.extend(
        [
            "",
            "## Limitations",
        ]
    )
    for lim in LIMITATIONS:
        sections.append(f"- {lim}")

    sections.extend(
        [
            "",
            "## Reproduce",
            "```bash",
            "python -m src.benchmark --run standard",
            "python -m src.report --build",
            "streamlit run app.py",
            "```",
            "",
        ]
    )

    return "\n".join(sections)


def export_report(
    bench: BenchmarkResult,
    out_dir: Union[str, pathlib.Path] = "outputs/report",
    theme: str = "light",
) -> Dict[str, str]:
    """Generate all benchmark charts, format REPORT.md, and serialize artifacts to disk.

    Args:
        bench: BenchmarkResult to document.
        out_dir: Target output directory.
        theme: Theme name for generated charts ('light' or 'dark').

    Returns:
        Dictionary mapping artifact identifiers to absolute file path strings.
    """
    out_path = pathlib.Path(out_dir)
    charts_dir = out_path / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Export benchmark charts
    chart_paths_map = export_benchmark_charts(bench=bench, out_dir=charts_dir, theme=theme)
    chart_list: List[Tuple[str, str]] = [
        ("Nodes Expanded vs. Obstacle Density", "charts/01_nodes_vs_density.png"),
        ("Nodes Expanded vs. Grid Size", "charts/02_nodes_vs_size.png"),
        ("Median Runtime vs. Grid Size", "charts/03_runtime_vs_size.png"),
        ("Search Expansion Efficiency", "charts/04_expansion_efficiency.png"),
        ("Cost Quality and Optimality", "charts/05_cost_quality.png"),
        ("Runtime vs. Nodes Expanded", "charts/06_runtime_vs_expansions.png"),
        ("Win Rate Distribution", "charts/07_winners.png"),
    ]

    # 2. Check for scenario figure comparisons
    figure_list: List[Tuple[str, str]] = []
    trap_path = (out_path / "../paths/trap_12x20__comparison.png").resolve()
    mud_path = (out_path / "../paths/mud_barrier_15x25__comparison.png").resolve()

    if trap_path.exists():
        figure_list.append(("Trap Scenario Multi-Algorithm Comparison", "../paths/trap_12x20__comparison.png"))
    if mud_path.exists():
        figure_list.append(("Mud Barrier Multi-Algorithm Comparison", "../paths/mud_barrier_15x25__comparison.png"))

    # 3. Assemble findings and report
    findings = build_findings(bench.results)
    report_md = build_report_markdown(
        bench=bench,
        findings=findings,
        charts=chart_list,
        figures=figure_list,
    )

    report_file = out_path / "REPORT.md"
    report_file.write_text(report_md, encoding="utf-8")

    result_map: Dict[str, str] = {"report": str(report_file.resolve())}
    for k, v in chart_paths_map.items():
        result_map[k] = v

    return result_map


def copy_docs_assets(
    report_dir: Union[str, pathlib.Path] = "outputs/report",
    docs_dir: Union[str, pathlib.Path] = "docs",
    paths_dir: Union[str, pathlib.Path] = "outputs/paths",
    animations_dir: Union[str, pathlib.Path] = "outputs/animations",
) -> List[str]:
    """Copy key visualization artifacts and charts to the docs/ folder for documentation.

    Returns:
        List of file paths written to docs_dir.
    """
    d_dir = pathlib.Path(docs_dir)
    d_dir.mkdir(parents=True, exist_ok=True)

    r_dir = pathlib.Path(report_dir)
    p_dir = pathlib.Path(paths_dir)
    a_dir = pathlib.Path(animations_dir)

    mappings = [
        (p_dir / "mud_barrier_15x25__comparison.png", d_dir / "mud_barrier_comparison.png"),
        (a_dir / "mud_barrier_15x25__race.gif", d_dir / "mud_barrier_race.gif"),
        (r_dir / "charts" / "04_expansion_efficiency.png", d_dir / "benchmark_expansion_efficiency.png"),
        (r_dir / "charts" / "05_cost_quality.png", d_dir / "benchmark_cost_quality.png"),
        (r_dir / "charts" / "06_runtime_vs_expansions.png", d_dir / "benchmark_runtime_vs_expansions.png"),
    ]

    copied: List[str] = []
    for src, dst in mappings:
        if src.exists():
            dst.write_bytes(src.read_bytes())
            copied.append(str(dst.resolve()))

    return copied


# =====================================================================
# D1.6 — CLI & Self-Test Suite
# =====================================================================

if __name__ == "__main__":
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description="Pathfinding benchmark report exporter.")
    parser.add_argument("--build", action="store_true", help="Build REPORT.md from results CSV.")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Path to results CSV.")
    parser.add_argument("--out", type=str, default="outputs/report", help="Report output folder.")
    parser.add_argument("--docs", action="store_true", help="Copy report assets to docs/ folder.")

    args = parser.parse_args()

    if args.build:
        b_res = load_benchmark(args.csv)
        exp_map = export_report(b_res, out_dir=args.out, theme="light")
        print("Generated benchmark report:")
        for k, v in exp_map.items():
            print(f"  {k}: {v}")

        if args.docs:
            docs_copied = copy_docs_assets(report_dir=args.out, docs_dir="docs")
            print(f"Copied {len(docs_copied)} assets to docs/")
        sys.exit(0)

    # Self-test block on quick suite run
    print("Executing Phase 8 report self-test suite...")

    quick_bench = run_benchmark("quick", repeats=2)

    # 1. headline_numbers
    hn = headline_numbers(quick_bench.results)
    assert set(hn.keys()) == {
        "scenarios",
        "runs",
        "algorithms",
        "categories",
        "optimal_pct",
        "best_saving_pct",
    }
    assert hn["optimal_pct"] == 100.0

    # 2. build_findings validation
    findings = build_findings(quick_bench.results)
    assert len(findings) >= 6, f"Expected >= 6 findings, got {len(findings)}"
    keys_found = [f.key for f in findings]
    assert len(keys_found) == len(set(keys_found)), "Finding keys must be unique"

    f_map = {f.key: f for f in findings}
    assert "correctness" in f_map
    assert f_map["correctness"].tone == "success"
    assert "inadmissible_demo" in f_map
    assert f_map["inadmissible_demo"].evidence["mean_cost_ratio"] >= 1.0
    assert "greedy_tradeoff" in f_map
    assert f_map["greedy_tradeoff"].evidence["worst_cost_ratio"] >= 1.38

    for f in findings:
        assert f.text and "nan" not in f.text.lower(), f"Finding {f.key} contained NaN: {f.text}"
        for ev_k, ev_v in f.evidence.items():
            assert math.isfinite(ev_v), f"Evidence {ev_k} in {f.key} was not finite: {ev_v}"

    # 3. Dynamic wording validation on synthetic frames
    sample_df = quick_bench.results.copy()
    # Case A: A* faster than Dijkstra in every pair
    pair_mask_a = (sample_df["base_algorithm"] == "A*") & (sample_df["admissible"] == True)
    pair_mask_d = sample_df["base_algorithm"] == "Dijkstra"
    sample_fast = sample_df.copy()
    sample_fast.loc[pair_mask_a, "runtime_median_ms"] = 0.01
    sample_fast.loc[pair_mask_d, "runtime_median_ms"] = 100.0
    f_fast = {f.key: f for f in build_findings(sample_fast)}
    if "wallclock" in f_fast:
        assert f_fast["wallclock"].tone == "info"

    # Case B: A* slower than Dijkstra
    sample_slow = sample_df.copy()
    sample_slow.loc[pair_mask_a, "runtime_median_ms"] = 100.0
    sample_slow.loc[pair_mask_d, "runtime_median_ms"] = 0.01
    f_slow = {f.key: f for f in build_findings(sample_slow)}
    if "wallclock" in f_slow:
        assert f_slow["wallclock"].tone == "warning"

    # Case C: single_route_mazes text direction ("less" vs "more")
    maze_mask = sample_df["category"] == "perfect_maze"
    other_mask = sample_df["category"] != "perfect_maze"
    # m > o -> "less"
    sample_less = sample_df.copy()
    sample_less.loc[maze_mask, "nodes_vs_dijkstra_pct"] = 90.0
    sample_less.loc[other_mask, "nodes_vs_dijkstra_pct"] = 20.0
    f_less = {f.key: f for f in build_findings(sample_less)}
    if "single_route_mazes" in f_less:
        assert "less" in f_less["single_route_mazes"].text

    # m <= o -> "more"
    sample_more = sample_df.copy()
    sample_more.loc[maze_mask, "nodes_vs_dijkstra_pct"] = 20.0
    sample_more.loc[other_mask, "nodes_vs_dijkstra_pct"] = 90.0
    f_more = {f.key: f for f in build_findings(sample_more)}
    if "single_route_mazes" in f_more:
        assert "more" in f_more["single_route_mazes"].text

    # 4. df_to_markdown formatting
    test_df = pd.DataFrame(
        [
            {"Name": "Test | A", "Val": 1.2345, "Missing": None},
            {"Name": "Test B", "Val": float("nan"), "Missing": "Data"},
        ]
    )
    md_table = df_to_markdown(test_df)
    assert "| Name | Val | Missing |" in md_table
    assert "| --- | --- | --- |" in md_table
    assert "Test \\| A" in md_table
    assert "1.23" in md_table
    assert "| - |" in md_table or "- | Data |" in md_table

    # 5. export_report in TemporaryDirectory
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = pathlib.Path(tmp_str)
        # Create mock scenario figures directory to exercise scenario figures rendering
        mock_paths = tmp_dir / "paths"
        mock_paths.mkdir()
        (mock_paths / "trap_12x20__comparison.png").write_bytes(b"dummy")
        (mock_paths / "mud_barrier_15x25__comparison.png").write_bytes(b"dummy")

        exp_res = export_report(quick_bench, out_dir=tmp_dir / "report")
        report_file = tmp_dir / "report" / "REPORT.md"
        assert report_file.exists()
        rep_text = report_file.read_text(encoding="utf-8")

        required_headings = [
            "## Objective",
            "## Method",
            "## Results",
            "## Scenario figures",
            "## Key findings",
            "## Limitations",
            "## Reproduce",
        ]
        for h in required_headings:
            assert h in rep_text, f"Missing heading {h} in report"

        # Verify image links resolve
        for line in rep_text.splitlines():
            if line.startswith("!["):
                rel_p = line.split("(")[1].split(")")[0]
                resolved_img = (tmp_dir / "report" / rel_p).resolve()
                assert resolved_img.exists(), f"Image link {rel_p} failed to resolve to existing file"

    # 6. Round trip: export_results -> load_benchmark
    with tempfile.TemporaryDirectory() as tmp_exp:
        from src.benchmark import export_results

        export_results(quick_bench, out_dir=tmp_exp)
        loaded_bench = load_benchmark(csv_path=pathlib.Path(tmp_exp) / "results.csv")
        assert len(loaded_bench.results) == len(quick_bench.results)
        assert loaded_bench.results["found"].dtype == bool
        assert loaded_bench.results["admissible"].dtype == bool
        assert len(loaded_bench.summary) == len(quick_bench.summary)

        # Removed column raises ValueError
        bad_csv = pathlib.Path(tmp_exp) / "bad.csv"
        drop_df = quick_bench.results.drop(columns=["cost_ratio"])
        drop_df.to_csv(bad_csv, index=False)
        try:
            load_benchmark(bad_csv)
            assert False, "Missing column must raise ValueError"
        except ValueError as err:
            assert "cost_ratio" in str(err)

    # Missing file raises FileNotFoundError
    try:
        load_benchmark("nonexistent_path.csv")
        assert False, "Missing file must raise FileNotFoundError"
    except FileNotFoundError:
        pass

    # 7. copy_docs_assets with temp dirs returns list without raising
    with tempfile.TemporaryDirectory() as tmp_docs:
        docs_list = copy_docs_assets(docs_dir=tmp_docs)
        assert isinstance(docs_list, list)

    # 8. Rule enforcement
    assert "streamlit" not in sys.modules, "Rule violation: streamlit must not be imported in src.report"
    assert "matplotlib.pyplot" not in sys.modules, "Rule violation: pyplot must not be imported in src.report"

    print("All report tests passed!")
