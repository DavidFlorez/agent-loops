"""Generate study plots under assets/ as publication-quality PNGs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _short_model_name(m: str) -> str:
    m = m.split("@")[0]
    m = m.split("/")[-1]
    for suf in ("-preview", "-20250929", "-20251001", "-20250805", "-20251101", "-0709", "-0905", "-Instruct-FP8"):
        m = m.replace(suf, "")
    return m[:26]


def plot_step_vs_dollar_share(
    tb_models: pd.DataFrame,
    swe_models: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot (a): loop share of steps vs loop share of cache-aware dollars per model."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=180)
    datasets = [
        ("Terminal-Bench 2.0 (by model)", tb_models, "#1f77b4", axes[0]),
        ("SWE-rebench July 2026 (by model)", swe_models, "#d62728", axes[1]),
    ]

    for title, df, color, ax in datasets:
        sub = df.dropna(subset=["loop_dollar_share"]).copy()
        sub = sub[sub["cost_runs"] >= 10]
        if sub.empty:
            ax.set_title(f"{title} (no cost data)")
            continue
        x = sub["loop_step_share"].to_numpy() * 100.0
        y = sub["loop_dollar_share"].to_numpy() * 100.0
        sizes = np.clip(sub["cost_runs"].to_numpy() * 0.25, 35, 360)

        lim = max(float(np.max(x)), float(np.max(y)), 5.0) * 1.15
        ax.plot([0, lim], [0, lim], linestyle="--", color="#666666", linewidth=1.2, label="y = x (equal share)")
        ax.scatter(x, y, s=sizes, c=color, alpha=0.78, edgecolors="white", linewidth=0.8)

        sub = sub.assign(_x=x, _y=y, _gap=y - x)
        picks = pd.concat(
            [
                sub.nlargest(5, "_y"),
                sub.nlargest(3, "_gap"),
                sub[sub["model"].str.lower().str.contains("gemini")].head(4),
            ]
        ).drop_duplicates(subset=["model"])

        for idx, (_, row) in enumerate(picks.iterrows()):
            rx, ry = float(row["_x"]), float(row["_y"])
            short = _short_model_name(str(row["model"]))
            if short == "claude-haiku-4-5":
                dx, dy, ha = 6, 11, "left"
            else:
                if rx < lim * 0.28:
                    dx, ha = 7, "left"
                elif rx > lim * 0.78:
                    dx, ha = -7, "right"
                else:
                    dx = 6 if idx % 2 == 0 else -6
                    ha = "left" if dx > 0 else "right"
                dy = 5 if idx % 2 == 0 else -8
            ax.annotate(
                short,
                (rx, ry),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.6,
                ha=ha,
                alpha=0.93,
            )

        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("Loop Share of Steps (%)", fontsize=10.5)
        ax.set_ylabel("Loop Share of Cache-Aware Dollars (%)", fontsize=10.5)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.legend(loc="upper left", fontsize=8.5, frameon=True)

    fig.suptitle(
        "Loop Steps vs Cache-Aware Dollar Share per Model",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _short_detector_name(name: str) -> str:
    mapping = {
        "cline_style": "Cline (5 adj)",
        "openhands_stuck": "OpenHands Stuck",
        "gemini_cli": "Gemini CLI",
        "openclaw_10": "OpenClaw-10",
        "openclaw_20": "OpenClaw-20",
        "turn_budget_1_25x_p90": "1.25x P90 Budget",
    }
    if name in mapping:
        return mapping[name]
    if name.startswith("step_cap_"):
        return f"Cap {name.split('_')[-1]}"
    return name


_DETECTOR_OFFSETS_TB = {
    "cline_style": (6, 5, "left"),
    "openhands_stuck": (6, -10, "left"),
    "gemini_cli": (-6, 6, "right"),
    "openclaw_10": (3, 10, "left"),
    "openclaw_20": (5, 6, "left"),
    "turn_budget_1_25x_p90": (6, 5, "left"),
    "step_cap_30": (5, -6, "left"),
    "step_cap_50": (6, 5, "left"),
    "step_cap_75": (6, 5, "left"),
    "step_cap_100": (6, 5, "left"),
    "step_cap_150": (6, -6, "left"),
}

_DETECTOR_OFFSETS_SWE = {
    "cline_style": (5, 7, "left"),
    "openhands_stuck": (6, -10, "left"),
    "gemini_cli": (-5, 7, "right"),
    "openclaw_10": (0, 8, "center"),
    "openclaw_20": (-5, 6, "right"),
    "turn_budget_1_25x_p90": (-6, 6, "right"),
    "step_cap_30": (5, -6, "left"),
    "step_cap_50": (6, 5, "left"),
    "step_cap_75": (6, 5, "left"),
    "step_cap_100": (6, 5, "left"),
    "step_cap_150": (6, 6, "left"),
}


def plot_detector_savings_vs_false_stop(
    tb_detectors: pd.DataFrame,
    swe_detectors: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot (b): dollars saved vs false-stop rate for each detector and its matched-random point."""
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8), dpi=180)
    panels = [
        ("Terminal-Bench 2.0", tb_detectors, axes[0], _DETECTOR_OFFSETS_TB),
        ("SWE-rebench July 2026", swe_detectors, axes[1], _DETECTOR_OFFSETS_SWE),
    ]

    for title, df, ax, offset_map in panels:
        if df.empty:
            ax.set_title(f"{title} (empty)")
            continue
        det_x = df["false_stop_rate"].to_numpy() * 100.0
        det_y = df["dollars_saved_pct"].to_numpy() * 100.0
        rnd_x = df["rand_false_stop_mean"].to_numpy() * 100.0
        rnd_y = df["rand_dollars_saved_pct_mean"].to_numpy() * 100.0
        rnd_xerr_low = np.maximum(0.0, rnd_x - df["rand_false_stop_ci_low"].to_numpy() * 100.0)
        rnd_xerr_high = np.maximum(0.0, df["rand_false_stop_ci_high"].to_numpy() * 100.0 - rnd_x)
        rnd_yerr_low = np.maximum(0.0, rnd_y - df["rand_dollars_saved_pct_ci_low"].to_numpy() * 100.0)
        rnd_yerr_high = np.maximum(0.0, df["rand_dollars_saved_pct_ci_high"].to_numpy() * 100.0 - rnd_y)

        for i in range(len(df)):
            ax.plot([det_x[i], rnd_x[i]], [det_y[i], rnd_y[i]], color="#bbbbbb", linestyle=":", linewidth=1.0, zorder=1)

        ax.errorbar(
            rnd_x,
            rnd_y,
            xerr=[rnd_xerr_low, rnd_xerr_high],
            yerr=[rnd_yerr_low, rnd_yerr_high],
            fmt="x",
            color="#7f7f7f",
            ecolor="#b0b0b0",
            elinewidth=1.0,
            capsize=2.5,
            markersize=6.5,
            label="Matched-Random (200 seeds, 95% CI)",
            zorder=2,
        )

        is_cap = df["name"].str.startswith("step_cap_") | (df["name"] == "turn_budget_1_25x_p90")
        ax.scatter(
            det_x[~is_cap],
            det_y[~is_cap],
            s=75,
            c="#1f77b4",
            marker="o",
            edgecolors="white",
            linewidth=0.8,
            label="Loop Detectors (Rule-Based)",
            zorder=4,
        )
        ax.scatter(
            det_x[is_cap],
            det_y[is_cap],
            s=70,
            c="#ff7f0e",
            marker="s",
            edgecolors="white",
            linewidth=0.8,
            label="Step Caps & Turn Budget",
            zorder=4,
        )

        for i, row in df.iterrows():
            dname = str(row["name"])
            lbl = _short_detector_name(dname)
            dx, dy, ha = offset_map.get(dname, (5, 5, "left"))
            ax.annotate(
                lbl,
                (det_x[i], det_y[i]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=7.5,
                ha=ha,
                alpha=0.95,
            )

        ax.set_xlabel("False-Stop Rate (% of fires on runs that succeeded)", fontsize=10.5)
        ax.set_ylabel("Cache-Aware Dollars Saved (%)", fontsize=10.5)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xlim(-2, max(float(np.max(det_x)), float(np.max(rnd_x)), 10.0) * 1.12)
        ax.set_ylim(-2.5, max(float(np.max(det_y)), float(np.max(rnd_y)), 5.0) * 1.15)
        ax.grid(True, linestyle=":", alpha=0.45)
        ax.legend(loc="upper left", fontsize=8.5, frameon=True)

    fig.suptitle(
        "Loop Detectors vs Matched-Random Stopping: Dollars Saved vs False-Stop Rate",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_loop_prevalence_by_harness(
    tb_harnesses: pd.DataFrame,
    swe_harnesses: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot (c): horizontal bar chart of loop prevalence per harness across both benchmarks."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=180)
    panels = [
        ("Terminal-Bench 2.0 (9 harnesses with steps)", tb_harnesses, axes[0]),
        ("SWE-rebench July 2026 (17 participants)", swe_harnesses, axes[1]),
    ]

    for title, df, ax in panels:
        if df.empty:
            ax.set_title(f"{title} (empty)")
            continue
        sdf = df.sort_values("loop_run_rate", ascending=True).copy()
        names = sdf["harness"].astype(str).tolist()
        any_pct = sdf["loop_run_rate"].to_numpy() * 100.0
        rot_pct = (sdf["rotation_runs"] / sdf["runs"]).to_numpy() * 100.0
        y_pos = np.arange(len(names))

        ax.barh(y_pos, any_pct, color="#4c72b0", height=0.65, label="Any Loop (% of runs)")
        ax.barh(y_pos, rot_pct, color="#dd8452", height=0.65, label="Period-k Rotation (k<=5)")

        for y_i, (a_val, n_runs) in enumerate(zip(any_pct, sdf["runs"].to_numpy())):
            ax.text(
                a_val + 0.8,
                y_i,
                f"{a_val:.1f}% (n={int(n_runs)})",
                va="center",
                fontsize=7.8,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8.8)
        ax.set_xlabel("Share of Runs (%)", fontsize=10.5)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        max_val = float(np.max(any_pct))
        if max_val > 70.0:
            ax.set_xlim(0, 118.0)
            ax.set_xticks([0, 20, 40, 60, 80, 100])
        else:
            ax.set_xlim(0, max(max_val, 10.0) * 1.26)
        ax.grid(True, axis="x", linestyle=":", alpha=0.45)
        ax.legend(loc="lower right", fontsize=8.5, frameon=True)

    fig.suptitle(
        "Loop Prevalence by Agent Harness / Participant",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


_OPEN_COLOR = "#D55E00"   # Okabe-Ito vermillion (Open-Weights)
_CLOSED_COLOR = "#0072B2"  # Okabe-Ito blue (Closed/Proprietary)
_GEMINI_COLOR = "#009E73"  # Okabe-Ito bluish-green (Google Gemini callouts)


def plot_loops_per_100_by_model_band(
    model_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot loops per 100 steps by model with 95% clustered CIs grouped by pass-rate band across primary cohorts."""
    panel_cohorts = [
        ("tb_terminus_2", "Terminal-Bench 2.0 (terminus-2)"),
        ("swe_rebench_swelike", "SWE-rebench Jul 2026 (swelike)"),
        ("longtail_swe_swelike", "Longtail-SWE (swelike)"),
        ("tau2_classic", "Sierra tau2 Classic (airline+retail+telecom)"),
        ("tau2_banking_knowledge_alltools", "Sierra tau2 Banking Knowledge (AllTools)"),
    ]
    active = [(c, lbl) for c, lbl in panel_cohorts if not model_df[model_df["cohort"] == c].empty]
    if not active:
        return

    n_panels = len(active)
    ncols = 2
    nrows = (n_panels + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.8 * nrows), dpi=180)
    axes_flat = np.atleast_1d(axes).ravel()

    band_order = {"B5 (>=75%)": 5, "B4 (55-75%)": 4, "B3 (40-55%)": 3, "B2 (25-40%)": 2, "B1 (<25%)": 1}

    for idx_p, (cname, title) in enumerate(active):
        ax = axes_flat[idx_p]
        sub = model_df[model_df["cohort"] == cname].copy()
        sub["_bo"] = sub["pass_rate_band"].map(lambda b: band_order.get(str(b), 0))
        sub = sub.sort_values(
            ["_bo", "executed_pass_rate", "loops_per_100_steps"],
            ascending=[True, True, False],
        ).reset_index(drop=True)

        y_pos = np.arange(len(sub))
        vals = sub["loops_per_100_steps"].to_numpy()
        xerr_lo = np.maximum(0.0, vals - sub["loops_per_100_steps_ci_low"].to_numpy())
        xerr_hi = np.maximum(0.0, sub["loops_per_100_steps_ci_high"].to_numpy() - vals)

        for yi, (_, row) in enumerate(sub.iterrows()):
            is_op = row["weights"] == "Open"
            col = _OPEN_COLOR if is_op else _CLOSED_COLOR
            marker = "s" if is_op else ("D" if row["family"] == "gemini" else "o")
            if row["family"] == "gemini":
                col = _GEMINI_COLOR
            ax.errorbar(
                vals[yi],
                yi,
                xerr=[[xerr_lo[yi]], [xerr_hi[yi]]],
                fmt=marker,
                color=col,
                ecolor=col,
                elinewidth=1.2,
                capsize=2.5,
                markersize=5.2,
                alpha=0.92,
                zorder=3,
            )

        # Horizontal dividers between pass-rate bands
        prev_band = None
        for yi, (_, row) in enumerate(sub.iterrows()):
            b = str(row["pass_rate_band"])
            if prev_band is not None and b != prev_band:
                ax.axhline(yi - 0.5, color="#cccccc", linestyle="--", linewidth=0.8, zorder=1)
            prev_band = b

        ylabels = [
            f"{str(r['canonical_name'])[:22]} ({r['executed_pass_rate']*100:.0f}%, {str(r['pass_rate_band']).split()[0]})"
            for _, r in sub.iterrows()
        ]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ylabels, fontsize=7.0 if len(sub) > 18 else 7.8)
        ax.set_xlabel("Loops per 100 Executed Steps (95% Task-Clustered CI)", fontsize=9.5)
        ax.set_title(title, fontsize=11.0, fontweight="bold")
        ax.set_xlim(0.0, max(5.0, float(np.nanmax(vals + xerr_hi)) * 1.12))
        ax.grid(True, axis="x", linestyle=":", alpha=0.4)

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].axis("off")

    # Legend in last empty slot or top figure legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=_OPEN_COLOR, markersize=8, label="Open-Weights"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_CLOSED_COLOR, markersize=8, label="Closed (Non-Gemini)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=_GEMINI_COLOR, markersize=8, label="Closed (Google Gemini)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.968), frameon=True, fontsize=10.0)
    fig.suptitle(
        "Loop Rate by Model & Pass-Rate Capability Band (Within-Scaffold, Shared-Task Cohorts)",
        fontsize=13.5,
        fontweight="bold",
        y=0.992,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_open_closed_gap_before_after_matching(
    matching: dict[str, dict[str, Any]],
    out_path: Path,
) -> None:
    """Plot Open - Closed difference (95% CI) before and after capability matching across all cohorts."""
    ordered = [c for c in matching.keys()]
    if not ordered:
        return
    ordered = list(reversed(ordered))

    short_labels = {
        "tb_terminus_2": "TB terminus-2",
        "tb_openhands": "TB openhands",
        "tb_mini_swe_agent": "TB mini-swe-agent",
        "swe_rebench_swelike": "SWE-rebench swelike",
        "longtail_swe_swelike": "Longtail-SWE swelike",
        "tau2_classic": "Sierra tau2 Classic",
        "tau2_banking_knowledge_alltools": "Sierra BK AllTools",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=180, sharey=True)
    specs = [
        ("loops_per_100_steps", 1.0, "Open - Closed: Loops per 100 Steps (95% CI)", axes[0]),
        ("any_loop_rate", 100.0, "Open - Closed: Any-Loop Run Share (percentage points, 95% CI)", axes[1]),
    ]
    y_base = np.arange(len(ordered), dtype=np.float64)

    for metric_key, scale, xlabel, ax in specs:
        ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1.2, zorder=1)
        for yi, cname in enumerate(ordered):
            entry = matching[cname][metric_key]
            u_pt = entry["unmatched_diff"] * scale
            u_lo = max(0.0, u_pt - entry["unmatched_diff_ci"][0] * scale)
            u_hi = max(0.0, entry["unmatched_diff_ci"][1] * scale - u_pt)
            ax.errorbar(
                u_pt,
                yi + 0.14,
                xerr=[[u_lo], [u_hi]],
                fmt="o",
                color="#7f7f7f",
                ecolor="#999999",
                elinewidth=1.6,
                capsize=3.5,
                markersize=7.0,
                label="Before Matching (Unmatched)" if yi == 0 else None,
                zorder=3,
            )

            if entry["overlap_bands"] > 0 and not np.isnan(entry["matched_diff"]):
                m_pt = entry["matched_diff"] * scale
                m_lo = max(0.0, m_pt - entry["matched_diff_ci"][0] * scale)
                m_hi = max(0.0, entry["matched_diff_ci"][1] * scale - m_pt)
                col = _OPEN_COLOR if m_pt > 0 else _CLOSED_COLOR
                ax.errorbar(
                    m_pt,
                    yi - 0.14,
                    xerr=[[m_lo], [m_hi]],
                    fmt="s",
                    color=col,
                    ecolor=col,
                    elinewidth=2.0,
                    capsize=3.5,
                    markersize=7.2,
                    label="After Pass-Rate Band Matching" if yi == 0 else None,
                    zorder=4,
                )
            else:
                ax.text(0.2, yi - 0.14, "no overlap", va="center", fontsize=7.5, color="#888888", style="italic")

        ax.set_xlabel(xlabel, fontsize=10.0)
        ax.grid(True, axis="x", linestyle=":", alpha=0.45)

    axes[0].set_yticks(y_base)
    axes[0].set_yticklabels([short_labels.get(c, c) for c in ordered], fontsize=10.0)
    axes[0].set_title("Loop Rate Density Gap (Loops / 100 Steps)", fontsize=11.5, fontweight="bold")
    axes[1].set_title("Run-Level Loop Prevalence Gap (pp)", fontsize=11.5, fontweight="bold")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="#7f7f7f", markersize=7.5, lw=1.6, label="Before Matching (Unmatched)"),
        Line2D([0], [0], marker="s", color=_OPEN_COLOR, markersize=7.5, lw=2.0, label="After Matching (Open > Closed)"),
        Line2D([0], [0], marker="s", color=_CLOSED_COLOR, markersize=7.5, lw=2.0, label="After Matching (Open < Closed Reversal)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.945), frameon=True, fontsize=9.5)
    fig.suptitle(
        "Open-Weights vs Closed Loop Gap Before and After Capability-Band Matching",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_open_vs_closed_across_four_datasets(
    model_df: pd.DataFrame,
    regressions: dict[str, Any],
    out_path: Path,
) -> None:
    """Plot pass rate vs loops/100 steps across all 4 trajectory corpora highlighting Open, Closed & Gemini."""
    panels = [
        ("Terminal-Bench 2.0 (terminus-2)", ["tb_terminus_2"], regressions.get("tb_terminus_2")),
        ("SWE-rebench Jul 2026 (swelike)", ["swe_rebench_swelike"], regressions.get("swe_rebench_swelike")),
        ("Longtail-SWE (swelike)", ["longtail_swe_swelike"], regressions.get("longtail_swe_swelike")),
        (
            "Sierra tau2 (Classic & Banking AllTools)",
            ["tau2_classic", "tau2_banking_knowledge_alltools"],
            regressions.get("tau2_banking_knowledge_alltools"),
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.2), dpi=180)
    axes_flat = axes.ravel()

    for idx_p, (title, clist, reg) in enumerate(panels):
        ax = axes_flat[idx_p]
        sub = model_df[model_df["cohort"].isin(clist)].copy()
        if sub.empty:
            ax.set_title(f"{title} (empty)")
            continue

        x = sub["executed_pass_rate"].to_numpy() * 100.0
        y = sub["loops_per_100_steps"].to_numpy()

        for _, row in sub.iterrows():
            rx = float(row["executed_pass_rate"]) * 100.0
            ry = float(row["loops_per_100_steps"])
            is_op = row["weights"] == "Open"
            is_gem = row["family"] == "gemini"
            is_cl = row["cohort"] == "tau2_classic"
            col = _GEMINI_COLOR if is_gem else (_OPEN_COLOR if is_op else _CLOSED_COLOR)
            if len(clist) > 1 and is_cl:
                marker = "^"
            else:
                marker = "D" if is_gem else ("s" if is_op else "o")
            ax.scatter(rx, ry, s=72, c=col, marker=marker, alpha=0.88, edgecolors="white", linewidth=0.8, zorder=3)

        # Annotate selected extremes + highest-capability Open + Gemini models cleanly
        sub = sub.assign(_x=x, _y=y)
        picks = pd.concat(
            [
                sub.nlargest(2, "_y"),
                sub[sub["weights"] == "Open"].nlargest(2, "_x"),
                sub[sub["weights"] == "Closed"].nlargest(1, "_x"),
                sub[sub["family"] == "gemini"].head(2),
            ]
        ).drop_duplicates(subset=["cohort", "model"])

        for k_i, (_, row) in enumerate(picks.iterrows()):
            rx, ry = float(row["_x"]), float(row["_y"])
            short = str(row["canonical_name"])[:19]
            dx = -6 if rx > 68 else 6
            ha = "right" if dx < 0 else "left"
            dy = 5 if k_i % 2 == 0 else -8
            ax.annotate(short, (rx, ry), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, ha=ha, alpha=0.92)

        # Add regression IRR box
        if reg and (reg.get("negbin_offset") or {}).get("coefs", {}).get("is_open_weight"):
            c = reg["negbin_offset"]["coefs"]["is_open_weight"]
            txt = f"NB Offset Adj IRR = {c['irr_or']:.2f} [{c['ci_low']:.2f}, {c['ci_high']:.2f}]\np_2way = {c['p']:.3f}"
            ax.text(
                0.97,
                0.95,
                txt,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8.3,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.92),
            )

        ax.set_xlabel("Executed Pass Rate (%)", fontsize=10.0)
        ax.set_ylabel("Loops per 100 Executed Steps", fontsize=10.0)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xlim(-3, 103)
        ax.set_ylim(-0.5, max(5.0, float(np.max(y)) * 1.18))
        ax.grid(True, linestyle=":", alpha=0.45)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=_OPEN_COLOR, markersize=8.5, label="Open-Weights"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_CLOSED_COLOR, markersize=8.5, label="Closed (Non-Gemini)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=_GEMINI_COLOR, markersize=8.5, label="Closed (Google Gemini)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#666666", markersize=8.5, label="Sierra tau2 Classic (Panel 4 triangles)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.956), frameon=True, fontsize=9.5)
    fig.suptitle(
        "Loop Rate vs Task Pass Rate Across Four Public Benchmarks (Uniform-Scaffold Cohorts)",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.935])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

