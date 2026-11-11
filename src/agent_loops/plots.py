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


def plot_hero_raw_vs_controlled(
    length_matched_df: pd.DataFrame,
    two_stage_df: pd.DataFrame,
    strict_comp_df: pd.DataFrame,
    regressions: dict[str, Any],
    out_path: Path,
) -> None:
    """1600x900 px hero figure: SWE-rebench Open-vs-Closed gap at 3 control stages + strict labeler check."""
    swe_len = length_matched_df[length_matched_df["cohort"] == "swe_rebench_swelike"]
    swe_2s = two_stage_df[two_stage_df["cohort"] == "swe_rebench_swelike"]
    swe_str = strict_comp_df[
        (strict_comp_df["cohort"] == "swe_rebench_swelike") & (strict_comp_df["metric"] == "loops_per_100_steps")
    ]

    if swe_len.empty:
        return
    r_len = swe_len.iloc[0]
    r_2s = swe_2s.iloc[0] if not swe_2s.empty else None
    r_str = swe_str.iloc[0] if not swe_str.empty else None

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=100)

    # Left panel: Loop density gap (Open - Closed loops per 100 steps) across 5 stages
    ax = axes[0]
    stage_labels = [
        "1. Raw Unmatched\n(All SWE-rebench runs)",
        "2. Skill-Matched\n(Same Pass-Rate Bands)",
        "3. Length-Matched\n(Same Step-Count Deciles)",
        "4. Skill + Length Matched\n(Same Band × Step Decile)",
    ]
    pts = [
        float(r_len["raw_diff_loops_per_100"]),
        float(r_len["cap_matched_diff_loops_per_100"]),
        float(r_len["length_matched_diff_loops_per_100"]),
        float(r_len["cap_and_length_matched_diff_loops_per_100"]),
    ]
    ci_lo = [
        float(r_len["raw_diff_ci_low"]),
        float(r_len["cap_matched_ci_low"]),
        float(r_len["length_matched_ci_low"]),
        float(r_len["cap_and_length_matched_ci_low"]),
    ]
    ci_hi = [
        float(r_len["raw_diff_ci_high"]),
        float(r_len["cap_matched_ci_high"]),
        float(r_len["length_matched_ci_high"]),
        float(r_len["cap_and_length_matched_ci_high"]),
    ]

    if r_str is not None:
        stage_labels.append("5. Strict Labeler (Matched)\n(>=3x Exact / ABA / Rotation)")
        pts.append(float(r_str["strict_matched_diff"]))
        ci_lo.append(float(r_str["strict_matched_ci_low"]))
        ci_hi.append(float(r_str["strict_matched_ci_high"]))

    y_pos = np.arange(len(stage_labels))[::-1]
    colors = ["#d95f02", "#e6550d", "#fd8d3c", "#fdae6b", "#756bb1"][: len(stage_labels)]

    ax.barh(y_pos, pts, height=0.52, color=colors, edgecolor="#222222", linewidth=1.2, alpha=0.90)
    for yp, pt, lo, hi in zip(y_pos, pts, ci_lo, ci_hi):
        if not np.isnan(lo) and not np.isnan(hi) and hi >= lo:
            ax.errorbar(
                pt,
                yp,
                xerr=[[max(0.0, pt - lo)], [max(0.0, hi - pt)]],
                fmt="none",
                ecolor="#111111",
                elinewidth=2.5,
                capsize=6,
                zorder=5,
            )
            label_txt = f"{pt:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            x_txt = max(pt, hi) + 0.08
        else:
            label_txt = f"{pt:+.2f}"
            x_txt = pt + 0.08
        ax.text(x_txt, yp, label_txt, va="center", ha="left", fontsize=13, fontweight="bold", color="#111111")

    if r_2s is not None:
        tw_lo = float(r_2s["new_twostage_ci_low"])
        tw_hi = float(r_2s["new_twostage_ci_high"])
        yp_skill = y_pos[1]
        ax.plot([tw_lo, tw_hi], [yp_skill - 0.18, yp_skill - 0.18], color="#1f78b4", lw=4.0, zorder=6)
        ax.text(
            tw_hi + 0.08,
            yp_skill - 0.22,
            f"2-Stage (model+task) CI: [{tw_lo:+.2f}, {tw_hi:+.2f}]",
            va="top",
            ha="left",
            fontsize=11.5,
            color="#1f78b4",
            fontweight="bold",
        )

    ax.axvline(0.0, color="#222222", linestyle="--", linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stage_labels, fontsize=13.5, fontweight="bold")
    ax.set_xlabel("Open minus Closed Loops per 100 Steps", fontsize=14, fontweight="bold")
    ax.set_title("A. SWE-rebench Loop Density Gap Across Controls", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlim(-0.5, max(ci_hi) + 1.8)
    ax.tick_params(axis="x", labelsize=12.5)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)

    # Right panel: Run-level Any-Loop Odds Ratio before vs after controlling for trajectory length log(steps)
    ax2 = axes[1]
    swe_reg = regressions.get("swe_rebench_swelike", {})
    pool_reg = regressions.get("pooled_within_scaffold", {})
    swe_lg = ((swe_reg.get("logit") or {}).get("coefs") or {}).get("is_open_weight", {})
    swe_lgs = ((swe_reg.get("logit_with_steps") or {}).get("coefs") or {}).get("is_open_weight", {})
    pool_lg = ((pool_reg.get("logit") or {}).get("coefs") or {}).get("is_open_weight", {})
    pool_lgs = ((pool_reg.get("logit_with_steps") or {}).get("coefs") or {}).get("is_open_weight", {})

    or_labels = [
        "SWE-rebench Logit\n(No Step-Count Control)",
        "SWE-rebench Logit + log(steps)\n(Controlled for Run Length)",
        "Pooled 7-Cohort Logit\n(No Step-Count Control)",
        "Pooled 7-Cohort Logit + log(steps)\n(Controlled for Run Length)",
    ]
    or_dicts = [swe_lg, swe_lgs, pool_lg, pool_lgs]
    y2 = np.arange(len(or_labels))[::-1]
    or_cols = ["#d95f02", "#2b8cbe", "#d95f02", "#2b8cbe"]

    for yp, d, col in zip(y2, or_dicts, or_cols):
        if not d:
            continue
        val = float(d["irr_or"])
        lo = float(d["ci_low"])
        hi = float(d["ci_high"])
        pval = float(d["p"])
        ax2.barh(yp, val - 1.0, left=1.0, height=0.46, color=col, edgecolor="#222222", linewidth=1.2, alpha=0.88)
        ax2.errorbar(
            val,
            yp,
            xerr=[[max(0.0, val - lo)], [max(0.0, min(hi, 13.5) - val)]],
            fmt="o",
            color="#111111",
            ecolor="#111111",
            elinewidth=2.2,
            capsize=5,
            markersize=7,
        )
        sig_str = f"OR = {val:.2f} (p={pval:.3f})"
        ax2.text(max(val, min(hi, 12.0)) + 0.25, yp, sig_str, va="center", ha="left", fontsize=13, fontweight="bold")

    ax2.axvline(1.0, color="#222222", linestyle="--", linewidth=1.8, label="No Difference (OR = 1.0)")
    ax2.set_yticks(y2)
    ax2.set_yticklabels(or_labels, fontsize=13.5, fontweight="bold")
    ax2.set_xlim(0.0, 15.5)
    ax2.set_xlabel("Odds Ratio (OR) for Having >=1 Loop in a Run", fontsize=14, fontweight="bold")
    ax2.set_title("B. Why More Open Runs Flag a Loop: Longer Trajectories", fontsize=16, fontweight="bold", pad=12)
    ax2.tick_params(axis="x", labelsize=12.5)
    ax2.grid(True, axis="x", linestyle=":", alpha=0.5)

    note_txt = (
        f"SWE-rebench Open models take {float(r_len['open_mean_steps']):.1f} steps/run vs "
        f"{float(r_len['closed_mean_steps']):.1f} for Closed (+{(float(r_len['open_mean_steps'])/float(r_len['closed_mean_steps'])-1)*100:.0f}% longer).\n"
        "Controlling for log(steps) collapses SWE-rebench Any-Loop OR from 8.43 (p<0.001) to 1.34 (p=0.58, non-significant),\n"
        "and Pooled Any-Loop OR from 2.25 (p=0.024) to 1.47 (p=0.25, non-significant)."
    )
    ax2.text(
        0.02,
        0.04,
        note_txt,
        transform=ax2.transAxes,
        fontsize=11.5,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", edgecolor="#2b8cbe", linewidth=1.5),
    )

    fig.suptitle(
        "SWE-rebench Open vs Closed Loop Gap: Raw vs Skill, Run-Length & Labeler Controls",
        fontsize=20,
        fontweight="bold",
        y=0.97,
    )
    fig.subplots_adjust(left=0.21, right=0.97, top=0.86, bottom=0.11, wspace=0.42)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_cohort_verdicts(
    verdicts_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """1600x900 px forest plot of matched gap per cohort with 1-stage & 2-stage CIs, colored by verdict rule."""
    if verdicts_df.empty:
        return

    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    df = verdicts_df.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(df))

    verdict_colors = {
        "higher for open": "#d95f02",
        "lower for open": "#1f78b4",
        "mixed": "#e6ab02",
        "null": "#666666",
    }

    ax.axvline(0.0, color="#111111", linestyle="--", linewidth=1.8, zorder=2)

    for idx, row in df.iterrows():
        v_one = str(row["verdict_task_boot"])
        col = verdict_colors.get(v_one, "#666666")
        pt = float(row["matched_diff_lp100"])
        if np.isnan(pt):
            pt = float(row["unmatched_diff_lp100"])
            lo1, hi1 = pt, pt
            lo2, hi2 = pt, pt
            is_unmatched_fallback = True
        else:
            lo1 = float(row["task_boot_matched_ci_low"])
            hi1 = float(row["task_boot_matched_ci_high"])
            lo2 = float(row["twostage_boot_matched_ci_low"])
            hi2 = float(row["twostage_boot_matched_ci_high"])
            is_unmatched_fallback = False

        # Outer 2-stage (model + task) CI bar
        if not np.isnan(lo2) and not np.isnan(hi2) and hi2 > lo2:
            ax.plot([lo2, hi2], [idx, idx], color=col, linewidth=3.0, alpha=0.45, zorder=3)
            ax.plot([lo2, lo2], [idx - 0.14, idx + 0.14], color=col, linewidth=2.2, alpha=0.55, zorder=3)
            ax.plot([hi2, hi2], [idx - 0.14, idx + 0.14], color=col, linewidth=2.2, alpha=0.55, zorder=3)

        # Inner 1-stage (task-only) CI bar + point marker
        ax.errorbar(
            pt,
            idx,
            xerr=[[max(0.0, pt - lo1)], [max(0.0, hi1 - pt)]],
            fmt="D" if is_unmatched_fallback else "o",
            color=col,
            ecolor=col,
            elinewidth=5.5,
            capsize=7,
            markersize=11,
            markeredgecolor="#111111",
            markeredgewidth=1.2,
            zorder=5,
        )

        # Right-hand callout annotation
        v_two = str(row["verdict_twostage_boot"])
        nb_reliable = str(row["irr_status"]) == "reliable"
        if nb_reliable and not np.isnan(float(row["nb_adj_irr"])):
            irr_txt = f"NB IRR={float(row['nb_adj_irr']):.2f} (p={float(row['nb_adj_p_2way']):.3f})"
        else:
            irr_txt = f"Raw Ratio={float(row['raw_rate_ratio']):.2f}x (NB collinear)"

        ci_txt = f"[{lo1:+.2f}, {hi1:+.2f}]" if not is_unmatched_fallback else "(no band overlap)"
        badge = f"{pt:+.2f} {ci_txt}  |  {irr_txt}  ->  {v_one.upper()}"
        if v_two != v_one:
            badge += f" (2-stage: {v_two.upper()})"
        ax.text(
            8.2,
            idx,
            badge,
            va="center",
            ha="left",
            fontsize=12,
            fontweight="bold",
            color=col,
        )

    short_names = {
        "tb_terminus_2": "Terminal-Bench 2.0\n(terminus-2, 16O vs 16C)",
        "swe_rebench_swelike": "SWE-rebench Jul 2026\n(swelike, 7O vs 6C)",
        "longtail_swe_swelike": "Longtail-SWE\n(swelike, 5O vs 4C)*",
        "tau2_banking_knowledge_alltools": "Sierra tau2 Banking\n(AllTools, 3O vs 17C)",
        "tau2_classic": "Sierra tau2 Classic\n(Air/Ret/Tel, 2O vs 6C)",
        "tb_openhands": "Terminal-Bench 2.0\n(openhands, 5O vs 7C)",
        "tb_mini_swe_agent": "Terminal-Bench 2.0\n(mini-swe-agent, 5O vs 8C)",
    }
    ax.set_yticks(y_pos)
    ax.set_yticklabels([short_names.get(str(r["cohort"]), str(r["cohort_label"])) for _, r in df.iterrows()], fontsize=13, fontweight="bold")
    ax.set_xlim(-3.0, 20.5)
    ax.set_xlabel("Matched Open minus Closed Gap (Loops per 100 Executed Steps)", fontsize=14.5, fontweight="bold")
    ax.tick_params(axis="x", labelsize=13)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="#d95f02", lw=5, markersize=10, label="Higher for Open (CI > 0 & NB IRR > 1, p<0.05)"),
        Line2D([0], [0], marker="o", color="#e6ab02", lw=5, markersize=10, label="Mixed (Matched CI != 0, but NB IRR p>=0.05 or collinear)"),
        Line2D([0], [0], marker="o", color="#666666", lw=5, markersize=10, label="Null (Matched CI includes 0 & NB IRR p>=0.05)"),
        Line2D([0], [0], color="#333333", lw=2.5, alpha=0.45, label="Thin Outer Cap = 2-Stage (Model + Task) Bootstrap 95% CI"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=12, frameon=True, facecolor="#f8f9fa")
    ax.text(
        0.01,
        0.02,
        "*Longtail-SWE has 0 overlapping pass-rate bands (diamond shows unmatched gap); Verdict Rule requires both matched CI > 0 and reliable adjusted NB IRR > 1 (p<0.05).",
        transform=ax.transAxes,
        fontsize=11.5,
        style="italic",
        color="#444444",
    )

    fig.suptitle(
        "Pre-Specified Cohort Verdicts: Matched Loop Gap, Two-Stage CIs & Adjusted NB IRR",
        fontsize=19,
        fontweight="bold",
        y=0.96,
    )
    fig.subplots_adjust(left=0.22, right=0.98, top=0.88, bottom=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_detectors_vs_random_16x9(
    tb_det_df: pd.DataFrame,
    swe_det_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """1600x900 px figure: production loop detectors vs step-distribution-matched random stoppers with 95% CIs."""
    if tb_det_df.empty and swe_det_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 9), dpi=100, sharey=True)
    datasets = [
        (tb_det_df, "A. Terminal-Bench 2.0 (34,462 Runs / 5,011 Priced)"),
        (swe_det_df, "B. SWE-rebench Jul 2026 (9,435 Runs / 6,381 Priced)"),
    ]
    rules_order = [
        ("gemini_cli", "Gemini-CLI\n(Tool/Chant Guard)"),
        ("openclaw_10", "OpenClaw-10\n(Rolling 30-Step Window)"),
        ("cline_style", "Cline-Style\n(5 Identical Actions)"),
        ("openhands_stuck", "OpenHands Stuck\n(4x Action+Obs / 3x ABA)"),
        ("openclaw_20", "OpenClaw-20\n(Strict Rolling Window)"),
    ]

    y_base = np.arange(len(rules_order))[::-1]
    height = 0.34

    for ax_i, (df_sub, ds_title) in enumerate(datasets):
        ax = axes[ax_i]
        if df_sub.empty:
            continue
        sub = df_sub.set_index("name")

        for idx_r, (rule_key, r_label) in enumerate(rules_order):
            yp = y_base[idx_r]
            if rule_key not in sub.index:
                continue
            row = sub.loc[rule_key]
            p_dol = float(row["dollars_saved_pct"]) * 100.0
            p_fsr = float(row["false_stop_rate"]) * 100.0
            p_loss = float(row["success_loss_pp"])

            r_dol = float(row["rand_dollars_saved_pct_mean"]) * 100.0
            r_dol_lo = float(row["rand_dollars_saved_pct_ci_low"]) * 100.0
            r_dol_hi = float(row["rand_dollars_saved_pct_ci_high"]) * 100.0
            r_fsr = float(row["rand_false_stop_mean"]) * 100.0
            r_fsr_lo = float(row["rand_false_stop_ci_low"]) * 100.0
            r_fsr_hi = float(row["rand_false_stop_ci_high"]) * 100.0

            # Plot False Stop Rate (% of stopped runs that would have succeeded) with 95% CIs on random baseline
            ax.barh(
                yp + height / 2,
                p_fsr,
                height=height,
                color="#2b8cbe",
                edgecolor="#111111",
                linewidth=1.2,
                label="Production Guard (False-Stop Rate %)" if idx_r == 0 else None,
            )
            ax.text(
                p_fsr + 0.8,
                yp + height / 2,
                f"Guard FSR: {p_fsr:.1f}% (Pass lost: {p_loss:.2f}pp | $ saved: {p_dol:.2f}%)",
                va="center",
                ha="left",
                fontsize=11.5,
                fontweight="bold",
                color="#0b4f6c",
            )

            ax.barh(
                yp - height / 2,
                r_fsr,
                height=height,
                color="#de2d26",
                edgecolor="#111111",
                linewidth=1.2,
                alpha=0.85,
                label="Matched Random Stopper (200-Seed 95% CI)" if idx_r == 0 else None,
            )
            ax.errorbar(
                r_fsr,
                yp - height / 2,
                xerr=[[max(0.0, r_fsr - r_fsr_lo)], [max(0.0, r_fsr_hi - r_fsr)]],
                fmt="none",
                ecolor="#111111",
                elinewidth=2.5,
                capsize=5,
            )
            ax.text(
                max(r_fsr, r_fsr_hi) + 0.8,
                yp - height / 2,
                f"Random FSR: {r_fsr:.1f}% [{r_fsr_lo:.1f}%, {r_fsr_hi:.1f}%] ($ saved: {r_dol:.2f}% [{r_dol_lo:.2f}%, {r_dol_hi:.2f}%])",
                va="center",
                ha="left",
                fontsize=11.5,
                fontweight="bold",
                color="#a50f15",
            )

        ax.set_yticks(y_base)
        ax.set_yticklabels([r[1] for r in rules_order], fontsize=13.5, fontweight="bold")
        ax.set_xlabel("False-Stop Rate (% of Triggered Stops That Killed a Passing Run)", fontsize=13.5, fontweight="bold")
        ax.set_title(ds_title, fontsize=16, fontweight="bold", pad=12)
        ax.set_xlim(0, 68)
        ax.tick_params(axis="x", labelsize=12.5)
        ax.grid(True, axis="x", linestyle=":", alpha=0.5)

    axes[0].legend(loc="lower right", fontsize=12, frameon=True, facecolor="#f8f9fa")
    fig.suptitle(
        "Production Loop Detectors vs Step-Matched Random Stoppers (with 200-Seed 95% CIs)",
        fontsize=19.5,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.02,
        "On Terminal-Bench, production guards cut false-stop rates in half (e.g., gemini_cli 10.8% vs 23.2% random) while saving more cache-aware dollars (5.71% vs 4.33% random).",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color="#222222",
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.12, wspace=0.06)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_glm52_vs_opus5(
    model_df: pd.DataFrame,
    strict_by_model_df: pd.DataFrame,
    swe_by_model_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """1600x900 px comparison of SWE-rebench pass rate, cost/run, and loop rate for glm-5.2, opus-5, grok-4.5, gpt-5.6-sol."""
    targets = ["glm-5.2", "opus-5", "grok-4.5", "gpt-5.6-sol"]
    sub = model_df[(model_df["cohort"] == "swe_rebench_swelike") & (model_df["model"].isin(targets))].copy()
    if sub.empty:
        return

    sub["order"] = sub["model"].map({m: i for i, m in enumerate(targets)})
    sub = sub.sort_values("order").reset_index(drop=True)

    strict_sub = strict_by_model_df[
        (strict_by_model_df["cohort"] == "swe_rebench_swelike") & (strict_by_model_df["model"].isin(targets))
    ].set_index("model")

    swe_cost_map: dict[str, float] = {}
    if not swe_by_model_df.empty and "group" in swe_by_model_df.columns:
        for _, r in swe_by_model_df.iterrows():
            c_runs = int(r.get("cost_runs", 0))
            tot_c = float(r.get("total_cost_usd", 0.0))
            if c_runs > 0:
                swe_cost_map[str(r["group"])] = tot_c / c_runs

    fig, axes = plt.subplots(1, 3, figsize=(16, 9), dpi=100)
    x = np.arange(len(sub))
    display_labels = [
        f"{r['model']}\n({'Open' if r['weights']=='Open' else 'Closed'})"
        for _, r in sub.iterrows()
    ]
    bar_colors = [_OPEN_COLOR if r["weights"] == "Open" else _CLOSED_COLOR for _, r in sub.iterrows()]

    # Panel 1: Executed Pass Rate (%)
    ax0 = axes[0]
    pr_vals = sub["executed_pass_rate"].to_numpy() * 100.0
    ax0.bar(x, pr_vals, width=0.55, color=bar_colors, edgecolor="#111111", linewidth=1.3, alpha=0.9)
    for xi, val in zip(x, pr_vals):
        ax0.text(xi, val + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax0.set_xticks(x)
    ax0.set_xticklabels(display_labels, fontsize=13, fontweight="bold")
    ax0.set_ylabel("Executed Pass Rate (%)", fontsize=14, fontweight="bold")
    ax0.set_title("1. Task Pass Rate (%)", fontsize=16, fontweight="bold", pad=12)
    ax0.set_ylim(0, 88)
    ax0.tick_params(axis="y", labelsize=12.5)
    ax0.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Panel 2: Rebuilt Cache-Aware Cost per Run ($ USD)
    ax1 = axes[1]
    cost_vals = np.array([swe_cost_map.get(str(m), 0.0) for m in sub["model"]], dtype=np.float64)
    ax1.bar(x, cost_vals, width=0.55, color=bar_colors, edgecolor="#111111", linewidth=1.3, alpha=0.9)
    for xi, (_, r) in enumerate(sub.iterrows()):
        val = float(cost_vals[xi])
        ld_sh = float(r["loop_dollar_share"]) * 100.0 if not np.isnan(float(r["loop_dollar_share"])) else 0.0
        ax1.text(xi, val + 0.06, f"${val:.2f}/run\n({ld_sh:.1f}% $ in loops)", ha="center", va="bottom", fontsize=12.5, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_labels, fontsize=13, fontweight="bold")
    ax1.set_ylabel("Mean Cache-Aware Cost per Run (USD)", fontsize=14, fontweight="bold")
    ax1.set_title("2. Cost per Run & Loop $ Share", fontsize=16, fontweight="bold", pad=12)
    ax1.set_ylim(0, max(1.0, float(np.max(cost_vals)) * 1.28))
    ax1.tick_params(axis="y", labelsize=12.5)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.5)

    # Panel 3: Loops per 100 Steps (Default vs Strict Labeler)
    ax2 = axes[2]
    w = 0.34
    def_vals = sub["loops_per_100_steps"].to_numpy()
    def_lo = sub["loops_per_100_steps_ci_low"].to_numpy()
    def_hi = sub["loops_per_100_steps_ci_high"].to_numpy()
    str_vals = np.array([
        float(strict_sub.loc[m, "strict_loops_per_100_steps"]) if m in strict_sub.index else 0.0
        for m in sub["model"]
    ])

    ax2.bar(x - w / 2, def_vals, width=w, color=bar_colors, edgecolor="#111111", linewidth=1.3, alpha=0.92, label="Default Labeler")
    ax2.errorbar(
        x - w / 2,
        def_vals,
        yerr=[np.maximum(0.0, def_vals - def_lo), np.maximum(0.0, def_hi - def_vals)],
        fmt="none",
        ecolor="#111111",
        elinewidth=2.0,
        capsize=4,
    )
    ax2.bar(x + w / 2, str_vals, width=w, color="#756bb1", edgecolor="#111111", linewidth=1.3, alpha=0.88, label="Strict Labeler (>=3x)")

    for xi, dv, dhi, sv in zip(x, def_vals, def_hi, str_vals):
        ax2.text(xi - w / 2, dhi + 0.05, f"{dv:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax2.text(xi + w / 2, sv + 0.05, f"{sv:.2f}", ha="center", va="bottom", fontsize=12, fontweight="bold", color="#54278f")

    ax2.set_xticks(x)
    ax2.set_xticklabels(display_labels, fontsize=13, fontweight="bold")
    ax2.set_ylabel("Loops per 100 Executed Steps", fontsize=14, fontweight="bold")
    ax2.set_title("3. Loop Density (Default vs Strict)", fontsize=16, fontweight="bold", pad=12)
    ax2.set_ylim(0, max(1.0, float(np.max(def_hi)) * 1.28))
    ax2.tick_params(axis="y", labelsize=12.5)
    ax2.legend(loc="upper right", fontsize=12, frameon=True)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        "Frontier Showdown on SWE-rebench (swelike): GLM-5.2 vs Claude Opus 5, Grok 4.5 & GPT-5.6-sol",
        fontsize=19,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.02,
        "GLM-5.2 (Open) matches frontier pass rates on identical swelike tasks at a fraction of Opus-5 and Grok-4.5 per-run cost.",
        ha="center",
        fontsize=13,
        fontweight="bold",
        color="#222222",
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.14, wspace=0.24)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def plot_harness_dumbbell(
    harness_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """1600x900 px dumbbell chart: one row per model, one dot per harness, sorted by within-model spread."""
    if harness_df.empty:
        return

    model_order = (
        harness_df.groupby(["dataset", "canonical_model", "weights"], as_index=False)["spread_loops_per_100_steps"]
        .first()
        .sort_values(["spread_loops_per_100_steps", "canonical_model"], ascending=[True, True])
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    y_pos = np.arange(len(model_order))

    scaffold_palette = {
        "terminus-2": ("#d95f02", "o", "terminus-2 (Bare Terminal Harness)"),
        "claude-code": ("#1f78b4", "s", "claude-code (Anthropic Agent Harness)"),
        "codex": ("#2ca02c", "^", "codex (OpenAI Agent Harness)"),
        "gemini-cli": ("#17becf", "D", "gemini-cli (Google CLI Harness)"),
        "openhands": ("#9467bd", "P", "openhands (OpenHands Scaffold)"),
        "mini-swe-agent": ("#e377c2", "v", "mini-swe-agent (Minimal SWE Harness)"),
        "terminus-3-3": ("#8c564b", "X", "terminus-3-3 (Terminal Harness)"),
        "deepagent-harbor": ("#7f7f7f", "<", "deepagent-harbor (Harbor Agent)"),
        "junie": ("#bcbd22", ">", "junie (JetBrains Junie)"),
        "swelike": ("#ff7f0e", "h", "swelike (SWE-rebench Common Scaffold)"),
    }

    seen_scaffolds: set[str] = set()

    for yi, mrow in model_order.iterrows():
        ds = str(mrow["dataset"])
        cm = str(mrow["canonical_model"])
        w = str(mrow["weights"])
        spread = float(mrow["spread_loops_per_100_steps"])
        sub = harness_df[(harness_df["dataset"] == ds) & (harness_df["canonical_model"] == cm)].sort_values("loops_per_100_steps")

        vals = sub["loops_per_100_steps"].to_numpy()
        lo_v, hi_v = float(np.min(vals)), float(np.max(vals))
        line_col = _OPEN_COLOR if w == "Open" else _CLOSED_COLOR
        ax.plot([lo_v, hi_v], [yi, yi], color=line_col, linewidth=3.2, alpha=0.55, zorder=2)

        for _, srow in sub.iterrows():
            scaf = str(srow["harness"])
            lp = float(srow["loops_per_100_steps"])
            col, marker, leg_lbl = scaffold_palette.get(scaf, ("#333333", "o", scaf))
            seen_scaffolds.add(scaf)
            ax.scatter(
                lp,
                yi,
                s=135,
                color=col,
                marker=marker,
                edgecolors="#111111",
                linewidth=1.2,
                zorder=4,
            )

        ax.text(
            hi_v + 0.35,
            yi,
            f"Spread: {spread:.2f}/100 ({lo_v:.2f} -> {hi_v:.2f})",
            va="center",
            ha="left",
            fontsize=11.5,
            fontweight="bold",
            color="#222222",
        )

    y_labels = [
        f"{r['canonical_model']} ({r['weights'][0]}, {'TB' if r['dataset']=='terminalbench' else 'SWE'})"
        for _, r in model_order.iterrows()
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=12, fontweight="bold")
    ax.set_xlabel("Loops per 100 Executed Steps (Shared Tasks Within Each Model)", fontsize=14, fontweight="bold")
    ax.set_xlim(-0.5, float(harness_df["loops_per_100_steps"].max()) + 4.8)
    ax.tick_params(axis="x", labelsize=12.5)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)

    from matplotlib.lines import Line2D
    handles = []
    for scaf_key, (col, marker, leg_lbl) in scaffold_palette.items():
        if scaf_key in seen_scaffolds:
            handles.append(
                Line2D([0], [0], marker=marker, color="w", markerfacecolor=col, markeredgecolor="#111111", markersize=10, label=leg_lbl)
            )
    ax.legend(handles=handles, loc="lower right", fontsize=11.5, frameon=True, facecolor="#f8f9fa", title="Agent Harness / Scaffold", title_fontsize=12.5)

    tb_subs = model_order[model_order["dataset"] == "terminalbench"]
    med_spread = float(tb_subs["spread_loops_per_100_steps"].median()) if not tb_subs.empty else 0.0
    max_spread = float(tb_subs["spread_loops_per_100_steps"].max()) if not tb_subs.empty else 0.0
    callout = (
        f"Same Weights, Different Harness (Terminal-Bench shared tasks):\n"
        f"• Within-model harness spread median = {med_spread:.2f} loops/100 steps (max = {max_spread:.2f})\n"
        f"• Comparable in scale to the capability-matched Open-vs-Closed gap (+4.20 on TB, +0.97 on SWE-rebench)"
    )
    ax.text(
        0.36,
        0.16,
        callout,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#fff8e7", edgecolor="#d95f02", linewidth=1.6),
    )

    fig.suptitle(
        "Same Model, Different Harness: Loop Rate Spread Across Agent Scaffolds (Sorted by Spread)",
        fontsize=19,
        fontweight="bold",
        y=0.96,
    )
    fig.subplots_adjust(left=0.22, right=0.98, top=0.90, bottom=0.10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

