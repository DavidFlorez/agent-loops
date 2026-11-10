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
