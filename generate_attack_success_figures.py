from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ATTACK_ROOT = ROOT / "llm_multiturn_attacks-main" / "llm_multiturn_attacks-main"
EXAMPLES_DIR = ATTACK_ROOT / "examples"
OUTPUT_DIR = ROOT / "figures"
SUMMARY_DIR = ROOT / "analysis_outputs"


def load_attack_results() -> pd.DataFrame:
    rows = []
    for csv_path in EXAMPLES_DIR.rglob("*.csv"):
        df = pd.read_csv(csv_path)
        name_parts = csv_path.stem.split("_")
        turns = None
        for index, part in enumerate(name_parts):
            if part == "N" and index + 1 < len(name_parts):
                turns = int(name_parts[index + 1])
                break
        tense = name_parts[-1]
        success_rate = pd.to_numeric(df["Human"], errors="coerce").mean()
        rows.append(
            {
                "model": csv_path.parent.name,
                "file": csv_path.name,
                "turns": turns,
                "tense": tense,
                "n_rows": len(df),
                "success_rate": success_rate,
            }
        )

    results = pd.DataFrame(rows).sort_values(["model", "tense", "turns"])
    results["success_pct"] = results["success_rate"] * 100.0
    return results


def save_summary_tables(results: pd.DataFrame) -> None:
    SUMMARY_DIR.mkdir(exist_ok=True)

    overall = (
        results.groupby(["turns", "tense"], as_index=False)["success_rate"]
        .mean()
        .rename(columns={"success_rate": "mean_success_rate"})
    )
    overall["mean_success_pct"] = overall["mean_success_rate"] * 100.0
    overall.to_csv(SUMMARY_DIR / "attack_success_overall.csv", index=False)

    wide = results.pivot_table(
        index=["model", "tense"], columns="turns", values="success_rate"
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(
        columns={1: "single_turn_success", 2: "two_turn_success", 3: "three_turn_success"}
    )
    wide["hidden_risk_gap"] = wide["three_turn_success"] - wide["single_turn_success"]
    wide.to_csv(SUMMARY_DIR / "attack_success_by_model.csv", index=False)


def plot_success_by_turns(results: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    colors = {
        "gemini-2.0-flash": "#1f77b4",
        "gpt-4o-mini": "#ff7f0e",
        "llama-2-7b-chat": "#2ca02c",
        "qwen2-7b-instruct": "#d62728",
    }

    for ax, tense in zip(axes, ["present", "past"]):
        subset = results[results["tense"] == tense]
        for model in subset["model"].unique():
            model_data = subset[subset["model"] == model].sort_values("turns")
            ax.plot(
                model_data["turns"],
                model_data["success_pct"],
                marker="o",
                linewidth=2,
                label=model,
                color=colors.get(model),
            )
        ax.set_title(f"{tense.title()} phrasing")
        ax.set_xlabel("Attack length (turns)")
        ax.set_xticks([1, 2, 3])
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Jailbreak success rate (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Multi-turn attacks raise jailbreak success even when single-turn success is low")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "attack_success_by_turns.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_hidden_risk_gap(results: pd.DataFrame) -> None:
    wide = results.pivot_table(
        index=["model", "tense"], columns="turns", values="success_pct"
    ).reset_index()
    wide.columns.name = None
    wide["hidden_risk_gap"] = wide[3] - wide[1]
    wide = wide.sort_values(["tense", "hidden_risk_gap"], ascending=[True, False])

    labels = [f"{row.model}\n{row.tense}" for row in wide.itertuples()]
    colors = ["#b22222" if tense == "past" else "#4169e1" for tense in wide["tense"]]

    fig, ax = plt.subplots(figsize=(11.5, 5))
    ax.bar(labels, wide["hidden_risk_gap"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Three-turn minus single-turn success (percentage points)")
    ax.set_title("Hidden conversational risk is large for several model-condition pairs")
    ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hidden_risk_gap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_single_to_multi_correlation(results: pd.DataFrame) -> None:
    wide = results.pivot_table(
        index=["model", "tense"], columns="turns", values="success_rate"
    ).reset_index()
    wide.columns.name = None
    correlation = wide[1].corr(wide[3])

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    palette = {"present": "#4169e1", "past": "#b22222"}
    for row in wide.itertuples():
        single_turn = getattr(row, "_3") if hasattr(row, "_3") else getattr(row, "1")
        three_turn = getattr(row, "_5") if hasattr(row, "_5") else getattr(row, "3")
        ax.scatter(single_turn * 100, three_turn * 100, s=90, color=palette[row.tense], alpha=0.85)
        ax.text(
            single_turn * 100 + 0.8,
            three_turn * 100 + 0.8,
            f"{row.model}\n{row.tense}",
            fontsize=8,
        )

    x = wide[1].astype(float)
    y = wide[3].astype(float)
    coeffs = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ys = coeffs[0] * xs + coeffs[1]
    ax.plot(xs * 100, ys * 100, linestyle="--", color="black", linewidth=1.5)

    ax.set_xlabel("Single-turn success rate (%)")
    ax.set_ylabel("Three-turn success rate (%)")
    ax.set_title(f"Cheap single-turn stress tests track multi-turn failure (r = {correlation:.2f})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "single_to_multi_correlation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results = load_attack_results()
    save_summary_tables(results)
    plot_success_by_turns(results)
    plot_hidden_risk_gap(results)
    plot_single_to_multi_correlation(results)
    print(f"Wrote figures to: {OUTPUT_DIR}")
    print(f"Wrote summaries to: {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
