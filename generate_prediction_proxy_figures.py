from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "llm_multiturn_attacks-main" / "llm_multiturn_attacks-main" / "examples"
OUTPUT_DIR = ROOT / "figures"
SUMMARY_DIR = ROOT / "analysis_outputs"


def load_prompt_level_results() -> pd.DataFrame:
    rows = []
    for csv_path in EXAMPLES_DIR.rglob("*.csv"):
        df = pd.read_csv(csv_path)
        parts = csv_path.stem.split("_")
        turns = None
        for idx, part in enumerate(parts):
            if part == "N" and idx + 1 < len(parts):
                turns = int(parts[idx + 1])
                break
        tense = parts[-1]
        model = csv_path.parent.name
        final_df = df if turns == 1 else df[df["Multi Step"] == turns]
        tmp = final_df[["Unique ID", "Human"]].copy()
        tmp["Human"] = pd.to_numeric(tmp["Human"], errors="coerce").fillna(0).astype(int)
        tmp["model"] = model
        tmp["tense"] = tense
        tmp["turns"] = turns
        rows.append(tmp)

    all_df = pd.concat(rows, ignore_index=True)
    wide = all_df.pivot_table(
        index=["model", "Unique ID"], columns=["tense", "turns"], values="Human", aggfunc="first"
    ).reset_index()
    wide.columns = ["model", "Unique ID", "past_1", "past_2", "past_3", "present_1", "present_2", "present_3"]
    wide["any_single_turn_warning"] = ((wide["present_1"] == 1) | (wide["past_1"] == 1)).astype(int)
    wide["any_three_turn_jailbreak"] = ((wide["present_3"] == 1) | (wide["past_3"] == 1)).astype(int)
    wide["hidden_present_failure"] = ((wide["present_1"] == 0) & (wide["present_3"] == 1)).astype(int)
    return wide


def confusion_metrics(df: pd.DataFrame, predictor: str, outcome: str, label: str) -> dict:
    tp = int(((df[predictor] == 1) & (df[outcome] == 1)).sum())
    fp = int(((df[predictor] == 1) & (df[outcome] == 0)).sum())
    fn = int(((df[predictor] == 0) & (df[outcome] == 1)).sum())
    tn = int(((df[predictor] == 0) & (df[outcome] == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "predictor": label,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


def save_prediction_summaries(wide: pd.DataFrame) -> pd.DataFrame:
    SUMMARY_DIR.mkdir(exist_ok=True)

    hidden_by_model = (
        wide.groupby("model", as_index=False)["hidden_present_failure"]
        .mean()
        .rename(columns={"hidden_present_failure": "hidden_failure_rate"})
    )
    hidden_by_model["hidden_failure_pct"] = hidden_by_model["hidden_failure_rate"] * 100.0
    hidden_by_model.to_csv(SUMMARY_DIR / "hidden_failure_by_model.csv", index=False)

    metrics = pd.DataFrame(
        [
            confusion_metrics(
                wide, "present_1", "any_three_turn_jailbreak", "Direct single-turn"
            ),
            confusion_metrics(
                wide, "past_1", "any_three_turn_jailbreak", "Reframed single-turn"
            ),
            confusion_metrics(
                wide,
                "any_single_turn_warning",
                "any_three_turn_jailbreak",
                "Either single-turn warning",
            ),
        ]
    )
    metrics.to_csv(SUMMARY_DIR / "predictor_metrics.csv", index=False)

    conditional = pd.DataFrame(
        [
            {
                "condition": "Direct single-turn safe",
                "three_turn_present_jailbreak_rate": wide.loc[wide["present_1"] == 0, "present_3"].mean(),
            },
            {
                "condition": "Direct single-turn jailbreak",
                "three_turn_present_jailbreak_rate": wide.loc[wide["present_1"] == 1, "present_3"].mean(),
            },
            {
                "condition": "Reframed single-turn safe",
                "three_turn_past_jailbreak_rate": wide.loc[wide["past_1"] == 0, "past_3"].mean(),
            },
            {
                "condition": "Reframed single-turn jailbreak",
                "three_turn_past_jailbreak_rate": wide.loc[wide["past_1"] == 1, "past_3"].mean(),
            },
        ]
    )
    conditional.to_csv(SUMMARY_DIR / "conditional_jailbreak_rates.csv", index=False)
    return metrics


def plot_hidden_failures(wide: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    hidden = (
        wide.groupby("model", as_index=False)["hidden_present_failure"]
        .mean()
        .sort_values("hidden_present_failure", ascending=False)
    )
    hidden["pct"] = hidden["hidden_present_failure"] * 100.0

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(hidden["model"], hidden["pct"], color="#b22222")
    ax.axhline(wide["hidden_present_failure"].mean() * 100.0, linestyle="--", color="black", linewidth=1.2)
    ax.text(
        0.02,
        wide["hidden_present_failure"].mean() * 100.0 + 1.2,
        f"Overall hidden-failure rate = {wide['hidden_present_failure'].mean() * 100.0:.1f}%",
        transform=ax.get_yaxis_transform(),
        fontsize=9,
    )
    ax.set_ylabel("Rate of hidden failures (%)")
    ax.set_title("Many prompts look safe in one turn but fail by the third turn")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hidden_present_failures.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_predictor_comparison(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharex=True)
    colors = ["#4169e1", "#ff8c00", "#2e8b57"]

    axes[0].bar(metrics["predictor"], metrics["precision"] * 100.0, color=colors)
    axes[0].set_title("Precision")
    axes[0].set_ylabel("Percentage")
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(metrics["predictor"], metrics["recall"] * 100.0, color=colors)
    axes[1].set_title("Recall")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("A cheap stressed single-turn screen catches more future multi-turn jailbreaks")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "predictor_precision_recall.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_conditional_risk(wide: pd.DataFrame) -> None:
    rows = [
        {
            "comparison": "Direct prompt",
            "condition": "single-turn safe",
            "rate": wide.loc[wide["present_1"] == 0, "present_3"].mean() * 100.0,
        },
        {
            "comparison": "Direct prompt",
            "condition": "single-turn jailbreak",
            "rate": wide.loc[wide["present_1"] == 1, "present_3"].mean() * 100.0,
        },
        {
            "comparison": "Reframed prompt",
            "condition": "single-turn safe",
            "rate": wide.loc[wide["past_1"] == 0, "past_3"].mean() * 100.0,
        },
        {
            "comparison": "Reframed prompt",
            "condition": "single-turn jailbreak",
            "rate": wide.loc[wide["past_1"] == 1, "past_3"].mean() * 100.0,
        },
    ]
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = [0, 1, 3, 4]
    colors = ["#7aa6ff", "#1f4db8", "#ffbf66", "#cc6f00"]
    ax.bar(x, df["rate"], color=colors)
    ax.set_xticks(x, [f"{a}\n{b}" for a, b in zip(df["comparison"], df["condition"])])
    ax.set_ylabel("Three-turn jailbreak rate (%)")
    ax.set_title("Single-turn stress signals future multi-turn risk")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "conditional_multi_turn_risk.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    wide = load_prompt_level_results()
    metrics = save_prediction_summaries(wide)
    plot_hidden_failures(wide)
    plot_predictor_comparison(metrics)
    plot_conditional_risk(wide)
    print(f"Wrote figures to: {OUTPUT_DIR}")
    print(f"Wrote summaries to: {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
