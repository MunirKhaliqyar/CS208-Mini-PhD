from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CHAT_PATH = ROOT / "ChatAlpaca-main" / "ChatAlpaca-main" / "data" / "chatalpaca-10k.json"
CONTURE_PATH = ROOT / "conture-main" / "conture-main" / "data" / "data.json"
OUTPUT_DIR = ROOT / "figures"
SUMMARY_DIR = ROOT / "analysis_outputs"


def load_chatalpaca() -> list[list[dict]]:
    conversations = []
    for line in CHAT_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        conversations.append(item["conversations"])
    return conversations


def load_conture() -> list[dict]:
    with CONTURE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_dialog_summaries(chat_data: list[list[dict]], conture_data: list[dict]) -> None:
    SUMMARY_DIR.mkdir(exist_ok=True)

    user_turn_counts = [
        sum(1 for turn in conversation if turn["from"] == "human") for conversation in chat_data
    ]
    chat_summary = pd.DataFrame(
        sorted(Counter(user_turn_counts).items()), columns=["user_turns", "conversation_count"]
    )
    chat_summary.to_csv(SUMMARY_DIR / "chatalpaca_turn_distribution.csv", index=False)

    turn_rows = []
    for dialog in conture_data:
        for turn_index, turn in enumerate(dialog["turns"], start=1):
            turn_rows.append(
                {"turn_index": turn_index, "overall_impression": turn["overall impression"]}
            )
    conture_summary = (
        pd.DataFrame(turn_rows)
        .groupby("turn_index", as_index=False)["overall_impression"]
        .agg(mean_score="mean", count="count")
    )
    conture_summary.to_csv(SUMMARY_DIR / "conture_turn_quality.csv", index=False)


def plot_chatalpaca_distribution(chat_data: list[list[dict]]) -> None:
    user_turn_counts = [
        sum(1 for turn in conversation if turn["from"] == "human") for conversation in chat_data
    ]
    distribution = Counter(user_turn_counts)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    xs = sorted(distribution.keys())
    ys = [distribution[x] for x in xs]
    ax.bar(xs, ys, color="#008b8b")
    ax.set_title("ChatAlpaca conversations are usually 4 to 6 user turns long")
    ax.set_xlabel("User turns per conversation")
    ax.set_ylabel("Number of conversations")
    ax.set_xticks(xs)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "chatalpaca_turn_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_conture_quality(conture_data: list[dict]) -> None:
    rows = []
    for dialog in conture_data:
        for turn_index, turn in enumerate(dialog["turns"], start=1):
            rows.append(
                {"turn_index": turn_index, "overall_impression": turn["overall impression"]}
            )

    summary = (
        pd.DataFrame(rows)
        .groupby("turn_index", as_index=False)["overall_impression"]
        .mean()
        .rename(columns={"overall_impression": "mean_score"})
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(summary["turn_index"], summary["mean_score"], marker="o", linewidth=2, color="#8b4513")
    ax.set_title("ConTurE shows turn quality shifts across a conversation")
    ax.set_xlabel("Turn index")
    ax.set_ylabel("Mean human quality score (0 to 2)")
    ax.set_xticks(summary["turn_index"])
    ax.set_ylim(0.8, 1.7)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "conture_turn_quality.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    chat_data = load_chatalpaca()
    conture_data = load_conture()
    save_dialog_summaries(chat_data, conture_data)
    plot_chatalpaca_distribution(chat_data)
    plot_conture_quality(conture_data)
    print(f"Wrote figures to: {OUTPUT_DIR}")
    print(f"Wrote summaries to: {SUMMARY_DIR}")


if __name__ == "__main__":
    main()
