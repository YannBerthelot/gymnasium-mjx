"""The README's learning-curve figure and table, from the JSON lines that
Ajax's benchmarks/mujoco_curves.py writes (one row per seed: task, side
"gymnasium" or "mujoco", steps, returns).

Each panel shows the mean evaluation return over seeds with a 95% band
(normal approximation). The table compares the mean of each seed's last 10
evaluations (the last 100,000 steps): the difference in percent of
Gymnasium's mean, with a 90% interval (two one-sided tests at 5%).

    uv run --with matplotlib python tools/plot_curves.py curves.jsonl docs/
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

SIDES = {"gymnasium": "Gymnasium (C MuJoCo)", "mujoco": "gymnasium-mjx (MJX)"}
# The reference palette's first two categorical slots, stepped per surface.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e1e0d9",
        "gymnasium": "#2a78d6",
        "mujoco": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#2c2c2a",
        "gymnasium": "#3987e5",
        "mujoco": "#d95926",
    },
}
LAST_EVALUATIONS = 10


def read_runs(path: Path) -> dict:
    runs: dict = defaultdict(lambda: defaultdict(list))
    for line in path.read_text().splitlines():
        row = json.loads(line)
        runs[row["task"]][row["side"]].append(row)
    # Only tasks with both sides make a comparison.
    return {task: sides for task, sides in runs.items() if len(sides) == 2}


def figure(runs: dict, theme: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    tasks = sorted(runs)
    plt.rcParams.update({"font.size": 9, "text.color": theme["ink"]})
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(3.0 * len(tasks), 2.4), squeeze=False
    )
    fig.patch.set_facecolor(theme["surface"])
    for axis, task in zip(axes[0], tasks, strict=True):
        axis.set_facecolor(theme["surface"])
        seeds = []
        for side in SIDES:
            rows = runs[task][side]
            steps = np.asarray(rows[0]["steps"]) / 1e6
            returns = np.asarray([row["returns"] for row in rows])
            mean = returns.mean(axis=0)
            band = 1.96 * returns.std(axis=0, ddof=1) / np.sqrt(len(rows))
            color = theme[side]
            axis.fill_between(
                steps, mean - band, mean + band, color=color, alpha=0.2, lw=0
            )
            axis.plot(steps, mean, color=color, lw=2, label=SIDES[side])
            seeds.append(str(len(rows)))
        axis.set_title(
            f"{task}  ({' vs '.join(seeds)} seeds)", fontsize=9, color=theme["ink"]
        )
        axis.set_xlabel("environment steps (millions)", color=theme["muted"])
        axis.grid(axis="y", color=theme["grid"], lw=0.6)
        axis.tick_params(colors=theme["muted"], length=0)
        for side, spine in axis.spines.items():
            spine.set_visible(side == "bottom")
            spine.set_color(theme["grid"])
    axes[0][0].set_ylabel("evaluation return", color=theme["muted"])
    handles, labels = axes[0][0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    for text in legend.get_texts():
        text.set_color(theme["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=150, facecolor=theme["surface"])
    plt.close(fig)


def table(runs: dict) -> str:
    lines = [
        "| Task | Seeds | Gymnasium | gymnasium-mjx | Difference (90% CI) |",
        "|---|---|---|---|---|",
    ]
    for task in sorted(runs):
        final = {
            side: [statistics.mean(row["returns"][-LAST_EVALUATIONS:]) for row in rows]
            for side, rows in runs[task].items()
        }
        reference, ours = final["gymnasium"], final["mujoco"]
        base = statistics.mean(reference)
        difference = statistics.mean(ours) - base
        error = math.sqrt(
            statistics.variance(reference) / len(reference)
            + statistics.variance(ours) / len(ours)
        )
        low, high = (
            100 * (difference + sign * 1.645 * error) / base for sign in (-1, 1)
        )
        lines.append(
            f"| {task} | {len(reference)} / {len(ours)} | {base:,.0f}"
            f" | {statistics.mean(ours):,.0f} | {100 * difference / base:+.1f}%"
            f" ({low:+.1f}% to {high:+.1f}%) |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("curves", type=Path)
    parser.add_argument("out", type=Path)
    arguments = parser.parse_args()
    runs = read_runs(arguments.curves)
    arguments.out.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        figure(runs, theme, arguments.out / f"curves-{name}.png")
    print(table(runs))


if __name__ == "__main__":
    main()
