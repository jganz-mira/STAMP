"""Create the compact independent-test AUROC comparison figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


DEFAULT_RESULTS = Path(
    "/home/jonathan/data/CPTAC_STAMP/mil_test_independent/summary_mean_std.csv"
)
DEFAULT_OUTPUT = Path("figures/testset_auroc_barplot")

MODEL_ORDER = ("mira", "ignite", "nextgen", "imagenet", "openmidnight")
MODEL_LABELS = {
    "mira": "MIRA",
    "ignite": "IGNITE",
    "nextgen": "NextGen",
    "imagenet": "ImageNet",
    "openmidnight": "OpenMidnight",
}
MODEL_COLORS = {
    "mira": "#FC8D62",
    "ignite": "#66C2A5",
    "nextgen": "#8DA0CB",
    "imagenet": "#E78AC3",
    "openmidnight": "#A6D854",
}


def read_aurocs(path: Path) -> dict[str, tuple[float, float]]:
    """Read fold-aggregated AUROCs; absent models remain empty placeholders."""
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            row["feature_set"].strip().lower(): (
                float(row["auroc_mean"]),
                float(row["auroc_std"]),
            )
            for row in rows
        }


def create_plot(
    results_path: Path,
    output_stem: Path,
    *,
    dpi: int = 300,
) -> None:
    results = read_aurocs(results_path)

    # Requested physical size: 5.14 cm wide and 3.92 cm high.
    fig, ax = plt.subplots(figsize=(5.14 / 2.54, 3.92 / 2.54))
    x_positions = range(len(MODEL_ORDER))

    for x, model in zip(x_positions, MODEL_ORDER, strict=True):
        if model not in results:
            continue
        mean, std = results[model]
        ax.bar(
            x,
            mean,
            width=0.64,
            color=MODEL_COLORS[model],
            edgecolor="#555555",
            linewidth=0.35,
            yerr=std,
            error_kw={
                "ecolor": "#333333",
                "elinewidth": 0.65,
                "capsize": 2.0,
                "capthick": 0.65,
            },
            zorder=3,
        )

    ax.set_xlim(-0.55, len(MODEL_ORDER) - 0.45)
    ax.set_ylim(0.5, 1.0)
    ax.set_xticks(
        list(x_positions),
        ["MIRA", "IGNITE", "Next-\nGen", "Image-\nNet", "Open-\nMidnight"],
    )
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])

    ax.tick_params(axis="x", labelsize=5.4, length=2.2, width=0.45, pad=1.5)
    ax.tick_params(axis="y", labelsize=5.8, length=2.2, width=0.45, pad=1.5)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(0.5)

    legend_handles = [
        Patch(
            facecolor=MODEL_COLORS[model],
            edgecolor="#555555",
            linewidth=0.35,
            label=MODEL_LABELS[model],
        )
        for model in MODEL_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=3,
        frameon=False,
        fontsize=5.2,
        handlelength=0.75,
        handleheight=0.75,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0,
    )

    # Fixed margins preserve the exact requested canvas dimensions on export.
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.23, top=0.76)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".svg"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    create_plot(args.results, args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
