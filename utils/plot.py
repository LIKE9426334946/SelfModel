import csv
from pathlib import Path


import matplotlib.pyplot as plt


def plot_history(output_dir, title="Training Curves"):

    output_dir = Path(output_dir)

    with open(output_dir / "metrics.csv", "r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    epochs = [int(row["epoch"]) for row in rows]

    panels = [
        ("Loss", ["train_loss", "val_loss"]),
        ("Validation IoU / F1", ["val_iou", "val_f1"]),
        ("Validation Precision / Recall", ["val_precision", "val_recall"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (panel_title, columns) in zip(axes, panels):
        for column in columns:
            values = [float(row[column]) for row in rows]
            ax.plot(epochs, values, label=column, linewidth=2, marker="o", markersize=3)
        ax.set_title(panel_title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylim(bottom=0)
    axes[1].set_ylim(0, 1)
    axes[2].set_ylim(0, 1)

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=200)
