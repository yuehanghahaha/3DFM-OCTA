"""Plot the saved validation histories without changing model selection."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
WAVELET = ROOT.parent / "haar3d_200_2026-10-21" / "wavelet_history.jsonl"


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    histories = {
        "3D voxel baseline": (records(ROOT / "history_200.jsonl"), "#2563a6"),
        "3D Haar wavelet": (records(WAVELET), "#bb5a3c"),
    }
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    for label, (history, color) in histories.items():
        rows = [row for row in history if "validation" in row]
        for ax, key in zip(axes, ("mae", "psnr")):
            xs = [row["epoch"] for row in rows]
            ys = [row["validation"][key] for row in rows]
            ax.plot(xs, ys, color=color, lw=1.25, alpha=0.75, label=label)
            best = (min if key == "mae" else max)(rows, key=lambda row: row["validation"][key])
            ax.scatter([best["epoch"]], [best["validation"][key]], color=color,
                       s=55, zorder=5, edgecolors="white")
            ax.annotate(f"{label.split()[1]} e{best['epoch']}: {best['validation'][key]:.4f}"
                        if key == "mae" else
                        f"{label.split()[1]} e{best['epoch']}: {best['validation'][key]:.2f}",
                        (best["epoch"], best["validation"][key]),
                        xytext=(7, 7), textcoords="offset points", fontsize=8, color=color)
    for ax in axes:
        ax.axvline(100, color="#666", linestyle="--", lw=1)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Validation MAE (lower is better)")
    axes[1].set_ylabel("Validation PSNR, dB (higher is better)")
    axes[1].set_xlabel("Epoch; baseline validated every epoch after 100, Haar every 5 epochs")
    fig.suptitle("OCTA-500 3M: same 10 validation patients", fontsize=13)
    fig.savefig(ROOT / "validation_200_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
