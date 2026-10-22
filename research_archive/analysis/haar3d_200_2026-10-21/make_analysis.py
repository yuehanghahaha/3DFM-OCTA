"""Summarize the completed 3D Haar flow matching run against the voxel baseline."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def read_history(name):
    return [json.loads(line) for line in (ROOT / name).read_text(encoding="utf-8").splitlines()]


def validation_plot(histories):
    colors = {"voxel": "#2C5F8A", "haar3d": "#BF5B3C"}
    labels = {"voxel": "Voxel baseline (100 epochs)", "haar3d": "3D Haar (200 epochs)"}
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, layout="constrained")
    for representation, history in histories.items():
        records = [row for row in history if "validation" in row]
        x = [row["epoch"] for row in records]
        color = colors[representation]
        for ax, key in zip(axes, ("mae", "psnr")):
            y = [row["validation"][key] for row in records]
            ax.plot(x, y, color=color, linewidth=1.6, marker="o", markersize=3.5,
                    label=labels[representation])
            best = (min if key == "mae" else max)(records, key=lambda row: row["validation"][key])
            ax.scatter(best["epoch"], best["validation"][key], color=color,
                       s=95, zorder=4, edgecolor="white", linewidth=1.2)
            offset = (-105, -18) if key == "psnr" and best["epoch"] > 180 else (8, 9)
            ax.annotate(f"{best['epoch']}: {best['validation'][key]:.4f}" if key == "mae"
                        else f"{best['epoch']}: {best['validation'][key]:.2f}",
                        (best["epoch"], best["validation"][key]), xytext=offset,
                        textcoords="offset points", fontsize=9, color=color)
        axes[0].legend(loc="upper right", frameon=False)
    for ax in axes:
        ax.axvline(100, color="#929292", linewidth=1, linestyle="--")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Patient-macro MAE ↓")
    axes[1].set_ylabel("Patient-macro PSNR (dB) ↑")
    axes[1].set_xlabel("Epoch")
    fig.suptitle("OCTA-500 3M validation (10 patients, every 5 epochs)", fontsize=14)
    fig.savefig(ROOT / "validation_trajectory.png", dpi=170)
    plt.close(fig)


def paired_test_analysis():
    wavelet_path = ROOT / "wavelet_test_metrics.json"
    if not wavelet_path.exists():
        return None
    wavelet = read_json(wavelet_path.name)
    baseline = read_json("baseline_test_metrics.json")
    wc = {case["patient_id"]: case for case in wavelet["cases"]}
    bc = {case["patient_id"]: case for case in baseline["cases"]}
    assert set(wc) == set(bc) and len(wc) == 50
    assert wavelet["checkpoint_epoch"] == 195 and baseline["checkpoint_epoch"] == 90
    assert wavelet["config"]["manifest_sha256"] == baseline["config"]["manifest_sha256"]
    assert wavelet["config"]["ode_steps"] == baseline["config"]["ode_steps"] == 8
    for key in ("fov", "axis_order", "intensity_scale", "depth", "patch", "base_channels",
                "condition", "overlap"):
        assert wavelet["config"][key] == baseline["config"][key]

    ids = sorted(wc)
    rows = []
    for patient_id in ids:
        w, b = wc[patient_id], bc[patient_id]
        rows.append({"patient_id": patient_id,
                     "mae_baseline": b["mae"], "mae_haar3d": w["mae"],
                     "mae_difference_haar3d_minus_baseline": w["mae"] - b["mae"],
                     "psnr_baseline": b["psnr"], "psnr_haar3d": w["psnr"],
                     "psnr_difference_haar3d_minus_baseline": w["psnr"] - b["psnr"]})
    with (ROOT / "test_patient_comparison.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    rng = np.random.default_rng(2026)
    indices = rng.integers(0, len(rows), size=(20000, len(rows)))
    summary = {"patients": len(rows), "baseline_checkpoint_epoch": 90,
               "haar3d_checkpoint_epoch": 195}
    for metric in ("mae", "psnr"):
        b = np.array([row[f"{metric}_baseline"] for row in rows])
        w = np.array([row[f"{metric}_haar3d"] for row in rows])
        delta = w - b
        summary[metric] = {"baseline": float(b.mean()), "haar3d": float(w.mean()),
                           "difference_haar3d_minus_baseline": float(delta.mean()),
                           "paired_bootstrap_95pct_ci": np.percentile(delta[indices].mean(axis=1),
                                                                       [2.5, 97.5]).tolist(),
                           "haar3d_better_patients": int(np.count_nonzero(delta < 0 if metric == "mae"
                                                                             else delta > 0))}
        assert np.isclose(summary[metric]["baseline"], baseline["metrics"][metric])
        assert np.isclose(summary[metric]["haar3d"], wavelet["metrics"][metric])
    summary["mae"]["relative_difference_pct"] = (summary["mae"]["difference_haar3d_minus_baseline"]
                                                  / summary["mae"]["baseline"] * 100)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for ax, metric in zip(axes, ("mae", "psnr")):
        x = np.array([row[f"{metric}_baseline"] for row in rows])
        y = np.array([row[f"{metric}_haar3d"] for row in rows])
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        pad = (hi - lo) * .08
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#555555", linewidth=1,
                linestyle="--")
        ax.scatter(x, y, color="#BF5B3C", s=29, alpha=.8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"Voxel baseline {metric.upper()}")
        ax.set_ylabel(f"3D Haar {metric.upper()}")
        ax.set_title(f"{metric.upper()}: {summary[metric]['haar3d_better_patients']}/50 Haar wins")
        ax.grid(alpha=.18)
    fig.suptitle("Paired full-volume test results (50 unseen patients)", fontsize=14)
    fig.savefig(ROOT / "test_patient_scatter.png", dpi=170)
    plt.close(fig)
    return summary


def paired_projection_analysis():
    wavelet = read_json("wavelet_projection_metrics.json")
    baseline = read_json("baseline_projection_metrics.json")
    assert wavelet["assets"] == baseline["assets"]
    wc = {case["patient_id"]: case for case in wavelet["cases"]}
    bc = {case["patient_id"]: case for case in baseline["cases"]}
    assert set(wc) == set(bc) and len(wc) == 50
    ids = sorted(wc)
    rng = np.random.default_rng(2026)
    indices = rng.integers(0, len(ids), size=(20000, len(ids)))
    summary = {}
    for slab in ("FULL", "ILM_OPL", "OPL_BM"):
        summary[slab] = {}
        for metric in ("mae", "psnr", "correlation"):
            b = np.array([bc[patient_id]["projections"][slab][metric] for patient_id in ids])
            w = np.array([wc[patient_id]["projections"][slab][metric] for patient_id in ids])
            delta = w - b
            summary[slab][metric] = {
                "baseline": float(b.mean()), "haar3d": float(w.mean()),
                "difference_haar3d_minus_baseline": float(delta.mean()),
                "paired_bootstrap_95pct_ci": np.percentile(delta[indices].mean(axis=1),
                                                            [2.5, 97.5]).tolist(),
                "haar3d_better_patients": int(np.count_nonzero(delta < 0 if metric == "mae"
                                                                  else delta > 0))}
            assert np.isclose(summary[slab][metric]["baseline"],
                              baseline["projection_metrics_patient_macro"][slab][metric])
            assert np.isclose(summary[slab][metric]["haar3d"],
                              wavelet["projection_metrics_patient_macro"][slab][metric])
    return summary


def main():
    histories = {"voxel": read_history("baseline_history.jsonl"),
                 "haar3d": read_history("wavelet_history.jsonl")}
    validation_plot(histories)
    summary = {"validation": {}, "test": paired_test_analysis(),
               "projection": paired_projection_analysis()}
    for representation, history in histories.items():
        records = [row for row in history if "validation" in row]
        best = min(records, key=lambda row: row["validation"]["mae"])
        summary["validation"][representation] = {"epochs": len(history),
                                                   "best_mae_epoch": best["epoch"],
                                                   "best_checkpoint_metrics": best["validation"],
                                                   "final_metrics": records[-1]["validation"]}
    (ROOT / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
