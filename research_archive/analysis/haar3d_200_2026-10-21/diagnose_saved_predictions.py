"""Read-only intensity audit of the same 50 saved baseline and Haar test cases."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_volume(path):
    with np.load(path, allow_pickle=False) as data:
        if str(data["axis_order"].item()) != "ZYX":
            raise ValueError(f"Unexpected axis order in {path}")
        return data["octa"], data["octa_gt"]


def load_projection(path):
    with np.load(path, allow_pickle=False) as data:
        return data["prediction_ILM_OPL"], data["reference_z256_ILM_OPL"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--wavelet", type=Path, required=True)
    parser.add_argument("--baseline-projections", type=Path, required=True)
    parser.add_argument("--wavelet-projections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    b_report = json.loads((args.baseline / "metrics.json").read_text())
    w_report = json.loads((args.wavelet / "metrics.json").read_text())
    ids = sorted(str(row["case_id"]) for row in b_report["cases"])
    if len(ids) != 50 or ids != sorted(str(row["case_id"]) for row in w_report["cases"]):
        raise ValueError("Expected identical 50 test case IDs")
    rows = []
    for i, case_id in enumerate(ids, 1):
        b, gt = load_volume(args.baseline / f"{case_id}.npz")
        w, gt_w = load_volume(args.wavelet / f"{case_id}.npz")
        if not np.array_equal(gt, gt_w) or b.shape != gt.shape or w.shape != gt.shape:
            raise ValueError(f"Volume/reference mismatch for {case_id}")
        bp, gp = load_projection(args.baseline_projections / f"{case_id}_projections.npz")
        wp, gp_w = load_projection(args.wavelet_projections / f"{case_id}_projections.npz")
        if not np.array_equal(gp, gp_w) or bp.shape != gp.shape or wp.shape != gp.shape:
            raise ValueError(f"Projection/reference mismatch for {case_id}")
        bright = gt > 0.1  # Intensity diagnostic, not a vessel segmentation.
        gt_mean = float(gt.mean(dtype=np.float64))
        gt_std = float(gt.std(dtype=np.float64))
        row = {"case_id": case_id, "reference_mean": gt_mean,
               "reference_std": gt_std, "reference_above_0p1_fraction": float(bright.mean()),
               "zero_mae": gt_mean,
               "zero_psnr": -10 * math.log10(max(float(np.square(gt, dtype=np.float64).mean()), 1e-12)),
               "reference_ilm_opl_p99": float(np.percentile(gp, 99))}
        for name, pred, proj in (("baseline", b, bp), ("wavelet", w, wp)):
            error = pred - gt
            mae = float(np.abs(error).mean(dtype=np.float64))
            mse = float(np.square(error, dtype=np.float64).mean())
            row[f"{name}_mae"] = mae
            row[f"{name}_psnr"] = -10 * math.log10(max(mse, 1e-12))
            row[f"{name}_mean"] = float(pred.mean(dtype=np.float64))
            row[f"{name}_std_ratio"] = float(pred.std(dtype=np.float64) / gt_std)
            row[f"{name}_above_0p1_fraction"] = float((pred > 0.1).mean())
            row[f"{name}_mean_on_reference_above_0p1"] = float(pred[bright].mean())
            row[f"{name}_ilm_opl_p99"] = float(np.percentile(proj, 99))
            row[f"{name}_ilm_opl_std_ratio"] = float(proj.std(dtype=np.float64) / gp.std(dtype=np.float64))
        rows.append(row)
        print(f"Audited {i}/50: {case_id}", flush=True)
    means = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key != "case_id"}
    result = {"protocol": "50 fixed test cases; exact shared Z256 reference verified; intensity >0.1 is not a vessel mask; all values on [0,1] raw scale", "mean": means, "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["mean"], indent=2), flush=True)


if __name__ == "__main__":
    main()
