"""Compare the fixed 50-patient test evaluations of three selected checkpoints."""

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OLDER = ROOT.parent / "haar3d_200_2026-10-21"
REPORTS = {
    "baseline_90": OLDER / "baseline_test_metrics.json",
    "baseline_106": ROOT / "test_best106_metrics.json",
    "haar3d_195": OLDER / "wavelet_test_metrics.json",
}


def main():
    reports = {name: json.loads(path.read_text(encoding="utf-8"))
               for name, path in REPORTS.items()}
    by_case = {name: {row["case_id"]: row for row in report["cases"]}
               for name, report in reports.items()}
    ids = sorted(by_case["baseline_90"])
    if len(ids) != 50 or any(sorted(rows) != ids for rows in by_case.values()):
        raise ValueError("Expected the same 50 test cases")
    if any(report["split"] != "test" or report["patients"] != 50 for report in reports.values()):
        raise ValueError("Expected 50-patient test evaluations")
    if len({report["config"]["manifest_sha256"] for report in reports.values()}) != 1:
        raise ValueError("Manifest hash mismatch")
    if {report["config"]["ode_steps"] for report in reports.values()} != {8}:
        raise ValueError("Inference steps differ")
    rows = []
    for case_id in ids:
        row = {"case_id": case_id}
        for model, cases in by_case.items():
            row.update({f"{model}_{key}": cases[case_id][key] for key in ("mae", "psnr")})
        rows.append(row)
    with (ROOT / "test_50_comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rng = np.random.default_rng(2026)
    indices = rng.integers(0, 50, size=(20000, 50))
    summary = {"patients": 50,
               "checkpoints": {name: report["checkpoint_epoch"] for name, report in reports.items()},
               "means": {name: report["metrics"] for name, report in reports.items()},
               "paired": {}}
    for comparator in ("baseline_90", "haar3d_195"):
        pair = {}
        for metric in ("mae", "psnr"):
            delta = np.array([row[f"baseline_106_{metric}"] - row[f"{comparator}_{metric}"]
                              for row in rows])
            pair[metric] = {"mean_difference_106_minus_comparator": float(delta.mean()),
                            "bootstrap_95pct_ci": np.percentile(delta[indices].mean(axis=1),
                                                                  [2.5, 97.5]).tolist(),
                            "baseline_106_better_cases": int(np.count_nonzero(delta < 0 if metric == "mae"
                                                                             else delta > 0))}
        summary["paired"][f"baseline_106_vs_{comparator}"] = pair
    (ROOT / "test_50_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
