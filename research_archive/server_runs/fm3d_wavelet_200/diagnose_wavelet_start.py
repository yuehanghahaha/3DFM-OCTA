"""Read-only Haar start-versus-teacher-path check on fixed training/validation patches."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fm3d.data import load_pair
from fm3d.model import make_model
from fm3d.wavelets import haar_dwt3d, haar_idwt3d


def scores(prediction, target):
    a, b = prediction[0, 0].cpu().numpy(), target[0, 0].cpu().numpy()
    pa, pb = a.max(0), b.max(0)
    return {"volume_mae": float(np.abs(a - b).mean()),
            "max_projection_pcc": float(np.corrcoef(pa.ravel(), pb.ravel())[0, 1]),
            "max_projection_p99": float(np.percentile(pa, 99)),
            "target_max_projection_p99": float(np.percentile(pb, 99))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config.get("representation") != "haar3d":
        raise ValueError("Expected Haar checkpoint")
    model = make_model(config["base_channels"], config["condition"], "haar3d")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    result = {"checkpoint_epoch": checkpoint["epoch"],
              "protocol": "First case by ID in train and validation; fixed 256x64x64 patch at Y=X=64; max-Z projection; teacher paths contain true OCTA and are not generation scores.",
              "cases": []}
    with torch.inference_mode():
        for split in ("train", "validation"):
            row = sorted((r for r in checkpoint["manifest"] if r["split"] == split),
                         key=lambda r: r["case_id"])[0]
            source, target = load_pair(row["path"], config["axis_order"],
                                       config["intensity_scale"], config["depth"])
            source, target = (v[None, :, :, 64:128, 64:128] for v in (source, target))
            s, y = haar_dwt3d(source), haar_dwt3d(target)
            case = {"case_id": row["case_id"], "split": split}
            for t in (0.0, 0.05, 0.5):
                state = (1 - t) * s + t * y
                velocity = model(state, torch.tensor([t]), s)
                endpoint = haar_idwt3d(state + (1 - t) * velocity).clamp(0, 1)
                case[f"teacher_t_{t}"] = scores(endpoint, target)
            state = s.clone()
            for step in range(8):
                state += model(state, torch.tensor([step / 8]), s) / 8
            case["free_rollout_8"] = scores(haar_idwt3d(state).clamp(0, 1), target)
            result["cases"].append(case)
            print(json.dumps(case), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
