# Three-level evaluation of saved 3M predictions

The first complete result is `artifacts/fm3d_metrics_20261021/index.html`.
It contains all 50 held-out patients, 15,200 B-scans, three projections per
patient, and exploratory FID over 50 ILM–OPL projections. No retraining or model
selection is performed by these commands. Use the same protocol for later baselines.

## Requirements

Keep the existing PyTorch installation. Install a compatible torchvision (for
torch 2.5.1, torchvision 0.20.1), plus scipy, scikit-image and lpips 0.1.4.
Optional FID requires clean-fid 0.1.35; report generation needs matplotlib.
The first evaluation used numpy 1.26.4, scipy 1.14.1/scikit-image 0.24.0 on the
server for volumes, and the package versions recorded in the local projection
protocol for projections. Each output records its versions explicitly.

LPIPS downloads the official pretrained AlexNet weights; use a persistent,
personal `TORCH_HOME` on a shared server. Never substitute random weights.
The package provides the LPIPS v0.1 calibration weights. SHA256 fingerprints
are saved in protocol.json. GPU selection should be limited to your assigned GPU.

## Volume and B-scan metrics

From the project root, with the personal conda environment activated:

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_HOME=/srv/oct_research/torch_cache \
python -m fm3d.metrics_suite volume \
  --input runs/fm3d_3M_run01_test_best_20261020 \
  --manifest /srv/oct_research/OCTA500_common_v1/split_3M_z256.csv \
  --output runs/fm3d_metrics_20261021/volume \
  --lpips --device cuda:0
```

Input directory must contain the original inference `metrics.json` and each
`<case>.npz` with `octa`, `octa_gt`, and `axis_order='ZYX'`.
This frozen runner validates exactly 50 unique test patients and the complete
ID set. It computes whole-volume MAE/MSE/RMSE/PSNR/SSIM, then separately
averages 304 ZX B-scan scores and LPIPS within each patient. PSNR has a 120 dB
ceiling for exact equality. SSIM uses fixed range1, uniform window7, sample
covariance, K1=.01/K2=.03, no channel axis. True3D SSIM is distinct from B-scan
SSIM. No intensity stretching is used.

## Projection and vessel metrics

```bash
python -m fm3d.metrics_suite projection \
  --input artifacts/fm3d_test_20261020/report \
  --manifest data/OCTA500_common_v1/split_3M_z256.csv \
  --labels data/OCTA500_common_v1/labels_native \
  --output artifacts/fm3d_metrics_20261021/projection \
  --lpips --device cuda:0
```

Input is the existing `fm3d.projection_report` output containing raw float32
projection NPZs, not display PNGs. Basic metrics and LPIPS cover FULL,
ILM_OPL and OPL_BM. Vessel metrics cover ILM_OPL only:

- Reference annotation is `(GT_LargeVessel >0) OR (GT_Capillary >0)`, unchanged
  official BMP orientation [X,Y]. FAZ is not included.
- MAE-V/PSNR-V/SSIM-V multiply both images by the same weight map, 1 for
  annotated vessels and .1 elsewhere. The metric range remains1.
- VDE/VDC threshold each raw image strictly above its own global mean.
  VDE is absolute occupancy difference; VDC is Pearson correlation between
  nonoverlapping 16×16 patch occupancy vectors. No morphology or test-tuned threshold.
- Dice/IoU/precision/recall/hard2D clDice compare the prediction threshold mask
  with the official annotation. A second set evaluates the reference threshold
  mask against that annotation, to expose threshold-method limitations.
  Another explicitly named set compares the two threshold masks.
- clDice uses Zhang skeletonization. Both empty masks score1; one empty scores0.
  Zero-variance Pearson correlations are null and excluded with counts disclosed.

These are adapted, explicitly frozen evaluation definitions, not claims to
reproduce the published numerical tables. Threshold-derived 2D clDice is not
3D topology or cbDice, and does not use a learned vessel segmentation network.

## Exploratory FID

Download the clean-fid official NVIDIA Inception weights into a personal
directory using `cleanfid.inception_torchscript.InceptionV3W(path)` once.
Then run:

```bash
python -m fm3d.fid_projection \
  --input artifacts/fm3d_test_20261020/report \
  --output artifacts/fm3d_metrics_20261021/fid \
  --weights artifacts/fm3d_metrics_20261021/weights
```

One ILM_OPL image per patient is converted by round(255*x), replicated RGB and
resized by clean-fid clean mode. No contrast stretch. All 50 IDs are listed
explicitly to avoid Windows case-insensitive duplicate glob matches.
Features are saved and the covariance FID is checked against an independent
low-rank computation. At n=50 the2048-D covariance is rank deficient: report
this FID only as exploratory, not as a per-patient accuracy measure or a value
directly comparable with papers using different image units/preprocessing.

## Outputs and verification

Each mode saves `protocol.json`, `metrics.json`, `per_case_metrics.csv`, and
`input_sha256.json`; projections also save all masks. Existing result files are
updated when explicitly rerunning the same output directory, so use a new
directory for another model. Patient summaries are mean ± sample SD.

```bash
python -m unittest fm3d.test_metrics_suite fm3d.test_projection_report
```

The report assembler `tools/build_fm3d_metrics_report.py` is specific to this
completed run. It checks both 50-patient ID sets and the original MAE/PSNR,
then writes HTML, Markdown, combined CSV, JSON and diagnostic figures. Original
float32 tensor reductions and new float64 reductions differ only at numerical
roundoff (max MAE difference1.33e-10, PSNR1.04e-7dB in this run).

Sources: TransPro §4.2 (local audited paper),
[OCTA-500 §3.3](https://arxiv.org/pdf/2012.07261),
[LPIPS](https://github.com/richzhang/PerceptualSimilarity),
[clean-fid](https://github.com/GaParmar/clean-fid),
[clDice](https://github.com/jocpae/clDice).
