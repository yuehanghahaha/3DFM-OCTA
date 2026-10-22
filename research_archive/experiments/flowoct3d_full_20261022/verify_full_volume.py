"""Audit the additional real full-FOV OCT-only inference without using test data."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE))

import numpy as np
from scipy.io import loadmat
from fm3d.data import load_pair
from flowoct3d.metrics import panel, sample_metrics
from flowoct3d.utils import read_json, sha256, write_json


data = WORKSPACE/'data/OCTA500_common_v1'
destination = ROOT/'verification/full_volume_10319'
source_path = data/'z256/3M/10319.npz'
source, target = load_pair(source_path, 'ZYX', 255., 256)
with np.load(destination/'prediction.npz', allow_pickle=False) as values:
    prediction = values['octa']
    assert str(values['axis_order'].item()) == 'ZYX'
assert prediction.shape == (256, 304, 304)
assert np.isfinite(prediction).all() and prediction.min() >= 0 and prediction.max() <= 1
sample = dict(oct=source[0].numpy(), octa=target[0].numpy(),
              layers=np.asarray(loadmat(data/'labels_native/GT_Layers/10319.mat')['Layer'], dtype=np.float32),
              case_id='10319', patient_id='10319', split='validation', patch_id='full_volume')
metrics, projections = sample_metrics(prediction, sample, full_ssim=True)
panel(sample, prediction, projections, destination/'comparison.png')
protocol = read_json(destination/'prediction.json')
write_json(ROOT/'verification/full_volume_smoke.json', dict(passed=True, case_id='10319', split='validation',
           shape=list(prediction.shape), source_sha256=sha256(source_path),
           prediction_sha256=sha256(destination/'prediction.npz'),
           checkpoint_sha256=protocol['checkpoint_sha256'], metrics=metrics,
           scope='One additional complete real validation volume, OCT-only 8-step coefficient-grid sliding inference. Not the patch comparison table or a full test-set evaluation.'))
print('Full-volume inference verified:', prediction.shape, metrics['volume_psnr'])

