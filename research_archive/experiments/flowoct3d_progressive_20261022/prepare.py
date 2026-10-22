"""Freeze a public FlowOCT reference and a cumulative 3D removal ladder."""
from dataclasses import replace
from pathlib import Path
import sys
import shutil

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE))
import numpy as np
from flowoct3d.losses import LossConfig
from flowoct3d.utils import read_json, sha256, write_json


def configurations():
    full = LossConfig()
    no_rdca = replace(full, rdca_weight=0.)
    no_projection = replace(no_rdca, projection_weight=0.)
    no_structure = replace(no_projection, structure=False)
    bare = replace(no_structure, reconstruction_weight=0.)
    return {
        '00_flowoct_public': dict(kind='public', label='Public FlowOCT', representation='haar2d'),
        '01_full3d': dict(kind='3d', label='Full 3D', representation='haar3d', losses=full.to_dict(), start_fraction=0.),
        '02_minus_rdca': dict(kind='3d', label='- RDCA', representation='haar3d', losses=no_rdca.to_dict(), start_fraction=0.),
        '03_minus_rdca_projection': dict(kind='3d', label='- RDCA - projection', representation='haar3d', losses=no_projection.to_dict(), start_fraction=0.),
        '04_minus_rdca_projection_structure': dict(kind='3d', label='- RDCA - projection - structure', representation='haar3d', losses=no_structure.to_dict(), start_fraction=0.),
        '05_haar_fm_only': dict(kind='3d', label='Haar FM only', representation='haar3d', losses=bare.to_dict(), start_fraction=0.),
        '06_voxel_fm_only': dict(kind='3d', label='Voxel FM only', representation='voxel', losses=bare.to_dict(), start_fraction=0.),
        '07_full_half_start': dict(kind='3d', label='Full 3D + 50% t=0', representation='haar3d', losses=full.to_dict(), start_fraction=.5),
    }


def main():
    if (ROOT/'protocol.json').exists():
        raise FileExistsError('Protocol already frozen')
    previous = WORKSPACE/'experiments/flowoct3d_full_20261022'
    for source_folder, destination in ((previous/'data', ROOT/'data'),
                                      (previous/'source/flowoct3d', ROOT/'source/flowoct3d'),
                                      (previous/'source/fm3d', ROOT/'source/fm3d'),
                                      (WORKSPACE/'flowoct_reproduction/flowoct', ROOT/'source/flowoct')):
        destination.mkdir(parents=True, exist_ok=True)
        for path in source_folder.rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            if source_folder.name == 'flowoct' and path.name not in ('__init__.py', 'model.py', 'wavelets.py', 'wavelet_functions.py', 'roi.py'):
                continue
            target = destination/path.relative_to(source_folder)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    # Reuse the exact original 800-step prefix; append two balanced frozen blocks.
    prefix = previous/'03_full/few_case/schedule.npz'
    with np.load(prefix, allow_pickle=False) as d:
        indices, times = [d[k].copy() for k in ('indices', 'times')]
    assert len(indices) == len(times) == 800
    for block in (1, 2):
        rng = np.random.default_rng(20261023+block*10000)
        indices = np.concatenate([indices, np.concatenate([rng.permutation(8) for _ in range(100)])])
        times = np.concatenate([times, rng.random(800).astype(np.float32)])
    half_times = times.copy()
    for block in range(3):
        mask_rng = np.random.default_rng(20261023+991+block*10000)
        selected = mask_rng.permutation(800)[:400]+block*800
        half_times[selected] = 0.
    np.savez_compressed(ROOT/'data/schedule.npz', indices=indices, times=times, half_times=half_times)
    protocol = dict(schema='public_flowoct_and_cumulative_3d_removal_v1', seed=20261021,
        updates=2400, eval_every=400, milestones=[800, 1600, 2400], base_channels=16,
        lr=2e-4, amp=True, ode_steps=8, condition=True, inference_chunk=16,
        arms=configurations(), prior_study=str(previous),
        prior_prefix_sha256=sha256(prefix), schedule_sha256=sha256(ROOT/'data/schedule.npz'),
        selection_sha256=sha256(ROOT/'data/selection.json'),
        scope='Four training/four validation patients, two full-Z fixed patches each, one seed, no test images',
        comparison='Cumulative removal along a fixed path, not independent component main effects. Public FlowOCT is a method control with its own optimizer and slice/time sampling.',
        public=dict(commit='2309126956d2b80a95f2d307d85a50799cc658a6',
                    weights=[10., .5, .005, .1], optimizer='Adam beta1=0.5 beta2=0.999',
                    amp=False, gradient_clipping=None, network_chunk=0, train_slices=16,
                    native_solver='Euler3 with per-step clamp', common_solver='Euler8 with final clamp',
                    preprocessing='Exactly the frozen [0,1] tensors; lossless ZYX <-> YXZ only'))
    source_files = [p for p in (ROOT/'source').rglob('*.py') if '__pycache__' not in p.parts]
    protocol['source_sha256'] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted(source_files)}
    write_json(ROOT/'protocol.json', protocol)
    print('Frozen original FlowOCT and seven 3D arms; 2400 updates each; 19200 total.')


if __name__ == '__main__':
    main()

