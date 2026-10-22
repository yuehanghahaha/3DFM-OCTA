"""Freeze four training and four validation patients, selected without image scores."""
import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))

import numpy as np
from PIL import Image
from scipy.io import loadmat
from fm3d.data import load_pair, read_rows, validate_manifest
from engine import ARMS, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--one-updates', type=int, default=400)
    parser.add_argument('--few-updates', type=int, default=800)
    args = parser.parse_args()
    data = args.data_root.resolve()
    if (ROOT / 'data/selection.json').exists():
        raise FileExistsError('Frozen selection already exists; do not silently select patients again')
    rows = read_rows(data / 'split_3M_z256.csv')
    validate_manifest(rows, '3M')
    chosen = []
    # Include one per disease; NORMAL first makes the one-case check unambiguous.
    for phase in ('train', 'validation'):
        for disease in ('NORMAL', 'DR', 'AMD', 'CNV'):
            candidates = sorted([r for r in rows if r['split'] == phase and r['disease'] == disease],
                                key=lambda r: r['case_id'])
            chosen.append(candidates[0])
    if len({r['patient_id'] for r in chosen}) != len(chosen):
        raise ValueError('Selected patients overlap')
    config = dict(schema='fm3d_small_patient_ablation_v1', seed=20261021, patch_ZYX=[256, 64, 64],
                  base_channels=16, condition=True, intensity_scale=255, axis_order='ZYX',
                  lr=2e-4, weight_decay=0., grad_clip=1., amp=True, ode_steps=8,
                  checkpoint_rollout=False, eval_every=200, batch_size=1,
                  time_sampling='uniform [0,1); same precomputed sequence for all arms',
                  selection_metric='validation patient-macro ILM-OPL SSIM; equal-budget last checkpoints also reported',
                  stages=[dict(name='one_case', updates=args.one_updates, seed=20261022),
                          dict(name='few_case', updates=args.few_updates, seed=20261023)],
                  comparison='Each stage starts from scratch; one_case and few_case are independent, not continued training')
    write_json(ROOT / 'suite.json', config)
    for name, (representation, lp, lr) in ARMS.items():
        destination = ROOT / name
        destination.mkdir(exist_ok=True)
        write_json(destination / 'config.json', dict(name=name, representation=representation,
                   lambda_projection=lp, lambda_reconstruction=lr))
    (ROOT / 'data/patches').mkdir(parents=True, exist_ok=True)
    selection = dict(schema='frozen_3M_train_validation_patches_v1', manifest_sha256=sha256(data/'split_3M_z256.csv'),
        selection='First ID within each fixed train/validation disease stratum; no test case; no image/metric selection',
        source_data_root=str(data), source_axis_order='ZYX', source_intensity_scale=255,
        patch_ZYX=[256,64,64], patch_origins_ZYX=[[0,64,64],[0,120,120]],
        mask_orientation='Official BMP XY -> transpose to YX -> same crop as volume',
        layers='Native MATLAB1-based 6YX; post-hoc evaluation only; never used by generator/loss',
        identity_basis='OCTA500_common_v1 public subject IDs; see original protocol.json',
        patients=[], patches=[])
    for row in chosen:
        source, target = load_pair(row['path'], 'ZYX', 255, 256)
        source, target = source[0].numpy(), target[0].numpy()
        case = row['case_id']
        layer_path = data / 'labels_native/GT_Layers' / (case+'.mat')
        layers = np.asarray(loadmat(layer_path)['Layer'], dtype=np.float32)
        mask_paths = [data/'labels_native'/kind/(case+'.bmp') for kind in ('GT_LargeVessel','GT_Capillary')]
        masks = [np.asarray(Image.open(p)) > 0 for p in mask_paths]
        vessel_mask = (masks[0] | masks[1]).T
        if layers.shape != (6,*source.shape[1:]) or vessel_mask.shape != source.shape[1:]:
            raise ValueError(f'Annotation geometry mismatch: {case}')
        patient = {k: row[k] for k in ('case_id','patient_id','split','disease')}
        patient.update(source_npz_sha256=sha256(row['path']), layers_sha256=sha256(layer_path),
                       masks_sha256={p.parent.name:sha256(p) for p in mask_paths})
        selection['patients'].append(patient)
        for y,x in ((64,64),(120,120)):
            patch_id = f'y{y:03d}_x{x:03d}'
            path = ROOT / 'data/patches' / (case+'_'+patch_id+'.npz')
            np.savez_compressed(path, oct=source[:,y:y+64,x:x+64], octa=target[:,y:y+64,x:x+64],
                layers=layers[:,y:y+64,x:x+64], vessel_mask=vessel_mask[y:y+64,x:x+64].astype(np.uint8))
            selection['patches'].append({**{k:row[k] for k in ('case_id','patient_id','split','disease')},
                'patch_id':patch_id, 'origin_ZYX':[0,y,x], 'file':path.relative_to(ROOT).as_posix(), 'sha256':sha256(path)})
        print(f'Prepared {case} {row["split"]} {row["disease"]}', flush=True)
    selection['one_case_patch'] = chosen[0]['case_id']+'_y064_x064'
    write_json(ROOT / 'data/selection.json', selection)
    shutil.copy2(data/'protocol.json', ROOT/'data/original_data_protocol.json')
    print('Frozen 8 patients / 16 patches. Test subjects were not loaded.', flush=True)


if __name__ == '__main__':
    main()
