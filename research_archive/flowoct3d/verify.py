"""Independent audit of saved predictions, step logs, module updates and budgets."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity
import torch

from .utils import read_json, sha256, write_json


def independent_metrics(prediction, target, layers):
    error = prediction.astype(np.float64)-target
    centers = ((np.arange(len(target))+.5)*640/len(target)-.5)[:, None, None]
    mask = (centers >= layers[0]-1) & (centers <= layers[2]-1)
    # Evaluated volumes are clipped nonnegative, so zero outside regions is valid.
    a, b = [np.where(mask, x, 0).max(0).astype(np.float64) for x in (prediction, target)]
    x, y = a-a.mean(), b-b.mean()
    denom = np.sqrt(np.sum(x*x)*np.sum(y*y))
    return dict(volume_mae=float(np.abs(error).mean()),
                volume_psnr=float(-10*np.log10(max(float(np.mean(error**2)), 1e-12))),
                ilm_opl_ssim=float(structural_similarity(a, b, data_range=1., win_size=7)),
                ilm_opl_pcc=float(np.sum(x*y)/denom) if denom > 1e-15 else None)


def verify(root):
    root = Path(root)
    config, selection = read_json(root/'suite.json'), read_json(root/'data/selection.json')
    inputs = {}
    for r in selection['patches']:
        path = root/r['file']
        assert sha256(path) == r['sha256'], f'Changed input: {path}'
        with np.load(path, allow_pickle=False) as a:
            inputs[r['case_id']+'_'+r['patch_id']] = {k: a[k].copy() for k in ('octa', 'layers')}
    all_records, common_initial, total_updates, predictions = [], {}, 0, 0
    foreground_records = []
    for stage in config['stages']:
        for arm in stage['arms']:
            folder = root/arm/stage['name']
            final = read_json(folder/'final_metrics.json')
            saved = torch.load(folder/'latest.pt', map_location='cpu', weights_only=False)
            for filename, digest in saved['protocol']['sources'].items():
                assert sha256(root/'source'/filename) == digest, f'Training source changed: {filename}'
            initial = torch.load(folder/'initial.pt', map_location='cpu', weights_only=False)['model']
            assert sha256(folder/'latest.pt') == final['checkpoint_sha256']
            assert saved['updates'] == final['updates'] == stage['updates']
            optimizer_steps = {int(s['step']) for s in saved['optimizer']['state'].values()}
            assert optimizer_steps == {stage['updates']}
            logs = [__import__('json').loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
            assert [r['update'] for r in logs] == list(range(1, stage['updates']+1))
            schedule = np.load(folder/'schedule.npz')
            np.testing.assert_array_equal(np.array([r['time'] for r in logs], dtype=np.float32), schedule['times'])
            training_ids = {r['case_id'] for r in selection['patches'] if r['split'] == 'train'}
            assert all(r['case_id'] in training_ids for r in logs)
            training = [r for r in selection['patches'] if r['split'] == 'train' and
                        (stage['name'] == 'few_case' or r['case_id']+'_'+r['patch_id'] == selection['one_case_patch'])]
            for log, index in zip(logs, schedule['indices']):
                assert (log['case_id'], log['patch_id']) == (training[index]['case_id'], training[index]['patch_id'])
            representation = config['arms'][arm]['representation']
            if representation in common_initial:
                for k in initial:
                    assert torch.equal(initial[k], common_initial[representation][k]), 'Initialization differs'
            else:
                common_initial[representation] = initial
            head_delta = sum(float((saved['model'][k]-initial[k]).abs().sum()) for k in initial if k.startswith('depth_encoder.'))
            has_rdca = bool(config['arms'][arm]['losses']['rdca_weight'])
            assert (head_delta > 0) == has_rdca, f'RDCA encoder update mismatch: {arm}'
            expected_keys = ({selection['one_case_patch']} if stage['name'] == 'one_case' else
                             {r['case_id']+'_'+r['patch_id'] for r in selection['patches'] if r['split'] == 'train'})
            expected_keys |= {r['case_id']+'_'+r['patch_id'] for r in selection['patches'] if r['split'] == 'validation'}
            assert {r['case_id']+'_'+r['patch_id'] for r in final['cases']} == expected_keys
            maximum_error = 0.
            for record in final['cases']:
                key = record['case_id']+'_'+record['patch_id']
                path = folder/'predictions'/(key+'.npz')
                assert sha256(path) == record['prediction_sha256']
                with np.load(path, allow_pickle=False) as d:
                    prediction, target = d['octa'], d['octa_gt']
                    assert str(d['axis_order'].item()) == 'ZYX'
                    assert np.isfinite(prediction).all() and prediction.min() >= 0 and prediction.max() <= 1
                    np.testing.assert_array_equal(target, inputs[key]['octa'])
                recomputed = independent_metrics(prediction, target, inputs[key]['layers'])
                mask = target > .05*target.max()
                err = prediction.astype(np.float64)-target
                foreground_records.append(dict(stage=stage['name'], arm=arm, case_id=record['case_id'],
                    patient_id=record['patient_id'], split=record['split'], patch_id=record['patch_id'],
                    foreground_fraction=float(mask.mean()),
                    foreground_mae=float(np.abs(err[mask]).mean()) if mask.any() else None,
                    foreground_psnr=float(-10*np.log10(max(float(np.mean(err[mask]**2)), 1e-12))) if mask.any() else None,
                    background_mae=float(np.abs(err[~mask]).mean()) if (~mask).any() else None))
                for metric, value in recomputed.items():
                    if value is None:
                        assert record['metrics'][metric] is None
                    else:
                        discrepancy = abs(value-record['metrics'][metric])
                        maximum_error = max(maximum_error, discrepancy)
                        assert discrepancy < 1e-7, (arm, key, metric, discrepancy)
                predictions += 1
            for split, summary in final['summary'].items():
                by_patient = defaultdict(list)
                for r in final['cases']:
                    if r['split'] == split:
                        by_patient[r['patient_id']].append(r['metrics'])
                for key in ('volume_mae', 'volume_psnr', 'ilm_opl_ssim', 'ilm_opl_pcc'):
                    patient_values = [np.mean([r[key] for r in rs if r[key] is not None]) for rs in by_patient.values()]
                    assert abs(float(np.mean(patient_values))-summary['metrics'][key]) < 1e-10
            total_updates += stage['updates']
            all_records.append(dict(stage=stage['name'], arm=arm, updates=saved['updates'],
                                    attempts=saved['attempts'], head_parameter_L1_change=head_delta,
                                    predictions=len(final['cases']), maximum_metric_discrepancy=maximum_error))
            print(f'Verified {stage["name"]} {arm}: {stage["updates"]} updates', flush=True)
    result = dict(passed=True, runs=len(all_records), successful_updates=total_updates,
                  predictions=predictions, records=all_records,
                  protocol='Independent NumPy/SSIM recomputation, input/checkpoint/output hashes, optimizer counts, RDCA updates, fixed schedules, patient aggregation')
    write_json(root/'verification/results_audit.json', result)
    write_json(root/'verification/foreground_metrics.json', dict(
        protocol='Post-hoc target>0.05*per-patch max. Same mask for all methods, not expert vessel segmentation or the original paper evaluation.',
        records=foreground_records))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(verify(args.root))

