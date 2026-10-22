"""Independent numerical, budget, exposure and module-change audit."""
from collections import defaultdict
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source'))
import numpy as np
from skimage.metrics import structural_similarity
import torch
from flowoct3d.utils import read_json, sha256, write_json


def recompute(prediction, target, layers):
    error = prediction.astype(np.float64)-target
    centers = ((np.arange(len(target))+.5)*640/len(target)-.5)[:, None, None]
    mask = (centers >= layers[0]-1) & (centers <= layers[2]-1)
    a, b = [np.where(mask, v, 0).max(0).astype(np.float64) for v in (prediction, target)]
    x, y = a-a.mean(), b-b.mean()
    denominator = np.sqrt(np.sum(x*x)*np.sum(y*y))
    return dict(volume_mae=float(np.abs(error).mean()),
        volume_psnr=float(-10*np.log10(max(float(np.mean(error**2)), 1e-12))),
        ilm_opl_pcc=float(np.sum(x*y)/denominator) if denominator > 1e-15 else None,
        ilm_opl_ssim=float(structural_similarity(a, b, data_range=1., win_size=7)))


def verify():
    config, selection = read_json(ROOT/'protocol.json'), read_json(ROOT/'data/selection.json')
    assert sha256(ROOT/'data/schedule.npz') == config['schedule_sha256']
    assert sha256(ROOT/'data/selection.json') == config['selection_sha256']
    for path, digest in config['source_sha256'].items():
        assert sha256(ROOT/path) == digest, path
    with np.load(ROOT/'data/schedule.npz') as data:
        schedule = {key: data[key].copy() for key in data.files}
    inputs = {}
    for record in selection['patches']:
        path = ROOT/record['file']
        assert sha256(path) == record['sha256']
        assert record['split'] in ('train', 'validation')
        with np.load(path, allow_pickle=False) as data:
            inputs[record['case_id']+'_'+record['patch_id']] = {k: data[k].copy() for k in ('octa', 'layers')}
    training = [r for r in selection['patches'] if r['split'] == 'train']
    initial_by_representation = {}
    audited, total_predictions, total_updates, maximum_error = [], 0, 0, 0.
    for name, arm in config['arms'].items():
        folder = ROOT/name
        checkpoint = torch.load(folder/'latest.pt', map_location='cpu', weights_only=False)
        assert checkpoint['updates'] == config['updates']
        assert {int(s['step']) for s in checkpoint['optimizer']['state'].values()} == {config['updates']}
        assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
        initial = torch.load(folder/'initial.pt', map_location='cpu', weights_only=False)['model']
        if arm['kind'] == '3d':
            representation = arm['representation']
            if representation in initial_by_representation:
                for key in initial:
                    assert torch.equal(initial[key], initial_by_representation[representation][key])
            else:
                initial_by_representation[representation] = initial
        prefix = 'contrast_projection_head.' if arm['kind'] == 'public' else 'depth_encoder.'
        delta = sum(float((checkpoint['model'][key]-initial[key]).abs().sum()) for key in initial if key.startswith(prefix))
        contrast_enabled = arm['kind'] == 'public' or bool(arm['losses']['rdca_weight'])
        assert (delta > 0) == contrast_enabled, name
        logs = [json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
        assert [r['update'] for r in logs] == list(range(1, config['updates']+1))
        for i, row in enumerate(logs):
            expected = training[int(schedule['indices'][i])]
            assert (row['case_id'], row['patch_id']) == (expected['case_id'], expected['patch_id'])
            assert np.isfinite(row['loss'])
            if arm['kind'] == 'public':
                assert row['seed'] == 20261023*10000+i+1
                assert row['sampled_Bscans'] == 16
                expected_loss = 10*row['vloss']+.5*row['L1loss']+.005*row['contrastloss']+.1*row['proj_loss']
            else:
                times = schedule['half_times' if arm.get('start_fraction') else 'times']
                assert np.float32(row['time']) == times[i]
                weights = arm['losses']
                expected_loss = sum(row[k]*weights[w] for k, w in
                    (('fm', 'fm_weight'), ('reconstruction', 'reconstruction_weight'),
                     ('projection', 'projection_weight'), ('rdca', 'rdca_weight')))
                for key, weight in (('reconstruction', 'reconstruction_weight'), ('projection', 'projection_weight'), ('rdca', 'rdca_weight')):
                    if weights[weight] == 0:
                        assert row[key] == 0.
            assert abs(expected_loss-row['loss']) < 2e-6*max(1., abs(expected_loss))
        exposure = defaultdict(int)
        for row in logs:
            exposure[row['case_id']+'_'+row['patch_id']] += 1
        assert set(exposure.values()) == {300}
        metric_files = [(folder/f'metrics_{n:05d}.json', folder/f'predictions_{n:05d}', n) for n in config['milestones']]
        if arm['kind'] == 'public':
            metric_files.append((folder/'final_native3.json', folder/'predictions_native3', config['updates']))
        arm_error = 0.
        for metric_file, prediction_folder, update in metric_files:
            result = read_json(metric_file)
            assert result['updates'] == update
            assert {r['case_id']+'_'+r['patch_id'] for r in result['cases']} == set(inputs)
            for row in result['cases']:
                key = row['case_id']+'_'+row['patch_id']
                path = prediction_folder/(key+'.npz')
                assert sha256(path) == row['prediction_sha256']
                with np.load(path, allow_pickle=False) as data:
                    prediction, target = data['octa'], data['octa_gt']
                    assert str(data['axis_order'].item()) == 'ZYX'
                    assert prediction.shape == (256, 64, 64)
                    assert np.isfinite(prediction).all() and prediction.min() >= 0 and prediction.max() <= 1
                    np.testing.assert_array_equal(target, inputs[key]['octa'])
                scores = recompute(prediction, target, inputs[key]['layers'])
                for metric, value in scores.items():
                    assert value is not None
                    error = abs(value-row['metrics'][metric])
                    assert error < 1e-7, (name, update, key, metric, error)
                    arm_error = max(arm_error, error)
                total_predictions += 1
            for split, summary in result['summary'].items():
                patients = defaultdict(list)
                for row in result['cases']:
                    if row['split'] == split:
                        patients[row['patient_id']].append(row['metrics'])
                for key in ('volume_mae', 'volume_psnr', 'ilm_opl_pcc', 'ilm_opl_ssim'):
                    value = float(np.mean([np.mean([r[key] for r in rows]) for rows in patients.values()]))
                    assert abs(value-summary['metrics'][key]) < 1e-10
        final = read_json(folder/'final_metrics.json')
        assert final['checkpoint_sha256'] == sha256(folder/'latest.pt')
        assert final['optimizer_steps'] == [2400]
        total_updates += config['updates']
        maximum_error = max(maximum_error, arm_error)
        audited.append(dict(arm=name, updates=config['updates'], attempts=checkpoint['attempts'],
            head_parameter_L1_change=delta, exposures_per_training_patch=300, maximum_metric_error=arm_error))
        print('Verified', name, flush=True)
    result = dict(passed=True, arms=len(audited), successful_updates=total_updates,
        predictions=total_predictions, maximum_metric_error=maximum_error, records=audited,
        checks='Source/input/checkpoint/prediction hashes, equal initialization within representation, actual optimizer counts, fixed exposures/times, cumulative disabled terms, independent NumPy/SSIM recomputation and patient aggregation')
    write_json(ROOT/'verification/results_audit.json', result)
    print(result)


if __name__ == '__main__':
    verify()

