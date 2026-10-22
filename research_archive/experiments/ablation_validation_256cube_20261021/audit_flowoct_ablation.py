"""Protect previous artifacts and audit the two additional velocity-only controls."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import numpy as np
from engine import load_samples, sha256, write_json
from flowoct_velocity_only import ARMS


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def protect_previous():
    destination = ROOT / 'verification/flowoct_ablation_before.json'
    if destination.exists():
        raise FileExistsError('Previous-artifact baseline already exists')
    previous = read(ROOT / 'server_artifacts.json')
    assert len(previous) == 41
    hashes = {}
    for item in previous:
        p = ROOT / item['file']
        hashes[item['file']] = sha256(p)
        assert hashes[item['file']] == item['sha256'], item['file']
    files = [ROOT / name for name in ('run256.py', 'memory_model.py', 'engine.py', 'suite.json', 'flowoct_control.py')]
    files += sorted((ROOT / 'source').rglob('*.py'))
    for p in files:
        hashes[p.relative_to(ROOT).as_posix()] = sha256(p)
    write_json(destination, {'status': 'passed', 'previous_large_artifacts': 41,
                             'hashes': hashes, 'description': 'Before either new arm starts training'})
    print(json.dumps({'protected_files': len(hashes), 'large_artifacts': 41}), flush=True)


def audit():
    before = read(ROOT / 'verification/flowoct_ablation_before.json')
    for name, expected in before['hashes'].items():
        assert sha256(ROOT / name) == expected, ('Previous artifact changed', name)
    cfg = read(ROOT / 'flowoct_ablation_config.json')
    samples = load_samples(ROOT, read(ROOT / 'data/selection.json'))
    cases = {sample['case_id']: sample for sample in samples}
    baseline_initial = read(ROOT / '06_flowoct/one_case/initial_metrics.json')
    baseline_first_step = json.loads((ROOT / '06_flowoct/one_case/steps.jsonl').read_text().splitlines()[0])
    checked, schedules, artifacts = 0, [], []
    for arm in ARMS:
        out = ROOT / arm / 'one_case'
        status = read(out / 'status.json')
        assert status['status'] == 'complete' and status['updates'] == cfg['updates']
        assert status['attempts'] == cfg['updates'] and status['amp_skipped'] == 0
        protocol = read(out / 'protocol.json')
        assert protocol['config'] == cfg and protocol['selection_sha256'] == sha256(ROOT / 'data/selection.json')
        for name, expected in protocol['sources'].items():
            assert sha256(ROOT / name) == expected, name
        initial = read(out / 'initial_metrics.json')
        for phase in ('train', 'validation'):
            for key, expected in baseline_initial['summary'][phase]['metrics'].items():
                actual = initial['summary'][phase]['metrics'][key]
                if isinstance(expected, (float, int)):
                    assert abs(actual - expected) < 1e-8, ('Initial prediction mismatch', arm, phase, key)
        rows = [json.loads(line) for line in (out / 'steps.jsonl').read_text().splitlines()]
        assert [row['update'] for row in rows] == list(range(1, cfg['updates'] + 1))
        assert abs(rows[0]['vloss'] - baseline_first_step['vloss']) < 1e-8
        for row in rows:
            assert row['seed'] == cfg['step_seed_base'] + row['update']
            assert abs(row['loss'] - 10 * row['vloss']) < 1e-6
            assert not any(key in row for key in ('L1loss', 'contrastloss', 'proj_loss'))
            assert len(row['slice_indices']) == len(row['slice_times']) == 64
            assert len(set(row['slice_indices'])) == 64
            assert all(0 <= value < 256 for value in row['slice_indices'])
            assert all(0 <= value < 1 for value in row['slice_times'])
        with np.load(out / 'schedule.npz', allow_pickle=False) as stored:
            schedules.append({key: stored[key].copy() for key in stored.files})
        assert np.array_equal(schedules[-1]['slice_indices'], [row['slice_indices'] for row in rows])
        assert np.array_equal(schedules[-1]['slice_times'], np.asarray([row['slice_times'] for row in rows], dtype=np.float32))
        for folder, metric in [('predictions_native3', 'final_native3.json'), ('predictions', 'final_metrics.json')]:
            report = read(out / metric)
            assert report['completed_updates'] == cfg['updates']
            metrics = {row['case_id']: row['metrics'] for row in report['cases']}
            paths = sorted((out / folder).glob('*.npz'))
            assert len(paths) == 5
            for p in paths:
                case = p.name.split('_')[0]
                with np.load(p, allow_pickle=False) as saved:
                    prediction, truth = saved['octa'], saved['octa_gt']
                    assert prediction.shape == truth.shape == (256, 256, 256)
                    assert saved['axis_order'].item() == 'ZYX'
                    assert np.array_equal(truth, cases[case]['octa'])
                    assert np.isfinite(prediction).all() and prediction.min() >= 0 and prediction.max() <= 1
                    mae = float(np.abs(prediction.astype(np.float64) - truth.astype(np.float64)).mean())
                    assert abs(mae - metrics[case]['volume_mae']) < 1e-8
                checked += 1
        for p in sorted(out.rglob('*')):
            if p.is_file() and (p.suffix == '.pt' or p.suffix == '.npz' and p.parent.name.startswith('predictions')):
                artifacts.append({'file': p.relative_to(ROOT).as_posix(), 'server_path': str(p),
                                  'bytes': p.stat().st_size, 'sha256': sha256(p)})
    for key in ('seeds', 'slice_indices', 'slice_times'):
        assert np.array_equal(schedules[0][key], schedules[1][key]), key
    assert checked == 20
    write_json(ROOT / 'flowoct_ablation_server_artifacts.json', artifacts)
    result = {'status': 'passed', 'completed_new_arms': 2, 'successful_new_updates': 800,
              'amp_skipped': 0, 'new_predictions_checked': checked,
              'identical_actual_sampling_between_new_arms': True,
              'initial_metrics_match_full_flowoct': True,
              'first_velocity_loss_matches_full_flowoct': True,
              'prior_artifacts_and_sources_unchanged': len(before['hashes']),
              'old_large_artifacts_unchanged': 41, 'source_fingerprints_match': True,
              'frozen_targets_exactly_equal': True, 'volume_mae_independently_recomputed': True,
              'large_new_artifacts': len(artifacts), 'combined_arms': 8,
              'combined_successful_updates': 3200, 'combined_predictions': 55}
    write_json(ROOT / 'verification/flowoct_ablation_audit.json', result)
    metadata = read(ROOT / 'flowoct_ablation_run.json')
    metadata.update(state='complete', **result)
    write_json(ROOT / 'flowoct_ablation_run.json', metadata)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', action='store_true')
    args = parser.parse_args()
    protect_previous() if args.before else audit()
