"""Verify exported paired metrics and regenerate review charts without patient data or weights."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
COLORS = {'voxel': '#285F89', 'haar3d': '#B95636'}
LABELS = {'voxel': 'Voxel baseline (100 epochs)', 'haar3d': '3D Haar (200 epochs)'}


def read_csv(name):
    with (ROOT / 'tables' / name).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def paired_stats(baseline, haar, indices, lower_is_better):
    delta = haar - baseline
    return {'baseline': float(baseline.mean()), 'haar3d': float(haar.mean()),
            'difference_haar3d_minus_baseline': float(delta.mean()),
            'paired_bootstrap_95pct_ci': np.percentile(delta[indices].mean(axis=1), [2.5, 97.5]).tolist(),
            'haar3d_better_patients': int(np.count_nonzero(delta < 0 if lower_is_better else delta > 0))}


def check_stats(actual, expected):
    for key, value in actual.items():
        np.testing.assert_allclose(value, expected[key], rtol=1e-10, atol=1e-12, err_msg=key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Directory for recomputed JSON and three charts')
    args = parser.parse_args()
    protocol, expected = read_json('experiment_protocol.json'), read_json('metrics_summary.json')
    rows = read_csv('test_metrics.csv')
    pairs = {model: {} for model in COLORS}
    for row in rows:
        model, sample = row['model'], row['sample_id']
        if sample in pairs[model]:
            raise ValueError(f'Duplicate metric record: {model}/{sample}')
        pairs[model][sample] = row
    samples = sorted(pairs['voxel'])
    assert len(samples) == 50 and pairs['voxel'].keys() == pairs['haar3d'].keys()
    arrays = {model: {key: np.array([float(pairs[model][sample][key]) for sample in samples])
                      for key in rows[0] if key not in ('model', 'sample_id')} for model in COLORS}
    assert all(np.isfinite(array).all() for values in arrays.values() for array in values.values())
    bootstrap = protocol['bootstrap']
    rng = np.random.default_rng(bootstrap['seed'])
    indices = rng.integers(0, len(samples), size=(bootstrap['resamples'], len(samples)))
    result = {'test': {}, 'projection': {}, 'validation': {}}
    for scope in ('volume', 'FULL', 'ILM_OPL', 'OPL_BM'):
        stats = result['test'] if scope == 'volume' else result['projection'].setdefault(scope, {})
        metrics = ('mae', 'psnr') if scope == 'volume' else ('mae', 'psnr', 'correlation')
        reference = expected['test'] if scope == 'volume' else expected['projection'][scope]
        for metric in metrics:
            key = f'{scope}_{metric}'
            stats[metric] = paired_stats(arrays['voxel'][key], arrays['haar3d'][key], indices, metric == 'mae')
            check_stats(stats[metric], reference[metric])
    result['test']['patients'] = len(samples)
    result['test']['mae']['relative_difference_pct'] = (
        result['test']['mae']['difference_haar3d_minus_baseline'] / result['test']['mae']['baseline'] * 100)
    histories = {model: [] for model in COLORS}
    for row in read_csv('training_history.csv'):
        histories[row['model']].append(row)
    for model, history in histories.items():
        history.sort(key=lambda row: int(row['epoch']))
        config = protocol['models'][model]
        assert [int(row['epoch']) for row in history] == list(range(1, config['config']['epochs'] + 1))
        assert sum(int(row['optimizer_updates']) for row in history) == config['optimizer_updates']
        assert sum(int(row['amp_skipped_updates']) for row in history) == config['amp_skipped_updates']
        validated = [row for row in history if row['validation_mae']]
        best = min(validated, key=lambda row: float(row['validation_mae']))
        assert int(best['epoch']) == config['checkpoint_epoch'] == expected['validation'][model]['best_mae_epoch']
        for key in ('mae', 'psnr'):
            np.testing.assert_allclose(float(best[f'validation_{key}']),
                                       expected['validation'][model]['best_checkpoint_metrics'][key], atol=1e-12)
        result['validation'][model] = {'epochs': len(history), 'checkpoint_epoch': int(best['epoch'])}
        result['test'][f'{"baseline" if model == "voxel" else "haar3d"}_checkpoint_epoch'] = int(best['epoch'])
    selected = read_csv('selected_examples.csv')
    delta = arrays['haar3d']['volume_mae'] - arrays['voxel']['volume_mae']
    ordered = [samples[index] for index in np.argsort(delta, kind='stable')]
    for row, rank in zip(selected, protocol['example_selection']['one_based_ranks'], strict=True):
        assert int(row['mae_delta_rank']) == rank and row['sample_id'] == ordered[rank - 1]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'recomputed_metrics.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, (model, history) in zip(axes, histories.items()):
        ax.plot([int(row['epoch']) for row in history], [float(row['fm_loss']) for row in history],
                color=COLORS[model], linewidth=1.5)
        ax.set(title=LABELS[model], xlabel='Epoch', ylabel='Training FM velocity MSE', yscale='log')
        ax.grid(alpha=0.2)
    fig.suptitle('Training-path losses; assess generation using held-out outputs', fontsize=12)
    fig.savefig(args.output / 'training_curves.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout='constrained')
    for model, history in histories.items():
        validated = [row for row in history if row['validation_mae']]
        epoch = protocol['models'][model]['checkpoint_epoch']
        selected_row = next(row for row in validated if int(row['epoch']) == epoch)
        for ax, key in zip(axes, ('mae', 'psnr')):
            x, y = [int(row['epoch']) for row in validated], [float(row[f'validation_{key}']) for row in validated]
            ax.plot(x, y, color=COLORS[model], marker='o', markersize=3, linewidth=1.4, label=LABELS[model])
            selected_y = float(selected_row[f'validation_{key}'])
            ax.scatter(epoch, selected_y, s=75, edgecolor='white', linewidth=1, zorder=5, color=COLORS[model])
            offset = (-125, 12) if epoch > 150 else (8, 10)
            if key == 'psnr' and epoch > 150:
                offset = (-125, -18)
            ax.annotate(f'MAE-selected epoch {epoch}', (epoch, selected_y), xytext=offset,
                        textcoords='offset points', color=COLORS[model], fontsize=9)
    for ax in axes:
        ax.axvline(100, color='#888888', linewidth=1, linestyle='--')
        ax.grid(alpha=0.2)
        ax.set_xlim(0, 210)
    axes[0].set_ylabel('Validation MAE (lower is better)')
    axes[1].set(ylabel='Validation PSNR / dB (higher is better)', xlabel='Epoch')
    axes[0].legend(frameon=False)
    fig.suptitle('OCTA-500 3M: 10 validation cases, evaluated every 5 epochs', fontsize=13)
    fig.savefig(args.output / 'validation_trajectory.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout='constrained')
    for ax, metric in zip(axes, ('mae', 'psnr')):
        b, w = arrays['voxel'][f'volume_{metric}'], arrays['haar3d'][f'volume_{metric}']
        low, high = min(b.min(), w.min()), max(b.max(), w.max())
        pad = (high - low) * 0.08
        ax.plot([low - pad, high + pad], [low - pad, high + pad], color='#777777', linestyle='--', linewidth=1)
        ax.scatter(b, w, color=COLORS['haar3d'], s=26, alpha=0.8)
        ax.set(xlim=(low - pad, high + pad), ylim=(low - pad, high + pad),
               xlabel=f'Voxel baseline {metric.upper()}', ylabel=f'3D Haar {metric.upper()}',
               title=f"{metric.upper()}: Haar improves {result['test'][metric]['haar3d_better_patients']}/50 cases")
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.2)
    fig.suptitle('Paired test results: baseline epoch 90 / Haar epoch 195', fontsize=13)
    fig.savefig(args.output / 'test_patient_scatter.png', dpi=160)
    plt.close(fig)
    print('Verified 50 paired test cases, 300 epoch records and five example-selection ranks.')
    print(f'Recomputed statistics and three charts: {args.output}')


if __name__ == '__main__':
    main()
