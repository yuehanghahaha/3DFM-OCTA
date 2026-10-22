"""Run only the two new FlowOCT arms; all existing six-arm training files stay frozen."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
import monai
import numpy as np
import torch
from engine import load_samples, save_checkpoint, seed_all, sha256, write_json
from flowoct_control import to_flowoct, evaluate_predictor, predict
from flowoct_velocity_only import ARMS, FlowOCTVelocityOnly


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def make_model(arm, cfg, device):
    seed_all(cfg['seed'])
    opt = SimpleNamespace(gpu_ids=[device.index or 0] if device.type == 'cuda' else [],
                          isTrain=True, lr=cfg['lr'], beta=cfg['beta'],
                          train_slices=cfg['train_slices'],
                          network_chunk=cfg['pixel_network_chunk'] if ARMS[arm] == 'pixel' else cfg['network_chunk'])
    return FlowOCTVelocityOnly(opt, ARMS[arm])


def signature(arm, cfg):
    paths = [ROOT / name for name in ('run_flowoct_ablation.py', 'flowoct_velocity_only.py',
                                     'engine.py', 'flowoct_control.py')]
    paths += sorted((ROOT / 'source/flowoct').glob('*.py'))
    record = {'arm': arm, 'config': cfg, 'selection_sha256': sha256(ROOT / 'data/selection.json'),
              'sources': {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}}
    return record, hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def evaluate(model, samples, mode, cfg, destination=None):
    previous = model.training
    model.eval()
    chunk = cfg['pixel_inference_chunk'] if model.representation == 'pixel' else cfg['inference_chunk']
    try:
        return evaluate_predictor(lambda source: predict(
            model, torch.from_numpy(source)[None, None].to(model.device), mode, chunk_size=chunk),
            samples, destination)
    finally:
        model.train(previous)


def run(args):
    torch.set_num_threads(2)
    cfg = read(ROOT / 'flowoct_ablation_config.json')
    assert cfg['step_seed_base'] == 20261022 * 10000
    assert cfg['network_chunk'] == 8 and cfg['inference_chunk'] == 16
    device = torch.device('cuda')
    selection = read(ROOT / 'data/selection.json')
    samples = load_samples(ROOT, selection)
    train = [s for s in samples if s['split'] == 'train']
    assert len(train) == 1 and train[0]['case_id'] == cfg['training_case']
    assert [s['case_id'] for s in samples if s['split'] == 'validation'] == cfg['validation_cases']
    source, target = [torch.from_numpy(train[0][key])[None, None].to(device) for key in ('oct', 'octa')]
    assert list(source.shape[2:]) == cfg['shape_ZYX']
    model = make_model(args.arm, cfg, device)
    model.set_input({'A': to_flowoct(source), 'B': to_flowoct(target)})
    if args.benchmark:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        tick, rows = time.perf_counter(), []
        for update in range(1, 3):
            seed_all(cfg['step_seed_base'] + update)
            model.forward()
            rows.append({'loss': float(model.loss.detach()), **model.get_current_loss()})
        torch.cuda.synchronize()
        update_seconds = (time.perf_counter() - tick) / 2
        chunk = cfg['pixel_inference_chunk'] if ARMS[args.arm] == 'pixel' else cfg['inference_chunk']
        probes = {}
        model.eval()
        for mode in ('native3', 'shared8'):
            prediction = predict(model, source[:, :, :, :chunk, :], mode, chunk_size=chunk)
            assert torch.isfinite(prediction).all() and prediction.min() >= 0 and prediction.max() <= 1
            probes[mode] = {'shape_ZYX': list(prediction.shape[2:]), 'finite': True}
        torch.cuda.synchronize()
        result = {'status': 'passed', 'arm': args.arm, 'shape_ZYX': cfg['shape_ZYX'],
                  'seconds_per_update': update_seconds, 'solver_probes': probes,
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                  'generator_parameters': sum(p.numel() for p in model.netG.parameters()),
                  'network_chunk': model.opt.network_chunk,
                  'sampled_Bscans': len(model.last_sampling['slice_indices']), 'amp': False,
                  'updates': rows}
        write_json(ROOT / 'verification' / ('benchmark_' + args.arm + '.json'), result)
        print(json.dumps(result), flush=True)
        return

    out = ROOT / args.arm / 'one_case'
    out.mkdir(parents=True, exist_ok=True)
    protocol, digest = signature(args.arm, cfg)
    latest = out / 'latest.pt'
    completed, elapsed = 0, 0.
    if latest.exists():
        if not args.resume:
            raise FileExistsError('Existing checkpoint; use --resume')
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        assert saved['signature_sha256'] == digest, 'Protocol, input or source changed'
        model.load_state_dict(saved['model'])
        model.optimizer_G.load_state_dict(saved['optimizer'])
        completed, elapsed = saved['completed_updates'], saved['elapsed_seconds']
        del saved
        # Records beyond the last durable checkpoint cannot silently be appended twice.
        if (out / 'steps.jsonl').exists():
            lines = [json.loads(line) for line in (out / 'steps.jsonl').read_text().splitlines()]
            if len(lines) != completed:
                raise RuntimeError('Partial records after checkpoint; preserve and reconcile before resume')
        if completed == cfg['updates'] and (out / 'final_metrics.json').exists() and (out / 'final_native3.json').exists():
            print('Already complete', args.arm, flush=True)
            return
    elif (out / 'steps.jsonl').exists():
        raise FileExistsError('Training records exist without a checkpoint')
    write_json(out / 'protocol.json', protocol)
    write_json(out / 'environment.json', {
        'torch': torch.__version__, 'monai': monai.__version__, 'numpy': np.__version__,
        'gpu': torch.cuda.get_device_name(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'generator_parameters': sum(p.numel() for p in model.netG.parameters()),
        'contrast_head_parameters': 0, 'amp': False, 'gradient_clipping': False,
        'representation': ARMS[args.arm], 'sampled_Bscans_per_update': 64,
        'network_chunk': model.opt.network_chunk,
        'inference_chunk': cfg['pixel_inference_chunk'] if ARMS[args.arm] == 'pixel' else cfg['inference_chunk'],
        'native_model_axes': 'BCYXZ', 'independent_OCT_condition': False})
    write_json(out / 'status.json', {'status': 'initial_evaluation', 'arm': args.arm, 'updates': completed,
                                    'target': cfg['updates']})
    if not (out / 'initial_metrics.json').exists():
        write_json(out / 'initial_metrics.json', evaluate(model, samples, 'shared8', cfg))
    tick = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model.train()
    for update in range(completed + 1, cfg['updates'] + 1):
        update_seed = cfg['step_seed_base'] + update
        seed_all(update_seed)
        model.forward()
        row = {'update': update, 'seed': update_seed, 'loss': float(model.loss.detach()),
               **model.get_current_loss(), **model.last_sampling}
        assert abs(row['loss'] - cfg['velocity_weight'] * row['vloss']) < 1e-6
        with (out / 'steps.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        if update % 10 == 0:
            status = {'status': 'running', 'arm': args.arm, 'updates': update, 'target': cfg['updates'],
                      'seconds': elapsed + time.perf_counter() - tick, 'last_loss': row['loss']}
            write_json(out / 'status.json', status)
            print(json.dumps(status), flush=True)
        if update % cfg['eval_every'] == 0 or update == cfg['updates']:
            report = evaluate(model, samples, 'shared8', cfg)
            write_json(out / f'metrics_{update:05d}.json', report)
            duration = elapsed + time.perf_counter() - tick
            payload = {'model': model.state_dict(), 'optimizer': model.optimizer_G.state_dict(),
                       'completed_updates': update, 'elapsed_seconds': duration,
                       'signature_sha256': digest, 'protocol': protocol}
            save_checkpoint(latest, payload)
            if update != cfg['updates']:
                save_checkpoint(out / f'checkpoint_{update:05d}.pt', payload)
            with (out / 'history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'update': update, 'seconds': duration,
                                         'summary': report['summary']}, allow_nan=False) + '\n')
            print(json.dumps({'evaluated': update, 'train': report['summary']['train']['metrics']['ilm_opl_pcc'],
                              'validation': report['summary']['validation']['metrics']['ilm_opl_pcc']}), flush=True)

    for mode, folder, metric in [('native3', 'predictions_native3', 'final_native3.json'),
                                 ('shared8', 'predictions', 'final_metrics.json')]:
        write_json(out / 'status.json', {'status': 'final_evaluation', 'arm': args.arm,
                                        'updates': cfg['updates'], 'target': cfg['updates'], 'solver': mode})
        report = evaluate(model, samples, mode, cfg, out / folder)
        report.update(arm=args.arm, completed_updates=cfg['updates'], solver=mode, amp_skipped=0,
                      scope='Same frozen central256 data; one train and four validation patients')
        write_json(out / metric, report)
    records = [json.loads(line) for line in (out / 'steps.jsonl').read_text().splitlines()]
    np.savez(out / 'schedule.npz',
             seeds=np.asarray([row['seed'] for row in records], dtype=np.int64),
             slice_indices=np.asarray([row['slice_indices'] for row in records], dtype=np.int64),
             slice_times=np.asarray([row['slice_times'] for row in records], dtype=np.float32))
    status = {'status': 'complete', 'arm': args.arm, 'updates': cfg['updates'],
              'attempts': cfg['updates'], 'amp_skipped': 0,
              'seconds': elapsed + time.perf_counter() - tick,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    write_json(out / 'status.json', status)
    print(json.dumps(status), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        write_json(ROOT / 'verification' / ('error_' + args.arm + '.json'),
                   {'error': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
