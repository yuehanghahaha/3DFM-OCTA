"""Three isolated server diagnostics; never change the production runs."""
import argparse
import hashlib
import importlib
import json
import math
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parent
SERVER = Path('/docsystem/niuke/lvyuehang')
SEED = 20261021
TIMES = (0., .001, .01, .05, .1, .5, .9, 1.)


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def seed(value=SEED):
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def prepare():
    (ROOT / 'snapshots').mkdir(exist_ok=False)
    records = {}
    for arm, directory, run in (
        ('voxel', 'fm3d_ubuntu', 'fm3d_3M_run01'),
        ('haar', 'fm3d_wavelet_200', 'fm3d_haar3d_3M_200'),
    ):
        origin = SERVER / directory
        shutil.copytree(origin / 'fm3d', ROOT / 'source' / arm / 'fm3d',
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        checkpoint = ROOT / 'snapshots' / (arm + '.pt')
        shutil.copy2(origin / 'runs' / run / 'best.pt', checkpoint)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        records[arm] = dict(checkpoint=str(checkpoint), sha256=sha256(checkpoint),
                            original=str(origin / 'runs' / run / 'best.pt'),
                            epoch=payload['epoch'], config=payload['config'])
        if arm == 'voxel':
            manifest = payload['manifest']
        elif payload['manifest'] != manifest:
            raise ValueError('Checkpoint manifests differ')
    write_json(ROOT / 'manifest.json', manifest)
    write_json(ROOT / 'protocol.json', dict(
        created=stamp(), seed=SEED, physical_gpu=0, checkpoints=records,
        sample_selection='First two IDs in train and validation; each at Z=0,Y=X=64 and 120; no test cases',
        patch_ZYX=[256, 64, 64], times=TIMES, noise_seeds=[SEED, SEED+1, SEED+2],
        noise_path_scope='Out-of-training-distribution probe: these checkpoints were trained with OCT source, not Gaussian noise',
        supervised=dict(model='Plain 3D U-Net, base16, two downsamplings; OCT input, OCTA output; no time, wavelets or FM',
                        loss='Unclamped full-voxel MSE only', updates=2400,
                        lr='2e-4 through update1600; 5e-5 thereafter', optimizer='Adam, no weight decay',
                        precision='FP32; TF32 disabled', selection='Only first train ID, Y=X=64',
                        diagnostic_success='Training full-Z max-projection PCC>=0.90, SSIM>=0.80, MSE<=10% of target-mean predictor; descriptive threshold, not a generalization claim'),
        solver_scope='Direct fixed patches; Euler1/8/32, plus Euler128 and Heun8/32 on first train/validation patch; production sampler separately checked with an analytic oracle',
        projection_scope='Full-Z maximum intensity projection; not ILM-OPL or a vessel segmentation metric',
        source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'source').rglob('*.py'))},
    ))
    print(json.dumps({k: dict(epoch=v['epoch'], sha256=v['sha256']) for k, v in records.items()}), flush=True)


def module(arm, name):
    return importlib.import_module(f'source.{arm}.fm3d.{name}')


def encode(x, arm):
    return module('haar', 'wavelets').haar_dwt3d(x) if arm == 'haar' else x


def decode(x, arm):
    return module('haar', 'wavelets').haar_idwt3d(x) if arm == 'haar' else x


def load_model(arm, device):
    payload = torch.load(ROOT / 'snapshots' / (arm + '.pt'), map_location='cpu', weights_only=False)
    config = payload['config']
    mod = module(arm, 'model')
    model = mod.make_model(config['base_channels'], config['condition'], 'haar3d') if arm == 'haar' else mod.VelocityUNet3D(config['base_channels'], config['condition'])
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device).eval()


def load_samples():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    config = json.loads((ROOT / 'protocol.json').read_text())['checkpoints']['voxel']['config']
    loader = module('haar', 'data').load_pair
    samples, audit = [], []
    (ROOT / 'inputs').mkdir(exist_ok=True)
    for split in ('train', 'validation'):
        for row in sorted((r for r in manifest if r['split'] == split), key=lambda r: r['case_id'])[:2]:
            x, y = loader(row['path'], config['axis_order'], config['intensity_scale'], config['depth'])
            for loc in (64, 120):
                region = (slice(None), slice(None), slice(loc, loc+64), slice(loc, loc+64))
                a, b = x[region].contiguous()[None], y[region].contiguous()[None]
                assert a.shape == b.shape == (1, 1, 256, 64, 64)
                identity = f"{split}_{row['case_id']}_y{loc:03d}_x{loc:03d}"
                path = ROOT / 'inputs' / (identity + '.npz')
                if not path.exists():
                    np.savez_compressed(path, oct=a.numpy(), octa=b.numpy(), axis_order=np.array('BCZYX'))
                meta = dict(id=identity, case_id=row['case_id'], patient_id=row['patient_id'], split=split,
                            origin_ZYX=[0, loc, loc], file=str(path), sha256=sha256(path),
                            shape=list(a.shape), oct_range=[float(a.min()), float(a.max())],
                            octa_range=[float(b.min()), float(b.max())])
                samples.append(dict(meta=meta, x=a, y=b))
                audit.append(meta)
    write_json(ROOT / 'input_audit.json', audit)
    return samples


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    denominator = torch.sqrt(a.square().sum() * b.square().sum()).item()
    return float((a*b).sum().item() / denominator) if denominator > 1e-15 else None


def scores(prediction, target):
    prediction, target = prediction.detach().float().cpu(), target.detach().float().cpu()
    error = prediction.double() - target.double()
    a, b = prediction[0, 0].amax(0), target[0, 0].amax(0)
    mse = error.square().mean().item()
    high = target > .1
    return dict(mae=error.abs().mean().item(), mse=mse, psnr=-10*math.log10(max(mse, 1e-12)),
                max_projection_pcc=pcc(a, b),
                max_projection_ssim=float(structural_similarity(a.numpy(), b.numpy(), data_range=1., win_size=7)),
                high_signal_mae=error[high].abs().mean().item() if high.any() else None,
                fraction_outside_0_1=float(((prediction < 0) | (prediction > 1)).float().mean()))


def maps(output, prediction):
    output.parent.mkdir(exist_ok=True, parents=True)
    a = prediction.detach().float().cpu()[0, 0]
    np.savez_compressed(output, max_z=a.amax(0).numpy(), mean_z=a.mean(0).numpy(), bscan=a[:, 32].numpy())


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU(),
                                    nn.Conv3d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.SiLU())

    def forward(self, x):
        return self.layers(x)


class PlainUNet3D(nn.Module):
    def __init__(self, b=16):
        super().__init__()
        self.enc0, self.enc1, self.enc2 = Block(1, b), Block(b, b*2), Block(b*2, b*4)
        self.dec1, self.dec0 = Block(b*6, b*2), Block(b*3, b)
        self.output = nn.Conv3d(b, 1, 1)

    def forward(self, x):
        a = self.enc0(x)
        b = self.enc1(F.avg_pool3d(a, 2))
        c = self.enc2(F.avg_pool3d(b, 2))
        h = self.dec1(torch.cat((F.interpolate(c, size=b.shape[2:], mode='trilinear', align_corners=False), b), 1))
        h = self.dec0(torch.cat((F.interpolate(h, size=a.shape[2:], mode='trilinear', align_corners=False), a), 1))
        return self.output(h)


def supervised(samples, device):
    output = ROOT / 'experiment1_supervised'
    output.mkdir(exist_ok=False)
    seed()
    model = PlainUNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=0)
    x, y = (samples[0][key].to(device) for key in ('x', 'y'))
    constant = scores(torch.full_like(y, y.mean()), y)
    history = []
    tick = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(2401):
        if step == 0 or step % 200 == 0:
            model.eval()
            with torch.inference_mode():
                raw = model(x)
                record = dict(update=step, seconds=time.perf_counter()-tick,
                              raw=scores(raw, y), final_clamp=scores(raw.clamp(0, 1), y))
            history.append(record)
            write_json(output / 'history.json', history)
            maps(output / f'maps_{step:04d}.npz', raw.clamp(0, 1))
            print('SUPERVISED', step, json.dumps(record['raw']), 'seconds', round(record['seconds'], 1), flush=True)
            if step and step % 400 == 0:
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), update=step), output / f'update_{step:04d}.pt')
        if step == 2400:
            break
        model.train()
        for group in optimizer.param_groups:
            group['lr'] = 2e-4 if step < 1600 else 5e-5
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x).float(), y)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite supervised loss')
        loss.backward()
        if step % 100 == 0:
            grad_norm = math.sqrt(sum(p.grad.detach().double().square().sum().item() for p in model.parameters() if p.grad is not None))
            if not 0 < grad_norm < float('inf'):
                raise FloatingPointError('Missing or nonfinite gradient')
            print('GRADIENT', step, grad_norm, flush=True)
        optimizer.step()
    model.eval()
    evaluation = []
    with torch.inference_mode():
        for sample in samples:
            prediction = model(sample['x'].to(device)).cpu()
            evaluation.append(dict(**sample['meta'], raw=scores(prediction, sample['y']),
                                   final_clamp=scores(prediction.clamp(0, 1), sample['y']),
                                   trained_on_this_patch=sample is samples[0]))
            maps(output / (sample['meta']['id'] + '.npz'), prediction.clamp(0, 1))
    final = history[-1]['raw']
    passed = final['max_projection_pcc'] >= .9 and final['max_projection_ssim'] >= .8 and final['mse'] <= .1*constant['mse']
    write_json(output / 'result.json', dict(complete=True, updates=2400, successful_updates=2400,
                elapsed_seconds=time.perf_counter()-tick, parameter_count=sum(p.numel() for p in model.parameters()),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), constant_mean_predictor=constant,
                memorization_threshold_passed=passed, evaluation=evaluation,
                scope='One training patch, single seed. Other patches are diagnostic holdouts, not tuned on.'))


@torch.inference_mode()
def rollout(model, c, steps, method='euler'):
    state = c.clone()
    for step in range(steps):
        t = torch.full((1,), step/steps, device=c.device)
        v = model(state, t, c).float()
        if method == 'heun':
            v2 = model(state+v/steps, torch.full((1,), (step+1)/steps, device=c.device), c).float()
            state = state + (v+v2)/(2*steps)
        else:
            state = state + v/steps
    return state


@torch.inference_mode()
def velocity_probes(samples, device):
    output = ROOT / 'experiment2_velocity'
    output.mkdir(exist_ok=False)
    records = []
    for arm in ('voxel', 'haar'):
        model = load_model(arm, device)
        for sample in samples:
            x, y = (sample[key].to(device) for key in ('x', 'y'))
            c, truth = encode(x, arm), encode(y, arm)
            target_velocity = truth-c
            case = dict(arm=arm, **sample['meta'], teacher=[], noise_ood=[], solvers=[])
            for value in TIMES:
                state = (1-value)*c+value*truth
                velocity = model(state, torch.tensor([value], device=device), c).float()
                mse = F.mse_loss(velocity, target_velocity).item()
                endpoint = decode(state+(1-value)*velocity, arm)
                err_voxel = decode(velocity-target_velocity, arm)
                case['teacher'].append(dict(time=value, velocity_mse=mse,
                    zero_velocity_mse=target_velocity.square().mean().item(),
                    relative_rmse=math.sqrt(mse/max(target_velocity.square().mean().item(), 1e-15)),
                    velocity_pcc=pcc(velocity, target_velocity),
                    high_signal_velocity_mse=err_voxel[y>.1].square().mean().item(),
                    endpoint_raw=scores(endpoint, y), endpoint_clamp=scores(endpoint.clamp(0, 1), y),
                    target_contained_in_state=value>0))
                if sample['meta']['origin_ZYX'][1] == 64:
                    maps(output / f"{arm}_{sample['meta']['id']}_teacher_{value}.npz", endpoint.clamp(0, 1))
            for noise_seed in (SEED, SEED+1, SEED+2):
                generator = torch.Generator(device=device).manual_seed(noise_seed)
                noise_voxel = torch.randn(x.shape, generator=generator, device=device)
                noise = encode(noise_voxel, arm)
                v_truth = truth-noise
                for value in (.1, .5, .9):
                    state = (1-value)*noise+value*truth
                    velocity = model(state, torch.tensor([value], device=device), c).float()
                    mse = F.mse_loss(velocity, v_truth).item()
                    case['noise_ood'].append(dict(time=value, seed=noise_seed, velocity_mse=mse,
                        zero_velocity_mse=v_truth.square().mean().item(),
                        relative_rmse=math.sqrt(mse/max(v_truth.square().mean().item(), 1e-15)),
                        velocity_pcc=pcc(velocity, v_truth)))
            settings = [('euler', 1), ('euler', 8), ('euler', 32)]
            if sample['meta']['id'] in (samples[0]['meta']['id'], samples[4]['meta']['id']):
                settings += [('euler', 128), ('heun', 8), ('heun', 32)]
            for method, steps in settings:
                state = rollout(model, c, steps, method)
                prediction = decode(state, arm)
                case['solvers'].append(dict(method=method, steps=steps,
                    raw=scores(prediction, y), final_clamp=scores(prediction.clamp(0, 1), y)))
                maps(output / f"{arm}_{sample['meta']['id']}_{method}{steps}.npz", prediction.clamp(0, 1))
            records.append(case)
            write_json(output / 'result.json', dict(complete=False, cases=records))
            print('VELOCITY', arm, sample['meta']['id'],
                  [(r['time'], round(r['velocity_mse'], 8)) for r in case['teacher']], flush=True)
        del model
        torch.cuda.empty_cache()
    write_json(output / 'result.json', dict(complete=True, cases=records))


class KnownVelocity(nn.Module):
    def __init__(self, arm, sign=1):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.arm, self.sign = arm, sign
        self.times = []
        self.first_input_equals_source = None

    def forward(self, state, t, source):
        if self.first_input_equals_source is None:
            self.first_input_equals_source = torch.equal(state, source)
        self.times.append(float(t.item()))
        bias = torch.zeros_like(source)
        if self.arm == 'haar':
            bias[:, 0] = .1*math.sqrt(8)
        else:
            bias += .1
        return self.sign*(-.4*source+bias)


@torch.inference_mode()
def formula_audit(device):
    output = ROOT / 'experiment3_formulas'
    output.mkdir(exist_ok=False)
    seed()
    x = torch.rand(1, 1, 16, 16, 16, device=device)
    y = torch.rand_like(x)
    tests = []
    for arm in ('voxel', 'haar'):
        model_module = module(arm, 'model')
        c, truth = encode(x, arm), encode(y, arm)
        for value in (.0, .1, .5, .9, 1.):
            class Recorder(nn.Module):
                def forward(self, state, t, source):
                    self.state, self.condition, self.time = state.clone(), source.clone(), t.clone()
                    return truth-c
            recorder = Recorder()
            kwargs = dict(representation='haar3d') if arm == 'haar' else {}
            loss = model_module.flow_loss(recorder, x, y, torch.tensor([value], device=device), **kwargs)
            state_error = (recorder.state-((1-value)*c+value*truth)).abs().max().item()
            tests.append(dict(arm=arm, check='production_training_state_target_and_condition', time=value,
                              loss_with_exact_velocity=float(loss), max_state_error=state_error,
                              fixed_condition_matches_source=torch.equal(recorder.condition, c),
                              passed=float(loss)<1e-12 and state_error<1e-6 and torch.equal(recorder.condition, c)))
        inference = module(arm, 'inference')
        source = torch.linspace(0, .8, 18*21*23).reshape(1, 1, 18, 21, 23)
        expected = .6*source+.1
        for steps in (1, 8, 32):
            oracle = KnownVelocity(arm).to(device).train()
            kwargs = dict(representation='haar3d') if arm == 'haar' else {}
            generated = inference.generate_volume(oracle, source, [16, 16, 16], steps=steps,
                                                   overlap=.5, amp=False, **kwargs)
            error = (generated-expected).abs().max().item()
            actual_times = sorted(set(oracle.times))
            correct_times = np.allclose(actual_times, np.arange(steps)/steps, atol=1e-7, rtol=0)
            tests.append(dict(arm=arm, check='production_sampler_odd_shape_multiple_overlapping_windows',
                              steps=steps, max_endpoint_error=error, time_grid=actual_times,
                              starts_from_oct=oracle.first_input_equals_source,
                              training_mode_restored=oracle.training,
                              passed=error<1e-5 and bool(correct_times) and oracle.first_input_equals_source and oracle.training))
        reverse = KnownVelocity(arm, sign=-1).to(device)
        kwargs = dict(representation='haar3d') if arm == 'haar' else {}
        wrong = inference.generate_volume(reverse, source, [16, 16, 16], steps=8, overlap=.5, **kwargs)
        wrong_error = (wrong-expected).abs().max().item()
        tests.append(dict(arm=arm, check='negative_control_reversed_velocity_must_fail',
                          max_endpoint_error=wrong_error, passed=wrong_error>.1))
    recovered = decode(encode(x, 'haar'), 'haar')
    inverse_error = (recovered-x).abs().max().item()
    voxel_mse, coefficient_mse = (y-x).square().mean().item(), (encode(y, 'haar')-encode(x, 'haar')).square().mean().item()
    tests.append(dict(check='haar_inverse_and_mse_units', inverse_max_error=inverse_error,
                      voxel_mse=voxel_mse, coefficient_mse=coefficient_mse,
                      passed=inverse_error<1e-6 and abs(voxel_mse-coefficient_mse)<1e-6))
    # The picture's Gaussian path is algebraically consistent, but is not these checkpoints' training path.
    noise = torch.randn_like(x)
    for value in (.1, .5, .9):
        state = (1-value)*noise+value*y
        error = (state+(1-value)*(y-noise)-y).abs().max().item()
        tests.append(dict(check='screenshot_noise_path_endpoint_identity', time=value,
                          max_endpoint_error=error, passed=error<1e-6))
    write_json(output / 'result.json', dict(complete=True, all_passed=all(t['passed'] for t in tests),
                checks=tests, training='state=(1-t)*OCT+t*OCTA; velocity=OCTA-OCT; Haar is linear transform of both',
                inference='state0=OCT; t_k=k/N; state+=v/N; positive time direction; only final voxel clamp',
                screenshot='state=(1-t)*epsilon+t*OCTA; velocity=OCTA-epsilon; requires training for that source distribution'))
    if not all(t['passed'] for t in tests):
        raise AssertionError('Production formula audit failed; see experiment3_formulas/result.json')
    print('FORMULA AUDIT', len(tests), 'passed', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'run'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.phase == 'prepare':
        prepare()
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError('Server diagnostics require CUDA')
    device = torch.device('cuda:0')
    write_json(ROOT / 'status.json', dict(state='running', started=stamp(), stage='formula_audit',
               gpu=torch.cuda.get_device_name(), pytorch=torch.__version__))
    formula_audit(device)
    samples = load_samples()
    for sample in samples:
        maps(ROOT / 'reference_maps' / (sample['meta']['id']+'_oct.npz'), sample['x'])
        maps(ROOT / 'reference_maps' / (sample['meta']['id']+'_octa.npz'), sample['y'])
    write_json(ROOT / 'status.json', dict(state='running', updated=stamp(), stage='velocity_probes'))
    velocity_probes(samples, device)
    write_json(ROOT / 'status.json', dict(state='running', updated=stamp(), stage='supervised_memorization'))
    supervised(samples, device)
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    snapshots_unchanged = all(sha256(ROOT / 'snapshots' / (arm+'.pt')) == record['sha256'] for arm, record in protocol['checkpoints'].items())
    source_unchanged = all(sha256(ROOT / path) == digest for path, digest in protocol['source_hashes'].items())
    write_json(ROOT / 'final_audit.json', dict(snapshot_hashes_unchanged=snapshots_unchanged,
               copied_production_source_hashes_unchanged=source_unchanged,
               no_test_cases=all(s['meta']['split']!='test' for s in samples), completed=stamp()))
    write_json(ROOT / 'status.json', dict(state='complete', completed=stamp(), stages=3))
    print('ALL THREE DIAGNOSTICS COMPLETE', stamp(), flush=True)


if __name__ == '__main__':
    main()
