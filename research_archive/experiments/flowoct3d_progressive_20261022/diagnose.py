"""Measure endpoint/free-generation gaps, detail attenuation and weighted gradients."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source'))
import numpy as np
import torch
from flowoct3d.losses import LossConfig, objectives
from flowoct3d.metrics import sample_metrics
from flowoct3d.model import FlowOCT3D
from flowoct3d.small_data import load_samples
from flowoct3d.utils import seed_all, sha256, write_json
from fm3d.wavelets import haar_dwt3d


def quality(model, sample):
    device = next(model.parameters()).device
    source, target = [torch.from_numpy(sample[k])[None, None].to(device) for k in ('oct', 'octa')]
    output = []
    with torch.inference_mode():
        condition, truth = model.encode(source), model.encode(target)
        for name, value in (('pure_oct_t0_endpoint', 0.), ('target_assisted_t05_endpoint', .5), ('pure_oct_euler8', None)):
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                if value is None:
                    raw = model.rollout(source, 8)
                    velocity_mse = None
                else:
                    state = (1-value)*condition+value*truth
                    v = model(state, torch.full((1,), value, device=device), condition)
                    raw = model.decode(state+(1-value)*v.float())
                    velocity_mse = float((v.float()-(truth-condition)).square().mean())
            prediction = raw.clamp(0, 1)[0, 0].float().cpu().numpy()
            metrics, _ = sample_metrics(prediction, sample)
            a, b = [haar_dwt3d(v).float()[:, 1:] for v in (raw.clamp(0, 1), target)]
            hf_ratio = float(a.square().mean()/b.square().mean().clamp_min(1e-12))
            output.append(dict(mode=name, target_in_state=value == .5, velocity_mse=velocity_mse,
                ilm_opl_pcc=metrics['ilm_opl_pcc'], ilm_opl_ssim=metrics['ilm_opl_ssim'],
                hf_energy_ratio=hf_ratio, prediction_std=float(prediction.std()), target_std=float(sample['octa'].std())))
    return output


def gradients(model, sample, config):
    device = next(model.parameters()).device
    source, target = [torch.from_numpy(sample[k])[None, None].to(device) for k in ('oct', 'octa')]
    layers = torch.from_numpy(sample['layers'])[None].to(device)
    parameters = [p for p in model.velocity.parameters() if p.requires_grad]
    results = []
    for value in (0., .5):
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
            total, parts = objectives(model, source, target, torch.tensor([value], device=device), config, layers)
        weights = dict(fm=config.fm_weight, reconstruction=config.reconstruction_weight,
                       projection=config.projection_weight, rdca=config.rdca_weight)
        vectors = {}
        terms = {}
        for name, weight in weights.items():
            loss = parts[name]*weight
            grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
            vector = torch.cat([(g.float() if g is not None else torch.zeros_like(p)).flatten()
                                for g, p in zip(grads, parameters)])
            vectors[name] = vector.detach()
            terms[name] = dict(raw_loss=float(parts[name].detach()), weighted_loss=float(loss.detach()),
                               generator_gradient_norm=float(vector.norm()))
        pairs = {}
        names = list(vectors)
        for i, a in enumerate(names):
            for b in names[i+1:]:
                denominator = vectors[a].norm()*vectors[b].norm()
                pairs[a+'__'+b] = float(torch.dot(vectors[a], vectors[b])/denominator) if denominator > 1e-12 else None
        results.append(dict(time=value, terms=terms, gradient_cosines=pairs))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--new-final', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    _, samples = load_samples(ROOT)
    selected = [s for s in samples if s['patch_id'] == 'y064_x064']
    previous = ROOT.parents[1]/'experiments/flowoct3d_full_20261022'
    checks = [('prior_full_800', previous/'03_full/few_case/latest.pt', 'haar3d'),
              ('prior_haar_800', previous/'02_haar_fm/few_case/latest.pt', 'haar3d')]
    if args.new_final:
        checks = [('new_full_2400', ROOT/'01_full3d/latest.pt', 'haar3d'),
                  ('new_half_start_2400', ROOT/'07_full_half_start/latest.pt', 'haar3d')]
    records = []
    for name, path, representation in checks:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        seed_all(20261021)
        model = FlowOCT3D(16, True, representation).cuda().eval()
        model.load_state_dict(saved['model'])
        for sample in selected:
            row = dict(model=name, checkpoint_sha256=sha256(path), case_id=sample['case_id'],
                       split=sample['split'], quality=quality(model, sample))
            if sample['split'] == 'train' and 'full' in name:
                row['weighted_gradients'] = gradients(model, sample, LossConfig())
            records.append(row)
            print(name, sample['case_id'], sample['split'], flush=True)
        del model
        torch.cuda.empty_cache()
    write_json(ROOT/'verification'/('diagnosis_final.json' if args.new_final else 'diagnosis_prior800.json'), dict(
        records=records, scope='Fixed first patch of each frozen train/validation patient. Target-assisted estimates are diagnostic only. Gradient norms use loss weights and generator parameters; negative cosine indicates local conflict, not proof of training failure.'))


if __name__ == '__main__':
    main()

