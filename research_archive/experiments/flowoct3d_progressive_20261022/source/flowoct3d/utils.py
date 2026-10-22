import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, payload):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def code_hashes():
    root = Path(__file__).resolve().parents[1]
    files = list((root/'flowoct3d').glob('*.py'))
    files += [root/'fm3d'/name for name in ('model.py', 'wavelets.py', 'inference.py', 'data.py')]
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(files)}


def apply_update(loss, model, optimizer, scaler):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=not scaler.is_enabled())
    scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() < scale, float(norm)

