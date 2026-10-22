"""Explicit volume contract, subject-independent manifests, and BMP packaging."""
import argparse
import csv
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


COUNTS = {'3M': {'train': 140, 'validation': 10, 'test': 50},
          '6M': {'train': 180, 'validation': 20, 'test': 100}}
FIELDS = ['case_id', 'patient_id', 'path', 'split']


def read_rows(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Empty CSV')
    for row in rows:
        for key in ('case_id', 'patient_id', 'path'):
            if not row.get(key, '').strip():
                raise ValueError(f'Missing {key}; patient_id must come from verified metadata')
        row['path'] = str((path.parent / row['path']).resolve())
    return rows


def validate_manifest(rows, fov):
    if Counter(r.get('split') for r in rows) != Counter(COUNTS[fov]):
        raise ValueError(f'{fov} requires exactly {COUNTS[fov]}')
    subjects, cases, paths, content_groups = {}, set(), set(), {}
    for row in rows:
        cid, pid, phase = row['case_id'], row['patient_id'], row['split']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', cid):
            raise ValueError('case_id must use only letters, numbers, underscore or hyphen')
        path = Path(row['path']).resolve()
        if cid in cases or path in paths:
            raise ValueError(f'Duplicate case or file: {cid}')
        if pid in subjects and subjects[pid] != phase:
            raise ValueError(f'Patient crosses splits: {pid}')
        group = row.get('leakage_group', '').strip()
        if group:
            if group in content_groups and content_groups[group] != phase:
                raise ValueError(f'Duplicate image content crosses splits: {group}')
            content_groups[group] = phase
        if not path.is_file() or path.suffix.lower() != '.npz':
            raise ValueError(f'Missing NPZ: {path}')
        subjects[pid] = phase
        cases.add(cid)
        paths.add(path)


def load_pair(path, axis_order, intensity_scale, depth):
    """Resize in stored intensity units, then fixed-scale before paired cropping."""
    if sorted(axis_order) != ['X', 'Y', 'Z']:
        raise ValueError('axis_order must be a permutation of ZYX')
    if not math.isfinite(intensity_scale) or intensity_scale <= 0 or depth < 0:
        raise ValueError('intensity_scale must be finite and >0; depth must be >=0')
    tensors = []
    with np.load(path, allow_pickle=False) as data:
        if 'axis_order' in data and str(data['axis_order'].item()) != axis_order:
            raise ValueError(f'NPZ axis metadata conflicts with configuration: {path}')
        for key in ('oct', 'octa'):
            a = np.asarray(data[key], dtype=np.float32)
            if a.ndim != 3 or min(a.shape) < 8 or not np.isfinite(a).all():
                raise ValueError(f'{path}: {key} must be finite 3D with dimensions >=8')
            if a.min() < 0 or a.max() > intensity_scale:
                raise ValueError(f'{path}: {key} outside [0,{intensity_scale}]; verify scale')
            a = np.transpose(a, [axis_order.index(axis) for axis in 'ZYX']).copy()
            tensors.append(torch.from_numpy(a)[None])
    source, target = tensors
    if source.shape != target.shape:
        raise ValueError(f'Unaligned OCT/OCTA shapes: {path}')
    if depth and source.shape[1] != depth:
        size = (depth, *source.shape[2:])
        source, target = [F.interpolate(x[None], size=size, mode='trilinear',
                                       align_corners=False)[0] for x in tensors]
    return (source / intensity_scale).contiguous(), (target / intensity_scale).contiguous()


class VolumeDataset(Dataset):
    def __init__(self, rows, split, axis_order, intensity_scale, depth, patch=None):
        self.rows = [r for r in rows if r['split'] == split]
        if not self.rows:
            raise ValueError(f'No {split} cases')
        self.axis_order, self.intensity_scale, self.depth = axis_order, intensity_scale, depth
        self.patch = tuple(patch) if patch is not None else None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        source, target = load_pair(row['path'], self.axis_order, self.intensity_scale, self.depth)
        if self.patch is not None:
            if any(n < p for n, p in zip(source.shape[1:], self.patch)):
                raise ValueError(f'{row["case_id"]}: patch larger than volume; choose a smaller patch')
            starts = [random.randint(0, n - p) for n, p in zip(source.shape[1:], self.patch)]
            region = (slice(None),) + tuple(slice(s, s + p) for s, p in zip(starts, self.patch))
            source, target = source[region].contiguous(), target[region].contiguous()
        return {'oct': source, 'octa': target, 'case_id': row['case_id']}


def make_splits(inventory, output, fov, seed):
    rows = read_rows(inventory)
    if len(rows) != sum(COUNTS[fov].values()):
        raise ValueError(f'{fov} inventory requires {sum(COUNTS[fov].values())} cases')
    if any(r.get('split', '').strip() for r in rows):
        raise ValueError('Inventory already has split assignments; validate that manifest instead')
    if len({r['patient_id'] for r in rows}) != len(rows):
        raise ValueError('Repeated patients: assign grouped splits manually; do not split scans independently')
    if output.exists():
        raise FileExistsError(f'Will not replace fixed split: {output}')
    rows.sort(key=lambda r: r['case_id'])
    random.Random(seed).shuffle(rows)
    start = 0
    for split, count in COUNTS[fov].items():
        for row in rows[start:start + count]:
            row['split'] = split
        start += count
    validate_manifest(rows, fov)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row[k] for k in FIELDS},
                             'path': os.path.relpath(row['path'], output.parent.resolve())})
    output.with_suffix('.meta.json').write_text(json.dumps({
        'fov': fov, 'seed': seed, 'counts': COUNTS[fov],
        'protocol': 'User-defined random patient split; NOT the original paper IDs; no disease stratification'
    }, indent=2), encoding='utf-8')


def pack_bmp(oct_dir, octa_dir, output, axis_order):
    from PIL import Image
    if sorted(axis_order) != ['X', 'Y', 'Z']:
        raise ValueError('axis_order describes the verified [stack, row, column] axes')
    if output.suffix.lower() != '.npz' or output.exists():
        raise ValueError('output must be a new .npz file')
    names_a = {p.name for p in oct_dir.glob('*.bmp')}
    names_b = {p.name for p in octa_dir.glob('*.bmp')}
    if not names_a or names_a != names_b:
        raise ValueError('OCT/OCTA slice filenames must match exactly')
    def number(name):
        if not Path(name).stem.isdigit():
            raise ValueError('Use numeric BMP slice names; otherwise explicitly audit ordering first')
        return int(Path(name).stem)
    names = sorted(names_a, key=number)
    indices = [number(n) for n in names]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError('Duplicate or missing slice indices')
    arrays = []
    for folder in (oct_dir, octa_dir):
        slices = []
        for name in names:
            with Image.open(folder / name) as image:
                if image.mode != 'L':
                    raise ValueError(f'Expected native 8-bit grayscale mode L: {folder / name}; got {image.mode}')
                a = np.asarray(image).copy()
            if a.ndim != 2:
                raise ValueError(f'Native grayscale BMP required: {folder / name}')
            slices.append(a)
        volume = np.stack(slices)
        arrays.append(volume.transpose([axis_order.index(k) for k in 'ZYX']))
    if arrays[0].shape != arrays[1].shape or min(arrays[0].shape) < 8:
        raise ValueError('Expected paired volumes with equal shapes and dimensions >=8')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, oct=arrays[0], octa=arrays[1], axis_order=np.array('ZYX'),
                        slice_names=np.array(names))
    print(f'{output}: {arrays[0].shape}, axis_order=ZYX, no intensity/geometry changes')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    pack = sub.add_parser('pack')
    pack.add_argument('--oct-dir', type=Path, required=True)
    pack.add_argument('--octa-dir', type=Path, required=True)
    pack.add_argument('--output', type=Path, required=True)
    pack.add_argument('--axis-order', required=True, help='Verified axes of [stack,row,column], e.g. YZX')
    split = sub.add_parser('split')
    split.add_argument('--inventory', type=Path, required=True)
    split.add_argument('--output', type=Path, required=True)
    split.add_argument('--fov', choices=COUNTS, required=True)
    split.add_argument('--seed', type=int, default=2026)
    check = sub.add_parser('check')
    check.add_argument('--manifest', type=Path, required=True)
    check.add_argument('--fov', choices=COUNTS, required=True)
    check.add_argument('--axis-order', required=True)
    check.add_argument('--intensity-scale', type=float, required=True)
    args = parser.parse_args()
    if args.command == 'pack':
        pack_bmp(args.oct_dir, args.octa_dir, args.output, args.axis_order.upper())
    elif args.command == 'split':
        make_splits(args.inventory, args.output, args.fov, args.seed)
    else:
        rows = read_rows(args.manifest)
        validate_manifest(rows, args.fov)
        for row in rows:
            load_pair(row['path'], args.axis_order.upper(), args.intensity_scale, depth=0)
        print(f'Validated {len(rows)} paired volumes: {COUNTS[args.fov]}')


if __name__ == '__main__':
    main()
