"""Reuse the previously frozen 8 subjects / 16 patches without reselection."""
import argparse
from pathlib import Path
import shutil

import numpy as np

from .utils import read_json, sha256, write_json


def prepare(previous, output):
    previous, output = Path(previous), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new experiment directory')
    selection = read_json(previous/'data/selection.json')
    selection['previous_selection_sha256'] = sha256(previous/'data/selection.json')
    selection['layers'] = ('Native 1-based 6YX. Training subjects: optional projection supervision; '
                           'validation subjects: post-hoc evaluation only. Never model inputs.')
    selection['vessel_mask_usage'] = 'Official vessel masks only for evaluation; not loss masks'
    for record in selection['patches']:
        source = previous/record['file']
        if sha256(source) != record['sha256']:
            raise ValueError(f'Frozen patch changed: {source}')
        destination = output/record['file']
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    write_json(output/'data/selection.json', selection)
    shutil.copy2(previous/'data/original_data_protocol.json', output/'data/original_data_protocol.json')
    return load_samples(output)


def load_samples(root):
    root = Path(root)
    selection = read_json(root/'data/selection.json')
    patients, paths, samples, hashes = {}, set(), [], {}
    for record in selection['patches']:
        if record['split'] not in ('train', 'validation'):
            raise ValueError('Small study must not load test subjects')
        pid, split = record['patient_id'], record['split']
        if pid in patients and patients[pid] != split:
            raise ValueError('Patient crosses splits')
        patients[pid] = split
        path = (root/record['file']).resolve()
        if path in paths or sha256(path) != record['sha256']:
            raise ValueError('Duplicate or changed patch')
        if record['sha256'] in hashes and hashes[record['sha256']] != split:
            raise ValueError('Identical patch crosses splits')
        hashes[record['sha256']] = split
        paths.add(path)
        with np.load(path, allow_pickle=False) as data:
            sample = {key: data[key].copy() for key in ('oct', 'octa', 'layers', 'vessel_mask')}
        for key in ('oct', 'octa'):
            a = sample[key]
            if list(a.shape) != selection['patch_ZYX'] or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
                raise ValueError('Expected finite normalized fixed-geometry patches')
        sample.update(record)
        samples.append(sample)
    return selection, samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    selection, samples = prepare(args.previous, args.output)
    print(f'Frozen {len(samples)} patches; original case selection and pixel hashes preserved.')

