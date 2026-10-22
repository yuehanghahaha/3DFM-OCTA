"""Export only the chosen split's layer labels/native projections/official BMPs."""
import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fm3d.projection_report import project


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--split', choices=['validation', 'test'], required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with (args.data / 'split_3M_z256.csv').open(encoding='utf-8-sig') as file:
        rows = [r for r in csv.DictReader(file) if r['split'] == args.split]
    args.output.mkdir(parents=True)
    records = []
    for row in rows:
        case = str(row['case_id'])
        path = args.data / 'labels_native/GT_Layers' / f'{case}.mat'
        layers = loadmat(path)['Layer']
        with np.load(args.data / 'native/3M' / f'{case}.npz', allow_pickle=False) as data:
            volume = data['octa'].astype(np.float32) / 255
        native, empty = project(volume, layers)
        arrays = {'layers': layers, **{f'native_{key}': value for key, value in native.items()}}
        hashes = {'layers': hashlib.sha256(path.read_bytes()).hexdigest()}
        for key in native:
            path = args.data / f'official_projections/3M/OCTA({key})/{case}.bmp'
            with Image.open(path) as im:
                arrays[f'official_{key}'] = np.asarray(im.convert('L'))
            hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        np.savez_compressed(args.output / f'{case}.npz', **arrays)
        records.append({'case_id': case, 'source_sha256': hashes, 'native_empty_rays': empty})
        print(f'{len(records)}/{len(rows)}: {case}', flush=True)
    metadata = {'split': args.split, 'case_ids': [r['case_id'] for r in rows], 'cases': records,
                'source': str(args.data.resolve()), 'source_native_depth': 640,
                'layer_order': ['ILM', 'IPL', 'OPL', 'ISOS', 'RPE', 'BM'],
                'layer_coordinates': 'MATLAB 1-based, shape 6YX; inclusive boundaries',
                'orientation': 'Projection arrays XY, matching official BMP',
                'usage': 'Ground-truth annotations for post-hoc evaluation only, not generator inputs'}
    (args.output / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    with zipfile.ZipFile(args.output.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in args.output.iterdir():
            archive.write(path, path.name)


if __name__ == '__main__':
    main()
