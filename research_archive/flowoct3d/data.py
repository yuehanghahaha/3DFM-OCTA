"""Full-volume data adapter; retain original fixed splits and intensity units."""
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch

from fm3d.data import load_pair
from .losses import layer_masks


def read_layers(directory, case_id, lateral_shape, native_depth=640):
    if directory is None:
        return None
    path = Path(directory)/(case_id+'.mat')
    layers = np.asarray(loadmat(path)['Layer'], dtype=np.float32)
    if layers.shape != (6, *lateral_shape):
        raise ValueError(f'{case_id}: layer geometry does not match ZYX volume')
    # Validate without allocating the full volume mask.
    layer_masks(torch.from_numpy(layers)[None], 1, native_depth)
    return layers


def load_case(row, config, with_layers=True):
    source, target = load_pair(row['path'], config['axis_order'], config['intensity_scale'], config['depth'])
    sample = dict(oct=source[0].numpy(), octa=target[0].numpy(), case_id=row['case_id'],
                  patient_id=row['patient_id'], split=row['split'], disease=row.get('disease', 'unknown'),
                  patch_id='full_volume')
    if with_layers and config.get('layers_dir'):
        sample['layers'] = read_layers(config['layers_dir'], row['case_id'], source.shape[-2:],
                                       config['losses']['native_depth'])
    return sample


def crop_case(sample, patch, rng, representation):
    shape = sample['oct'].shape
    if patch[0] != shape[0] or any(p > n for p, n in zip(patch, shape)):
        raise ValueError('Training patch must span the entire preprocessed Z and fit the lateral FOV')
    stride = 2 if representation == 'haar3d' else 1
    y, x = [int(rng.integers((n-p)//stride+1))*stride for n, p in zip(shape[1:], patch[1:])]
    region = (slice(None), slice(y, y+patch[1]), slice(x, x+patch[2]))
    out = dict(sample)
    out.update(oct=sample['oct'][region].copy(), octa=sample['octa'][region].copy(), origin_ZYX=[0, y, x])
    if 'layers' in sample:
        out['layers'] = sample['layers'][region].copy()
    return out

