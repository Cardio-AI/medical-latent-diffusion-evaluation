"""CSV-based image and precomputed-latent input for the training commands."""
from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai import transforms
from monai.data import DataLoader, Dataset
from monai.transforms import MapTransform


class SelectFirstChanneld(MapTransform):
    def __call__(self, data):
        result = dict(data)
        for key in self.keys:
            result[key] = result[key][:1]
        return result


def read_csv(path):
    path = Path(path).resolve()
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or 'image' not in reader.fieldnames:
            raise ValueError(f'{path} must contain an image column')
        rows = []
        for row in reader:
            image = Path(row['image']).expanduser()
            if not image.is_absolute():
                image = path.parent / image
            if not image.is_file():
                raise FileNotFoundError(image)
            rows.append({'image': str(image.resolve()), 'timepoint': int(row.get('timepoint') or 0)})
    if not rows:
        raise ValueError(f'{path} contains no images')
    return rows


def ae_transform(dataset, train, roi=None):
    ops = [transforms.LoadImaged(keys=['image']), transforms.EnsureChannelFirstd(keys=['image'])]
    if dataset == 'xray':
        ops += [SelectFirstChanneld(keys=['image']), transforms.ScaleIntensityd(keys=['image'], minv=-1., maxv=1.)]
        if train:
            ops += [transforms.RandRotated(keys=['image'], range_x=0.0, range_y=0.0872665, prob=0.2), transforms.RandFlipd(keys=['image'], spatial_axis=1, prob=0.5)]
    elif dataset == 'ct':
        ops += [transforms.ThresholdIntensityd(keys=['image'], threshold=1000, above=False, cval=1000),
                transforms.ThresholdIntensityd(keys=['image'], threshold=-1000, above=True, cval=-1000),
                transforms.ScaleIntensityd(keys=['image'], minv=-1., maxv=1.)]
        if train:
            ops += [transforms.RandRotated(keys=['image'], range_x=0.0872665, range_y=0.0872665, range_z=0.0872665, prob=0.2),
                    transforms.RandFlipd(keys=['image'], spatial_axis=1, prob=0.5)]
        if roi is not None:
            ops += [transforms.SpatialPadd(keys=['image'], spatial_size=roi),
                    transforms.RandSpatialCropd(keys=['image'], roi_size=roi, random_size=False)]
    else:
        raise ValueError(dataset)
    ops += [transforms.ToTensord(keys=['image'])]
    return transforms.Compose(ops)


def encoding_transform(dataset, ct_spatial_size=(512, 512, 256)):
    ops = [transforms.LoadImaged(keys=['image']), transforms.EnsureChannelFirstd(keys=['image'])]
    if dataset == 'xray':
        ops += [SelectFirstChanneld(keys=['image']), transforms.ScaleIntensityd(keys=['image'], minv=-1., maxv=1.)]
    elif dataset == 'ct':
        size = tuple(ct_spatial_size)
        if len(size) != 3 or any(v < 1 or v % 32 for v in size):
            raise ValueError('CT spatial dimensions must be positive multiples of 32')
        ops += [transforms.ThresholdIntensityd(keys=['image'], threshold=1000, above=False, cval=1000),
                transforms.ThresholdIntensityd(keys=['image'], threshold=-1000, above=True, cval=-1000),
                transforms.ScaleIntensityd(keys=['image'], minv=-1., maxv=1.),
                transforms.CenterSpatialCropd(keys=['image'], roi_size=size),
                transforms.SpatialPadd(keys=['image'], spatial_size=size, mode='constant', constant_values=-1.)]
    else:
        raise ValueError(dataset)
    return transforms.Compose([*ops, transforms.ToTensord(keys=['image'])])


def image_loaders(train_csv, validation_csv, dataset, batch_size, workers, train_roi=None, val_roi=None):
    train = Dataset(data=read_csv(train_csv), transform=ae_transform(dataset, True, train_roi))
    valid = Dataset(data=read_csv(validation_csv), transform=ae_transform(dataset, False, val_roi))
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=workers, persistent_workers=workers > 0),
            DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=workers, persistent_workers=workers > 0))


class Latents(torch.utils.data.Dataset):
    def __init__(self, manifest, spatial_dims):
        self.rows = read_csv(manifest)
        self.spatial_dims = spatial_dims
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        array = np.asarray(nib.load(row['image']).dataobj, dtype=np.float32)
        if array.ndim != self.spatial_dims + 1:
            raise ValueError(f"{row['image']}: expected channel-first {self.spatial_dims + 1}D latent; got {array.shape}")
        return {'image': torch.from_numpy(np.ascontiguousarray(array)), 'timepoint': row['timepoint']}
