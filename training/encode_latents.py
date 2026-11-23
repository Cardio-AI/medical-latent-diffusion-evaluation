"""Encode image manifests as channel-first NIfTI latent arrays."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('xray', 'ct'), required=True)
    p.add_argument('--ids-csv', required=True)
    p.add_argument('--split', choices=('train', 'validation'), required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--ct-spatial-size', type=int, nargs=3, default=(512, 512, 256))
    p.add_argument('--overwrite', action='store_true')
    return p


def clean_state(payload):
    state = payload.get('state_dict', payload)
    return {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}


def main():
    args = parser().parse_args()
    import nibabel as nib
    import numpy as np
    import torch
    from monai.data import DataLoader, Dataset
    from omegaconf import OmegaConf
    from data import read_csv, encoding_transform
    from models.vqvae import VQVAE

    cfg = OmegaConf.load(args.config).stage1.params
    if int(cfg.spatial_dims) != (2 if args.dataset == 'xray' else 3):
        raise ValueError('dataset and model dimensionality disagree')
    items = read_csv(args.ids_csv)
    loader = DataLoader(Dataset(items, encoding_transform(args.dataset, args.ct_spatial_size)), batch_size=1, num_workers=args.workers, shuffle=False)
    device = torch.device(args.device)
    model = VQVAE(**cfg).to(device)
    model.load_state_dict(clean_state(torch.load(args.checkpoint, map_location='cpu', weights_only=False)), strict=True)
    model.eval()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows, names = [], set()
    for item, batch in zip(items, loader):
        path = Path(item['image']); name = path.name[:-7] if path.name.endswith('.nii.gz') else path.stem
        name += '_00.nii.gz'
        if name in names: raise ValueError(f'duplicate latent filename: {name}')
        names.add(name)
        target = out/name
        if target.exists() and not args.overwrite:
            raise FileExistsError(f'{target} already exists; use --overwrite')
        image = batch['image']
        image = image.as_tensor() if hasattr(image, 'as_tensor') else torch.as_tensor(image)
        with torch.inference_mode():
            latent = model.encode_stage_2_inputs(image.to(device))
        array = latent.detach().cpu().float().numpy()[0]
        affine = getattr(batch['image'], 'meta', {}).get('affine')
        if affine is None:
            affine = np.eye(4)
        else:
            affine = np.asarray(affine.cpu() if hasattr(affine, 'cpu') else affine)
            if affine.ndim == 3:
                affine = affine[0]
        nib.save(nib.Nifti1Image(array, affine), target)
        rows.append({'image': name, 'timepoint': item['timepoint']})
    with (out/f'{args.split}.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=('image', 'timepoint')); writer.writeheader(); writer.writerows(rows)
    print(f'encoded {len(rows)} images to {out}', flush=True)


if __name__ == '__main__':
    main()
