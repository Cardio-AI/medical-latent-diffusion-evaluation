"""Save VQ-GAN reconstructions and paired float arrays for evaluation."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('xray', 'ct'), required=True)
    p.add_argument('--ids-csv', required=True)
    p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--workers', type=int, default=1)
    p.add_argument('--ct-spatial-size', type=int, nargs=3, default=(512, 512, 256))
    p.add_argument('--overwrite', action='store_true')
    return p


def main():
    args = parser().parse_args()
    import nibabel as nib
    import numpy as np
    import torch
    from PIL import Image
    from monai.data import DataLoader, Dataset
    from omegaconf import OmegaConf
    from data import read_csv, encoding_transform
    from encode_latents import clean_state
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
    if (out/'paired.csv').exists() and not args.overwrite:
        raise FileExistsError(f'{out}/paired.csv already exists; use --overwrite')
    rows = []
    for index, (item, batch) in enumerate(zip(items, loader)):
        tensor = batch['image']
        affine = getattr(tensor, 'meta', {}).get('affine')
        if affine is None:
            affine = np.eye(4)
        else:
            affine = np.asarray(affine.cpu() if hasattr(affine, 'cpu') else affine)
            if affine.ndim == 3:
                affine = affine[0]
        tensor = tensor.as_tensor() if hasattr(tensor, 'as_tensor') else torch.as_tensor(tensor)
        with torch.inference_mode():
            reconstruction, _ = model(tensor.to(device))
        reference = tensor.detach().cpu().float().numpy()[0, 0]
        reconstruction = reconstruction.detach().cpu().float().numpy()[0, 0]
        array_name = f'pair_{index:04d}.npz'
        preview_name = f'reconstruction_{index:04d}.png' if args.dataset == 'xray' else f'reconstruction_{index:04d}.nii.gz'
        for target in (out/array_name, out/preview_name):
            if target.exists() and not args.overwrite: raise FileExistsError(target)
        np.savez_compressed(out/array_name, reference=reference, reconstruction=reconstruction)
        if args.dataset == 'xray':
            # PILReader's model-space array is transposed relative to the input PNG.
            preview = np.rint((np.clip(reconstruction.T, -1, 1)+1)*127.5).astype(np.uint8)
            Image.fromarray(preview).save(out/preview_name)
        else:
            nib.save(nib.Nifti1Image(reconstruction, affine), out/preview_name)
        rows.append({'array': array_name, 'image': item['image'], 'reconstruction': preview_name})
    with (out/'paired.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=('array', 'image', 'reconstruction'))
        writer.writeheader(); writer.writerows(rows)
    print(f'wrote {len(rows)} reconstruction pairs to {out}', flush=True)


if __name__ == '__main__':
    main()
