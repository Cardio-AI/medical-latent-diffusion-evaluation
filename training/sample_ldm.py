"""Generate X-ray PNGs or CT NIfTI volumes with a VQ-GAN and a U-Net LDM."""
from __future__ import annotations

import argparse
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('xray', 'ct'), required=True)
    p.add_argument('--ae-config', required=True); p.add_argument('--ae-checkpoint', required=True)
    p.add_argument('--ldm-config', required=True); p.add_argument('--ldm-checkpoint', required=True)
    p.add_argument('--latent-shape', type=int, nargs='+', required=True, help='C H W [D]')
    p.add_argument('--output', required=True)
    p.add_argument('--count', type=int, default=1); p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--steps', type=int, default=300); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--scale-factor', type=float, default=1.)
    p.add_argument('--device', default='cuda')
    p.add_argument('--xray-orientation', choices=('legacy', 'input'), default='legacy',
                   help='Legacy clockwise rotation or transpose to match input PNG orientation')
    p.add_argument('--disable-flash-attention', action='store_true', help='Use regular attention instead of xformers')
    return p


def xray_pixels(array, orientation):
    import numpy as np
    pixels = np.rint((np.clip(array, -1, 1) + 1) * 127.5).astype(np.uint8)
    pixels = pixels.T if orientation == 'input' else np.rot90(pixels, k=-1)
    return np.ascontiguousarray(pixels)


def state_dict(payload):
    if isinstance(payload, dict):
        for key in ('state_dict', 'diffusion'):
            if key in payload and isinstance(payload[key], dict):
                payload = payload[key]; break
    return {k[7:] if k.startswith('module.') else k: v for k, v in payload.items()}


def main():
    args = parser().parse_args()
    import nibabel as nib
    import numpy as np
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    from models.vqvae import VQVAE
    from models.unet import DiffusionModelUNet
    from models.ddpm import DDPMScheduler

    ae_cfg = OmegaConf.load(args.ae_config).stage1.params
    ldm_cfg = OmegaConf.load(args.ldm_config).ldm
    if args.disable_flash_attention:
        ldm_cfg.params.use_flash_attention = False
    dims = 2 if args.dataset == 'xray' else 3
    if int(ae_cfg.spatial_dims) != dims or int(ldm_cfg.params.spatial_dims) != dims:
        raise ValueError('dataset and model dimensionality disagree')
    if (len(args.latent_shape) != dims+1 or args.latent_shape[0] != int(ldm_cfg.params.in_channels)
            or args.latent_shape[0] != int(ae_cfg.get('embedding_dim', 64))):
        raise ValueError('latent-shape must be C H W [D] and match LDM channels')
    if min(args.count, args.batch_size, args.steps, *args.latent_shape) < 1 or args.scale_factor <= 0:
        raise ValueError('count, batch size, steps, shape and scale factor must be positive')
    device = torch.device(args.device)
    ae = VQVAE(**ae_cfg).to(device).eval(); ae.load_state_dict(state_dict(torch.load(args.ae_checkpoint, map_location='cpu', weights_only=False)))
    model = DiffusionModelUNet(**ldm_cfg.params).to(device).eval(); model.load_state_dict(state_dict(torch.load(args.ldm_checkpoint, map_location='cpu', weights_only=False)))
    scheduler = DDPMScheduler(**ldm_cfg.scheduler)
    if args.steps != scheduler.num_train_timesteps:
        raise ValueError('this DDPM implementation requires one inference step per training timestep')
    scheduler.set_timesteps(args.steps)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    suffix = '.png' if dims == 2 else '.nii.gz'
    if any((out / f'sample_{i:04d}{suffix}').exists() for i in range(args.count)):
        raise FileExistsError('sample outputs already exist; use a new directory')
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    with torch.inference_mode():
        for start in range(0, args.count, args.batch_size):
            n = min(args.batch_size, args.count-start)
            latent = torch.randn((n, *args.latent_shape), generator=generator).to(device)
            for t in scheduler.timesteps:
                timestep = int(t)
                pred = model(x=latent, timesteps=torch.full((n,), timestep, device=device, dtype=torch.long))
                latent = scheduler.step(pred, timestep, latent, generator=generator)[0]
            decoded = ae.decode_stage_2_outputs(latent / args.scale_factor).detach().float().clamp(-1, 1).cpu().numpy()[:, 0]
            for offset, array in enumerate(decoded):
                if dims == 2:
                    image = xray_pixels(array, args.xray_orientation)
                    Image.fromarray(image).save(out/f'sample_{start+offset:04d}.png')
                else:
                    image = np.rint(array*1000).astype(np.int16)
                    nib.save(nib.Nifti1Image(image, np.eye(4)), out/f'sample_{start+offset:04d}.nii.gz')
    print(f'generated {args.count} samples in {out}', flush=True)


if __name__ == '__main__':
    main()
