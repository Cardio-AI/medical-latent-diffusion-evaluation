"""Train the unconditional U-Net diffusion model on pre-encoded VQ-VAE latents."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-csv', required=True, help='CSV of channel-first NIfTI latent arrays')
    p.add_argument('--val-csv', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--eval-every', type=int, default=2)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--scale-factor', type=float, default=1.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', type=Path)
    p.add_argument('--disable-flash-attention', action='store_true', help='Use regular attention instead of xformers')
    return p


def main():
    args = parser().parse_args()
    if min(args.batch_size, args.epochs, args.eval_every) < 1 or args.workers < 0 or args.scale_factor <= 0:
        raise ValueError('epochs, evaluation interval and batch size must be positive; workers cannot be negative')
    import torch
    import torch.nn.functional as F
    from monai.utils import set_determinism
    from omegaconf import OmegaConf
    from tensorboardX import SummaryWriter
    from data import Latents
    from models.unet import DiffusionModelUNet
    from models.ddpm import DDPMScheduler

    set_determinism(args.seed)
    cfg = OmegaConf.load(args.config).ldm
    if args.disable_flash_attention:
        cfg.params.use_flash_attention = False
    device = torch.device(args.device)
    dims = int(cfg.params.spatial_dims)
    train = torch.utils.data.DataLoader(Latents(args.train_csv, dims), batch_size=args.batch_size,
                                        shuffle=True, num_workers=args.workers)
    val = torch.utils.data.DataLoader(Latents(args.val_csv, dims), batch_size=args.batch_size,
                                      shuffle=False, num_workers=args.workers)
    model = DiffusionModelUNet(**cfg.params).to(device)
    ema = copy.deepcopy(model).eval()
    scheduler = DDPMScheduler(**cfg.scheduler)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.base_lr))
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=.9995)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((out / name).exists() for name in ('checkpoint.pth', 'final_model.pth')):
        raise FileExistsError('output already contains trained weights; use a new directory or --resume')
    start, best = 0, float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['diffusion']); ema.load_state_dict(checkpoint['ema_model'])
        opt.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint: lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start, best = int(checkpoint['epoch']), float(checkpoint['best_loss'])

    def run_epoch(loader, training):
        model.train(training)
        total = 0.0
        for batch in loader:
            images = batch['image'].to(device) * args.scale_factor
            if images.shape[1] != int(cfg.params.in_channels):
                raise ValueError(f'latent channels {images.shape[1]} do not match model in_channels {cfg.params.in_channels}')
            labels = batch['timepoint'].to(device).long()
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()
            if training: opt.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training), torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                noise = torch.randn_like(images)
                noisy = scheduler.add_noise(original_samples=images, noise=noise, timesteps=timesteps)
                prediction = model(x=noisy, timesteps=timesteps, class_labels=labels)
                if scheduler.prediction_type == 'v_prediction':
                    target = scheduler.get_velocity(images, noise, timesteps)
                elif scheduler.prediction_type == 'epsilon':
                    target = noise
                else:
                    raise ValueError(f'unsupported training prediction_type {scheduler.prediction_type}')
                loss = F.smooth_l1_loss(prediction.float(), target.float())
            if training:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                with torch.no_grad():
                    for target_param, source_param in zip(ema.parameters(), model.parameters()):
                        target_param.mul_(.9999).add_(source_param.detach(), alpha=.0001)
            total += float(loss.detach()) * images.shape[0]
        return total / len(loader.dataset)

    with SummaryWriter(str(out/'train')) as tw, SummaryWriter(str(out/'val')) as vw:
        for epoch in range(start, args.epochs):
            loss = run_epoch(train, True); lr_scheduler.step(); tw.add_scalar('loss', loss, epoch+1)
            if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
                validation = run_epoch(val, False); vw.add_scalar('loss', validation, epoch+1)
                best = min(best, validation)
                torch.save({'epoch': epoch+1, 'diffusion': model.state_dict(), 'ema_model': ema.state_dict(),
                            'optimizer': opt.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'best_loss': best}, out/'checkpoint.pth')
                print(f'epoch={epoch+1} train_loss={loss:.6g} validation_loss={validation:.6g}', flush=True)
        torch.save(model.state_dict(), out/'final_model.pth')


if __name__ == '__main__':
    main()
