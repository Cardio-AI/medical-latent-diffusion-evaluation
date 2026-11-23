"""Train the VQ-GAN first stage on PNG images or NIfTI volumes."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('xray', 'ct'), required=True)
    p.add_argument('--train-csv', required=True, help='CSV with image paths')
    p.add_argument('--val-csv', required=True, help='CSV with image paths')
    p.add_argument('--config', required=True)
    p.add_argument('--output', required=True, help='Output directory')
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=2)
    p.add_argument('--device', default='cuda')
    p.add_argument('--adv-weight', type=float, default=.01)
    p.add_argument('--perceptual-weight', type=float, default=.01)
    p.add_argument('--spectral-weight', type=float, default=1.)
    p.add_argument('--entropy-weight', type=float, default=0.)
    p.add_argument('--adv-start', type=int, default=0, help='First epoch with adversarial loss')
    p.add_argument('--warmup-epochs', type=int, default=0)
    p.add_argument('--ct-roi', type=int, nargs=3, help='Override progressive CT train crop')
    p.add_argument('--ct-val-roi', type=int, nargs=3, default=(256, 256, 192))
    p.add_argument('--resume', type=Path, help='Checkpoint created by this command')
    p.add_argument('--mlflow-experiment', help='Optionally log metrics and artifacts to configured MLflow')
    return p


class EMA:
    def __init__(self, model, decay):
        self.ema_model = copy.deepcopy(model).eval()
        self.decay = decay
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def update(self, model):
        import torch
        with torch.no_grad():
            ema_state = self.ema_model.state_dict()
            for key, source in model.state_dict().items():
                if ema_state[key].is_floating_point():
                    ema_state[key].mul_(self.decay).add_(source.detach(), alpha=1.0-self.decay)
                else:
                    ema_state[key].copy_(source)

    def __call__(self, model_input):
        return self.ema_model(model_input)


def ct_roi(epoch):
    if epoch < 50:
        return (64, 64, 64)
    if epoch < 100:
        return (128, 128, 128)
    return (192, 192, 128)


def main():
    args = parser().parse_args()
    if min(args.epochs, args.eval_every, args.batch_size) < 1 or args.workers < 0:
        raise ValueError('epochs, evaluation interval and batch size must be positive; workers cannot be negative')
    if args.dataset == 'ct' and any(v < 1 or v % 32 for v in (*tuple(args.ct_roi or ()), *tuple(args.ct_val_roi))):
        raise ValueError('CT crop sizes must be positive multiples of 32')
    import torch
    from generative.losses.perceptual import PerceptualLoss
    from generative.networks.nets.patchgan_discriminator import PatchDiscriminator
    from monai.utils import set_determinism
    from omegaconf import OmegaConf
    from tensorboardX import SummaryWriter
    from models.vqvae import VQVAE
    from ae_loop import train_epoch_vqgan, eval_vqgan
    from data import image_loaders

    config = OmegaConf.load(args.config)
    dims = int(config.stage1.params.spatial_dims)
    if dims != (2 if args.dataset == 'xray' else 3):
        raise ValueError('dataset and stage1 spatial_dims disagree')
    set_determinism(args.seed)
    device = torch.device(args.device)
    model = VQVAE(**config.stage1.params).to(device)
    discriminator = PatchDiscriminator(**config.discriminator.params).to(device)
    perceptual = PerceptualLoss(**config.perceptual_network.params).to(device) if args.perceptual_weight else None
    ema = EMA(model, float(config.stage1.get('ema_decay', .999)))
    opt_g = torch.optim.AdamW(model.parameters(), lr=float(config.stage1.base_lr))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=float(config.stage1.disc_lr))
    sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=.999)
    sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=.999)
    scaler_g = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    scaler_d = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((out / name).exists() for name in ('checkpoint.pth', 'final_model.pth')):
        raise FileExistsError('output already contains trained weights; use a new directory or --resume')
    start, best = 0, float('inf')
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        ema.ema_model.load_state_dict(checkpoint['ema_model'])
        opt_g.load_state_dict(checkpoint['optimizer_g']); opt_d.load_state_dict(checkpoint['optimizer_d'])
        if 'scheduler_g' in checkpoint: sch_g.load_state_dict(checkpoint['scheduler_g'])
        if 'scheduler_d' in checkpoint: sch_d.load_state_dict(checkpoint['scheduler_d'])
        start, best = int(checkpoint['epoch']), float(checkpoint['best_loss'])
    with SummaryWriter(str(out/'train')) as tw, SummaryWriter(str(out/'val')) as vw:
        last_roi, train, val = None, None, None
        for epoch in range(start, args.epochs):
            roi = tuple(args.ct_roi) if args.ct_roi else ct_roi(epoch)
            if train is None or (args.dataset == 'ct' and roi != last_roi):
                train, val = image_loaders(args.train_csv, args.val_csv, args.dataset, args.batch_size,
                                           args.workers, roi if args.dataset == 'ct' else None,
                                           tuple(args.ct_val_roi) if args.dataset == 'ct' else None)
                last_roi = roi
            warmup = min(1.0, (epoch + 1) / args.warmup_epochs) if args.warmup_epochs else 1.0
            adv = args.adv_weight * warmup if epoch >= args.adv_start else 0.0
            perc = args.perceptual_weight * warmup
            train_epoch_vqgan(model, ema, discriminator, perceptual, train, opt_g, opt_d, device, epoch, tw,
                              adv, perc, scaler_g, scaler_d, args.entropy_weight, dims, args.spectral_weight)
            sch_g.step(); sch_d.step()
            if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
                loss = eval_vqgan(model, ema, discriminator, perceptual, val, device, (epoch+1)*len(train), vw,
                                  adv, perc, args.entropy_weight, dims, args.spectral_weight)
                best = min(best, loss)
                checkpoint = {'epoch': epoch+1, 'state_dict': model.state_dict(), 'ema_model': ema.ema_model.state_dict(),
                              'discriminator': discriminator.state_dict(), 'optimizer_g': opt_g.state_dict(),
                              'optimizer_d': opt_d.state_dict(), 'scheduler_g': sch_g.state_dict(),
                              'scheduler_d': sch_d.state_dict(), 'best_loss': best}
                torch.save(checkpoint, out/'checkpoint.pth')
                print(f'epoch={epoch+1} validation_loss={loss:.6g}', flush=True)
        torch.save(model.state_dict(), out/'final_model.pth')
    if args.mlflow_experiment:
        import mlflow
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run():
            mlflow.log_params({'dataset': args.dataset, 'epochs': args.epochs, 'batch_size': args.batch_size})
            mlflow.log_metric('best_validation_loss', best)
            mlflow.log_artifact(str(out/'final_model.pth'))


if __name__ == '__main__':
    main()
