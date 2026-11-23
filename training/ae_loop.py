"""VQ-GAN update and validation functions."""
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.spectral_loss import JukeboxLoss
from tensorboardX import SummaryWriter
from tqdm import tqdm

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_epoch_vqgan(
    model: nn.Module,
    ema_model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    adv_weight: float,
    perceptual_weight: float,
    scaler_g: GradScaler,
    scaler_d: GradScaler,
    entropy_weight: float,
    spatial_dims: int = 2,
    spectral_weight: float = 1.0
) -> None:
    model.train()
    discriminator.train()

    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)
    jukebox_loss = JukeboxLoss(spatial_dims=spatial_dims) if spectral_weight else None
    perplexity_weight = 0.001
    pbar = tqdm(enumerate(loader), total=len(loader))
    for step, x in pbar:
        images = x["image"]
        images = (images.as_tensor() if hasattr(images, "as_tensor") else torch.as_tensor(images)).to(device)
        # GENERATOR
        optimizer_g.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):  # bf16 if available
            reconstruction, quantization_loss = model(images)
            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float()) if perceptual_loss is not None else reconstruction.new_zeros(())
            j_loss = jukebox_loss(reconstruction.float(), images.float()) if jukebox_loss is not None else reconstruction.new_zeros(())

            p_loss = torch.clamp(p_loss.float(), max=10.0)
            j_loss = torch.clamp(j_loss.float(), max=10.0)

            if adv_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1].float() #added .float() for 2D precision
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False).float() #added for 2D stability
            else:
                generator_loss = torch.tensor([0.0]).to(device)


            # ADDED ENTROPY LOSS TO ENFORCE DIVERSE CODEBOOK - ON EMA CLUSTERS FOR SMOOTH LEARNING

            cluster_size = model.quantizer.quantizer.ema_cluster_size
            smoothing = 0.5
            denom = cluster_size.sum() + smoothing * cluster_size.numel()

            if denom < 1e-12:
                entropy_loss = torch.zeros((), device=images.device, dtype=torch.float32)
            else:
                cluster_prob = (cluster_size + smoothing) / denom
                entropy = -(cluster_prob * torch.log(cluster_prob + 1e-12)).sum()
                entropy_loss = -entropy

            entropy_weight = entropy_weight

            loss = l1_loss + quantization_loss + perceptual_weight * p_loss + adv_weight * generator_loss + spectral_weight * j_loss + entropy_weight * entropy_loss

            loss = loss.mean()
            l1_loss = l1_loss.mean()
            p_loss = p_loss.mean()
            quantization_loss = quantization_loss.mean()
            g_loss = generator_loss.mean()
            j_loss = j_loss.mean()
            entropy_loss = entropy_loss.mean()


            losses = OrderedDict(
                loss=loss,
                l1_loss=l1_loss,
                p_loss=p_loss,
                quantization_loss=quantization_loss,
                g_loss=g_loss,
                j_loss=j_loss,
                entropy_loss=entropy_loss,

            )

        scaler_g.scale(losses["loss"]).backward()
        scaler_g.unscale_(optimizer_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.8) #changed from 0.8
        scaler_g.step(optimizer_g)
        ema_model.update(model)
        scaler_g.update()


        # DISCRIMINATOR
        if adv_weight > 0:
            optimizer_d.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=False):
                logits_fake = discriminator(reconstruction.contiguous().detach().float())[-1].float()
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True).float()

                logits_real = discriminator(images.contiguous().detach().float())[-1].float()
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True).float()

                discriminator_loss = ((loss_d_fake + loss_d_real) * 0.5).float()
                d_loss = (adv_weight * discriminator_loss).mean()

            scaler_d.scale(d_loss).backward()
            scaler_d.unscale_(optimizer_d)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.8)  # changed from 1.0
            scaler_d.step(optimizer_d)
            scaler_d.update()
        else:
            discriminator_loss = torch.zeros((), device=device, dtype=torch.float32)

        losses["d_loss"] = discriminator_loss

        writer.add_scalar("lr_g", get_lr(optimizer_g), epoch * len(loader)/2 + step)
        writer.add_scalar("lr_d", get_lr(optimizer_d), epoch * len(loader)/2 + step)
        for k, v in losses.items():
            writer.add_scalar(f"{k}", v.item(), epoch * len(loader) + step)

    pbar.set_postfix(
        {
            "epoch": epoch,
            "loss": f"{losses['loss'].item():.6f}",
            "l1_loss": f"{losses['l1_loss'].item():.6f}",
            "p_loss": f"{losses['p_loss'].item():.6f}",
            "g_loss": f"{losses['g_loss'].item():.6f}",
            "d_loss": f"{losses['d_loss'].item():.6f}",
            "lr_g": f"{get_lr(optimizer_g):.6f}",
            "lr_d": f"{get_lr(optimizer_d):.6f}",
        },
    )


@torch.no_grad()
def eval_vqgan(
    model: nn.Module,
    ema_model: nn.Module,
    discriminator: nn.Module,
    perceptual_loss: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    step: int,
    writer: SummaryWriter,
    adv_weight: float,
    perceptual_weight: float,
    entropy_weight: float,
    spatial_dims: int = 2,
    spectral_weight: float = 1.0
) -> float:
    model.eval()
    discriminator.eval()

    adv_loss = PatchAdversarialLoss(criterion="least_squares", no_activation_leastsq=True)
    jukebox_loss = JukeboxLoss(spatial_dims=spatial_dims) if spectral_weight else None

    total_losses = OrderedDict()

    for x in loader:
        images = x["image"]
        images = (images.as_tensor() if hasattr(images, "as_tensor") else torch.as_tensor(images)).to(device)

        # Forward pass
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            reconstruction, quantization_loss = ema_model(images)
            l1_loss = F.l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float()) if perceptual_loss is not None else reconstruction.new_zeros(())
            j_loss = jukebox_loss(reconstruction.float(), images.float()) if jukebox_loss is not None else reconstruction.new_zeros(())
            p_loss = torch.clamp(p_loss.float(), max=10.0)
            j_loss = torch.clamp(j_loss.float(), max=10.0)

            if adv_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1].float()
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False).float()
            else:
                generator_loss = torch.zeros((), device=device, dtype=torch.float32)

            if adv_weight > 0:
                logits_fake = discriminator(reconstruction.contiguous().detach().float())[-1].float()
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True).float()
                logits_real = discriminator(images.contiguous().detach().float())[-1].float()
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True).float()
                discriminator_loss = ((loss_d_fake + loss_d_real) * 0.5).float()
            else:
                discriminator_loss = torch.zeros((), device=device, dtype=torch.float32)

            # Entropy (safe version)
            cluster_size = model.quantizer.quantizer.ema_cluster_size
            smoothing = 0.5
            denom = cluster_size.sum() + smoothing * cluster_size.numel()
            if denom < 1e-12:
                entropy_loss = torch.zeros((), device=images.device, dtype=torch.float32)
            else:
                cluster_prob = (cluster_size + smoothing) / denom
                entropy = -(cluster_prob * torch.log(cluster_prob + 1e-8)).sum()
                entropy_loss = -entropy


            # Final loss
            loss = (
                l1_loss
                + quantization_loss
                + perceptual_weight * p_loss
                + adv_weight * generator_loss
                + spectral_weight * j_loss
                + entropy_weight * entropy_loss
            )

        # Aggregate
        losses = OrderedDict(
            loss=loss.mean(),
            l1_loss=l1_loss.mean(),
            p_loss=p_loss.mean(),
            quantization_loss=quantization_loss.mean(),
            g_loss=generator_loss.mean(),
            d_loss=discriminator_loss.mean(),
            j_loss=j_loss.mean(),
            entropy_loss=entropy_loss.mean(),
        )

        for k, v in losses.items():
            total_losses[k] = total_losses.get(k, 0.0) + v.item() * images.shape[0]

    # Normalize by dataset size
    for k in total_losses.keys():
        total_losses[k] /= len(loader.dataset)

    # Log to tensorboard
    for k, v in total_losses.items():
        writer.add_scalar(f"{k}", v, step)


    return total_losses["loss"]  # or "l1_loss" if you want that instead
