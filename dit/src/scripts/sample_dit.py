"""
Unconditional sampling from a trained DiT3D Latent Diffusion Model.
"""

import argparse
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch
import torch.multiprocessing as mp
import numpy as np
import nibabel as nib
from omegaconf import OmegaConf
from PIL import Image
from torch import amp

from src.models.dit import DiT2D, DiT3D
from src.models.vqvae import VQVAE
from src.models.load_stage1 import load_stage1_weights
from src.models.ddimscheduler import DDIMScheduler
from src.models.ddpmscheduler import DDPMScheduler
from src.config_utils import get_dit_params, get_dit_scheduler, get_stage1_params


def parse_args():
    parser = argparse.ArgumentParser(description="Sample from a trained DiT3D LDM")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage1_cfg", type=str, required=True)
    parser.add_argument("--diff_cfg", type=str, required=True)
    parser.add_argument("--diff_ckpt", type=str, default=None,
                        help="Path to a single DiT checkpoint")
    parser.add_argument("--diff_run_dir", type=str, default=None,
                        help="Directory containing checkpoint_epoch_N.pth files")
    parser.add_argument("--epoch_start", type=int, default=None)
    parser.add_argument("--epoch_end", type=int, default=None)
    parser.add_argument("--epoch_step", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="samples")
    parser.add_argument("--n_samples", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--scheduler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--reference_nii", type=str, default=None,
                        help="Reference .nii.gz to copy affine from")
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--latent_shape", type=int, nargs="+", required=True,
                        help="Latent spatial dimensions; two values for 2D or three for 3D")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of latents to denoise in parallel before decoding one-by-one.")
    parser.add_argument("--seed_base", type=int, default=2026082400,
                        help="Sample i always uses seed_base+i at every checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--xray-orientation", choices=("legacy", "input"), default="legacy",
                        help="Legacy clockwise rotation or transpose to match input PNG orientation")
    args = parser.parse_args()

    if args.diff_ckpt is None and args.diff_run_dir is None:
        parser.error("Specify either --diff_ckpt or --diff_run_dir with --epoch_start/--epoch_end")
    if args.diff_run_dir is not None and (args.epoch_start is None or args.epoch_end is None):
        parser.error("--diff_run_dir requires --epoch_start and --epoch_end")

    return args


def load_stage1(cfg_path, ckpt_path, device):
    cfg = OmegaConf.load(cfg_path)
    model = VQVAE(**get_stage1_params(cfg))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    load_stage1_weights(model, ckpt)

    return model.to(device).eval().requires_grad_(False)


def load_dit(cfg_path, ckpt_path, device):
    cfg = OmegaConf.load(cfg_path)
    model_class = DiT2D if int(cfg.get("spatial_dims", 3)) == 2 else DiT3D
    model = model_class(**get_dit_params(cfg))

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # Start with the full model state (includes non-trainable params like pos_embed),
    # then override trainable parameters with EMA shadow weights if available.
    state_dict = ckpt.get("model", ckpt)
    ema = ckpt.get("ema")
    if ema is not None:
        state_dict = dict(state_dict)  # copy so we don't mutate the checkpoint
        state_dict.update(ema["shadow"])
    model.load_state_dict(state_dict)

    return model.to(device).eval().requires_grad_(False), cfg


def _sample_path(out_dir, index, spatial_dims):
    suffix = ".png" if spatial_dims == 2 else ".nii.gz"
    return out_dir / f"sample_{index:04d}{suffix}"


def _seeded_noise(indices, in_channels, latent_shape, device, seed_base):
    samples = []
    for index in indices:
        generator = torch.Generator(device=device).manual_seed(seed_base + index)
        samples.append(torch.randn(
            (1, in_channels, *latent_shape), generator=generator, device=device
        ))
    return torch.cat(samples, dim=0)


def _save_decoded(recon, out_path, spatial_dims, affine, xray_orientation="legacy"):
    array = np.clip(recon[0, 0].float().cpu().numpy(), -1.0, 1.0)
    if spatial_dims == 2:
        pixels = np.rint((array + 1.0) * 127.5).astype(np.uint8)
        # The legacy path preserves the established LDM export convention.
        pixels = pixels.T if xray_orientation == "input" else np.rot90(pixels, k=-1)
        pixels = np.ascontiguousarray(pixels)
        temporary = out_path.with_name(f".{out_path.stem}.tmp.{os.getpid()}.png")
        Image.fromarray(pixels).save(temporary, format="PNG")
    else:
        hu = np.rint(array * 1000.0).astype(np.int16)
        temporary = out_path.with_name(
            f".{out_path.name}.tmp.{os.getpid()}.nii.gz"
        )
        nib.save(nib.Nifti1Image(hu, affine), temporary)
    os.replace(temporary, out_path)


def _run_diffusion(dit, stage1, scheduler, diff_cfg, indices, out_dir,
                   affine, scale_factor, latent_shape, device, gpu_id=None,
                   batch_size=1, seed_base=2026082400, xray_orientation="legacy"):
    in_channels = get_dit_params(diff_cfg).in_channels
    spatial_dims = int(diff_cfg.get("spatial_dims", len(latent_shape)))
    tag = f"[GPU {gpu_id}] " if gpu_id is not None else ""
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = [
        index for index in indices
        if not _sample_path(out_dir, index, spatial_dims).is_file()
    ]
    if not indices:
        print(f"    {tag}All requested samples already exist in {out_dir}", flush=True)
        return

    for chunk_start in range(0, len(indices), batch_size):
        chunk = indices[chunk_start : chunk_start + batch_size]
        batch = len(chunk)

        x = _seeded_noise(chunk, in_channels, latent_shape, device, seed_base)
        # DDPM adds fresh noise at every reverse step. Seed that stream as well
        # so an epoch comparison uses identical stochastic inputs, not merely
        # an identical starting latent. The screening launcher uses batch size 1,
        # making each sample reproducible even when a partially completed job is
        # resumed and only missing files are generated.
        step_generator = torch.Generator(device="cpu").manual_seed(
            seed_base + 1_000_000 + chunk[0]
        )
        with torch.no_grad(), amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            for t in scheduler.timesteps:
                t_batch = torch.full((batch,), t, device=device, dtype=torch.long)
                noise_pred = dit(x, t=t_batch, y=None)
                x, _ = scheduler.step(
                    noise_pred, t, x, generator=step_generator
                )
            x = x / scale_factor

        latents_cpu = x.float().cpu()
        del x
        torch.cuda.empty_cache()

        for b, i in enumerate(chunk):
            latent = latents_cpu[b : b + 1].to(device)
            with torch.no_grad(), amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                recon = stage1.decode_stage_2_outputs(latent)
            del latent

            out_path = _sample_path(out_dir, i, spatial_dims)
            _save_decoded(recon, out_path, spatial_dims, affine, xray_orientation)
            print(f"    {tag}Saved {out_path}", flush=True)

            del recon
            torch.cuda.empty_cache()


def _persistent_worker(rank, gpu_id, stage1_cfg, stage1_ckpt, diff_cfg_path,
                        job_queue, done_queue, scale_factor, latent_shape, batch_size,
                        seed_base, xray_orientation):
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    device = torch.device(f"cuda:{gpu_id}")

    stage1 = load_stage1(stage1_cfg, stage1_ckpt, device)

    diff_cfg = OmegaConf.load(diff_cfg_path)
    model_class = DiT2D if int(diff_cfg.get("spatial_dims", 3)) == 2 else DiT3D
    dit = model_class(**get_dit_params(diff_cfg)).to(device).eval().requires_grad_(False)

    while True:
        job = job_queue.get()
        if job is None:
            break

        ckpt_path, out_dir_str, indices, affine, scheduler_name, timesteps = job

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_dict = dict(ckpt.get("model", ckpt))
        ema = ckpt.get("ema")
        if ema is not None:
            state_dict.update(ema["shadow"])
        dit.load_state_dict(state_dict)

        if scheduler_name == "ddpm":
            scheduler = DDPMScheduler(**get_dit_scheduler(diff_cfg))
        else:
            scheduler = DDIMScheduler(**get_dit_scheduler(diff_cfg))
        scheduler.set_timesteps(timesteps)

        _run_diffusion(dit, stage1, scheduler, diff_cfg, indices, Path(out_dir_str),
                       affine, scale_factor, tuple(latent_shape), device, gpu_id=gpu_id,
                       batch_size=batch_size, seed_base=seed_base, xray_orientation=xray_orientation)

        done_queue.put(rank)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    n_gpus = torch.cuda.device_count()
    use_multi_gpu = args.device == "cuda" and n_gpus > 1
    print(f"Detected {n_gpus} GPU(s) — {'multi-GPU persistent workers' if use_multi_gpu else str(device)}.")

    affine = None
    if args.reference_nii is not None:
        ref = nib.load(args.reference_nii)
        affine = ref.affine
        print(f"Using affine from {args.reference_nii}")
    if affine is None:
        affine = np.eye(4)

    diff_cfg_for_shape = OmegaConf.load(args.diff_cfg)
    train_timesteps = int(get_dit_scheduler(diff_cfg_for_shape).get("num_train_timesteps", 1000))
    if args.timesteps < 1 or args.timesteps > train_timesteps:
        raise ValueError(f"timesteps must be between 1 and {train_timesteps}")
    if args.scheduler == "ddpm" and args.timesteps != train_timesteps:
        raise ValueError("DDPM requires all training timesteps; use DDIM for fewer steps")
    spatial_dims = int(diff_cfg_for_shape.get("spatial_dims", len(args.latent_shape)))
    if len(args.latent_shape) != spatial_dims:
        raise ValueError(
            f"latent_shape={args.latent_shape} has {len(args.latent_shape)} dimensions, "
            f"but the DiT config declares spatial_dims={spatial_dims}"
        )
    configured_shape = tuple(int(value) for value in get_dit_params(diff_cfg_for_shape).input_size)
    if tuple(args.latent_shape) != configured_shape:
        raise ValueError(
            f"latent_shape={tuple(args.latent_shape)} does not match configured "
            f"input_size={configured_shape}"
        )

    if args.diff_run_dir is not None:
        run_dir = Path(args.diff_run_dir)
        jobs = []
        for epoch in range(args.epoch_start, args.epoch_end + 1, args.epoch_step):
            ckpt = run_dir / f"checkpoint_epoch_{epoch-1}.pth"
            if not ckpt.exists():
                print(f"Warning: checkpoint not found, skipping: {ckpt}")
                continue
            jobs.append((epoch, ckpt, Path(args.output_dir) / f"epoch_{epoch}"))
    else:
        jobs = [(None, Path(args.diff_ckpt), Path(args.output_dir))]

    if not jobs:
        print("No requested checkpoints are currently available; nothing to sample.")
        return

    # ── Multi-GPU path ────────────────────────────────────────────────────
    if use_multi_gpu:
        nprocs = min(n_gpus, args.n_samples)
        indices_per_rank = [list(range(i, args.n_samples, nprocs)) for i in range(nprocs)]
        active_ranks = [r for r in range(nprocs) if indices_per_rank[r]]

        ctx = mp.get_context("spawn")
        job_queues = [ctx.Queue() for _ in range(nprocs)]
        done_queue = ctx.Queue()

        workers = []
        for rank in active_ranks:
            p = ctx.Process(
                target=_persistent_worker,
                args=(rank, rank, args.stage1_cfg, args.stage1_ckpt, args.diff_cfg,
                      job_queues[rank], done_queue, args.scale_factor, args.latent_shape,
                      args.batch_size, args.seed_base, args.xray_orientation),
                daemon=True,
            )
            p.start()
            workers.append(p)

        print(f"Spawned {len(active_ranks)} persistent worker(s) (GPUs {active_ranks}).")

        for epoch, ckpt_path, out_dir in jobs:
            label = f"epoch {epoch}" if epoch is not None else ckpt_path.name
            print(f"\n[{label}] {ckpt_path}")

            for rank in active_ranks:
                job_queues[rank].put((
                    str(ckpt_path), str(out_dir), indices_per_rank[rank],
                    affine, args.scheduler, args.timesteps,
                ))

            for _ in active_ranks:
                while True:
                    try:
                        done_queue.get(timeout=10)
                        break
                    except Exception:
                        # Check if all workers are still alive; raise if any died
                        dead = [p for p in workers if not p.is_alive()]
                        if dead:
                            raise RuntimeError(
                                f"{len(dead)} worker(s) died unexpectedly. "
                                "Check GPU memory and model config."
                            )

        for rank in active_ranks:
            job_queues[rank].put(None)
        for p in workers:
            p.join()

    # ── Single-GPU path ───────────────────────────────────────────────────
    else:
        print("Loading VQ-GAN...")
        stage1 = load_stage1(args.stage1_cfg, args.stage1_ckpt, device)

        diff_cfg_obj = OmegaConf.load(args.diff_cfg)
        model_class = DiT2D if int(diff_cfg_obj.get("spatial_dims", 3)) == 2 else DiT3D
        dit = model_class(**get_dit_params(diff_cfg_obj)).to(device).eval().requires_grad_(False)

        for epoch, ckpt_path, out_dir in jobs:
            label = f"epoch {epoch}" if epoch is not None else ckpt_path.name
            print(f"\n[{label}] {ckpt_path}")

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state_dict = dict(ckpt.get("model", ckpt))
            ema = ckpt.get("ema")
            if ema is not None:
                state_dict.update(ema["shadow"])
            dit.load_state_dict(state_dict)

            if args.scheduler == "ddpm":
                scheduler = DDPMScheduler(**get_dit_scheduler(diff_cfg_obj))
            else:
                scheduler = DDIMScheduler(**get_dit_scheduler(diff_cfg_obj))
            scheduler.set_timesteps(args.timesteps)

            indices = list(range(args.n_samples))
            print(f"  Sampling {args.n_samples} volumes ({args.scheduler.upper()}, {args.timesteps} steps)...")
            _run_diffusion(dit, stage1, scheduler, diff_cfg_obj, indices, out_dir,
                           affine, args.scale_factor, tuple(args.latent_shape), device,
                           batch_size=args.batch_size, seed_base=args.seed_base,
                           xray_orientation=args.xray_orientation)

    print("\nDone.")


if __name__ == "__main__":
    main()
