"""Evaluate paired reference/reconstruction arrays in the model intensity range."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch


def _array(value: np.ndarray, dims: int) -> torch.Tensor:
    value = np.asarray(value, dtype=np.float32)
    while value.ndim > dims and value.shape[0] == 1:
        value = value[0]
    if value.ndim != dims or not np.isfinite(value).all():
        raise ValueError(f"expected finite {dims}D single-channel array, got {value.shape}")
    return torch.from_numpy(np.ascontiguousarray(value))[None, None]


def _lpips_model(weights: Path, device: str):
    import lpips
    import torchvision
    # Instantiate without downloads and load the user's local ImageNet trunk.
    trunk = torchvision.models.squeezenet1_1(weights=None)
    trunk.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model = lpips.LPIPS(net="squeeze", pnet_rand=True, pretrained=True, verbose=False)
    for number in range(1, 8):
        for index, layer in getattr(model.net, f"slice{number}").named_children():
            layer.load_state_dict(trunk.features[int(index)].state_dict())
    return model.eval().to(device)


def _lpips_3d(model, reconstruction: torch.Tensor, target: torch.Tensor) -> float:
    rec, ref = reconstruction[0, 0], target[0, 0]
    values = []
    for axis, size in enumerate(ref.shape):
        count = max(1, int(round(size * 0.3)))
        indices = torch.linspace(0, size - 1, count, device=ref.device).round().long().unique()
        left, right = torch.index_select(rec, axis, indices), torch.index_select(ref, axis, indices)
        if axis == 1:
            left, right = left.permute(1, 0, 2), right.permute(1, 0, 2)
        elif axis == 2:
            left, right = left.permute(2, 0, 1), right.permute(2, 0, 1)
        for start in range(0, len(indices), 4):
            a = left[start:start + 4, None].repeat(1, 3, 1, 1)
            b = right[start:start + 4, None].repeat(1, 3, 1, 1)
            values.append(model(a, b, normalize=False).reshape(-1))
    return float(torch.cat(values).mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="CSV with array column; each NPZ has reference and reconstruction arrays")
    parser.add_argument("--dataset", choices=("chestxray", "luna"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="new per-image CSV file")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-range", type=float, default=2.0)
    parser.add_argument("--ssim", action="store_true", help="also calculate MONAI SSIM")
    parser.add_argument("--ms-ssim", action="store_true", help="also calculate MONAI multi-scale SSIM")
    parser.add_argument("--lpips-squeezenet-weights", type=Path,
                        help="local torchvision SqueezeNet 1.1 ImageNet checkpoint for LPIPS")
    args = parser.parse_args()
    if args.output.exists() or args.data_range <= 0:
        parser.error("output must be new and data range must be positive")
    dims = 2 if args.dataset == "chestxray" else 3
    ssim = ms_ssim = perceptual = None
    if args.ssim or args.ms_ssim:
        from monai.metrics import MultiScaleSSIMMetric, SSIMMetric
        ssim = SSIMMetric(spatial_dims=dims, data_range=args.data_range) if args.ssim else None
        ms_ssim = MultiScaleSSIMMetric(spatial_dims=dims, data_range=args.data_range, kernel_size=5) if args.ms_ssim else None
    if args.lpips_squeezenet_weights:
        if not args.lpips_squeezenet_weights.is_file():
            parser.error("LPIPS SqueezeNet weights must exist locally")
        perceptual = _lpips_model(args.lpips_squeezenet_weights, args.device)
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "array" not in reader.fieldnames:
            parser.error("CSV needs an array column")
        rows = list(reader)
    if not rows:
        parser.error("manifest is empty")
    result = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            path = Path(row["array"]).expanduser()
            if not path.is_absolute():
                path = args.manifest.parent / path
            with np.load(path, allow_pickle=False) as archive:
                target = _array(archive["reference"], dims).to(args.device)
                reconstruction = _array(archive["reconstruction"], dims).to(args.device)
            if target.shape != reconstruction.shape:
                raise ValueError(f"shape mismatch for row {index + 2}")
            mse = torch.mean((reconstruction - target) ** 2)
            denominator = torch.mean(target ** 2)
            values = {"image_index": index, "L1": float(torch.mean(torch.abs(reconstruction - target)).item()),
                      "PSNR": float(10.0 * math.log10(args.data_range ** 2 / max(float(mse.item()), 1e-20))),
                      "NMSE": float((mse / denominator.clamp_min(1e-20)).item())}
            if ssim is not None:
                values["SSIM"] = float(ssim(target, reconstruction).item())
            if ms_ssim is not None:
                values["MS_SSIM"] = float(ms_ssim(target, reconstruction).item())
            if perceptual is not None:
                values["LPIPS"] = (_lpips_3d(perceptual, reconstruction, target) if dims == 3 else
                                    float(perceptual(reconstruction.repeat(1, 3, 1, 1),
                                                     target.repeat(1, 3, 1, 1), normalize=False).mean().item()))
            result.append(values)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    print(f"Wrote {len(result)} image metrics to {args.output}")


if __name__ == "__main__":
    main()
