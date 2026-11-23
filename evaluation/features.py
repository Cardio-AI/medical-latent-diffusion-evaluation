"""Extract benchmark image embeddings from PNG or NIfTI manifests."""

from __future__ import annotations

import argparse
import csv
import enum
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from benchmark.transforms import Sample, centre_crop_or_pad, load_chestxray, load_luna


if not hasattr(enum, "StrEnum"):
    class _StrEnum(str, enum.Enum):
        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()

        def __str__(self):
            return str.__str__(self.value)

    enum.StrEnum = _StrEnum

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor" / "medgen_eval_lab"))

WEIGHT_SHA256 = {
    "radimagenet": "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734",
    "inception": "0cc3c7bd75056d25e46cba549dc184522069b81e9787eff6df84f397bd52a5ef",
    "medicalnet": "5b6189cafbee2f5604a7279b62bc163365aa6a86a377e1dc260a14275cacbd84",
}
MEDICALNET_FILES = {
    "hubconf.py": "1709645c653b42dc5c16b58c398d51fc21bf889f4a656e2a2987db6f8a8da5e4",
    "medicalnet_models/models/resnet.py": "26e487efeddabe0f8859b6bbb4cbef68bfaf34774420751a42683cb62b828979",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_sample(path: Path, dataset: str, source_id: str) -> Sample:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            data = np.asarray(archive["data"], dtype=np.float32)
            spacing = tuple(float(v) for v in archive["spacing"])
            recorded_dataset = str(archive["dataset"])
            if recorded_dataset != dataset or data.ndim != (2 if dataset == "chestxray" else 3):
                raise ValueError(f"array dataset or dimensionality mismatch in {path}")
            metadata = {"evaluation_affine": np.asarray(archive["affine"]).tolist()} if dataset == "luna" else {}
        if not np.isfinite(data).all():
            raise ValueError(f"non-finite input: {path}")
        return Sample(data, dataset, source_id, str(path), str(path), spacing, metadata)
    return load_chestxray(str(path), source_id) if dataset == "chestxray" else load_luna(str(path), source_id)


class ImageExtractor:
    def __init__(self, name: str, weights: Path, device: str):
        import torch
        import torchvision
        self.torch, self.device, self.name = torch, torch.device(device), name
        if name == "radimagenet":
            from .radimagenet_checkpoint import _radimagenet_resnet50_state_dict
            checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
            state = _radimagenet_resnet50_state_dict(checkpoint, tensor_predicate=torch.is_tensor)
            state = {key: value for key, value in state.items() if not key.startswith("fc.")}
            model = torchvision.models.resnet50(weights=None)
            missing, unexpected = model.load_state_dict(state, strict=False)
            missing = [key for key in missing if not key.startswith("fc.")]
            unexpected = [key for key in unexpected if not key.startswith("fc.")]
            if missing or unexpected:
                raise ValueError(f"incompatible RadImageNet state: missing={missing[:4]}, unexpected={unexpected[:4]}")
            self.size = 224
        elif name == "inception":
            model = torchvision.models.inception_v3(weights=None, aux_logits=True, init_weights=False)
            model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
            self.size = 299
        else:
            raise ValueError(name)
        model.fc = torch.nn.Identity()
        self.model = model.eval().to(self.device)

    def __call__(self, sample: Sample) -> np.ndarray:
        torch = self.torch
        tensor = torch.from_numpy(np.asarray(sample.data, dtype=np.float32))[None, None].repeat(1, 3, 1, 1)
        tensor = torch.nn.functional.interpolate(tensor, size=(self.size, self.size), mode="bilinear",
                                                 align_corners=False, antialias=True)
        mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        with torch.inference_mode():
            result = self.model(((tensor - mean) / std).to(self.device))
        if hasattr(result, "logits"):
            result = result.logits
        return result.float().cpu().numpy()[0]


class MedicalNetExtractor:
    def __init__(self, repository: Path, weights: Path, device: str,
                 crop: tuple[int, int, int] = (512, 512, 256), size: int = 112):
        import torch
        self.torch, self.device, self.crop, self.size = torch, torch.device(device), crop, size
        # Construct the same hub model without its automatic checkpoint download.
        sys.path.insert(0, str(repository.resolve()))
        from medicalnet_models.models.resnet import Bottleneck, ResNet
        model = ResNet(Bottleneck, [3, 4, 6, 3])
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
            raise ValueError("MedicalNet checkpoint must contain a state_dict")
        state = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}
        model.load_state_dict(state)
        self.model = model.eval().to(self.device)

    def __call__(self, sample: Sample) -> np.ndarray:
        torch = self.torch
        xyz = np.transpose(sample.data, (1, 2, 0)).astype(np.float32, copy=False)
        xyz = centre_crop_or_pad(xyz, self.crop, fill=-1000.0)
        xyz = (xyz - float(xyz.mean())) / max(float(xyz.std()), 1e-6)
        tensor = torch.from_numpy(np.ascontiguousarray(xyz))[None, None]
        tensor = torch.nn.functional.interpolate(tensor, size=(self.size,) * 3, mode="trilinear", align_corners=False)
        with torch.inference_mode():
            feature_map = self.model(tensor.to(self.device))
            output = torch.nn.functional.adaptive_avg_pool3d(feature_map, 1).flatten(1)
        return output.float().cpu().numpy()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("chestxray", "luna"), required=True)
    parser.add_argument("--extractor", choices=("radimagenet", "inception", "medicalnet"), required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV containing image and optional source_id")
    parser.add_argument("--weights", type=Path, required=True, help="local pretrained checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="new .npy matrix path")
    parser.add_argument("--medicalnet-repo", type=Path, help="trusted local MedicalNet-models hub repository")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-other-weights", action="store_true", help="permit weights different from the study checkpoint")
    parser.add_argument("--allow-other-medicalnet-source", action="store_true", help="permit a different local MedicalNet hub source")
    parser.add_argument("--medicalnet-crop", nargs=3, type=int, default=(512, 512, 256), metavar=("X", "Y", "Z"))
    parser.add_argument("--medicalnet-size", type=int, default=112)
    args = parser.parse_args()
    if (args.extractor == "medicalnet") != (args.dataset == "luna"):
        parser.error("medicalnet requires luna; 2D extractors require chestxray")
    if args.output.suffix != ".npy" or args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("output must be a new .npy file with no existing .json sidecar")
    if not args.weights.is_file():
        parser.error("weights must exist locally")
    weight_hash = digest(args.weights)
    if weight_hash != WEIGHT_SHA256[args.extractor] and not args.allow_other_weights:
        parser.error("weight hash does not match the study checkpoint; --allow-other-weights permits a different model")
    if any(v <= 0 for v in args.medicalnet_crop) or args.medicalnet_size <= 0:
        parser.error("medicalnet crop and size must be positive")
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "image" not in reader.fieldnames:
            parser.error("manifest CSV must contain an image column")
        records = list(reader)
    if not records:
        parser.error("manifest CSV is empty")
    if args.extractor == "medicalnet":
        if args.medicalnet_repo is None:
            parser.error("--medicalnet-repo is required")
        for relative, expected in MEDICALNET_FILES.items():
            path = args.medicalnet_repo / relative
            if not path.is_file() or (digest(path) != expected and not args.allow_other_medicalnet_source):
                parser.error(f"MedicalNet source missing or hash differs: {relative}")
        extractor = MedicalNetExtractor(args.medicalnet_repo, args.weights, args.device,
                                        tuple(args.medicalnet_crop), args.medicalnet_size)
    else:
        extractor = ImageExtractor(args.extractor, args.weights, args.device)
    rows = []
    for index, record in enumerate(records):
        image = Path(record["image"]).expanduser()
        if not image.is_absolute():
            image = args.manifest.parent / image
        sample = read_sample(image, args.dataset, record.get("source_id") or str(index))
        rows.append(extractor(sample))
    matrix = np.stack(rows).astype(np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("extractor returned invalid features")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.save(stream, matrix, allow_pickle=False)
    with args.output.with_suffix(".json").open("x", encoding="utf-8") as stream:
        json.dump({"dataset": args.dataset, "extractor": args.extractor, "weights_sha256": weight_hash,
                   "shape": list(matrix.shape), "medicalnet_crop": args.medicalnet_crop if args.dataset == "luna" else None,
                   "medicalnet_size": args.medicalnet_size if args.dataset == "luna" else None}, stream, indent=2)
        stream.write("\n")
    print(f"Wrote {matrix.shape} features to {args.output}")


if __name__ == "__main__":
    main()
