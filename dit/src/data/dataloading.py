# data/dataloading.py
import pandas as pd
from pathlib import Path
from typing import Tuple, Union, List
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import numpy as np

from monai.data import CacheDataset, PersistentDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ThresholdIntensityd,
    ScaleIntensityd,
    RandRotated,
    RandFlipd,
    RandSpatialCropd,
    CenterSpatialCropd,
    ToTensord,
    SpatialPadd
)

def get_datalist(ids_path: str, extended_report: bool = False) -> list[dict]:
    """
    Read paths from the 'image' column. Relative paths are relative to the CSV,
    matching the latent manifests produced by training/encode_latents.py.
    """
    df = pd.read_csv(ids_path, sep=",")
    csv_dir = Path(ids_path).resolve().parent
    data_dicts = []
    for _, row in df.iterrows():
        path = Path(str(row["image"]))
        data_dicts.append({"image": str(path if path.is_absolute() else csv_dir / path)})

    if extended_report:
        print("Data dict preview:")
        for i, d in enumerate(data_dicts[:5]):
            print(f"{i}: {d}")

    print(f"Found {len(data_dicts)} subjects.")
    return data_dicts

def get_dataloader(
    cache_dir: Union[str, Path],
    training_ids: str,
    validation_ids: str,
    batch_size: int,
    num_workers: int = 4,
    use_persistent: bool = False,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    rank: int = 0,
    world_size: int = 1,
    cache_rate: float = 0.0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns training and validation DataLoaders ready for DDP.

    Args:
        cache_dir: directory for persistent dataset cache
        training_ids: CSV file with training image paths
        validation_ids: CSV file with validation image paths
        batch_size: batch size for DataLoader
        num_workers: DataLoader workers
        use_persistent: use PersistentDataset instead of CacheDataset
        roi_size: crop size for random spatial cropping
        rank: rank of current process (DDP)
        world_size: total number of processes (DDP)
    """
    train_files = get_datalist(training_ids)
    val_files = get_datalist(validation_ids)

    train_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ThresholdIntensityd(keys=["image"], threshold=1000, above=False, cval=1000),
        ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
        ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
        RandRotated(keys=["image"], range_x=0.0872665, range_y=0.0872665, range_z=0.0872665, prob=0.2),
        RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
        RandSpatialCropd(keys=["image"], roi_size=roi_size, random_size=False),
        SpatialPadd(keys=["image"], spatial_size=roi_size, constant_values=-1.0),
        ToTensord(keys=["image"])
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        ThresholdIntensityd(keys=["image"], threshold=1000, above=False, cval=1000),
        ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
        ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
        CenterSpatialCropd(keys=["image"], roi_size=roi_size),
        SpatialPadd(keys=["image"], spatial_size=roi_size, constant_values=-1.0),
        ToTensord(keys=["image"])
    ])

    if use_persistent:
        train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir=str(Path(cache_dir) / "train"))
        val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir=str(Path(cache_dir) / "val"))
    else:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=cache_rate)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.0)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# -------------------------
# Shared safe loader for .pt latent files
# -------------------------
def _safe_load(path):
    path = str(path)
    if path.endswith((".nii", ".nii.gz")):
        import nibabel as nib
        return torch.from_numpy(
            np.asarray(nib.load(path).dataobj, dtype=np.float32)
        )
    with torch.serialization.safe_globals([
        np.ndarray,
        np.dtype,
        np._core.multiarray._reconstruct
    ]):
        return torch.load(path, weights_only=False)


# -------------------------
# Custom Dataset for precomputed latents
# -------------------------

class LatentDataset(Dataset):
    """
    Dataset for precomputed latent tensors stored as .pt files.
    Compatible with PyTorch 2.6+ safe loading.
    """
    def __init__(self, file_list: List, preload: bool = True):
        self.file_list = [
            f["image"] if isinstance(f, dict) else f
            for f in file_list
        ]
        self.preload = preload

        if preload:
            self.data = [_safe_load(f) for f in self.file_list]
        else:
            self.data = None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.preload:
            return {"image": self.data[idx]}
        else:
            return {"image": _safe_load(self.file_list[idx])}

# -------------------------
# DiT Dataloader function
# -------------------------
def get_dit_dataloader(
    training_ids: str,
    validation_ids: str,
    batch_size: int,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    use_precomputed_latents: bool = True,
    preload_latents: bool = True,
    use_persistent: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns training and validation DataLoaders for DiT training.
    Supports precomputed latents (.pt) or raw images.
    """
    train_files = get_datalist(training_ids)
    val_files = get_datalist(validation_ids)

    if use_precomputed_latents:
        train_ds = LatentDataset(train_files, preload=preload_latents)
        val_ds = LatentDataset(val_files, preload=preload_latents)
    else:
        train_transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ThresholdIntensityd(keys=["image"], threshold=1000, above=False, cval=1000),
            ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
            ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
            RandRotated(keys=["image"], range_x=0.0872665, range_y=0.0872665, range_z=0.0872665, prob=0.2),
            RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
            RandSpatialCropd(keys=["image"], roi_size=roi_size, random_size=False),
            ToTensord(keys=["image"])
        ])

        val_transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            ThresholdIntensityd(keys=["image"], threshold=1000, above=False, cval=1000),
            ThresholdIntensityd(keys=["image"], threshold=-1000, above=True, cval=-1000),
            ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0),
            CenterSpatialCropd(keys=["image"], roi_size=roi_size),
            SpatialPadd(keys=["image"], spatial_size=roi_size),
            ToTensord(keys=["image"])
        ])

        if use_persistent:
            train_ds = PersistentDataset(data=train_files, transform=train_transforms, cache_dir="/tmp/dit_train_cache")
            val_ds = PersistentDataset(data=val_files, transform=val_transforms, cache_dir="/tmp/dit_val_cache")
        else:
            train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.5)
            val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.0)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    # DiTTrainer validates on rank 0 only. Give it the full validation set.
    val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
