import random
from typing import List, Tuple
import torch

from ...shared.constants import IMAGENET_MEAN, IMAGENET_STD
from torch.utils.data import Dataset

from rasterio import open as ropen

from ..io_utils import (
    compute_grid,
    normalize01_then_standardize,
    pad_to_square,
    read_mask_tile,
    read_mask_tile_scaled,
    read_rgb_tile,
    read_rgb_tile_scaled,
)
import numpy as np
from typing import List, Tuple
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.dropout.coarse_dropout import CoarseDropout

class OrthoTileDataset(Dataset):
    def __init__(
        self,
        tif_path,
        tile_size=1024,
        overlap=512,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    ):
        self.tif_path = str(tif_path)
        self.T = int(tile_size)
        self.overlap = int(overlap)
        with ropen(self.tif_path) as ds:
            self.H, self.W = ds.height, ds.width
            if ds.count < 3:
                raise ValueError("Need at least 3 bands (RGB).")
        self.coords = compute_grid(self.H, self.W, self.T, self.overlap)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y0, x0 = self.coords[idx]
        with ropen(self.tif_path) as ds:
            rgb, (w, h) = read_rgb_tile(ds, x0, y0, self.T, self.W, self.H)
        rgb = normalize01_then_standardize(rgb, self.mean, self.std)
        rgb = pad_to_square(rgb, self.T)
        meta = {"y0": y0, "x0": x0, "h": h, "w": w}
        return torch.from_numpy(rgb), meta

    @classmethod
    def collate_tiles(cls, batch):
        tiles = torch.stack([b[0] for b in batch], dim=0)  # (B,3,T,T)
        metas = [b[1] for b in batch]
        return tiles, metas


class RasterPairTileDataset(Dataset):
    """
    Reads tiles from IMAGE_TIF and LABEL_TIF (same size/geo).
    LABEL_TIF: 0=background, 1=building, 255=ignore (optional).
    If reject_empty=True, tiles with building ratio < min_building_ratio are excluded.
    """

    def __init__(
        self,
        image_tif_path,
        label_tif_path,
        tile_size=1024,
        overlap=512,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        reject_empty: bool = True,
        min_building_ratio: float = 0.1,  # Minimum 10% building content
        train_scales: tuple[int, ...] = (1, 2, 4),
    ):
        self.img_path = str(image_tif_path)
        self.lbl_path = str(label_tif_path)
        self.T = int(tile_size)
        self.overlap = int(overlap)
        self.mean, self.std = mean, std
        self.reject_empty = reject_empty
        self.min_building_ratio = min_building_ratio if reject_empty else 0.0
        self.train_scales = tuple(sorted(set(train_scales)))

        with ropen(self.img_path) as ds:
            self.H, self.W = ds.height, ds.width
            if ds.count < 3:
                raise ValueError("Image must have at least 3 bands (RGB).")
        with ropen(self.lbl_path) as ds:
            if ds.height != self.H or ds.width != self.W:
                raise ValueError("Image and label rasters must match in size.")

        all_coords = compute_grid(self.H, self.W, self.T, self.overlap)

        if self.reject_empty:
            keep_coords = []
            with ropen(self.lbl_path) as dl:
                for y0, x0 in all_coords:
                    lab, (w, h) = read_mask_tile(dl, x0, y0, self.T, self.W, self.H)
                    # Calculate building ratio instead of just checking any
                    building_pixels = (lab == 1).sum()
                    total_pixels = lab.size
                    building_ratio = building_pixels / total_pixels if total_pixels > 0 else 0.0
                    
                    if building_ratio >= self.min_building_ratio:
                        keep_coords.append((y0, x0))
            if len(keep_coords) == 0:
                raise RuntimeError(
                    f"No tiles with ≥{self.min_building_ratio:.1%} building content found; "
                    f"check labels, reduce min_building_ratio, or disable reject_empty."
                )
            self.coords = keep_coords
            print(
                f"[RasterPairTileDataset] Kept {len(self.coords)} tiles with ≥{self.min_building_ratio:.1%} buildings "
                f"(rejected {len(all_coords)-len(self.coords)} sparse tiles)."
            )
        else:
            self.coords = all_coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y0, x0 = self.coords[idx]
        scale = random.choice(self.train_scales)

        with ropen(self.img_path) as di, ropen(self.lbl_path) as dl:
            rgb_s, (w, h) = read_rgb_tile_scaled(
                di, x0, y0, self.T, self.W, self.H, scale
            )
            lab_s, _ = read_mask_tile_scaled(dl, x0, y0, self.T, self.W, self.H, scale)

        # Normalize on the scaled data, then upsample to T×T
        rgb_s = normalize01_then_standardize(rgb_s, self.mean, self.std)  # (3,hs,ws)
        rgb_t = torch.from_numpy(rgb_s).unsqueeze(0)  # (1,3,hs,ws)
        rgb_t = torch.functional.F.interpolate(
            rgb_t, size=(self.T, self.T), mode="bilinear", align_corners=False
        ).squeeze(0)

        # Labels: binarize before upsample, keep ignore==255 with nearest
        lab = lab_s.astype(np.int32)
        ignore = lab == 255
        lab_bin = (lab == 1).astype(np.float32)  # (hs,ws)

        lab_t = torch.from_numpy(lab_bin).unsqueeze(0).unsqueeze(0)  # (1,1,hs,ws)
        lab_t = torch.functional.F.interpolate(
            lab_t, size=(self.T, self.T), mode="nearest"
        ).squeeze(
            0
        )  # (1,T,T)

        ign_t = torch.from_numpy(~ignore).unsqueeze(0).unsqueeze(0)
        ign_t = (
            torch.functional.F.interpolate(
                ign_t.float(), size=(self.T, self.T), mode="nearest"
            )
            .bool()
            .squeeze(0)
        )  # (1,T,T)

        sample = {
            "image": rgb_t.contiguous(),  # (3,T,T)
            "target": lab_t.contiguous(),  # (1,T,T)
            "ignore": ign_t.contiguous(),  # (1,T,T) True=keep
        }
        return sample

    @classmethod
    def collate_pairs(cls, batch):
        imgs = torch.stack([b["image"] for b in batch], dim=0)
        tars = torch.stack([b["target"] for b in batch], dim=0)
        keep = torch.stack([b["ignore"] for b in batch], dim=0)
        return imgs, tars, keep


# Expect these constants/functions to exist in your codebase:
# IMAGENET_MEAN, IMAGENET_STD
# ropen, compute_grid, read_rgb_tile_scaled, read_mask_tile, read_mask_tile_scaled
# normalize01_then_standardize (we'll call our own normalization to keep things tidy)


class MultiRasterPairTileDataset(Dataset):
    """
    Combines multiple ortho/mask pairs for training with augmentation.
    Each pair contributes tiles to a unified dataset with multi-scale training (Albumentations).
    """

    def __init__(
        self,
        ortho_mask_pairs: List[Tuple[str, str]],  # [(ortho1, mask1), (ortho2, mask2), ...]
        tile_size: int = 1024,
        overlap: int = 512,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        reject_empty: bool = True,
        min_building_ratio: float = 0.10,   # Minimum 10% building content
        train_scales: tuple[int, ...] = (1, 2, 4),
        augment: bool = True,
        augment_prob: float = 0.5,          # kept for API compatibility; not used directly
        jpeg_quality_range=(40, 80),
        downscale_range=(0.5, 0.9),
        rotation_deg: int = 10,
        shear_deg: int = 5,
        scale_range=(0.8, 1.2),
        translate_pct: float = 0.02,
        perspective_scale=(0.02, 0.05),
    ):
        self.T = int(tile_size)
        self.overlap = int(overlap)
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.reject_empty = reject_empty
        self.min_building_ratio = min_building_ratio if reject_empty else 0.0
        self.train_scales = tuple(sorted(set(train_scales)))
        self.augment = augment

        # Prepared during __init__
        self.tile_coords = []     # List[(dataset_idx, y0, x0)]
        self.datasets_info = []   # List[(img_path, lbl_path, H, W)]

        # Build augmentation pipeline once
        self.alb = self._build_albumentations_pipeline(
            tile_size=self.T,
            jpeg_quality_range=jpeg_quality_range,
            downscale_range=downscale_range,
            rotation_deg=rotation_deg,
            shear_deg=shear_deg,
            scale_range=scale_range,
            translate_pct=translate_pct,
            perspective_scale=perspective_scale,
        )

        # ----- Scan all datasets and collect valid tile coordinates -----
        for dataset_idx, (img_path, lbl_path) in enumerate(ortho_mask_pairs):
            img_path, lbl_path = str(img_path), str(lbl_path)

            # Validate files exist and have compatible dimensions
            with ropen(img_path) as ds:
                H, W = ds.height, ds.width
                if ds.count < 3:
                    raise ValueError(f"Image {img_path} must have at least 3 bands (RGB).")

            with ropen(lbl_path) as ds:
                if ds.height != H or ds.width != W:
                    raise ValueError(
                        f"Image and label rasters must match in size for pair {dataset_idx}."
                    )

            self.datasets_info.append((img_path, lbl_path, H, W))

            # Compute full grid for this image
            all_coords = compute_grid(H, W, self.T, self.overlap)

            if self.reject_empty:
                keep_coords = []
                with ropen(lbl_path) as dl:
                    for y0, x0 in all_coords:
                        lab, (w, h) = read_mask_tile(dl, x0, y0, self.T, W, H)
                        # building ratio (label 1 = building, 255 = ignore)
                        building_pixels = (lab == 1).sum()
                        total_pixels = lab.size
                        ratio = (building_pixels / total_pixels) if total_pixels > 0 else 0.0
                        if ratio >= self.min_building_ratio:
                            keep_coords.append((dataset_idx, y0, x0))

                if len(keep_coords) == 0:
                    print(
                        f"Warning: No tiles with \u2265{self.min_building_ratio:.1%} building content "
                        f"found in dataset {dataset_idx} ({img_path})"
                    )
                else:
                    self.tile_coords.extend(keep_coords)
                    print(
                        f"Dataset {dataset_idx}: Kept {len(keep_coords)} tiles with \u2265{self.min_building_ratio:.1%} buildings "
                        f"(rejected {len(all_coords)-len(keep_coords)} sparse tiles)"
                    )
            else:
                self.tile_coords.extend([(dataset_idx, y0, x0) for y0, x0 in all_coords])

        if len(self.tile_coords) == 0:
            raise RuntimeError(
                f"No tiles with \u2265{self.min_building_ratio:.1%} building content found across all datasets; "
                f"check labels, reduce min_building_ratio, or disable reject_empty."
            )

        print(
            f"MultiRasterPairTileDataset: {len(self.tile_coords)} total tiles across "
            f"{len(ortho_mask_pairs)} datasets"
        )

    # -------------------- Albumentations pipeline -------------------- #
    def _build_albumentations_pipeline(
        self,
        tile_size: int,
        **kwargs  # Ignore other parameters for now
    ):
        """Conservative augmentation pipeline for building segmentation"""
        return A.Compose([
            # --- Simple geometric transforms (building-friendly) ---
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),  # Less common but useful for aerial imagery
            A.RandomRotate90(p=0.6),  # Only 90-degree rotations to preserve ortho properties
            
            # --- Very light rotation only ---
            A.Rotate(limit=3, p=0.3, border_mode=0),  # Max 3 degrees, rare
            
            # --- Conservative photometric augmentations ---
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,  # Reduced from 0.08
                    contrast_limit=0.1,    # Reduced from 0.08
                    p=1.0
                ),
                A.ColorJitter(
                    brightness=0.05,       # Much more conservative
                    contrast=0.05,
                    saturation=0.02,      # Very light
                    hue=0.005,            # Minimal hue shift
                    p=1.0
                ),
            ], p=0.4),  # Apply to only 40% of samples
            
            # --- Size & normalization ---
            A.Resize(tile_size, tile_size, interpolation=1),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # Only here!
            ToTensorV2(transpose_mask=True),
        ])
            
    # -------------------- Dataset API -------------------- #
    def __len__(self):
        return len(self.tile_coords)

    def __getitem__(self, idx):
        dataset_idx, y0, x0 = self.tile_coords[idx]
        img_path, lbl_path, H, W = self.datasets_info[dataset_idx]

        # pick a random training scale (your multi-scale sampling)
        scale = random.choice(self.train_scales)

        # Read scaled tiles (numpy)
        with ropen(img_path) as di, ropen(lbl_path) as dl:
            rgb_s, _ = read_rgb_tile_scaled(di, x0, y0, self.T, W, H, scale)  # (3, hs, ws), float32 in [0,1] expected
            lab_s, _ = read_mask_tile_scaled(dl, x0, y0, self.T, W, H, scale)  # (hs, ws), uint8 with {0,1,255}

        # Albumentations expects HWC float32 image; mask uint8
        img_np = np.moveaxis(rgb_s, 0, -1).astype(np.float32)  # (hs,ws,3), values in [0,1] recommended
        mask_np = lab_s.astype(np.uint8)                        # (hs,ws), 0/1/255

        if self.augment:
            aug = self.alb(image=img_np, mask=mask_np)
            img_t = aug["image"].float()        # Already normalized by Albumentations
            mask_t = aug["mask"].long()
        else:
            # No-aug path
            img_t = torch.from_numpy(np.moveaxis(img_np, -1, 0)).unsqueeze(0)
            img_t = F.interpolate(img_t, size=(self.T, self.T), mode="bilinear", align_corners=False).squeeze(0)
            
            # Manual normalization for no-aug path
            img_t = (img_t - img_t.new_tensor(self.mean).view(3, 1, 1)) / img_t.new_tensor(self.std).view(3, 1, 1)
            
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float()
            mask_t = F.interpolate(mask_t, size=(self.T, self.T), mode="nearest").squeeze(0).squeeze(0).long()

        # Build binary target and keep mask
        # target: 1 where building; 0 elsewhere; ignore stays but we don't set it to 1
        target_t = (mask_t == 1).unsqueeze(0).float()   # (1,T,T)
        keep_t   = (mask_t != 255).unsqueeze(0)         # (1,T,T) bool True=keep

        sample = {
            "image": img_t.contiguous(),   # (3,T,T), standardized
            "target": target_t.contiguous(),  # (1,T,T), float32
            "keep": keep_t.contiguous(),      # (1,T,T), bool
        }
        return sample


    @classmethod
    def collate_pairs(cls, batch):
        imgs = torch.stack([b["image"] for b in batch], dim=0)  # (B,3,T,T)
        tars = torch.stack([b["target"] for b in batch], dim=0) # (B,1,T,T)
        keep = torch.stack([b["keep"] for b in batch], dim=0)   # (B,1,T,T)
        return imgs, tars, keep
