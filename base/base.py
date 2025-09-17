from __future__ import annotations
import os
from pathlib import Path
import tempfile
import time
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler

import rasterio
from rasterio.windows import Window
from segmentation_models_pytorch import Unet
from base.src.shared.mlflow_helpers import MlflowLogger
from shared.constants import DATASET_DIR, ORTHO_FILE_PATH
from tqdm.auto import tqdm

# ---------------------------- #
# ---------- CONFIG ---------- #
# ---------------------------- #

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------- #
# ---------- UTILS ----------- #
# ---------------------------- #


def open_profile_like(ref_tif: Path) -> dict:
    with rasterio.open(ref_tif) as ref:
        return ref.profile.copy()


def compute_grid(H: int, W: int, tile_size: int, overlap: int) -> List[Tuple[int, int]]:
    step = tile_size - overlap
    xs = list(range(0, W, step))
    ys = list(range(0, H, step))
    if xs[-1] + tile_size < W:
        xs.append(W - tile_size)
    if ys[-1] + tile_size < H:
        ys.append(H - tile_size)
    return [(y, x) for y in ys for x in xs]


def read_rgb_tile(
    ds: rasterio.io.DatasetReader, x0: int, y0: int, T: int, W: int, H: int
) -> Tuple[np.ndarray, Tuple[int, int]]:
    x1, y1 = min(x0 + T, W), min(y0 + T, H)
    win = Window(x0, y0, x1 - x0, y1 - y0)
    tile = ds.read(indexes=[1, 2, 3], window=win).astype(np.float32)  # (3,h,w)
    return tile, (x1 - x0, y1 - y0)


def read_mask_tile(
    ds: rasterio.io.DatasetReader, x0: int, y0: int, T: int, W: int, H: int
) -> Tuple[np.ndarray, Tuple[int, int]]:
    x1, y1 = min(x0 + T, W), min(y0 + T, H)
    win = Window(x0, y0, x1 - x0, y1 - y0)
    m = ds.read(1, window=win)
    return m, (x1 - x0, y1 - y0)


def normalize01_then_standardize(
    arr: np.ndarray, mean=IMAGENET_MEAN, std=IMAGENET_STD
) -> np.ndarray:
    if arr.max() > 1.5:
        arr = arr / 255.0
    mean = np.array(mean, np.float32)[:, None, None]
    std = np.array(std, np.float32)[:, None, None]
    return (arr - mean) / std


def pad_to_square(tile: np.ndarray, T: int) -> np.ndarray:
    ph, pw = T - tile.shape[1], T - tile.shape[2]
    if ph > 0 or pw > 0:
        tile = np.pad(tile, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    return tile


def write_geotiff_single_band(
    array: np.ndarray, out_path: Path, profile: dict, dtype: str
):
    p = profile.copy()
    p.update(
        driver="GTiff",
        dtype=dtype,
        count=1,
        compress="deflate",
        predictor=2 if "float" in dtype else 1,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(out_path, "w", **p) as dst:
        dst.write(array.astype(dtype), 1)


def dice_coeff(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean()


def iou_score(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1) - inter
    iou = (inter + eps) / (union + eps)
    return iou.mean()


def cosine_with_warmup(total_steps: int, warmup_steps: int = 1000):
    import math

    def _lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return _lambda


# ----------------------------------------- #
# ---------- DATASETS (Inference) --------- #
# ----------------------------------------- #


class OrthoTileDataset(Dataset):
    """Tiled RGB reader for a single ortho TIFF (inference)."""

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
        with rasterio.open(self.tif_path) as ds:
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
        with rasterio.open(self.tif_path) as ds:
            rgb, (w, h) = read_rgb_tile(ds, x0, y0, self.T, self.W, self.H)  # (3,h,w)
        rgb = normalize01_then_standardize(rgb, self.mean, self.std)
        rgb = pad_to_square(rgb, self.T)
        meta = {"y0": y0, "x0": x0, "h": h, "w": w}
        return torch.from_numpy(rgb), meta


def collate_tiles(batch):
    tiles = torch.stack([b[0] for b in batch], dim=0)  # (B,3,T,T)
    metas = [b[1] for b in batch]
    return tiles, metas


# --------------------------------------- #
# ---------- DATASETS (Training) -------- #
# --------------------------------------- #


class RasterPairTileDataset(Dataset):
    """
    Reads tiles from IMAGE_TIF and LABEL_TIF (same size/geo).
    LABEL_TIF: 0=background, 1=building, 255=ignore (optional).
    If reject_empty=True, tiles with no positive (==1) pixels are excluded.
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
    ):
        self.img_path = str(image_tif_path)
        self.lbl_path = str(label_tif_path)
        self.T = int(tile_size)
        self.overlap = int(overlap)
        self.mean, self.std = mean, std
        self.reject_empty = reject_empty

        with rasterio.open(self.img_path) as ds:
            self.H, self.W = ds.height, ds.width
            if ds.count < 3:
                raise ValueError("Image must have at least 3 bands (RGB).")
        with rasterio.open(self.lbl_path) as ds:
            if ds.height != self.H or ds.width != self.W:
                raise ValueError("Image and label rasters must match in size.")

        # Build initial full grid
        all_coords = compute_grid(self.H, self.W, self.T, self.overlap)

        # Optionally filter out tiles that are fully background (no label==1)
        if self.reject_empty:
            keep_coords = []
            with rasterio.open(self.lbl_path) as dl:
                for y0, x0 in all_coords:
                    lab, (w, h) = read_mask_tile(dl, x0, y0, self.T, self.W, self.H)
                    if (
                        lab == 1
                    ).any():  # keep only tiles with at least one positive pixel
                        keep_coords.append((y0, x0))
            if len(keep_coords) == 0:
                raise RuntimeError(
                    "No positive tiles found; check labels or disable reject_empty."
                )
            self.coords = keep_coords
            print(
                f"[RasterPairTileDataset] Kept {len(self.coords)} tiles with positives "
                f"(rejected {len(all_coords) - len(self.coords)} empty tiles)."
            )
        else:
            self.coords = all_coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        y0, x0 = self.coords[idx]
        with rasterio.open(self.img_path) as di, rasterio.open(self.lbl_path) as dl:
            rgb, (w, h) = read_rgb_tile(di, x0, y0, self.T, self.W, self.H)
            lab, _ = read_mask_tile(dl, x0, y0, self.T, self.W, self.H)

        rgb = normalize01_then_standardize(rgb, self.mean, self.std)
        rgb = pad_to_square(rgb, self.T)

        # label (H,W) -> (1,H,W), binarize 1, ignore 255
        lab = lab.astype(np.int32)
        ignore = lab == 255
        lab = (lab == 1).astype(np.float32)

        # pad to T×T with constant 0 for label & ignore
        ph, pw = self.T - lab.shape[0], self.T - lab.shape[1]
        if ph > 0 or pw > 0:
            lab = np.pad(lab, ((0, ph), (0, pw)), mode="constant", constant_values=0)
            ignore = np.pad(
                ignore, ((0, ph), (0, pw)), mode="constant", constant_values=False
            )

        sample = {
            "image": torch.from_numpy(rgb),  # (3,T,T)
            "target": torch.from_numpy(lab[None]),  # (1,T,T)
            "ignore": torch.from_numpy(~ignore[None]),  # keep-mask True=use pixel
        }
        return sample


def collate_pairs(batch):
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    targs = torch.stack([b["target"] for b in batch], dim=0)
    keep = torch.stack([b["ignore"] for b in batch], dim=0)
    return imgs, targs, keep


# --------------------------------- #
# ---------- TRAINER -------------- #
# --------------------------------- #


class Trainer:
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        tile_size=1024,
        overlap=512,
        batch_size=8,
        threshold=0.45,
        lr=3e-4,
        weight_decay=1e-2,
        device: Optional[str] = None,
    ):
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.threshold = float(threshold)
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
        ).to(self.device)

    def _make_loss(self, pos_weight: float = 2.0):
        bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight, device=self.device)
        )

        def dice_loss_from_logits(logits, target, keep_mask):
            probs = torch.sigmoid(logits)
            probs = probs[keep_mask]
            t = target[keep_mask]
            inter = (probs * t).sum()
            dice = 1 - (2 * inter + 1.0) / ((probs + t).sum() + 1.0)
            return dice

        def criterion(logits, target, keep_mask):
            if keep_mask is not None:
                logits = logits.masked_select(keep_mask)
                target = target.masked_select(keep_mask)
            loss_bce = bce(logits, target)
            loss_dice = dice_loss_from_logits(
                logits.view(-1, 1, 1, 1),
                target.view(-1, 1, 1, 1),
                torch.ones_like(target, dtype=torch.bool, device=target.device),
            )
            return 0.5 * loss_bce + 0.5 * loss_dice

        return criterion

    def _make_optimizer(self):
        return AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def load_checkpoint(
        self,
        ckpt_path: str | Path,
        strict: bool = True,
        map_location: str | torch.device | None = None,
    ):
        ckpt_path = Path(ckpt_path)
        device = self.device if map_location is None else map_location
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model", ckpt)  # support plain state_dict too
        missing, unexpected = self.model.load_state_dict(state, strict=strict)
        if missing:
            print(f"[load_checkpoint] Missing keys: {missing}")
        if unexpected:
            print(f"[load_checkpoint] Unexpected keys: {unexpected}")
        print(f"[load_checkpoint] Loaded weights from: {ckpt_path}")
        # Optional: record metadata if present
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.best_val = float(ckpt.get("val_dice", float("nan")))
        self.model.eval()  # set eval for inference

    def train(
        self,
        image_tif_path: Path,
        label_tif_path: Path,
        epochs: int = 60,
        val_split: float = 0.1,
        pos_weight: float = 2.0,
        num_workers: int = 2,
        ckpt_dir: Path | str = "ckpts",
        amp: bool = True,
        reject_empty: bool = True,
        experiment: str = "Footprints-UNet",
        run_name: str | None = None,
    ):

        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        dataset = RasterPairTileDataset(
            image_tif_path,
            label_tif_path,
            tile_size=self.tile_size,
            overlap=self.overlap,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            reject_empty=reject_empty,
        )

        n_total = len(dataset)
        n_val = max(1, int(val_split * n_total))
        n_train = n_total - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_pairs,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_pairs,
        )

        criterion = self._make_loss(pos_weight=pos_weight)
        optim = self._make_optimizer()
        total_steps = epochs * max(1, len(train_loader))
        scheduler = LambdaLR(optim, cosine_with_warmup(total_steps, warmup_steps=1000))
        scaler = GradScaler("cuda")

        best_val = -1.0

        # === MLflow start ===
        with MlflowLogger(experiment=experiment, run_name=run_name) as mlf:
            # Log static params once
            mlf.log_params(
                encoder_name=(
                    self.model.encoder_name
                    if hasattr(self.model, "encoder_name")
                    else "resnet34"
                ),
                encoder_weights="imagenet",
                tile_size=self.tile_size,
                overlap=self.overlap,
                batch_size=self.batch_size,
                lr=self.lr,
                weight_decay=self.weight_decay,
                epochs=epochs,
                pos_weight=pos_weight,
                val_split=val_split,
                amp=amp,
                reject_empty=reject_empty,
                device=self.device,
            )

            for ep in range(1, epochs + 1):
                # ---- Train ----
                self.model.train()
                loss_sum = 0.0
                n_imgs = 0
                t_start = time.time()

                train_bar = tqdm(
                    train_loader,
                    desc=f"Epoch {ep}/{epochs} [train]",
                    unit="batch",
                    leave=False,
                )

                for batch_idx, (imgs, targs, keep) in enumerate(train_bar, start=1):
                    imgs = imgs.to(self.device, non_blocking=True)
                    targs = targs.to(self.device, non_blocking=True)
                    keep = keep.to(self.device, non_blocking=True).bool()
                    n_imgs += imgs.size(0)

                    optim.zero_grad(set_to_none=True)
                    with autocast("cuda"):
                        logits = self.model(imgs)
                        loss = criterion(logits, targs, keep)

                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                    scheduler.step()

                    loss_sum += loss.item()
                    cur_lr = optim.param_groups[0]["lr"]
                    elapsed = max(1e-6, time.time() - t_start)
                    ips = n_imgs / elapsed

                    # live progress
                    train_bar.set_postfix(
                        {
                            "loss": f"{(loss_sum/batch_idx):.4f}",
                            "lr": f"{cur_lr:.2e}",
                            "ips": f"{ips:.2f}",
                        }
                    )

                train_loss = loss_sum / max(1, len(train_loader))

                # ---- Validate ----
                self.model.eval()
                dices, ious = [], []
                val_bar = tqdm(
                    val_loader,
                    desc=f"Epoch {ep}/{epochs} [val]  ",
                    unit="batch",
                    leave=False,
                )
                with torch.no_grad(), autocast("cuda"):
                    for imgs, targs, keep in val_bar:
                        imgs = imgs.to(self.device, non_blocking=True)
                        targs = targs.to(self.device, non_blocking=True)
                        keep = keep.to(self.device, non_blocking=True).bool()

                        logits = self.model(imgs)
                        probs = torch.sigmoid(logits)
                        probs_eval = probs * keep
                        targs_eval = targs * keep
                        preds_bin = (probs_eval >= 0.5).float()

                        d = dice_coeff(preds_bin, targs_eval).item()
                        i = iou_score(preds_bin, targs_eval).item()
                        dices.append(d)
                        ious.append(i)

                        # show running val metrics
                        val_bar.set_postfix(
                            {
                                "dice": f"{np.mean(dices):.4f}",
                                "iou": f"{np.mean(ious):.4f}",
                            }
                        )

                val_dice = float(np.mean(dices)) if dices else 0.0
                val_iou = float(np.mean(ious)) if ious else 0.0
                secs = max(1e-6, time.time() - t_start)
                ips_epoch = n_imgs / secs

                # ---- Log metrics to MLflow ----
                mlf.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_dice": val_dice,
                        "val_iou": val_iou,
                        "lr": optim.param_groups[0]["lr"],
                        "images_per_sec": ips_epoch,
                    },
                    step=ep,
                )

                # Clear one-line bars and print a concise epoch summary
                tqdm.write(
                    f"Epoch {ep:03d}/{epochs} | train_loss={train_loss:.4f} | "
                    f"val_dice={val_dice:.4f} | val_iou={val_iou:.4f} | ips={ips_epoch:.2f}"
                )

                # ---- Checkpoint (local only; not logged to MLflow) ----
                if val_dice > best_val:
                    best_val = val_dice
                    ckpt_path = Path(ckpt_dir) / "unet_best.pt"
                    torch.save(
                        {
                            "model": self.model.state_dict(),
                            "epoch": ep,
                            "val_dice": val_dice,
                            "val_iou": val_iou,
                        },
                        ckpt_path,
                    )
                    tqdm.write(
                        f"  ✔ Saved best checkpoint -> {ckpt_path} (val_dice {val_dice:.4f})"
                    )

        return best_val

    def predict(
        self,
        tif_path: Path,
        out_dir: Path | str = "out_small",
        num_workers: int = 6,
        amp: bool = True,
    ):
        tif_path = Path(tif_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        dataset = OrthoTileDataset(
            tif_path,
            tile_size=self.tile_size,
            overlap=self.overlap,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_tiles,
        )

        H, W = dataset.H, dataset.W
        prob_acc = np.zeros((H, W), dtype=np.float32)
        cnt_acc = np.zeros((H, W), dtype=np.uint16)

        self.model.eval()
        with torch.no_grad(), autocast("cuda"):
            for tiles, metas in loader:
                tiles = tiles.to(self.device, non_blocking=True)  # (B,3,T,T)
                logits = self.model(tiles)  # (B,1,T,T)
                probs = (
                    torch.sigmoid(logits).squeeze(1).float().cpu().numpy()
                )  # (B,T,T)
                for p, m in zip(probs, metas):
                    y0, x0, h, w = (
                        int(m["y0"]),
                        int(m["x0"]),
                        int(m["h"]),
                        int(m["w"]),
                    )
                    prob_acc[y0 : y0 + h, x0 : x0 + w] += p[:h, :w]
                    cnt_acc[y0 : y0 + h, x0 : x0 + w] += 1

        prob = np.zeros_like(prob_acc, dtype=np.float32)
        mask = cnt_acc > 0
        prob[mask] = prob_acc[mask] / cnt_acc[mask]

        profile = open_profile_like(tif_path)
        prob_path = out_dir / f"{tif_path.stem}_prob.tif"
        mask_path = out_dir / f"{tif_path.stem}_mask.tif"

        write_geotiff_single_band(prob, prob_path, profile, "float32")
        binary = (prob >= self.threshold).astype(np.uint8)
        write_geotiff_single_band(binary, mask_path, profile, "uint8")

        print(f"Saved:\n  Prob: {prob_path}\n  Mask: {mask_path}")
        return str(prob_path), str(mask_path)

    def predict_streaming(
        self,
        tif_path: Path,
        out_dir: Path | str = "out_stream",
        num_workers: int = 8,
        amp: bool = True,
        blocksize: int = 1024,
        tmp_dir: str | None = None,
        show_bar: bool = True,
    ):
        """
        Streaming prediction for huge orthos (e.g., 40 GB) with O(tiles) memory.
        Writes intermediate SUM and COUNT rasters to disk (tiled BigTIFF), updates them per-tile,
        then streams a final pass to produce PROB and binary MASK GeoTIFFs.
        """
        tif_path = Path(tif_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Dataset & loader
        dataset = OrthoTileDataset(
            tif_path,
            tile_size=self.tile_size,
            overlap=self.overlap,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
            pin_memory=True,
            collate_fn=collate_tiles,
        )

        H, W = dataset.H, dataset.W
        total_tiles = len(dataset)

        # Reference profile
        ref_profile = open_profile_like(tif_path)

        # Create temp folder
        tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="footprints_tmp_")
        sum_tif = Path(tmp_dir) / f"{tif_path.stem}_sum.tif"
        cnt_tif = Path(tmp_dir) / f"{tif_path.stem}_cnt.tif"

        # Create on-disk SUM (float32) and COUNT (uint16) rasters
        sum_profile = ref_profile.copy()
        sum_profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=blocksize,
            blockysize=blocksize,
            BIGTIFF="YES",
        )
        cnt_profile = ref_profile.copy()
        cnt_profile.update(
            driver="GTiff",
            dtype="uint16",
            count=1,
            compress="deflate",
            predictor=1,
            tiled=True,
            blockxsize=blocksize,
            blockysize=blocksize,
            BIGTIFF="YES",
        )

        # Create/initialize files
        with rasterio.open(sum_tif, "w", **sum_profile) as ds:
            pass
        with rasterio.open(cnt_tif, "w", **cnt_profile) as ds:
            pass

        # Speed/VRAM tweaks
        torch.backends.cudnn.benchmark = True
        self.model.eval()
        # channels_last can help with NHWC kernels on Ampere+
        self.model.to(memory_format=torch.channels_last)

        processed = 0
        t0 = time.time()
        iterator = (
            tqdm(
                loader,
                total=len(loader),
                unit="batch",
                desc="Predict (stream)",
                leave=False,
            )
            if show_bar
            else loader
        )

        # Pass 1: accumulate per tile into SUM/COUNT rasters on disk
        with torch.no_grad(), autocast("cuda"):
            for tiles, metas in iterator:
                tiles = tiles.to(self.device, non_blocking=True).to(
                    memory_format=torch.channels_last
                )
                logits = self.model(tiles)
                probs = (
                    torch.sigmoid(logits).squeeze(1).float().cpu().numpy()
                )  # (B,T,T)

                # Update SUM/COUNT windows
                with rasterio.open(
                    sum_tif, "r+", sharing=False
                ) as sum_ds, rasterio.open(cnt_tif, "r+", sharing=False) as cnt_ds:

                    for p, m in zip(probs, metas):
                        y0, x0, h, w = (
                            int(m["y0"]),
                            int(m["x0"]),
                            int(m["h"]),
                            int(m["w"]),
                        )
                        win = Window(x0, y0, w, h)

                        # read->modify->write small windows
                        cur_sum = sum_ds.read(1, window=win)
                        cur_cnt = cnt_ds.read(1, window=win)

                        cur_sum += p[:h, :w].astype(np.float32)
                        cur_cnt += 1  # uint16 is fine unless overlap is extreme

                        sum_ds.write(cur_sum, 1, window=win)
                        cnt_ds.write(cur_cnt, 1, window=win)

                processed += len(metas)
                if show_bar:
                    elapsed = max(1e-6, time.time() - t0)
                    tiles_per_sec = processed / elapsed
                    # Estimate coverage quickly from count blocks (optional extra read avoided)
                    iterator.set_postfix(
                        {
                            "tiles": f"{processed}/{total_tiles}",
                            "t/s": f"{tiles_per_sec:.2f}",
                        }
                    )

        # Pass 2: stream SUM/COUNT -> PROB and MASK, window by window
        prob_path = out_dir / f"{tif_path.stem}_prob.tif"
        mask_path = out_dir / f"{tif_path.stem}_mask.tif"

        prob_profile = ref_profile.copy()
        prob_profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=blocksize,
            blockysize=blocksize,
            BIGTIFF="YES",
        )
        mask_profile = ref_profile.copy()
        mask_profile.update(
            driver="GTiff",
            dtype="uint8",
            count=1,
            compress="deflate",
            predictor=1,
            tiled=True,
            blockxsize=blocksize,
            blockysize=blocksize,
            BIGTIFF="YES",
        )

        # Prepare block grid for streaming division
        bx, by = blocksize, blocksize
        xs = list(range(0, W, bx))
        ys = list(range(0, H, by))
        if xs[-1] + bx < W:
            xs.append(W - bx)
        if ys[-1] + by < H:
            ys.append(H - by)
        blocks = [(y, x) for y in ys for x in xs]

        it2 = (
            tqdm(blocks, desc="Finalize (sum/count→prob)", unit="block", leave=False)
            if show_bar
            else blocks
        )

        with rasterio.open(sum_tif, "r") as sum_ds, rasterio.open(
            cnt_tif, "r"
        ) as cnt_ds, rasterio.open(
            prob_path, "w", **prob_profile
        ) as prob_ds, rasterio.open(
            mask_path, "w", **mask_profile
        ) as mask_ds:

            for y0, x0 in it2:
                w = min(bx, W - x0)
                h = min(by, H - y0)
                win = Window(x0, y0, w, h)

                s = sum_ds.read(1, window=win).astype(np.float32)
                c = cnt_ds.read(1, window=win).astype(np.float32)

                # avoid div-by-zero
                prob_block = np.zeros_like(s, dtype=np.float32)
                nz = c > 0
                prob_block[nz] = s[nz] / c[nz]

                mask_block = (prob_block >= self.threshold).astype(np.uint8)

                prob_ds.write(prob_block, 1, window=win)
                mask_ds.write(mask_block, 1, window=win)

        # Optional: clean up tmp files
        try:
            os.remove(sum_tif)
            os.remove(cnt_tif)
        except Exception:
            pass

        total_secs = time.time() - t0
        if show_bar:
            tqdm.write(
                f"Streaming prediction done in {total_secs:.2f}s for {total_tiles} tiles"
            )

        print(f"Saved:\n  Prob: {prob_path}\n  Mask: {mask_path}")
        return str(prob_path), str(mask_path)


# ---------------------------- #
# ---------- MAIN ------------ #
# ---------------------------- #

if __name__ == "__main__":
    # Example usage:
    trainer = Trainer(tile_size=1024, overlap=512, batch_size=14, threshold=0.45)
    # trainer.train(
    #     image_tif_path=Path(DATASET_DIR / "ortho_cog_cropped.tif"),
    #     label_tif_path=Path(DATASET_DIR / "building_mask.tif"),
    #     epochs=60,
    #     reject_empty=True,
    # )
    trainer.load_checkpoint("./ckpts/unet_best.pt")
    trainer.predict_streaming(Path(DATASET_DIR / "ortho.tif"), out_dir="out_small")
    pass
