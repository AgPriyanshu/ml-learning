import os
import tempfile
import time
from pathlib import Path
from typing import  Optional

import numpy as np
import rasterio
import torch
import cv2
from scipy import ndimage
from configs import setup_environment
from segmentation_models_pytorch import DeepLabV3Plus
from segmentation_models_pytorch.base import SegmentationModel
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.datasets.ortho_dataset import OrthoTileDataset
from src.data.io_utils import get_file_profile, normalize01_then_standardize, read_rgb_tile_scaled
from src.models.postprocessors.tile_blenders import apply_tile_weights, create_gaussian_weight_map
from src.shared.constants import IMAGENET_MEAN, IMAGENET_STD
from src.models.metrics import dice_coeff, iou_score
from src.shared.mlflow_helpers import MlflowLogger
from rasterio.windows import Window

setup_environment()


class DeepLabV3Trainer:
    def __init__(
        self,
        model: SegmentationModel,
        batch_size,
        loss_function,
        postprocessor: Optional[callable] = None,
        # Train params.
        freeze_encoder_epochs: int = 1,  # warmup: freeze encoder N epochs
        # Training utils.
        device=None,
        ema=None,
        compile_model: bool = False,  # torch.compile for extra speed
        # grad_clip_norm: float = 1.0,
    ):
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.model = model
        self.device = device

        # Store train knobs
        self.freeze_encoder_epochs = max(0, int(freeze_encoder_epochs))

        # EMA
        self.ema = ema

        # (Optional) compile
        if compile_model and torch.cuda.is_available() and torch.__version__ >= "2.1":
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass  # fall back silently

    def filter_vegetation_false_positives(
        self, imgs: torch.Tensor, probs: torch.Tensor, vegetation_threshold: float = 0.15
    ) -> torch.Tensor:
        """
        Remove building predictions in areas likely to be vegetation using RGB analysis.
        
        Args:
            imgs: RGB images (B, 3, H, W) normalized with ImageNet stats
            probs: Building probability maps (B, 1, H, W)
            vegetation_threshold: Threshold for vegetation detection (higher = more strict)
        
        Returns:
            Filtered probability maps with reduced vegetation false positives
        """
        # Denormalize from ImageNet normalization to [0,1]
        mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)
        
        rgb = imgs * std + mean  # Back to [0,1]
        rgb = torch.clamp(rgb, 0, 1)
        
        # Extract RGB channels
        r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        
        # Vegetation indicators
        # 1. Excess Green Index: 2*G - R - B (higher for vegetation)
        exg = 2 * g - r - b
        
        # 2. Green-Red difference (vegetation has high green, low red)
        gr_diff = g - r
        
        # 3. Normalized green excess relative to overall brightness
        intensity = (r + g + b) / 3.0
        norm_green = (g - intensity) / (intensity + 1e-6)
        
        # 4. Green dominance: how much green exceeds other channels
        green_dominance = g - torch.max(r, b)
        
        # Combine vegetation indicators (higher = more vegetation-like)
        vegetation_score = (exg + gr_diff + norm_green + green_dominance) / 4.0
        
        # Create vegetation mask: areas likely to be vegetation
        vegetation_mask = (vegetation_score > vegetation_threshold).float()
        
        # Apply vegetation filter: reduce building probability in vegetation areas
        # Use 90% reduction in vegetation areas
        reduction_factor = 0.9
        filtered_probs = probs.squeeze(1) * (1.0 - vegetation_mask * reduction_factor)
        
        return filtered_probs.unsqueeze(1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scheduler,
        optimizer,
        epochs: int,
        ckpt_dir: Path | str = "checkpoints",
        amp: bool = True,  # Automatic Mixed Precision
        experiment: str = "Footprints-DeepLabV3Plus-Multidataset",
        run_name: str | None = None,
        early_stop_patience: int = 10,
    ):
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        criterion = self.loss_function()
        scaler = GradScaler(enabled=(amp and self.device == "cuda"))

        best_val = -1.0
        bad_epochs = 0

        # Optionally freeze encoder for warmup epochs
        if self.freeze_encoder_epochs > 0:
            for parameter in self.model.encoder.parameters():
                parameter.requires_grad = False

        with MlflowLogger(experiment=experiment, run_name=run_name) as mlf:
            mlf.log_params(
                batch_size=self.batch_size,
                epochs=epochs,
                amp=amp,
                device=self.device,
                freeze_encoder_epochs=self.freeze_encoder_epochs,
            )
            
            global_step = 0
            for ep in range(1, epochs + 1):
                # Unfreeze encoder after warmup
                if ep == (self.freeze_encoder_epochs + 1):
                    for parameter in self.model.encoder.parameters():
                        parameter.requires_grad = True

                self.model.train()
                loss_sum, n_imgs = 0.0, 0
                t_start = time.time()
                train_bar = tqdm(
                    train_loader,
                    desc=f"Epoch {ep}/{epochs} [train]",
                    unit="batch",
                    leave=False,
                )

                optimizer.zero_grad(set_to_none=True)
                for bidx, (imgs, targs, keep) in enumerate(train_bar, start=1):
                    imgs = imgs.to(self.device, non_blocking=True).to(
                        memory_format=torch.channels_last
                    )
                    targs = targs.to(self.device, non_blocking=True)
                    keep = keep.to(self.device, non_blocking=True).bool()
                    n_imgs += imgs.size(0)

                    with autocast(
                        device_type="cuda", enabled=(amp and self.device == "cuda")
                    ):
                        logits = self.model(imgs)
                        loss = criterion(logits, targs, keep)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    # EMA update after optimizer step
                    if self.ema is not None:
                        self.ema.update(self.model)

                    loss_sum += loss.item()
                    global_step += 1
                    
                    # Prevent GPU memory accumulation every 10 batches
                    if bidx % 10 == 0 and self.device == "cuda":
                        torch.cuda.empty_cache()

                    cur_lr = optimizer.param_groups[0]["lr"]
                    elapsed = max(1e-6, time.time() - t_start)
                    ips = n_imgs / elapsed
                    train_bar.set_postfix(
                        {
                            "loss": f"{(loss_sum/bidx):.4f}",
                            "lr": f"{cur_lr:.2e}",
                            "ips": f"{ips:.2f}",
                        }
                    )

                train_loss = loss_sum / max(1, len(train_loader))

                # ---- Validate (use EMA weights if enabled) ----
                self.model.eval()
                eval_model = self.model
                if self.ema is not None:
                    # temp swap: copy EMA weights into a shadow model for eval
                    eval_model = (
                        DeepLabV3Plus(
                            encoder_name="resnet50",
                            encoder_weights=None,
                            in_channels=3,
                            classes=1,
                        )
                        .to(self.device)
                        .to(memory_format=torch.channels_last)
                    )
                    eval_model.load_state_dict(self.model.state_dict(), strict=False)
                    self.ema.apply_to(eval_model)
                    eval_model.eval()

                dices, ious = [], []
                val_bar = tqdm(
                    val_loader,
                    desc=f"Epoch {ep}/{epochs} [val]  ",
                    unit="batch",
                    leave=False,
                )
                with torch.inference_mode(), autocast(
                    device_type="cuda", enabled=(amp and self.device == "cuda")
                ):
                    for imgs, targs, keep in val_bar:
                        imgs = imgs.to(self.device, non_blocking=True).to(
                            memory_format=torch.channels_last
                        )
                        targs = targs.to(self.device, non_blocking=True)
                        keep = keep.to(self.device, non_blocking=True).bool()

                        logits = eval_model(imgs)
                        probs = torch.sigmoid(logits)
                        # Apply vegetation filtering to reduce false positives
                        filtered_probs = self.filter_vegetation_false_positives(imgs, probs)
                        preds_bin = (filtered_probs >= 0.5).float() * keep
                        targs_eval = targs * keep

                        d = dice_coeff(preds_bin, targs_eval).item()
                        i = iou_score(preds_bin, targs_eval).item()
                        dices.append(d)
                        ious.append(i)
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

                
                
                # Force GPU memory cleanup to prevent leaks
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Log training metrics
                train_metrics = {
                    "loss": train_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    "images_per_sec": ips_epoch,
                }
                mlf.log_train_metrics(train_metrics, step=ep)

                # Log validation metrics
                val_metrics = {
                    "dice": val_dice,
                    "iou": val_iou,
                }
                mlf.log_val_metrics(val_metrics, step=ep)

                tqdm.write(
                    f"Epoch {ep:03d}/{epochs} | train_loss={train_loss:.4f} | "
                    f"val_dice={val_dice:.4f} | val_iou={val_iou:.4f} | ips={ips_epoch:.2f}"
                )

                # Checkpoint
                improved = val_dice > best_val
                if improved:
                    best_val = val_dice
                    bad_epochs = 0
                    ckpt_path = Path(ckpt_dir) / "deeplabv3p_best_overviews.pt"
                    to_save = self.model.state_dict()
                    if self.ema is not None:
                        # Save EMA weights (often better at inference)
                        ema_copy = {k: v.clone() for k, v in self.ema.shadow.items()}
                        for k, v in ema_copy.items():
                            if k in to_save:
                                to_save[k] = v
                    torch.save(
                        {
                            "model": to_save,
                            "epoch": ep,
                            "val_dice": val_dice,
                            "val_iou": val_iou,
                        },
                        ckpt_path,
                    )
                    tqdm.write(
                        f"  ✔ Saved best checkpoint -> {ckpt_path} (val_dice {val_dice:.4f})"
                    )
                else:
                    bad_epochs += 1
                    if bad_epochs >= early_stop_patience:
                        tqdm.write(
                            f"Early stopping at epoch {ep} (no improvement for {bad_epochs} epochs)."
                        )
                        break

        return best_val

    def _clean_mask_artifacts(
        self, 
        mask: np.ndarray, 
        min_object_size: int = 100,
        morphology_kernel_size: int = 3,
        fill_holes: bool = True,
        show_bar: bool = True
    ) -> np.ndarray:
        """
        Clean artifacts from binary mask using morphological operations and size filtering.
        
        Args:
            mask: Binary mask (0/1 values)
            min_object_size: Minimum object size in pixels
            morphology_kernel_size: Kernel size for morphological operations
            fill_holes: Whether to fill holes in objects
            show_bar: Whether to show progress messages
            
        Returns:
            Cleaned binary mask
        """
        if show_bar:
            tqdm.write("  Cleaning mask artifacts...")
        
        # Ensure binary mask
        cleaned_mask = (mask > 0).astype(np.uint8)
        
        # Apply morphological operations to reduce noise
        if morphology_kernel_size > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (morphology_kernel_size, morphology_kernel_size)
            )
            
            # Opening: remove small noise (erosion followed by dilation)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Closing: fill small gaps (dilation followed by erosion)
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Fill holes in objects
        if fill_holes:
            cleaned_mask = ndimage.binary_fill_holes(cleaned_mask).astype(np.uint8)
        
        # Remove small objects
        if min_object_size > 0:
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                cleaned_mask, connectivity=8
            )
            
            # Create mask for objects larger than min_object_size
            large_objects_mask = np.zeros_like(cleaned_mask)
            
            # Skip background (label 0)
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if area >= min_object_size:
                    large_objects_mask[labels == label] = 1
            
            cleaned_mask = large_objects_mask
            
            # Count removed artifacts
            removed_objects = num_labels - 1 - np.sum(large_objects_mask > 0)
            if show_bar and removed_objects > 0:
                tqdm.write(f"    Removed {removed_objects} small artifacts (< {min_object_size} pixels)")
        
        return cleaned_mask

    def predict(
        self,
        tif_path: Path,
        threshold: float,
        out_dir: Path | str = "out_stream",
        num_workers: int = 2,
        amp: bool = True,
        blocksize: int = 1024,
        tmp_dir: str | None = None,
        show_bar: bool = True,
        predict_scales: tuple[int, ...] = (1, 2),  # << NEW
        batch_size: int = 8,  # Increased default for better GPU utilization
        # Artifact removal parameters
        remove_artifacts: bool = True,
        min_object_size: int = 100,  # Minimum object size in pixels
        morphology_kernel_size: int = 3,  # Kernel size for morphological operations
        fill_holes: bool = True,  # Fill holes in detected objects
        # Performance parameters
        enable_profiling: bool = False,  # Enable performance profiling
        # Blending parameters
        gaussian_sigma_factor: float = 0.25,  # Controls Gaussian blending sharpness (lower = crisper)
    ):
        """
        Streaming prediction for huge orthos (e.g., 40 GB) with O(tiles) memory.
        Writes intermediate SUM and COUNT rasters to disk (tiled BigTIFF), updates them per-tile,
        then streams a final pass to produce PROB and binary MASK GeoTIFFs.
        
        Args:
            tif_path: Input orthophoto path
            threshold: Probability threshold for building detection
            out_dir: Output directory for results
            num_workers: Number of data loader workers
            amp: Use automatic mixed precision
            blocksize: Block size for tiled GeoTIFF output
            tmp_dir: Temporary directory for intermediate files
            show_bar: Show progress bars
            predict_scales: Scales for multi-scale prediction
            batch_size: Batch size for inference
            remove_artifacts: Whether to apply artifact removal post-processing
            min_object_size: Minimum object size in pixels to keep (removes small noise)
            morphology_kernel_size: Kernel size for morphological operations (noise reduction)
            fill_holes: Whether to fill holes in detected objects
            enable_profiling: Enable performance profiling to identify bottlenecks
            gaussian_sigma_factor: Controls Gaussian blending sharpness (0.1-0.5, lower = crisper boundaries)
            
        Returns:
            Tuple of (prob_path, mask_path)
        """
        tif_path = Path(tif_path)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.tile_size = 1024
        self.overlap = 512
        self.batch_size = batch_size

        # Dataset & loader
        dataset = OrthoTileDataset(
            tif_path,
            tile_size=self.tile_size,
            overlap=self.overlap,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,  # Increased prefetch
            pin_memory=True,
            collate_fn=dataset.collate_tiles,
        )

        H, W = dataset.H, dataset.W
        total_tiles = len(dataset)

        # Reference profile
        ref_profile = get_file_profile(tif_path)

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
            dtype="float32",  # Changed to float32 for weight accumulation
            count=1,
            compress="deflate",
            predictor=2,  # Use predictor=2 for float data
            tiled=True,
            blockxsize=blocksize,
            blockysize=blocksize,
            BIGTIFF="YES",
        )

        # Create/initialize files
        with rasterio.open(sum_tif, "w", **sum_profile):
            pass
        with rasterio.open(cnt_tif, "w", **cnt_profile):
            pass

        # Create weight map for smooth tile blending using Gaussian for crisp boundaries
        weight_map = create_gaussian_weight_map(self.tile_size, sigma_factor=gaussian_sigma_factor)
        
        if show_bar:
            tqdm.write(f"Using Gaussian blending (σ={gaussian_sigma_factor:.2f}) with tile_size={self.tile_size}, overlap={self.overlap}")

        # Speed/VRAM tweaks
        torch.backends.cudnn.benchmark = True
        self.model.eval()
        # channels_last can help with NHWC kernels on Ampere+
        self.model.to(memory_format=torch.channels_last)
        
        # Compile model for faster inference if supported
        if hasattr(torch, 'compile') and self.device == "cuda":
            try:
                self.model = torch.compile(self.model, mode="max-autotune")
                if show_bar:
                    tqdm.write("Model compiled for faster inference")
            except Exception as e:
                if show_bar:
                    tqdm.write(f"Model compilation failed, continuing without: {e}")

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
        
        # Initialize profiling
        profiler = None
        if enable_profiling:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            profiler.start()
            if show_bar:
                tqdm.write("Profiling enabled - logs will be saved to ./profiler_logs")

        # Pass 1: accumulate per tile into SUM/COUNT rasters on disk
        # Open base once to avoid re-opening per tile
        base_ds = rasterio.open(tif_path)

        with torch.no_grad(), autocast("cuda"):
            for batch_idx, (tiles, metas) in enumerate(iterator):
                # We will ignore 'tiles' (full-res) and read per-scale directly from base_ds
                ms_logits_sum = None  # will be torch tensor (B,1,T,T)

                for scale in predict_scales:
                    # Build a batch at this scale by reading windows with out_shape
                    batch_list = []
                    for m in metas:
                        y0, x0, h, w = (
                            int(m["y0"]),
                            int(m["x0"]),
                            int(m["h"]),
                            int(m["w"]),
                        )
                        # scaled read (3,hs,ws)
                        rgb_s, _ = read_rgb_tile_scaled(
                            base_ds, x0, y0, self.tile_size, dataset.W, dataset.H, scale
                        )
                        rgb_s = normalize01_then_standardize(
                            rgb_s, IMAGENET_MEAN, IMAGENET_STD
                        )
                        t = torch.from_numpy(rgb_s)  # (3,hs,ws)
                        # Only interpolate if needed
                        if rgb_s.shape[1] != self.tile_size or rgb_s.shape[2] != self.tile_size:
                            t = torch.nn.functional.interpolate(
                                t.unsqueeze(0),
                                size=(self.tile_size, self.tile_size),
                                mode="bilinear",
                                align_corners=False,
                                antialias=True  # Better quality interpolation
                            ).squeeze(0)
                        batch_list.append(t)
                    batch = torch.stack(batch_list, dim=0).to(
                        self.device, non_blocking=True
                    )  # (B,3,T,T)
                    batch = batch.to(memory_format=torch.channels_last)

                    # Mark step for CUDA graphs when using torch.compile
                    if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                        torch.compiler.cudagraph_mark_step_begin()
                    
                    logits = self.model(batch)  # (B,1,T,T)
                    
                    # Clone logits to prevent CUDA graph overwrites with torch.compile
                    logits = logits.clone()
                    
                    ms_logits_sum = (
                        logits if ms_logits_sum is None else (ms_logits_sum + logits)
                    )

                # Average across scales, then to probs (CPU) for accumulation
                logits_avg = ms_logits_sum / float(len(predict_scales))
                probs = (
                    torch.sigmoid(logits_avg).squeeze(1).float().cpu().numpy()
                )  # (B,T,T)
                
                # Clean up GPU memory after batch processing
                del ms_logits_sum, logits_avg, batch
                if self.device == "cuda":
                    torch.cuda.empty_cache()

                # Update SUM/COUNT windows with weighted blending
                # Process batch of updates to reduce I/O operations
                batch_updates = []
                for p, m in zip(probs, metas):
                    y0, x0, h, w = (
                        int(m["y0"]),
                        int(m["x0"]),
                        int(m["h"]),
                        int(m["w"]),
                    )
                    win = Window(x0, y0, w, h)

                    # Apply spatial weights to tile prediction
                    tile_pred = p[:h, :w].astype(np.float32)
                    tile_weights = weight_map[:h, :w]
                    weighted_pred = apply_tile_weights(tile_pred, tile_weights)
                    
                    batch_updates.append((win, weighted_pred, tile_weights))
                
                # Batch write to reduce file I/O overhead
                with rasterio.open(
                    sum_tif, "r+", sharing=False
                ) as sum_ds, rasterio.open(cnt_tif, "r+", sharing=False) as cnt_ds:
                    for win, weighted_pred, tile_weights in batch_updates:
                        # Read current accumulated values
                        cur_sum = sum_ds.read(1, window=win)
                        cur_cnt = cnt_ds.read(1, window=win)

                        # Accumulate weighted predictions and weights
                        cur_sum += weighted_pred
                        cur_cnt += tile_weights  # Accumulate weights instead of just counting

                        # Write back
                        sum_ds.write(cur_sum, 1, window=win)
                        cnt_ds.write(cur_cnt, 1, window=win)

                processed += len(metas)
                if show_bar:
                    elapsed = max(1e-6, time.time() - t0)
                    iterator.set_postfix(
                        {
                            "tiles": f"{processed}/{total_tiles}",
                            "t/s": f"{processed/elapsed:.2f}",
                        }
                    )
                
                # Update profiler
                if profiler:
                    profiler.step()

        base_ds.close()
        
        # Stop profiler
        if profiler:
            profiler.stop()
            if show_bar:
                tqdm.write("Profiling complete - check ./profiler_logs for results")
        
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

                # Normalize by accumulated weights (avoid div-by-zero)
                prob_block = np.zeros_like(s, dtype=np.float32)
                # Use a small threshold to avoid division by very small weights
                weight_threshold = 1e-6
                valid_weights = c > weight_threshold
                prob_block[valid_weights] = s[valid_weights] / c[valid_weights]

                mask_block = (prob_block >= threshold).astype(np.uint8)

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

        # Apply artifact removal post-processing if requested
        if remove_artifacts:
            if show_bar:
                tqdm.write("Applying artifact removal post-processing...")
            
            try:
                # Create cleaned mask path
                cleaned_mask_path = out_dir / f"{tif_path.stem}_mask_cleaned.tif"
                
                # Process mask in blocks to handle large files
                with rasterio.open(mask_path, "r") as src_mask:
                    profile = src_mask.profile.copy()
                    
                    with rasterio.open(cleaned_mask_path, "w", **profile) as dst_mask:
                        # Process in blocks
                        for y0, x0 in it2:
                            w = min(blocksize, W - x0)
                            h = min(blocksize, H - y0)
                            win = Window(x0, y0, w, h)
                            
                            # Read block
                            mask_block = src_mask.read(1, window=win)
                            
                            # Clean artifacts in this block
                            if np.any(mask_block > 0):  # Only process blocks with detections
                                cleaned_block = self._clean_mask_artifacts(
                                    mask_block,
                                    min_object_size=min_object_size,
                                    morphology_kernel_size=morphology_kernel_size,
                                    fill_holes=fill_holes,
                                    show_bar=False  # Suppress per-block messages
                                )
                            else:
                                cleaned_block = mask_block
                            
                            # Write cleaned block
                            dst_mask.write(cleaned_block, 1, window=win)
                
                if show_bar:
                    tqdm.write("  Artifact removal complete")
                    tqdm.write(f"  Original mask: {mask_path}")
                    tqdm.write(f"  Cleaned mask: {cleaned_mask_path}")
                
                print(f"Saved:\n  Prob: {prob_path}\n  Mask: {mask_path}\n  Cleaned Mask: {cleaned_mask_path}")
                return str(prob_path), str(mask_path), str(cleaned_mask_path)
                
            except Exception as e:
                if show_bar:
                    tqdm.write(f"Warning: Artifact removal failed: {e}")
                print(f"Warning: Artifact removal failed: {e}")
                print(f"Saved:\n  Prob: {prob_path}\n  Mask: {mask_path}")
                return str(prob_path), str(mask_path)
        else:
            print(f"Saved:\n  Prob: {prob_path}\n  Mask: {mask_path}")
            return str(prob_path), str(mask_path)

    def load_checkpoint(
        self,
        ckpt_path: str | Path,
        strict: bool = True,
        map_location: str | torch.device | None = None,
    ):
        ckpt_path = Path(ckpt_path)
        device = self.device if map_location is None else map_location
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model", ckpt)
        missing, unexpected = self.model.load_state_dict(state, strict=strict)
        if missing:
            print(f"[load_checkpoint] Missing keys: {missing}")
        if unexpected:
            print(f"[load_checkpoint] Unexpected keys: {unexpected}")
        print(f"[load_checkpoint] Loaded: {ckpt_path}")
        self.start_epoch = int(ckpt.get("epoch", 0)) + 1
        self.best_val = float(ckpt.get("val_dice", float("nan")))
        self.model.eval()
