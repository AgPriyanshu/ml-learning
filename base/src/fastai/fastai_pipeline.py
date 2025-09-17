from operator import itemgetter
from pathlib import Path
from src.shared.constants import DATASET_DIR
from src.shared.tiler import Tiler, BatchConfig
import numpy as np
import gc
from fastai.data.block import DataBlock
from fastai.data.transforms import RandomSplitter
from fastai.vision.all import *
from fastai.callback.all import EarlyStoppingCallback
from PIL import Image
import torch\

class MemoryEfficientTilerDataset:
    """Memory-optimized dataset - NO CACHING."""
    def __init__(self, image_tiler: Tiler, label_tiler: Tiler = None, min_building_ratio: float = 0.1):
        self.image_tiler = image_tiler
        self.label_tiler = label_tiler

        if label_tiler and min_building_ratio > 0:
            self.valid_indices = self._filter_tiles_by_content_efficient(min_building_ratio)
        else:
            self.valid_indices = list(range(len(image_tiler)))

        print(f"Memory-Efficient Tiler Dataset: {len(self.valid_indices)} valid tiles")

    def _filter_tiles_by_content_efficient(self, min_building_ratio):
        valid_indices = []
        for idx in range(0, len(self.image_tiler), 20):
            try:
                label_tile = self.label_tiler.get_tile_by_id(idx, as_numpy=True)  # C,H,W
                building_ratio = np.mean(label_tile == 1)
                if building_ratio >= min_building_ratio:
                    start_idx = max(0, idx - 2)
                    end_idx = min(len(self.image_tiler), idx + 3)
                    valid_indices.extend(range(start_idx, end_idx))
            except Exception as e:
                print(f"Warning: Could not check tile {idx}: {e}")
                continue
        return sorted(set(valid_indices))

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]

        # ----- IMAGE (ensure 3 channels: RGB) -----
        image_np = self.image_tiler.get_tile_by_id(actual_idx, as_numpy=True)  # C,H,W
        img_hwc = image_np.transpose(1, 2, 0)                                  # H,W,C
        if img_hwc.ndim == 2:  # single band -> RGB
            img_hwc = np.repeat(img_hwc[..., None], 3, axis=2)
        elif img_hwc.shape[2] >= 4:  # drop alpha/extra bands
            img_hwc = img_hwc[:, :, :3]
        image_pil = Image.fromarray(img_hwc.astype(np.uint8))
        if image_pil.mode != 'RGB':               # ensure RGB (fixes RGBA)
            image_pil = image_pil.convert('RGB')

        # ----- MASK (keep 1 channel; class ids 0/1) -----
        label_np = self.label_tiler.get_tile_by_id(actual_idx, as_numpy=True)  # C,H,W or H,W
        if label_np.ndim == 3:
            label_np = np.squeeze(label_np, axis=0) if label_np.shape[0] in (1, 3) else label_np[0]
        mask = (label_np == 1).astype(np.uint8)
        mask_pil = Image.fromarray(mask).convert('L')   # avoid 'mode=' deprecation
        return (image_pil, mask_pil)

def create_memory_efficient_tiler_dls(
    image_path: str,
    label_path: str = None,
    tile_size: int = 256,
    overlap: int = 32,
    batch_size: int = 2,
    valid_pct: float = 0.2,
    min_building_ratio: float = 0.1,
    **kwargs
):
    # MEMORY-OPTIMIZED CONFIG
    config = BatchConfig(
        batch_size=8,
        prefetch_batches=2,
        max_workers=4,
        enable_cache=False,
        memory_limit_mb=2048
    )

    image_tiler = Tiler(image_path, tile_size=tile_size, overlap=overlap, batch_config=config)
    label_tiler = Tiler(label_path, tile_size=tile_size, overlap=overlap, batch_config=config) if label_path else None

    dataset = MemoryEfficientTilerDataset(
        image_tiler=image_tiler,
        label_tiler=label_tiler,
        min_building_ratio=min_building_ratio
    )

    def get_items(_): return list(range(len(dataset)))

    def get_x(i):
        itm = dataset[i]
        return itm[0] if isinstance(itm, tuple) else itm

    if label_path:
        def get_y(i):
            _, m = dataset[i]
            return m

        dblock = DataBlock(
            blocks=(ImageBlock, MaskBlock(codes=['background', 'building'])),
            get_items=get_items,
            get_x=get_x,
            get_y=get_y,
            splitter=RandomSplitter(valid_pct=valid_pct, seed=42),
            batch_tfms=[IntToFloatTensor(), Normalize.from_stats(*imagenet_stats)]
        )
    else:
        dblock = DataBlock(
            blocks=(ImageBlock,),
            get_items=get_items,
            get_x=get_x,
            splitter=RandomSplitter(valid_pct=valid_pct, seed=42),
            batch_tfms=[IntToFloatTensor(), Normalize.from_stats(*imagenet_stats)]
        )

    # IMPORTANT: don't pass path=None; use '.' or omit entirely. Also use num_workers=0.
    return dblock.dataloaders(
        source=None,
        bs=batch_size,
        num_workers=0,
        # path='.'   # optional; or just omit the path argument
        **kwargs
    )
# Memory monitoring utility
def monitor_memory():
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.1f} MB")
    return memory_mb

def train_with_memory_monitoring(image_path: str, label_path: str, arch=resnet34, **kwargs):
    print("🧠 Initial memory usage:")
    monitor_memory()
    dls = create_memory_efficient_tiler_dls(
        image_path=image_path,
        label_path=label_path,
        **kwargs
    )
    print("🧠 After creating DataLoaders:")
    monitor_memory()
    print(f"🎯 Created DataLoaders with {len(dls.train)} batches")

    learn = unet_learner(
        dls,
        arch,
        metrics=[DiceMulti()],
        loss_func=CrossEntropyLossFlat(axis=1),
        cbs=[EarlyStoppingCallback(patience=3)],
        pretrained=True
    )
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    learn.model.to(dev)

    print("🧠 After creating learner:")
    monitor_memory()
    return learn, dls

if __name__ == "__main__":
    print("🧠 Starting memory usage:")
    monitor_memory()

    learn, dls = train_with_memory_monitoring(
        image_path=Path(DATASET_DIR / "ortho_cog_cropped.tif"),
        label_path=Path(DATASET_DIR / "building_mask.tif"),
        tile_size=1024,
        overlap=512,
        batch_size=4,
        min_building_ratio=0.05
    )

    print("🧠 Final memory usage:")
    monitor_memory()


    # try:
    #     xb, yb = dls.one_batch()
    #     dls.show_batch()
    #     learn.fine_tune(10)
    #     print(f"✅ Batch shapes: xb={xb.shape}  yb={yb.shape}")

        
    # except Exception as e:
    #     print(f"❌ Error loading batch: {e}")
    # ---- Quick prediction sanity-check on one batch ----
    xb, yb = dls.one_batch()
    learn.model.eval()
    with torch.inference_mode():
        logits = learn.model(xb)            # [B, C, H, W]
        pred = logits.argmax(dim=1)         # [B, H, W]  (0=background, 1=building)

    print("xb:", xb.shape, "yb:", yb.shape, "pred:", pred.shape)

    # visualize a couple of samples
    import matplotlib.pyplot as plt
    for i in range(min(3, xb.shape[0])):
        img = xb[i].cpu().permute(1,2,0).clamp(0,1).numpy()
        gt  = yb[i].cpu().numpy()
        pr  = pred[i].cpu().numpy()
        fig, axs = plt.subplots(1,3, figsize=(10,3))
        axs[0].imshow(img); axs[0].set_title("Image"); axs[0].axis('off')
        axs[1].imshow(gt, vmin=0, vmax=1); axs[1].set_title("GT"); axs[1].axis('off')
        axs[2].imshow(pr, vmin=0, vmax=1); axs[2].set_title("Pred"); axs[2].axis('off')
        plt.show()
