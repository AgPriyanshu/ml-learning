from pathlib import Path
from typing import List, Tuple
import numpy as np
import math
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio import open as ropen

from src.shared.constants import IMAGENET_MEAN, IMAGENET_STD


def get_file_profile(ref_tif: Path) -> dict:
    with ropen(ref_tif) as ref:
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


def read_rgb_tile(ds, x0, y0, T, W, H):
    x1, y1 = min(x0 + T, W), min(y0 + T, H)
    win = Window(x0, y0, x1 - x0, y1 - y0)
    tile = ds.read(indexes=[1, 2, 3], window=win).astype(np.float32)  # (3,h,w)
    return tile, (x1 - x0, y1 - y0)


def read_mask_tile(ds, x0, y0, T, W, H):
    x1, y1 = min(x0 + T, W), min(y0 + T, H)
    win = Window(x0, y0, x1 - x0, y1 - y0)
    m = ds.read(1, window=win)
    return m, (x1 - x0, y1 - y0)


def _scaled_out_hw(x0, y0, T, W, H, scale: int) -> tuple[int, int, int, int, int, int]:
    # Compute full-res window and the target scaled size that matches decimation
    x1, y1 = min(x0 + T, W), min(y0 + T, H)
    w, h = x1 - x0, y1 - y0
    # ensure at least 1 pixel at very small fragments
    out_w = max(1, math.ceil(w / scale))
    out_h = max(1, math.ceil(h / scale))
    return x0, y0, w, h, out_w, out_h


def read_rgb_tile_scaled(
    ds, x0, y0, T, W, H, scale: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Read an RGB tile at 'scale' (1=full, 2=half, 4=quarter...). Rasterio/GDAL
    will automatically use overviews if they exist for this decimation.
    """
    x0, y0, w, h, out_w, out_h = _scaled_out_hw(x0, y0, T, W, H, scale)
    win = Window(x0, y0, w, h)
    tile = ds.read(
        indexes=[1, 2, 3],
        window=win,
        out_shape=(3, out_h, out_w),
        resampling=Resampling.bilinear,
        boundless=False,
        fill_value=0,
    ).astype(np.float32)
    return tile, (w, h)  # return original w,h for placement


def read_mask_tile_scaled(
    ds, x0, y0, T, W, H, scale: int
) -> tuple[np.ndarray, tuple[int, int]]:
    x0, y0, w, h, out_w, out_h = _scaled_out_hw(x0, y0, T, W, H, scale)
    win = Window(x0, y0, w, h)
    m = ds.read(
        1,
        window=win,
        out_shape=(out_h, out_w),
        resampling=Resampling.nearest,
        boundless=False,
        fill_value=255,  # preserve ignore
    )
    return m, (w, h)


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
    with ropen(out_path, "w", **p) as dst:
        dst.write(array.astype(dtype), 1)


def normalize01_then_standardize(arr, mean=IMAGENET_MEAN, std=IMAGENET_STD):
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
