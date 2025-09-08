import numpy as np


def create_gaussian_weight_map(tile_size: int, sigma_factor: float = 0.3) -> np.ndarray:
    """
    Create a 2D Gaussian weight map for smooth tile blending.

    Args:
        tile_size: Size of the square tile
        sigma_factor: Controls the spread of the Gaussian (0.1-0.5)

    Returns:
        2D weight map with values from 0 to 1, peak at center
    """
    center = tile_size // 2
    sigma = tile_size * sigma_factor

    # Create coordinate grids
    y, x = np.ogrid[:tile_size, :tile_size]

    # Calculate squared distance from center
    dist_sq = (x - center) ** 2 + (y - center) ** 2

    # Gaussian weight map
    weight_map = np.exp(-dist_sq / (2 * sigma**2))

    return weight_map.astype(np.float32)


def create_distance_weight_map(tile_size: int, border_fade: int) -> np.ndarray:
    """
    Create a distance-based weight map that fades to 0 at edges.

    Args:
        tile_size: Size of the square tile
        border_fade: Width of border fade region (default: tile_size // 8)

    Returns:
        2D weight map with smooth fade to 0 at edges
    """
    if border_fade is None:
        border_fade = max(1, tile_size // 8)

    # Create 1D weight vector
    weights_1d = np.ones(tile_size, dtype=np.float32)

    # Apply fade at edges
    for i in range(border_fade):
        fade_val = (i + 1) / border_fade
        weights_1d[i] = fade_val
        weights_1d[-(i + 1)] = fade_val

    # Create 2D weight map by outer product
    weight_map = np.outer(weights_1d, weights_1d)

    return weight_map


def apply_tile_weights(tile_pred: np.ndarray, weight_map: np.ndarray) -> np.ndarray:
    """
    Apply spatial weights to tile predictions for smooth blending.

    Args:
        tile_pred: Tile prediction array (H, W)
        weight_map: Weight map (H, W)

    Returns:
        Weighted tile prediction
    """
    return tile_pred * weight_map
