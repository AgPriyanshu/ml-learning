import numpy as np


def random_flip_rotate(rgb: np.ndarray, mask: np.ndarray, prob: float = 0.5):
    """
    Apply random horizontal/vertical flips and 90-degree rotations.

    Args:
        rgb: (3, H, W) RGB image
        mask: (H, W) or (1, H, W) mask
        prob: Probability of applying each transformation

    Returns:
        Augmented rgb and mask with same shapes
    """
    import random

    # Ensure mask is 2D for processing
    mask_was_3d = mask.ndim == 3
    if mask_was_3d:
        mask = mask.squeeze(0)

    # Random horizontal flip
    if random.random() < prob:
        rgb = np.flip(rgb, axis=2).copy()
        mask = np.flip(mask, axis=1).copy()

    # Random vertical flip
    if random.random() < prob:
        rgb = np.flip(rgb, axis=1).copy()
        mask = np.flip(mask, axis=0).copy()

    # Random 90-degree rotation (0, 90, 180, 270 degrees)
    if random.random() < prob:
        k = random.randint(1, 3)  # 1-3 rotations of 90 degrees
        rgb = np.rot90(rgb, k=k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k=k, axes=(0, 1)).copy()

    # Restore mask shape if needed
    if mask_was_3d:
        mask = mask[None, ...]

    return rgb, mask


def random_brightness_contrast(
    rgb: np.ndarray,
    brightness_range: tuple = (-0.1, 0.1),
    contrast_range: tuple = (0.9, 1.1),
):
    """
    Apply random brightness and contrast adjustments to RGB image.

    Args:
        rgb: (3, H, W) RGB image (assumed to be normalized)
        brightness_range: Range for brightness adjustment
        contrast_range: Range for contrast adjustment

    Returns:
        Augmented RGB image
    """
    import random

    # Random brightness adjustment
    brightness = random.uniform(*brightness_range)
    rgb = rgb + brightness

    # Random contrast adjustment
    contrast = random.uniform(*contrast_range)
    rgb = rgb * contrast

    return rgb.copy()
