"""
Data Augmentation Module
Supports both FastAI on-the-fly augmentation and offline augmentation strategies
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from loguru import logger

from config import config


class OfflineAugmentor:
    """Offline data augmentation - pre-generates augmented images"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = (
            Path(output_dir)
            if output_dir
            else config.DATA_DIR.parent / "augmented_data"
        )
        self.transforms = config.OFFLINE_AUG_TRANSFORMS
        self.multiplier = config.OFFLINE_AUG_MULTIPLIER

    def augment_dataset(self, source_dir: Path) -> Dict[str, Any]:
        """Apply offline augmentation to entire dataset"""
        logger.info(f"🎨 Starting offline augmentation of {source_dir}")
        logger.info(f"📁 Output directory: {self.output_dir}")
        logger.info(f"🔢 Multiplier: {self.multiplier}x per image")

        stats = {
            "original_images": 0,
            "augmented_images": 0,
            "classes": {},
            "transforms_applied": list(self.transforms.keys()),
        }

        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for class_name in config.CLASSES:
            class_dir = source_dir / class_name
            if not class_dir.exists():
                continue

            output_class_dir = self.output_dir / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)

            # Get all images in class
            images = [
                p
                for p in class_dir.iterdir()
                if p.is_file() and p.suffix.lower() in config.ALLOWED_EXTENSIONS
            ]

            class_stats = {"original": len(images), "augmented": 0}

            # Copy original images
            for img_path in images:
                original_dest = output_class_dir / f"orig_{img_path.name}"
                img = Image.open(img_path)
                img.save(original_dest)
                class_stats["augmented"] += 1

            # Generate augmented versions
            for i, img_path in enumerate(images):
                try:
                    img = Image.open(img_path)

                    for aug_idx in range(self.multiplier):
                        augmented_img = self._apply_random_transforms(img)

                        # Save augmented image
                        base_name = img_path.stem
                        ext = img_path.suffix
                        aug_name = f"aug_{aug_idx}_{base_name}{ext}"
                        aug_path = output_class_dir / aug_name

                        augmented_img.save(aug_path)
                        class_stats["augmented"] += 1

                except Exception as e:
                    logger.error(f"Error augmenting {img_path}: {e}")

            stats["classes"][class_name] = class_stats
            stats["original_images"] += class_stats["original"]
            stats["augmented_images"] += class_stats["augmented"]

            logger.info(
                f"✅ {class_name}: {class_stats['original']} → {class_stats['augmented']} images"
            )

        logger.info(
            f"🎉 Augmentation complete: {stats['original_images']} → {stats['augmented_images']} images"
        )
        return stats

    def _apply_random_transforms(self, img: Image.Image) -> Image.Image:
        """Apply random subset of transforms to an image"""
        img = img.copy()

        # Rotation
        if "rotation" in self.transforms:
            angle_range = self.transforms["rotation"]["range"]
            angle = random.uniform(angle_range[0], angle_range[1])
            img = img.rotate(angle, fillcolor=(255, 255, 255))

        # Brightness
        if "brightness" in self.transforms:
            factor_range = self.transforms["brightness"]["range"]
            factor = random.uniform(factor_range[0], factor_range[1])
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)

        # Contrast
        if "contrast" in self.transforms:
            factor_range = self.transforms["contrast"]["range"]
            factor = random.uniform(factor_range[0], factor_range[1])
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)

        # Saturation
        if "saturation" in self.transforms:
            factor_range = self.transforms["saturation"]["range"]
            factor = random.uniform(factor_range[0], factor_range[1])
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(factor)

        # Horizontal flip
        if "horizontal_flip" in self.transforms:
            prob = self.transforms["horizontal_flip"]["probability"]
            if random.random() < prob:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Vertical flip
        if "vertical_flip" in self.transforms:
            prob = self.transforms["vertical_flip"]["probability"]
            if random.random() < prob:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # Gaussian blur
        if "gaussian_blur" in self.transforms:
            prob = self.transforms["gaussian_blur"]["probability"]
            if random.random() < prob:
                sigma_range = self.transforms["gaussian_blur"]["sigma"]
                sigma = random.uniform(sigma_range[0], sigma_range[1])
                img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Gaussian noise
        if "gaussian_noise" in self.transforms:
            prob = self.transforms["gaussian_noise"]["probability"]
            if random.random() < prob:
                img = self._add_gaussian_noise(img)

        return img

    def _add_gaussian_noise(self, img: Image.Image) -> Image.Image:
        """Add gaussian noise to image"""
        img_array = np.array(img)
        std_range = self.transforms["gaussian_noise"]["std"]
        std = random.uniform(std_range[0], std_range[1])

        noise = np.random.normal(0, std * 255, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_img)


class FastAIAugmentationConfig:
    """Configuration for FastAI augmentation transforms"""

    @staticmethod
    def get_transforms():
        """Get FastAI augmentation transforms based on config"""
        from fastai.vision.all import aug_transforms

        if not config.USE_AUGMENTATION:
            return None

        # Build transform list based on config
        transforms = []
        enabled_transforms = config.FASTAI_AUG_TRANSFORMS

        # Map config strings to FastAI transforms
        transform_map = {
            "flip_lr": "flip_vert",  # Will be handled by aug_transforms
            "rotate": "max_rotate",
            "zoom": "max_zoom",
            "lighting": "max_lighting",
            "contrast": "p_lighting",
        }

        # Default good augmentation for image classification
        aug_config = {
            "size": config.IMAGE_SIZE,
            "max_rotate": 30.0 if "rotate" in enabled_transforms else 0.0,
            "max_zoom": 1.1 if "zoom" in enabled_transforms else 1.0,
            "max_lighting": 0.2 if "lighting" in enabled_transforms else 0.0,
            "p_lighting": 0.75 if "contrast" in enabled_transforms else 0.0,
            "flip_vert": False,  # Usually not good for gadgets
            "do_flip": "flip_lr" in enabled_transforms,
        }

        logger.info(f"🎨 FastAI augmentation config: {aug_config}")
        return aug_transforms(**aug_config)


def create_augmented_dataloader(data_dir: Path, strategy: str = None):
    """Create dataloader with appropriate augmentation strategy"""
    from fastai.vision.all import ImageDataLoaders, Resize
    from src.training_pipeline import ConvertToRGB

    strategy = strategy or config.AUGMENTATION_STRATEGY

    if strategy == "fastai":
        logger.info("🎨 Using FastAI on-the-fly augmentation")

        # Get augmentation transforms
        item_tfms = [ConvertToRGB(), Resize(config.IMAGE_SIZE)]
        batch_tfms = FastAIAugmentationConfig.get_transforms()

        dls = ImageDataLoaders.from_folder(
            data_dir,
            train_pct=config.TRAIN_PCT,
            valid_pct=config.VALID_PCT,
            item_tfms=item_tfms,
            batch_tfms=batch_tfms,
            batch_size=config.BATCH_SIZE,
        )

    elif strategy == "offline":
        logger.info("🎨 Using offline pre-augmented data")

        # Check if augmented data exists
        aug_dir = data_dir.parent / "augmented_data"
        if not aug_dir.exists():
            logger.warning("Augmented data not found, creating it...")
            augmentor = OfflineAugmentor()
            augmentor.augment_dataset(data_dir)

        # Use augmented data directory
        dls = ImageDataLoaders.from_folder(
            aug_dir,
            train_pct=config.TRAIN_PCT,
            valid_pct=config.VALID_PCT,
            item_tfms=[ConvertToRGB(), Resize(config.IMAGE_SIZE)],
            batch_size=config.BATCH_SIZE,
        )

    elif strategy == "hybrid":
        logger.info("🎨 Using hybrid augmentation (offline + FastAI)")

        # Use augmented data + additional FastAI transforms
        aug_dir = data_dir.parent / "augmented_data"
        if not aug_dir.exists():
            augmentor = OfflineAugmentor()
            augmentor.augment_dataset(data_dir)

        item_tfms = [ConvertToRGB(), Resize(config.IMAGE_SIZE)]
        batch_tfms = FastAIAugmentationConfig.get_transforms()

        dls = ImageDataLoaders.from_folder(
            aug_dir,
            train_pct=config.TRAIN_PCT,
            valid_pct=config.VALID_PCT,
            item_tfms=item_tfms,
            batch_tfms=batch_tfms,
            batch_size=config.BATCH_SIZE,
        )

    else:
        raise ValueError(f"Unknown augmentation strategy: {strategy}")

    return dls


def run_offline_augmentation():
    """Standalone function to run offline augmentation"""
    logger.info("🎨 Running offline data augmentation...")

    augmentor = OfflineAugmentor()
    stats = augmentor.augment_dataset(config.DATA_DIR)

    logger.info("📊 Augmentation Summary:")
    logger.info(f"  Original images: {stats['original_images']}")
    logger.info(f"  Total augmented: {stats['augmented_images']}")
    logger.info(
        f"  Multiplication factor: {stats['augmented_images'] / stats['original_images']:.1f}x"
    )

    return stats


if __name__ == "__main__":
    run_offline_augmentation()
