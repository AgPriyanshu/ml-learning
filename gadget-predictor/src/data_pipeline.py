"""
Data Collection and Validation Pipeline
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from PIL import Image
import logging
from loguru import logger
from config import config


class DataValidator:
    """Validates and processes image data for training"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.classes = config.CLASSES
        self.min_images_per_class = config.MIN_IMAGES_PER_CLASS
        self.max_image_size_mb = config.MAX_IMAGE_SIZE_MB
        self.allowed_extensions = config.ALLOWED_EXTENSIONS

    def validate_directory_structure(self) -> Dict[str, any]:
        """Validate the directory structure and return statistics"""
        logger.info("Validating directory structure...")

        stats = {
            "total_classes": 0,
            "total_images": 0,
            "class_distribution": {},
            "invalid_images": [],
            "missing_classes": [],
            "valid": True,
            "errors": [],
        }

        # Check if data directory exists
        if not self.data_dir.exists():
            stats["valid"] = False
            stats["errors"].append(f"Data directory {self.data_dir} does not exist")
            return stats

        # Check each class directory
        for class_name in self.classes:
            class_dir = self.data_dir / class_name

            if not class_dir.exists():
                stats["missing_classes"].append(class_name)
                stats["valid"] = False
                continue

            # Count valid images in class directory
            valid_images = self._validate_class_images(class_dir)
            stats["class_distribution"][class_name] = len(valid_images)
            stats["total_images"] += len(valid_images)

            # Check minimum images requirement
            if len(valid_images) < self.min_images_per_class:
                stats["errors"].append(
                    f"Class {class_name} has only {len(valid_images)} images, "
                    f"minimum required: {self.min_images_per_class}"
                )
                stats["valid"] = False

        stats["total_classes"] = len(
            [c for c in self.classes if c not in stats["missing_classes"]]
        )

        return stats

    def _validate_class_images(self, class_dir: Path) -> List[Path]:
        """Validate images in a class directory"""
        valid_images = []

        for img_path in class_dir.iterdir():
            if (
                img_path.is_file()
                and img_path.suffix.lower() in self.allowed_extensions
            ):
                if self._is_valid_image(img_path):
                    valid_images.append(img_path)
                else:
                    logger.warning(f"Invalid image: {img_path}")

        return valid_images

    def _is_valid_image(self, img_path: Path) -> bool:
        """Check if an image file is valid"""
        try:
            # Check file size
            file_size_mb = img_path.stat().st_size / (1024 * 1024)
            if file_size_mb > self.max_image_size_mb:
                logger.warning(f"Image {img_path} is too large: {file_size_mb:.2f}MB")
                return False

            # Try to open and verify image
            with Image.open(img_path) as img:
                img.verify()

            # Try to load the image
            with Image.open(img_path) as img:
                img.load()

            return True

        except Exception as e:
            logger.error(f"Error validating image {img_path}: {e}")
            return False

    def clean_dataset(self, backup: bool = True) -> Dict[str, any]:
        """Clean the dataset by removing invalid images"""
        logger.info("Cleaning dataset...")

        cleaned_stats = {
            "removed_files": [],
            "corrupted_images": [],
            "oversized_images": [],
            "total_removed": 0,
        }

        if backup:
            backup_dir = self.data_dir.parent / f"{self.data_dir.name}_backup"
            if backup_dir.exists():
                shutil.rmtree(backup_dir)
            shutil.copytree(self.data_dir, backup_dir)
            logger.info(f"Backup created at {backup_dir}")

        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue

            for img_path in list(class_dir.iterdir()):
                if img_path.is_file():
                    if not self._is_valid_image(img_path):
                        cleaned_stats["removed_files"].append(str(img_path))
                        img_path.unlink()
                        cleaned_stats["total_removed"] += 1

        logger.info(
            f"Dataset cleaning completed. Removed {cleaned_stats['total_removed']} files."
        )
        return cleaned_stats


class DataCollector:
    """Collects and organizes new data"""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.validator = DataValidator(data_dir)

    def add_images(self, source_dir: Path, class_name: str) -> Dict[str, any]:
        """Add new images to a specific class"""
        logger.info(f"Adding images from {source_dir} to class {class_name}")

        if class_name not in config.CLASSES:
            raise ValueError(
                f"Unknown class: {class_name}. Valid classes: {config.CLASSES}"
            )

        class_dir = self.data_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        stats = {"added_images": 0, "skipped_images": 0, "errors": []}

        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Source directory {source_path} does not exist")

        # Get existing image count for naming
        existing_images = len(list(class_dir.glob("*")))

        for img_path in source_path.iterdir():
            if (
                img_path.is_file()
                and img_path.suffix.lower() in config.ALLOWED_EXTENSIONS
            ):
                try:
                    if self.validator._is_valid_image(img_path):
                        # Copy image with new name
                        new_name = f"{class_name}_{existing_images + stats['added_images']}{img_path.suffix}"
                        dest_path = class_dir / new_name
                        shutil.copy2(img_path, dest_path)
                        stats["added_images"] += 1
                    else:
                        stats["skipped_images"] += 1
                except Exception as e:
                    stats["errors"].append(f"Error processing {img_path}: {e}")
                    stats["skipped_images"] += 1

        logger.info(f"Added {stats['added_images']} images to class {class_name}")
        return stats

    def split_dataset(
        self, train_pct: float = None, valid_pct: float = None, test_pct: float = None
    ) -> Dict[str, any]:
        """Split dataset into train/validation/test sets (only if manual splitting is enabled)"""

        # Check if we should use FastAI's built-in splitting instead
        if config.USE_FASTAI_SPLITTING:
            logger.info(
                "✨ Skipping manual dataset splitting - FastAI will handle splitting automatically"
            )
            logger.info(
                f"📊 Split configuration: train={config.TRAIN_PCT}, valid={config.VALID_PCT}, seed={config.MANUAL_SPLIT_SEED}"
            )

            # Return basic stats without creating split directories
            total_files = 0
            class_distribution = {}

            for class_name in config.CLASSES:
                class_dir = self.data_dir / class_name
                if class_dir.exists():
                    images = [p for p in class_dir.iterdir() if p.is_file()]
                    class_distribution[class_name] = len(images)
                    total_files += len(images)

            return {
                "strategy": "fastai_splitting",
                "total_files": total_files,
                "class_distribution": class_distribution,
                "split_percentages": {
                    "train": config.TRAIN_PCT,
                    "valid": config.VALID_PCT,
                    "test": config.TEST_PCT,
                },
                "seed": config.MANUAL_SPLIT_SEED,
                "message": "Using FastAI's built-in splitting - no manual split directories created",
            }

        # Manual splitting logic (original approach)
        logger.info("📁 Creating manual dataset splits...")

        train_pct = train_pct or config.TRAIN_PCT
        valid_pct = valid_pct or config.VALID_PCT
        test_pct = test_pct or config.TEST_PCT

        # Ensure percentages sum to 1
        total_pct = train_pct + valid_pct + test_pct
        if abs(total_pct - 1.0) > 0.001:
            raise ValueError(f"Split percentages must sum to 1.0, got {total_pct}")

        logger.info(
            f"Splitting dataset: train={train_pct}, valid={valid_pct}, test={test_pct}"
        )

        split_stats = {
            "strategy": "manual_splitting",
            "train": {},
            "valid": {},
            "test": {},
            "total_files": 0,
        }

        # Create split directories
        splits_dir = self.data_dir.parent / "splits"
        for split in ["train", "valid", "test"]:
            split_dir = splits_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for class_name in config.CLASSES:
                (split_dir / class_name).mkdir(parents=True, exist_ok=True)

        # Split each class
        import random

        random.seed(config.MANUAL_SPLIT_SEED)  # For reproducible splits

        for class_name in config.CLASSES:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue

            # Get all images in class and shuffle for random split
            images = [p for p in class_dir.iterdir() if p.is_file()]
            random.shuffle(images)  # Shuffle with seed for reproducibility

            total_images = len(images)
            split_stats["total_files"] += total_images

            # Calculate split sizes
            train_size = int(total_images * train_pct)
            valid_size = int(total_images * valid_pct)
            test_size = total_images - train_size - valid_size

            # Split and copy files
            train_images = images[:train_size]
            valid_images = images[train_size : train_size + valid_size]
            test_images = images[train_size + valid_size :]

            splits = {"train": train_images, "valid": valid_images, "test": test_images}

            for split_name, split_images in splits.items():
                split_stats[split_name][class_name] = len(split_images)
                split_class_dir = splits_dir / split_name / class_name

                for img_path in split_images:
                    dest_path = split_class_dir / img_path.name
                    shutil.copy2(img_path, dest_path)

        logger.info(
            f"📁 Manual dataset split completed. Created splits in {splits_dir}"
        )
        return split_stats


def run_data_pipeline():
    """Run the complete data pipeline"""
    logger.info("🚀 Starting data pipeline...")

    # Initialize data collector and validator
    collector = DataCollector(config.DATA_DIR)
    validator = DataValidator(config.DATA_DIR)

    # Validate dataset
    logger.info("📋 Step 1: Validating dataset structure...")
    validation_stats = validator.validate_directory_structure()
    logger.info(f"Validation results: {validation_stats}")

    if not validation_stats["valid"]:
        logger.error("❌ Dataset validation failed!")
        for error in validation_stats["errors"]:
            logger.error(error)
        return False

    # Clean dataset
    logger.info("🧹 Step 2: Cleaning dataset...")
    clean_stats = validator.clean_dataset(backup=True)
    logger.info(f"Cleaning results: {clean_stats}")

    # Split dataset (conditional based on configuration)
    logger.info("📊 Step 3: Dataset splitting...")
    if config.USE_FASTAI_SPLITTING:
        logger.info("✨ Using FastAI's built-in splitting strategy")
        split_stats = (
            collector.split_dataset()
        )  # Will return stats without creating files
    else:
        logger.info("📁 Using manual splitting strategy")
        split_stats = collector.split_dataset()  # Will create split directories

    logger.info(f"Split results: {split_stats}")

    # Offline augmentation (if configured)
    aug_stats = None
    if config.USE_AUGMENTATION and config.AUGMENTATION_STRATEGY == "offline":
        logger.info("🎨 Step 4: Offline data augmentation...")
        from src.augmentation import OfflineAugmentor

        augmentor = OfflineAugmentor()
        aug_stats = augmentor.augment_dataset(config.DATA_DIR)
        logger.info(f"Augmentation results: {aug_stats}")
    elif config.USE_AUGMENTATION:
        logger.info(
            f"🎨 Step 4: Augmentation configured for training ({config.AUGMENTATION_STRATEGY})"
        )
    else:
        logger.info("📷 Step 4: No augmentation configured")

    # Summary
    strategy = split_stats.get("strategy", "unknown")
    total_files = split_stats.get("total_files", 0)

    logger.info("=" * 50)
    logger.info("📊 DATA PIPELINE SUMMARY")
    logger.info("=" * 50)
    logger.info(f"✅ Strategy: {strategy.replace('_', ' ').title()}")
    logger.info(f"📁 Total files: {total_files}")
    logger.info(f"🎯 Classes: {len(config.CLASSES)}")

    if strategy == "fastai_splitting":
        logger.info("🚀 Ready for FastAI training!")
    else:
        logger.info("📂 Manual splits created - ready for training!")

    logger.info("=" * 50)
    logger.info("✅ Data pipeline completed successfully!")
    return True


if __name__ == "__main__":
    run_data_pipeline()
