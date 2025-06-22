"""
Model Training Pipeline
Converts the Jupyter notebook training code into a modular pipeline
"""

import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import mlflow
import wandb
from loguru import logger

# FastAI imports
from fastai.vision.all import (
    ImageDataLoaders,
    Resize,
    Transform,
    vision_learner,
    resnet18,
    error_rate,
    accuracy,
    load_learner,
    PILImage,
)
from fastai.callback.tracker import TrackerCallback
from PIL import Image

from config import config

warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")


class ConvertToRGB(Transform):
    """Transform to convert images to RGB format"""

    def encodes(self, img):
        return self._convert_to_rgb(img)

    @staticmethod
    def _convert_to_rgb(img):
        """Convert images to RGB format, handling palette images with transparency"""
        if img.mode == "P":
            img = (
                img.convert("RGBA")
                if "transparency" in img.info
                else img.convert("RGB")
            )
        elif img.mode in ("RGBA", "LA"):
            img = img.convert("RGB")
        return img


class MLflowCallback(TrackerCallback):
    """Callback to log metrics to MLflow"""

    def __init__(self, run, monitor="valid_loss"):
        super().__init__(monitor=monitor)
        self.run = run

    def after_epoch(self):
        super().after_epoch()
        if self.run:
            try:
                last_values = self.learn.recorder.values[-1]
                metric_names = ["train_loss", "valid_loss"] + [
                    m.name for m in self.learn.metrics
                ]

                metrics = {}
                for i, value in enumerate(last_values):
                    if i < len(metric_names):
                        metrics[metric_names[i]] = float(value)

                # Add learning rate
                if hasattr(self.learn, "opt") and self.learn.opt:
                    metrics["learning_rate"] = self.learn.opt.hypers[0]["lr"]

                self.run.log_metrics(metrics, step=self.epoch)
                logger.info(f"✓ Epoch {self.epoch} metrics logged to MLflow")

            except Exception as e:
                logger.error(f"✗ Failed to log epoch {self.epoch} to MLflow: {e}")


class WandbMetricsCallback(TrackerCallback):
    """Custom callback to log training metrics to WandB"""

    def __init__(self, wandb_run, monitor="valid_loss"):
        super().__init__(monitor=monitor)
        self.wandb_run = wandb_run

    def after_epoch(self):
        super().after_epoch()
        if self.wandb_run and hasattr(self.wandb_run, "log"):
            try:
                last_values = self.learn.recorder.values[-1]
                metric_names = ["train_loss", "valid_loss"] + [
                    m.name for m in self.learn.metrics
                ]

                metrics = {"epoch": self.epoch}
                for i, value in enumerate(last_values):
                    if i < len(metric_names):
                        metrics[metric_names[i]] = float(value)

                # Add learning rate
                if hasattr(self.learn, "opt") and self.learn.opt:
                    metrics["learning_rate"] = self.learn.opt.hypers[0]["lr"]

                self.wandb_run.log(metrics)
                logger.info(f"✓ Epoch {self.epoch} metrics logged to WandB")

            except Exception as e:
                logger.error(f"✗ Failed to log epoch {self.epoch} to WandB: {e}")


class ModelTrainer:
    """Main training class that orchestrates the training process"""

    def __init__(self, data_dir: Path = None, model_dir: Path = None):
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR
        self.model_dir = Path(model_dir) if model_dir else config.MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.learn = None
        self.dls = None
        self.wandb_run = None
        self.mlflow_run = None

    def setup_experiment_tracking(self):
        """Initialize MLflow and WandB tracking"""
        try:
            # MLflow setup
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
            self.mlflow_run = mlflow.start_run()

            # Log config parameters to MLflow
            mlflow.log_params(
                {
                    "learning_rate": config.LEARNING_RATE,
                    "architecture": config.ARCHITECTURE,
                    "epochs": config.EPOCHS,
                    "batch_size": config.BATCH_SIZE,
                    "image_size": config.IMAGE_SIZE,
                    "pretrained": config.PRETRAINED,
                }
            )

            logger.info("✓ MLflow tracking initialized")

        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
            self.mlflow_run = None

        try:
            # WandB setup
            self.wandb_run = wandb.init(
                entity=config.WANDB_ENTITY,
                project=config.WANDB_PROJECT,
                config={
                    "learning_rate": config.LEARNING_RATE,
                    "architecture": config.ARCHITECTURE,
                    "epochs": config.EPOCHS,
                    "batch_size": config.BATCH_SIZE,
                    "image_size": config.IMAGE_SIZE,
                    "pretrained": config.PRETRAINED,
                },
            )
            logger.info("✓ WandB tracking initialized")

        except Exception as e:
            logger.warning(f"WandB initialization failed: {e}")
            self.wandb_run = None

    def create_data_loaders(self) -> ImageDataLoaders:
        """Create FastAI data loaders with configurable splitting and augmentation"""
        logger.info(f"Creating data loaders from {self.data_dir}")

        # Check if we should use augmentation
        if config.USE_AUGMENTATION and config.AUGMENTATION_STRATEGY in [
            "fastai",
            "offline",
            "hybrid",
        ]:
            logger.info(
                f"🎨 Using augmentation strategy: {config.AUGMENTATION_STRATEGY}"
            )
            from src.augmentation import create_augmented_dataloader

            # Use the augmentation module to create dataloaders
            self.dls = create_augmented_dataloader(
                self.data_dir, config.AUGMENTATION_STRATEGY
            )

        else:
            # No augmentation - use basic approach
            logger.info("📷 No augmentation - using basic data loading")

            # Get augmentation transforms (will be None if disabled)
            from src.augmentation import FastAIAugmentationConfig

            batch_tfms = (
                FastAIAugmentationConfig.get_transforms()
                if config.USE_AUGMENTATION
                else None
            )

            # Strategy 1: Use FastAI's built-in splitting (default and recommended)
            if config.USE_FASTAI_SPLITTING:
                logger.info("Using FastAI's built-in dataset splitting")

                # Import RandomSplitter for reproducible splits
                from fastai.data.transforms import RandomSplitter

                # Create splitter with seed for reproducibility
                splitter = RandomSplitter(
                    valid_pct=config.VALID_PCT, seed=config.MANUAL_SPLIT_SEED
                )

                self.dls = ImageDataLoaders.from_folder(
                    self.data_dir,
                    splitter=splitter,
                    item_tfms=[ConvertToRGB(), Resize(config.IMAGE_SIZE)],
                    batch_tfms=batch_tfms,
                    batch_size=config.BATCH_SIZE,
                )

            # Strategy 2: Use pre-split data directories (MLOps approach)
            else:
                splits_dir = self.data_dir.parent / "splits"
                if splits_dir.exists():
                    logger.info("Using pre-split dataset directories")
                    train_dir = splits_dir / "train"
                    valid_dir = splits_dir / "valid"

                    if not (train_dir.exists() and valid_dir.exists()):
                        raise FileNotFoundError(
                            "Pre-split directories not found. Run data pipeline first or "
                            "set USE_FASTAI_SPLITTING=True in config."
                        )

                    # Create data loaders from separate train/valid folders
                    self.dls = ImageDataLoaders.from_folder(
                        train_dir.parent,  # Points to splits directory
                        train="train",
                        valid="valid",
                        item_tfms=[ConvertToRGB(), Resize(config.IMAGE_SIZE)],
                        batch_tfms=batch_tfms,
                        batch_size=config.BATCH_SIZE,
                    )
                else:
                    # Fallback to FastAI splitting if no pre-split data
                    logger.warning(
                        "Pre-split data not found, falling back to FastAI splitting"
                    )
                    splitter = RandomSplitter(
                        valid_pct=config.VALID_PCT, seed=config.MANUAL_SPLIT_SEED
                    )

                    self.dls = ImageDataLoaders.from_folder(
                        self.data_dir,
                        splitter=splitter,
                        item_tfms=[ConvertToRGB(), Resize(config.IMAGE_SIZE)],
                        batch_tfms=batch_tfms,
                        batch_size=config.BATCH_SIZE,
                    )

        logger.info(f"Classes: {self.dls.vocab}")
        logger.info(f"Training samples: {len(self.dls.train_ds)}")
        logger.info(f"Validation samples: {len(self.dls.valid_ds)}")

        # Log augmentation info
        if config.USE_AUGMENTATION:
            logger.info(f"🎨 Augmentation enabled: {config.AUGMENTATION_STRATEGY}")
            if config.AUGMENTATION_STRATEGY == "fastai":
                logger.info(f"🔧 FastAI transforms: {config.FASTAI_AUG_TRANSFORMS}")
            elif config.AUGMENTATION_STRATEGY == "offline":
                logger.info(f"🔢 Offline multiplier: {config.OFFLINE_AUG_MULTIPLIER}x")
        else:
            logger.info("📷 No augmentation applied")

        return self.dls

    def create_model(self) -> "Learner":
        """Create the FastAI model"""
        logger.info(f"Creating {config.ARCHITECTURE} model")

        if not self.dls:
            raise ValueError(
                "Data loaders not created. Call create_data_loaders() first."
            )

        # Map architecture string to FastAI model
        arch_map = {
            "resnet18": resnet18,
            "resnet34": "resnet34",  # Add more as needed
            "resnet50": "resnet50",
        }

        if config.ARCHITECTURE not in arch_map:
            raise ValueError(f"Unsupported architecture: {config.ARCHITECTURE}")

        self.learn = vision_learner(
            self.dls,
            arch_map[config.ARCHITECTURE],
            metrics=[accuracy, error_rate],
            pretrained=config.PRETRAINED,
        )

        logger.info("✓ Model created successfully")
        return self.learn

    def train_model(self) -> Dict[str, float]:
        """Train the model with callbacks"""
        logger.info(f"Starting training for {config.EPOCHS} epochs")

        if not self.learn:
            raise ValueError("Model not created. Call create_model() first.")

        # Setup callbacks
        callbacks = []
        if self.mlflow_run:
            callbacks.append(MLflowCallback(self.mlflow_run))
        if self.wandb_run:
            callbacks.append(WandbMetricsCallback(self.wandb_run))

        # Train the model
        start_time = time.time()
        self.learn.fine_tune(config.EPOCHS, base_lr=config.LEARNING_RATE, cbs=callbacks)
        training_time = time.time() - start_time

        # Get final metrics
        final_metrics = {
            "final_train_loss": float(self.learn.recorder.values[-1][0]),
            "final_valid_loss": float(self.learn.recorder.values[-1][1]),
            "final_accuracy": float(self.learn.recorder.values[-1][2]),
            "final_error_rate": float(self.learn.recorder.values[-1][3]),
            "training_time_seconds": training_time,
        }

        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Final accuracy: {final_metrics['final_accuracy']:.4f}")
        logger.info(f"Final error rate: {final_metrics['final_error_rate']:.4f}")

        return final_metrics

    def evaluate_model(self) -> Dict[str, any]:
        """Evaluate the trained model"""
        logger.info("Evaluating model...")

        if not self.learn:
            raise ValueError("Model not trained. Call train_model() first.")

        # Get predictions on validation set
        preds, targets = self.learn.get_preds()

        # Calculate additional metrics
        from sklearn.metrics import classification_report, confusion_matrix
        import numpy as np

        pred_classes = preds.argmax(dim=1)

        # Convert to numpy for sklearn
        pred_classes_np = pred_classes.numpy()
        targets_np = targets.numpy()

        # Classification report
        class_names = self.dls.vocab
        report = classification_report(
            targets_np, pred_classes_np, target_names=class_names, output_dict=True
        )

        # Confusion matrix
        cm = confusion_matrix(targets_np, pred_classes_np)

        evaluation_results = {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "class_names": class_names,
            "num_classes": len(class_names),
            "validation_samples": len(targets),
        }

        logger.info("✓ Model evaluation completed")
        return evaluation_results

    def save_model(self, model_name: str = None) -> Dict[str, any]:
        """Save the trained model"""
        model_name = model_name or config.MODEL_NAME
        model_path = self.model_dir / f"{model_name}.pkl"

        logger.info(f"Saving model to {model_path}")

        try:
            # Export the trained model
            self.learn.export(model_path)
            time.sleep(1)  # Wait for file system sync

            # Verify model file exists
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")

            file_size = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Model saved ({file_size:.2f} MB)")

            # Create model metadata
            metadata = {
                "model_name": model_name,
                "model_path": str(model_path),
                "file_size_mb": file_size,
                "classes": list(self.dls.vocab),
                "architecture": config.ARCHITECTURE,
                "image_size": config.IMAGE_SIZE,
                "training_config": {
                    "learning_rate": config.LEARNING_RATE,
                    "epochs": config.EPOCHS,
                    "batch_size": config.BATCH_SIZE,
                },
            }

            # Save metadata
            import json

            metadata_path = self.model_dir / f"{model_name}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Log to MLflow
            if self.mlflow_run:
                mlflow.log_artifact(str(model_path))
                mlflow.log_artifact(str(metadata_path))
                mlflow.log_dict(metadata, "model_metadata.json")

            # Upload to WandB
            if self.wandb_run:
                model_artifact = wandb.Artifact(
                    name=model_name,
                    type="model",
                    description="FastAI ResNet18 model for gadget classification",
                    metadata=metadata,
                )
                model_artifact.add_file(str(model_path))
                model_artifact.add_file(str(metadata_path))
                self.wandb_run.log_artifact(model_artifact)
                logger.info("✓ Model uploaded to WandB")

            return metadata

        except Exception as e:
            logger.error(f"✗ Error saving model: {e}")
            raise

    def run_full_pipeline(self) -> Dict[str, any]:
        """Run the complete training pipeline"""
        logger.info("Starting full training pipeline...")

        pipeline_results = {
            "start_time": time.time(),
            "success": False,
            "data_stats": {},
            "training_metrics": {},
            "evaluation_results": {},
            "model_metadata": {},
        }

        try:
            # Setup experiment tracking
            self.setup_experiment_tracking()

            # Create data loaders
            dls = self.create_data_loaders()
            pipeline_results["data_stats"] = {
                "classes": list(dls.vocab),
                "num_classes": len(dls.vocab),
                "train_samples": len(dls.train_ds),
                "valid_samples": len(dls.valid_ds),
            }

            # Create and train model
            self.create_model()
            training_metrics = self.train_model()
            pipeline_results["training_metrics"] = training_metrics

            # Evaluate model
            evaluation_results = self.evaluate_model()
            pipeline_results["evaluation_results"] = evaluation_results

            # Save model
            model_metadata = self.save_model()
            pipeline_results["model_metadata"] = model_metadata

            pipeline_results["success"] = True
            pipeline_results["end_time"] = time.time()
            pipeline_results["total_time"] = (
                pipeline_results["end_time"] - pipeline_results["start_time"]
            )

            logger.info(
                f"✓ Training pipeline completed successfully in {pipeline_results['total_time']:.2f} seconds"
            )

        except Exception as e:
            logger.error(f"✗ Training pipeline failed: {e}")
            pipeline_results["error"] = str(e)
            raise

        finally:
            # Clean up experiment tracking
            if self.mlflow_run:
                mlflow.end_run()
            if self.wandb_run:
                self.wandb_run.finish()

        return pipeline_results


def run_training_pipeline():
    """Main function to run the training pipeline"""
    logger.info("Initializing training pipeline...")

    trainer = ModelTrainer()
    results = trainer.run_full_pipeline()

    # Print summary
    logger.info("=== Training Pipeline Summary ===")
    logger.info(f"Success: {results['success']}")
    logger.info(f"Total time: {results.get('total_time', 0):.2f} seconds")
    logger.info(
        f"Final accuracy: {results.get('training_metrics', {}).get('final_accuracy', 0):.4f}"
    )
    logger.info(f"Classes: {results.get('data_stats', {}).get('classes', [])}")

    return results


if __name__ == "__main__":
    run_training_pipeline()
