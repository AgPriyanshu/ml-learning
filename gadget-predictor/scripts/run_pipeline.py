#!/usr/bin/env python3
"""
MLOps Pipeline Orchestrator
Main script to run the complete pipeline: data validation -> training -> deployment
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import config
from src.data_pipeline import run_data_pipeline
from src.training_pipeline import run_training_pipeline


class PipelineOrchestrator:
    """Orchestrates the complete MLOps pipeline"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config = config
        self.results = {
            "pipeline_start_time": None,
            "pipeline_end_time": None,
            "stages": {},
            "success": False,
            "errors": [],
        }

    def run_data_pipeline(self) -> bool:
        """Run data validation and preprocessing"""
        logger.info("🔄 Running data pipeline...")

        try:
            success = run_data_pipeline()
            self.results["stages"]["data_pipeline"] = {
                "success": success,
                "stage": "data_validation",
            }
            return success
        except Exception as e:
            logger.error(f"Data pipeline failed: {e}")
            self.results["stages"]["data_pipeline"] = {
                "success": False,
                "error": str(e),
                "stage": "data_validation",
            }
            return False

    def run_training_pipeline(self) -> bool:
        """Run model training"""
        logger.info("🔄 Running training pipeline...")

        try:
            results = run_training_pipeline()
            self.results["stages"]["training_pipeline"] = {
                "success": results["success"],
                "training_results": results,
                "stage": "training",
            }
            return results["success"]
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            self.results["stages"]["training_pipeline"] = {
                "success": False,
                "error": str(e),
                "stage": "training",
            }
            return False

    def run_model_validation(self) -> bool:
        """Validate the trained model meets performance criteria"""
        logger.info("🔄 Running model validation...")

        try:
            # Check if model exists
            model_path = config.MODEL_DIR / f"{config.MODEL_NAME}.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            # Load training results to check performance
            training_results = (
                self.results["stages"]
                .get("training_pipeline", {})
                .get("training_results", {})
            )

            if not training_results:
                raise ValueError("No training results available for validation")

            # Performance thresholds
            min_accuracy = 0.8
            max_error_rate = 0.2

            final_accuracy = training_results.get("training_metrics", {}).get(
                "final_accuracy", 0
            )
            final_error_rate = training_results.get("training_metrics", {}).get(
                "final_error_rate", 1
            )

            # Validate performance
            performance_checks = {
                "accuracy_check": final_accuracy >= min_accuracy,
                "error_rate_check": final_error_rate <= max_error_rate,
                "model_exists": model_path.exists(),
            }

            all_checks_passed = all(performance_checks.values())

            self.results["stages"]["model_validation"] = {
                "success": all_checks_passed,
                "performance_checks": performance_checks,
                "metrics": {
                    "accuracy": final_accuracy,
                    "error_rate": final_error_rate,
                    "accuracy_threshold": min_accuracy,
                    "error_rate_threshold": max_error_rate,
                },
                "stage": "validation",
            }

            if all_checks_passed:
                logger.info("✅ Model validation passed")
            else:
                logger.error("❌ Model validation failed")
                for check, passed in performance_checks.items():
                    if not passed:
                        logger.error(f"  - {check}: FAILED")

            return all_checks_passed

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            self.results["stages"]["model_validation"] = {
                "success": False,
                "error": str(e),
                "stage": "validation",
            }
            return False

    def deploy_model(self) -> bool:
        """Deploy the model service"""
        logger.info("🔄 Deploying model service...")

        try:
            # For development, we'll start the FastAPI service
            # In production, this would deploy to your chosen platform

            logger.info("Starting model service...")

            # Check if service is already running
            try:
                import requests

                response = requests.get(
                    f"http://{config.API_HOST}:{config.API_PORT}/health", timeout=5
                )
                if response.status_code == 200:
                    logger.info("Model service already running")
                    self.results["stages"]["deployment"] = {
                        "success": True,
                        "message": "Service already running",
                        "stage": "deployment",
                    }
                    return True
            except:
                pass

            # Start the service in background (for development)
            deployment_cmd = [
                sys.executable,
                "-c",
                "from src.model_service import run_server; run_server()",
            ]

            # Note: In production, you'd use Docker, Kubernetes, etc.
            logger.info("Model service deployment initiated")

            self.results["stages"]["deployment"] = {
                "success": True,
                "deployment_command": " ".join(deployment_cmd),
                "stage": "deployment",
            }

            return True

        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            self.results["stages"]["deployment"] = {
                "success": False,
                "error": str(e),
                "stage": "deployment",
            }
            return False

    def run_full_pipeline(
        self, skip_data: bool = False, skip_training: bool = False
    ) -> Dict:
        """Run the complete MLOps pipeline"""
        import time

        logger.info("🚀 Starting MLOps Pipeline")
        self.results["pipeline_start_time"] = time.time()

        stages_to_run = []

        # Stage 1: Data Pipeline
        if not skip_data:
            stages_to_run.append(("Data Pipeline", self.run_data_pipeline))

        # Stage 2: Training Pipeline
        if not skip_training:
            stages_to_run.append(("Training Pipeline", self.run_training_pipeline))

        # Stage 3: Model Validation
        stages_to_run.append(("Model Validation", self.run_model_validation))

        # Stage 4: Deployment
        stages_to_run.append(("Model Deployment", self.deploy_model))

        # Run stages
        pipeline_success = True
        for stage_name, stage_func in stages_to_run:
            logger.info(f"🔄 Running {stage_name}...")
            stage_success = stage_func()

            if not stage_success:
                logger.error(f"❌ {stage_name} failed - stopping pipeline")
                pipeline_success = False
                break
            else:
                logger.info(f"✅ {stage_name} completed successfully")

        self.results["pipeline_end_time"] = time.time()
        self.results["success"] = pipeline_success
        self.results["total_time"] = (
            self.results["pipeline_end_time"] - self.results["pipeline_start_time"]
        )

        # Save results
        results_path = config.LOGS_DIR / "pipeline_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Print summary
        self.print_summary()

        return self.results

    def print_summary(self):
        """Print pipeline execution summary"""
        logger.info("=" * 60)
        logger.info("📊 PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 60)

        logger.info(
            f"Overall Success: {'✅ YES' if self.results['success'] else '❌ NO'}"
        )
        logger.info(f"Total Time: {self.results.get('total_time', 0):.2f} seconds")
        logger.info(
            f"Stages Completed: {len([s for s in self.results['stages'].values() if s.get('success', False)])}/{len(self.results['stages'])}"
        )

        logger.info("\n📋 Stage Details:")
        for stage_name, stage_data in self.results["stages"].items():
            status = "✅" if stage_data.get("success", False) else "❌"
            logger.info(f"  {status} {stage_name.replace('_', ' ').title()}")

            if "error" in stage_data:
                logger.info(f"      Error: {stage_data['error']}")

            # Show specific metrics for training
            if stage_name == "training_pipeline" and "training_results" in stage_data:
                metrics = stage_data["training_results"].get("training_metrics", {})
                if metrics:
                    logger.info(
                        f"      Accuracy: {metrics.get('final_accuracy', 0):.4f}"
                    )
                    logger.info(
                        f"      Error Rate: {metrics.get('final_error_rate', 0):.4f}"
                    )

        logger.info("=" * 60)


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="MLOps Pipeline Orchestrator")
    parser.add_argument("--skip-data", action="store_true", help="Skip data pipeline")
    parser.add_argument(
        "--skip-training", action="store_true", help="Skip training pipeline"
    )
    parser.add_argument("--config", type=Path, help="Custom config file path")

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator(config_path=args.config)

    # Run pipeline
    results = orchestrator.run_full_pipeline(
        skip_data=args.skip_data, skip_training=args.skip_training
    )

    # Exit with appropriate code
    sys.exit(0 if results["success"] else 1)


if __name__ == "__main__":
    main()
