"""
Tests for the MLOps pipeline components
"""

import pytest
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from src.data_pipeline import DataValidator, DataCollector
from config import config


class TestDataPipeline:
    """Test data pipeline components"""

    def test_data_validator_init(self):
        """Test DataValidator initialization"""
        validator = DataValidator(config.DATA_DIR)
        assert validator.data_dir == config.DATA_DIR
        assert validator.classes == config.CLASSES

    def test_image_validation(self):
        """Test image validation logic"""
        validator = DataValidator(config.DATA_DIR)

        # Create a test image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            # Create a simple RGB image
            img = Image.new("RGB", (100, 100), color="red")
            img.save(tmp.name)

            # Test validation
            is_valid = validator._is_valid_image(Path(tmp.name))
            assert is_valid is True

            # Clean up
            Path(tmp.name).unlink()

    def test_data_collector_init(self):
        """Test DataCollector initialization"""
        collector = DataCollector(config.DATA_DIR)
        assert collector.data_dir == config.DATA_DIR
        assert isinstance(collector.validator, DataValidator)


class TestConfiguration:
    """Test configuration management"""

    def test_config_attributes(self):
        """Test that config has required attributes"""
        required_attrs = [
            "PROJECT_NAME",
            "VERSION",
            "DATA_DIR",
            "MODEL_DIR",
            "CLASSES",
            "BATCH_SIZE",
            "EPOCHS",
            "LEARNING_RATE",
        ]

        for attr in required_attrs:
            assert hasattr(config, attr), f"Config missing attribute: {attr}"

    def test_classes_list(self):
        """Test that classes list is valid"""
        assert isinstance(config.CLASSES, list)
        assert len(config.CLASSES) > 0
        assert "smartphone" in config.CLASSES


class TestModelService:
    """Test model service components"""

    def test_preprocess_image(self):
        """Test image preprocessing"""
        from src.model_service import ModelService

        # Create a test RGB image
        img = Image.new("RGB", (100, 100), color="blue")

        # Mock ModelService (without loading actual model)
        service = ModelService.__new__(ModelService)
        service.model_loaded = False

        # Test preprocessing
        processed = service.preprocess_image(img)
        # Should return a FastAI PILImage
        assert processed is not None


class TestPipelineOrchestrator:
    """Test pipeline orchestration"""

    def test_orchestrator_init(self):
        """Test PipelineOrchestrator initialization"""
        from scripts.run_pipeline import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()
        assert orchestrator.config == config
        assert "stages" in orchestrator.results
        assert orchestrator.results["success"] is False


# Integration test fixtures
@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    return Image.new("RGB", (224, 224), color="green")


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary dataset directory structure"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create class directories
        for class_name in ["smartphone", "tablet"]:
            class_dir = tmp_path / class_name
            class_dir.mkdir()

            # Create a few sample images
            for i in range(3):
                img = Image.new("RGB", (100, 100), color="red")
                img.save(class_dir / f"{class_name}_{i}.jpg")

        yield tmp_path


class TestIntegration:
    """Integration tests"""

    def test_data_validation_with_temp_dataset(self, temp_dataset_dir):
        """Test data validation with temporary dataset"""
        validator = DataValidator(temp_dataset_dir)
        validator.classes = ["smartphone", "tablet"]  # Override for test

        stats = validator.validate_directory_structure()
        assert stats["valid"] is True
        assert stats["total_classes"] == 2
        assert "smartphone" in stats["class_distribution"]
        assert "tablet" in stats["class_distribution"]


# Mock tests for external dependencies
class TestMockIntegrations:
    """Test with mocked external services"""

    @pytest.mark.skipif(True, reason="Requires WandB setup")
    def test_wandb_integration(self):
        """Test WandB integration (requires setup)"""
        pass

    @pytest.mark.skipif(True, reason="Requires MLflow setup")
    def test_mlflow_integration(self):
        """Test MLflow integration (requires setup)"""
        pass


if __name__ == "__main__":
    pytest.main([__file__])
