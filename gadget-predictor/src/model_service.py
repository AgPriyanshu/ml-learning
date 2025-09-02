"""
Model Serving Service using FastAPI
Provides REST API endpoints for model inference
"""

import os
import io
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, field_validator
from PIL import Image
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
from loguru import logger

# FastAI imports
from fastai.vision.all import load_learner, PILImage

from config import config

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "model_predictions_total", "Total number of predictions made"
)
PREDICTION_LATENCY = Histogram(
    "model_prediction_duration_seconds", "Time spent on predictions"
)
MODEL_LOAD_TIME = Gauge("model_load_time_seconds", "Time taken to load the model")

app = FastAPI(
    title="Gadget Classification API",
    description="FastAI-based gadget classification service",
    version=config.VERSION,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    """Request model for predictions"""

    image_url: Optional[str] = None
    confidence_threshold: Optional[float] = 0.5

    @field_validator("confidence_threshold")
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1")
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions"""

    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    top_predictions: List[Dict[str, Union[str, float]]]
    processing_time_ms: float
    model_version: str


class ModelService:
    """Model service class that handles model loading and inference"""

    def __init__(self, model_path: Path = None):
        self.model_path = (
            Path(model_path)
            if model_path
            else config.MODEL_DIR / f"{config.MODEL_NAME}.pkl"
        )
        self.learn = None
        self.metadata = None
        self.model_loaded = False
        self.load_model()

    def load_model(self):
        """Load the trained FastAI model"""
        logger.info(f"Loading model from {self.model_path}")
        start_time = time.time()

        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            # Load the FastAI model
            self.learn = load_learner(self.model_path)

            # Load metadata if available
            metadata_path = (
                self.model_path.parent / f"{self.model_path.stem}_metadata.json"
            )
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    self.metadata = json.load(f)
            else:
                # Create minimal metadata from model
                self.metadata = {
                    "classes": list(self.learn.dls.vocab),
                    "model_name": config.MODEL_NAME,
                    "architecture": config.ARCHITECTURE,
                }

            load_time = time.time() - start_time
            MODEL_LOAD_TIME.set(load_time)

            self.model_loaded = True
            logger.info(f"✓ Model loaded successfully in {load_time:.2f} seconds")
            logger.info(f"✓ Classes: {self.metadata['classes']}")

        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            raise

    def preprocess_image(self, image: Image.Image) -> PILImage:
        """Preprocess image for model inference"""
        try:
            # Convert to RGB if needed
            if image.mode != "RGB":
                if image.mode == "P":
                    image = (
                        image.convert("RGBA")
                        if "transparency" in image.info
                        else image.convert("RGB")
                    )
                elif image.mode in ("RGBA", "LA"):
                    image = image.convert("RGB")

            # Create FastAI PILImage
            fastai_image = PILImage.create(image)
            return fastai_image

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise

    def predict(
        self, image: Image.Image, confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Make prediction on an image"""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)

            # Make prediction
            preds = self.learn.predict(processed_image)

            # Extract results
            predicted_class = str(preds[0])
            predicted_idx = preds[1]
            probabilities = preds[2]

            # Convert to numpy if needed
            if hasattr(probabilities, "numpy"):
                probs_array = probabilities.numpy()
            else:
                probs_array = probabilities

            # Get class names
            classes = self.metadata["classes"]

            # Create probability dictionary
            prob_dict = {
                str(cls): float(prob) for cls, prob in zip(classes, probs_array)
            }

            # Get top predictions above threshold
            top_predictions = [
                {"class": str(cls), "confidence": float(prob)}
                for cls, prob in zip(classes, probs_array)
                if prob > confidence_threshold
            ]

            # Sort by confidence
            top_predictions.sort(key=lambda x: x["confidence"], reverse=True)

            processing_time = (
                time.time() - start_time
            ) * 1000  # Convert to milliseconds

            result = {
                "predicted_class": predicted_class,
                "confidence": float(probs_array.max()),
                "probabilities": prob_dict,
                "top_predictions": top_predictions,
                "processing_time_ms": processing_time,
                "model_version": self.metadata.get("model_name", config.MODEL_NAME),
            }

            # Update metrics
            PREDICTION_COUNTER.inc()
            PREDICTION_LATENCY.observe(processing_time / 1000)

            return result

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise


# Global model service instance
model_service = ModelService()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Gadget Classification API",
        "status": "healthy",
        "model_loaded": model_service.model_loaded,
        "version": config.VERSION,
        "classes": model_service.metadata["classes"] if model_service.metadata else [],
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model_service.model_loaded else "unhealthy",
        "model_loaded": model_service.model_loaded,
        "model_path": str(model_service.model_path),
        "classes": model_service.metadata["classes"] if model_service.metadata else [],
        "timestamp": time.time(),
    }


@app.get("/model/info")
async def model_info():
    """Get model information"""
    if not model_service.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_metadata": model_service.metadata,
        "model_loaded": model_service.model_loaded,
        "classes": model_service.metadata["classes"],
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...), confidence_threshold: float = 0.5
):
    """Predict the class of an uploaded image"""

    if not model_service.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Make prediction
        result = model_service.predict(image, confidence_threshold)

        logger.info(
            f"Prediction made: {result['predicted_class']} ({result['confidence']:.2%})"
        )

        return PredictionResponse(**result)

    except Exception as e:
        logger.error(f"Error processing prediction request: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...), confidence_threshold: float = 0.5
):
    """Predict classes for multiple images"""

    if not model_service.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")

    results = []

    for i, file in enumerate(files):
        try:
            if not file.content_type.startswith("image/"):
                results.append(
                    {"filename": file.filename, "error": "File must be an image"}
                )
                continue

            # Read and process image
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))

            # Make prediction
            result = model_service.predict(image, confidence_threshold)
            result["filename"] = file.filename
            results.append(result)

        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return {"predictions": results}


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class ModelReloader:
    """Helper class to reload model without restarting service"""

    @staticmethod
    async def reload_model(model_path: str = None):
        """Reload the model from disk"""
        try:
            global model_service

            if model_path:
                model_service.model_path = Path(model_path)

            model_service.load_model()

            return {
                "status": "success",
                "message": "Model reloaded successfully",
                "model_path": str(model_service.model_path),
                "classes": model_service.metadata["classes"],
            }

        except Exception as e:
            logger.error(f"Failed to reload model: {e}")
            return {"status": "error", "message": f"Failed to reload model: {str(e)}"}


@app.post("/model/reload")
async def reload_model(model_path: Optional[str] = None):
    """Reload the model (admin endpoint)"""
    result = await ModelReloader.reload_model(model_path)

    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])

    return result


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app"""
    return app


def run_server():
    """Run the FastAPI server"""
    logger.info(
        f"Starting Gadget Classification API server on {config.API_HOST}:{config.API_PORT}"
    )

    uvicorn.run(
        "model_service:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_RELOAD,
        log_level=config.LOG_LEVEL.lower(),
    )


if __name__ == "__main__":
    run_server()
