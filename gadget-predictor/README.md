# Gadget Classification MLOps Pipeline

A complete MLOps pipeline for classifying gadgets (smartphone, tablet, smartwatch, headphones, camera, laptop) using FastAI and modern MLOps practices.

## 🚀 Features

- **End-to-end MLOps Pipeline**: Data validation, training, deployment, and monitoring
- **FastAI Integration**: State-of-the-art computer vision with ResNet18
- **Experiment Tracking**: MLflow and Weights & Biases integration
- **Model Serving**: FastAPI-based REST API with automatic documentation
- **Containerization**: Docker and Docker Compose for easy deployment
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **CI/CD**: GitHub Actions for automated testing and deployment
- **Data Validation**: Automated data quality checks and preprocessing

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [MLOps Pipeline](#mlops-pipeline)
- [API Documentation](#api-documentation)
- [Monitoring](#monitoring)
- [Development](#development)
- [Deployment](#deployment)
- [Contributing](#contributing)

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd gadget-predictor

# Set up environment variables
cp .env.example .env
# Edit .env with your WandB API key and other settings

# Start the complete MLOps stack
docker-compose up -d

# Access services:
# - Model API: http://localhost:8000
# - MLflow UI: http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
# - Jupyter: http://localhost:8888
```

### Option 2: Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python scripts/run_pipeline.py

# Start the API server
python -m src.model_service
```

## 📁 Project Structure

```
gadget-predictor/
├── src/
│   ├── data_pipeline.py          # Data validation and preprocessing
│   ├── training_pipeline.py      # Model training pipeline
│   ├── model_service.py          # FastAPI model serving
│   └── __init__.py
├── scripts/
│   └── run_pipeline.py           # Pipeline orchestrator
├── datasets/                     # Training data
│   ├── camera/
│   ├── headphones/
│   ├── laptop/
│   ├── smartphone/
│   ├── smartwatch/
│   └── tablet/
├── models/                       # Trained models
├── monitoring/                   # Monitoring configuration
│   ├── prometheus.yml
│   ├── alert_rules.yml
│   └── grafana/
├── .github/workflows/            # CI/CD pipelines
├── notebooks/                    # Jupyter notebooks
├── tests/                        # Unit tests
├── config.py                     # Configuration management
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container definition
├── docker-compose.yml           # Multi-service orchestration
└── README.md
```

## 🛠 Installation

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- Git

### Local Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up configuration
cp .env.example .env
```

### Environment Variables

Create a `.env` file with the following variables:

```bash
# WandB Configuration
WANDB_API_KEY=your_wandb_api_key
WANDB_PROJECT=gadgets-predictor
WANDB_ENTITY=your_username

# MLflow Configuration
MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
MODEL_NAME=gadget_classifier_model
CONFIDENCE_THRESHOLD=0.5
```

## 📖 Usage

### 1. Data Pipeline

```bash
# Validate and preprocess data (uses FastAI splitting by default)
python -m src.data_pipeline

# For manual splitting (set USE_FASTAI_SPLITTING=False in config)
# This creates physical train/valid/test directories
python -m src.data_pipeline

# Add new images to a class
python -c "
from src.data_pipeline import DataCollector
from pathlib import Path
collector = DataCollector(Path('datasets'))
collector.add_images('path/to/new/images', 'smartphone')
"
```

### 2. Training Pipeline

```bash
# Train model with default settings
python -m src.training_pipeline

# Or use the orchestrator
python scripts/run_pipeline.py --skip-data
```

### 3. Model Serving

```bash
# Start the FastAPI server
python -m src.model_service

# Test the API
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/image.jpg"
```

### 4. Complete Pipeline

```bash
# Run everything: data validation + training + deployment
python scripts/run_pipeline.py

# Skip specific stages
python scripts/run_pipeline.py --skip-data --skip-training
```

## 🔄 MLOps Pipeline

### Pipeline Stages

1. **Data Collection & Validation**

   - Validates dataset structure and image quality
   - Removes corrupted or invalid images
   - Splits data into train/validation/test sets
   - Generates data quality reports

2. **Model Training**

   - Loads and preprocesses data using FastAI
   - Trains ResNet18 model with transfer learning
   - Logs metrics to MLflow and WandB
   - Evaluates model performance
   - Saves model artifacts

3. **Model Validation**

   - Checks model performance against thresholds
   - Validates model file integrity
   - Generates evaluation reports

4. **Model Deployment**

   - Deploys model as FastAPI service
   - Performs health checks
   - Updates monitoring dashboards

5. **Monitoring**
   - Tracks prediction metrics
   - Monitors system resources
   - Alerts on performance degradation

### Experiment Tracking

The pipeline integrates with both MLflow and Weights & Biases:

- **MLflow**: Local experiment tracking and model registry
- **WandB**: Cloud-based experiment tracking with rich visualizations

Access MLflow UI at `http://localhost:5000` when running locally.

## 📊 API Documentation

### Endpoints

- `GET /` - Health check and service info
- `GET /health` - Detailed health status
- `GET /model/info` - Model metadata
- `POST /predict` - Single image prediction
- `POST /predict/batch` - Batch image prediction
- `GET /metrics` - Prometheus metrics
- `POST /model/reload` - Reload model (admin)

### Example Usage

```python
import requests

# Single prediction
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f},
        params={'confidence_threshold': 0.5}
    )
    result = response.json()
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

### Response Format

```json
{
  "predicted_class": "smartphone",
  "confidence": 0.95,
  "probabilities": {
    "smartphone": 0.95,
    "tablet": 0.03,
    "laptop": 0.01,
    "camera": 0.005,
    "smartwatch": 0.003,
    "headphones": 0.002
  },
  "top_predictions": [
    { "class": "smartphone", "confidence": 0.95 },
    { "class": "tablet", "confidence": 0.03 }
  ],
  "processing_time_ms": 45.2,
  "model_version": "gadget_classifier_model"
}
```

## 📈 Monitoring

### Metrics

The system exposes the following metrics:

- `model_predictions_total` - Total number of predictions
- `model_prediction_duration_seconds` - Prediction latency
- `model_load_time_seconds` - Model loading time
- System metrics (CPU, memory, disk usage)

### Dashboards

Access monitoring dashboards:

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Alerts

Configured alerts for:

- Model accuracy drops below 80%
- High prediction latency (>1s)
- Service downtime
- High resource usage

## 🔧 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ --cov=src

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Jupyter Development

```bash
# Start Jupyter server
jupyter lab

# Or use Docker
docker-compose up jupyter
```

## 🚢 Deployment

### Production Deployment

1. **Container Registry**

   ```bash
   # Build and push image
   docker build -t your-registry/gadget-predictor:latest .
   docker push your-registry/gadget-predictor:latest
   ```

2. **Kubernetes**

   ```bash
   # Apply Kubernetes manifests
   kubectl apply -f k8s/
   ```

3. **Cloud Platforms**
   - AWS ECS/Fargate
   - Google Cloud Run
   - Azure Container Instances

### Scaling

The API service is stateless and can be horizontally scaled:

```yaml
# docker-compose.yml
services:
  gadget-predictor:
    # ... configuration
    deploy:
      replicas: 3
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/gadget-predictor.git
cd gadget-predictor

# Create development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FastAI for the deep learning framework
- Weights & Biases for experiment tracking
- MLflow for model management
- FastAPI for the web framework
- Prometheus and Grafana for monitoring

## 📞 Support

For questions and support:

- Create an issue on GitHub
- Check the documentation
- Review existing issues and discussions

---

**Happy Machine Learning! 🤖✨**
