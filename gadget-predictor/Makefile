.PHONY: help install data train serve deploy test clean docker-build docker-up docker-down

# Default target
help:
	@echo "Available targets:"
	@echo "  install        - Install dependencies"
	@echo "  data          - Run data validation and preprocessing"
	@echo "  train         - Train the model"
	@echo "  serve         - Start the API server"
	@echo "  pipeline      - Run complete MLOps pipeline"
	@echo "  test          - Run tests"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-up     - Start Docker Compose stack"
	@echo "  docker-down   - Stop Docker Compose stack"
	@echo "  clean         - Clean generated files"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

# Data pipeline
data:
	python -m src.data_pipeline

# Data augmentation
augment:
	python -m src.augmentation

# Model training
train:
	python -m src.training_pipeline

# API server
serve:
	python -m src.model_service

# Complete pipeline
pipeline:
	python scripts/run_pipeline.py

pipeline-skip-data:
	python scripts/run_pipeline.py --skip-data

pipeline-skip-training:
	python scripts/run_pipeline.py --skip-training

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

# Code quality
format:
	black src/ tests/ scripts/

lint:
	flake8 src/ tests/ scripts/

type-check:
	mypy src/

quality: format lint type-check

# Docker operations
docker-build:
	docker build -t gadget-predictor:latest .

docker-build-dev:
	docker build -t gadget-predictor:dev --target development .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Monitoring
prometheus:
	docker-compose up -d prometheus

grafana:
	docker-compose up -d grafana

mlflow:
	docker-compose up -d mlflow

jupyter:
	docker-compose up -d jupyter

# Cleanup
clean:
	rm -rf logs/*.json
	rm -rf models/*.pkl
	rm -rf .pytest_cache
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

clean-docker:
	docker-compose down -v
	docker system prune -f

# Development workflow
dev-setup: install-dev
	@echo "Development environment setup complete"

dev-pipeline: data train serve

# Production deployment
prod-build:
	docker build -t gadget-predictor:prod --target production .

prod-deploy: prod-build
	@echo "Production deployment would go here"
	@echo "Configure your deployment target (K8s, ECS, etc.)"

# Health checks
health-check:
	curl -f http://localhost:8000/health || echo "Service not running"

api-test:
	@echo "Testing API endpoints..."
	curl -s http://localhost:8000/ | jq .
	@echo "Upload an image to test predictions:"
	@echo "curl -X POST http://localhost:8000/predict -F 'file=@path/to/image.jpg'"

# Monitoring checks
metrics:
	curl -s http://localhost:8000/metrics

prometheus-check:
	curl -s http://localhost:9090/-/healthy || echo "Prometheus not running"

grafana-check:
	curl -s http://localhost:3000/api/health || echo "Grafana not running"

# Quick start
quick-start: docker-build docker-up
	@echo "🚀 MLOps stack is starting up..."
	@echo "Services will be available at:"
	@echo "  - API: http://localhost:8000"
	@echo "  - MLflow: http://localhost:5000"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Grafana: http://localhost:3000 (admin/admin)"
	@echo "  - Jupyter: http://localhost:8888"

# Demo
demo: quick-start
	@echo "Waiting for services to start..."
	sleep 30
	make health-check
	make api-test 