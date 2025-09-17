import os
import mlflow

MLFLOW_TRACKING_URI = "http://100.107.183.71:5000"


class MlflowLogger:
    def __init__(
        self, 
        experiment="Footprints-UNet", 
        run_name=None, 
        enable_system_metrics=True,
        system_metrics_sampling_interval=10,
        system_metrics_samples_before_logging=1
    ):
        """
        Initialize MLflow logger with system metrics support.
        
        Args:
            experiment: Name of the MLflow experiment
            run_name: Optional name for the specific run
            enable_system_metrics: Whether to enable MLflow's built-in system metrics logging
            system_metrics_sampling_interval: Interval in seconds between system metrics samples
            system_metrics_samples_before_logging: Number of samples to aggregate before logging
        """
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment)
        
        self.run = None
        self.run_name = run_name
        self.enable_system_metrics = enable_system_metrics
        
        # Configure system metrics sampling if custom values provided
        if system_metrics_sampling_interval != 10:
            mlflow.set_system_metrics_sampling_interval(system_metrics_sampling_interval)
        if system_metrics_samples_before_logging != 1:
            mlflow.set_system_metrics_samples_before_logging(system_metrics_samples_before_logging)

    def __enter__(self):
        self.run = mlflow.start_run(
            run_name=self.run_name, 
            log_system_metrics=self.enable_system_metrics
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        mlflow.end_run()

    def log_params(self, **kwargs):
        # Filter out None
        params = {k: v for k, v in kwargs.items() if v is not None}
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int | None = None, category: str | None = None):
        """
        Log metrics with optional category prefix.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Optional step number for time series tracking
            category: Optional category prefix (e.g., 'system', 'model', 'train', 'val')
        """
        if category:
            # Add category prefix to metric names
            prefixed_metrics = {f"{category}.{k}": v for k, v in metrics.items()}
            mlflow.log_metrics(prefixed_metrics, step=step)
        else:
            mlflow.log_metrics(metrics, step=step)


    def log_model_metrics(self, metrics: dict, step: int | None = None):
        """Log training/model metrics (loss, accuracy, dice, etc.)."""
        self.log_metrics(metrics, step=step, category="model")

    def log_train_metrics(self, metrics: dict, step: int | None = None):
        """Log training-specific metrics."""
        self.log_metrics(metrics, step=step, category="train")

    def log_val_metrics(self, metrics: dict, step: int | None = None):
        """Log validation-specific metrics."""
        self.log_metrics(metrics, step=step, category="val")

    def log_artifact(self, path: str | os.PathLike):
        mlflow.log_artifact(str(path))

    def log_dict(self, d: dict, artifact_file: str = "metrics.json"):
        mlflow.log_dict(d, artifact_file)
    
    @staticmethod
    def enable_system_metrics_logging():
        """Enable MLflow system metrics logging globally."""
        mlflow.enable_system_metrics_logging()
    
    @staticmethod
    def disable_system_metrics_logging():
        """Disable MLflow system metrics logging globally."""
        mlflow.disable_system_metrics_logging()
    
    @staticmethod
    def set_system_metrics_config(sampling_interval: int = None, samples_before_logging: int = None):
        """Configure system metrics logging parameters.
        
        Args:
            sampling_interval: Interval in seconds between system metrics samples
            samples_before_logging: Number of samples to aggregate before logging
        """
        if sampling_interval is not None:
            mlflow.set_system_metrics_sampling_interval(sampling_interval)
        if samples_before_logging is not None:
            mlflow.set_system_metrics_samples_before_logging(samples_before_logging)
