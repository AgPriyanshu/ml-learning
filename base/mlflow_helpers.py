import time, json, os
import mlflow
import psutil
import torch

try:
    import GPUtil
except Exception:
    GPUtil = None

MLFLOW_TRACKING_URI = "http://100.107.183.71:5000"


def get_system_metrics(device: str):
    """Return dict of CPU/GPU/RAM metrics (best-effort)."""
    m = {}
    # CPU / RAM
    m["cpu_percent"] = float(psutil.cpu_percent(interval=None))
    vm = psutil.virtual_memory()
    m["ram_gb"] = round((vm.used / (1024**3)), 3)

    # GPU (if available)
    if device == "cuda" and torch.cuda.is_available():
        # torch-based mem (current process)
        m["gpu_mem_gb_alloc"] = round(torch.cuda.memory_allocated() / (1024**3), 3)
        m["gpu_mem_gb_reserved"] = round(torch.cuda.memory_reserved() / (1024**3), 3)
        # overall GPU util via GPUtil (optional)
        if GPUtil:
            try:
                g = GPUtil.getGPUs()[0]
                m["gpu_util"] = float(g.load) * 100.0
                m["gpu_mem_gb_total"] = round(g.memoryTotal / 1024, 3)
                m["gpu_mem_gb_used"] = round(g.memoryUsed / 1024, 3)
            except Exception:
                pass
    return m


class MlflowLogger:
    def __init__(self, experiment="Footprints-UNet", run_name=None):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment)
        self.run = None
        self.run_name = run_name

    def __enter__(self):
        self.run = mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, exc_type, exc, tb):
        mlflow.end_run()

    def log_params(self, **kwargs):
        # Filter out None
        params = {k: v for k, v in kwargs.items() if v is not None}
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int | None = None):
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str | os.PathLike):
        mlflow.log_artifact(str(path))

    def log_dict(self, d: dict, artifact_file: str = "metrics.json"):
        mlflow.log_dict(d, artifact_file)
