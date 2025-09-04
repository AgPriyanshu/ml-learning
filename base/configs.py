import os
import torch


def setup_environment():
    try:
        import torch.multiprocessing as mp

        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

    # modest but safe defaults for GDAL (tune up if RAM allows)
    os.environ["GDAL_CACHEMAX"] = os.environ.get("GDAL_CACHEMAX", "2048")  # MB
    os.environ["GDAL_NUM_THREADS"] = os.environ.get("GDAL_NUM_THREADS", "ALL_CPUS")

    # Speed up cuDNN autotuning for fixed image sizes
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")  # Ampere+
