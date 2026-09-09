"""Report the Python, PyTorch and CUDA environment used by this package."""

import json
import platform

import torch


details = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "gpu_count": torch.cuda.device_count(),
    "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
}
print(json.dumps(details, indent=2))
