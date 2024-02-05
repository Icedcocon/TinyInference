import torch


def cuda_count():
    # even if install torch cpu, this interface would return 0.
    return torch.cuda.device_count()