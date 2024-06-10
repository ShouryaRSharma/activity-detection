from enum import Enum

import torch


class DeviceType(Enum):
    MPS = "mps"
    CUDA = "cuda"
    CPU = "cpu"


def get_device() -> DeviceType:
    """Get the device type to use for the model inference.

    Returns:
        DeviceType: The device type to use
    """
    if torch.cuda.is_available():
        return DeviceType.CUDA
    elif torch.backends.mps.is_available():
        return DeviceType.MPS
    return DeviceType.CPU
