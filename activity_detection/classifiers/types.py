import numpy as np
from dataclasses import dataclass

from torch.functional import Tensor
from dataclasses import field


@dataclass
class Prediction:
    detected: bool
    bounding_box: np.ndarray | Tensor = field(default_factory=lambda: np.array([]))
