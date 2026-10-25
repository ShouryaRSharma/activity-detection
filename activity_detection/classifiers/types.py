from dataclasses import dataclass, field

import numpy as np
from torch.functional import Tensor


@dataclass
class Prediction:
    detected: bool
    bounding_box: np.ndarray | Tensor = field(default_factory=lambda: np.array([]))
