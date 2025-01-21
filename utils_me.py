import random
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image


def PILtoTorch(pil_image: Image.Image, resolution: Sequence[int]) -> torch.Tensor:
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)