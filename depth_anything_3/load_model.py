import os
import glob
import torch
import numpy as np
from typing import List, Optional, Tuple

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.read_write_model import (
    read_images_binary,
    read_cameras_binary,
)
from jhutil import cache_output
from easydict import EasyDict


device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
upsampler = None


# def load_model(model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE"):
def load_model(model_name: str = "depth-anything/DA3-Giant"):
    """Load Depth Anything 3 model. If model is already loaded, return it."""
    global model
    if model is None:
        print(f"Loading Depth Anything 3 ({model_name})...")
        model = DepthAnything3.from_pretrained(model_name)
        model = model.to(device=torch.device(device))
        model.eval()
        print(f"Model loaded on {device}")


if __name__ == "__main__":
    load_model()