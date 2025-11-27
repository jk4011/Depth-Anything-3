# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import torch
import numpy as np
from typing import List, Optional

from depth_anything_3.api import DepthAnything3
from jhutil import cache_output
from easydict import EasyDict

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None


def load_model(model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE"):
    """Load Depth Anything 3 model. If model is already loaded, return it."""
    global model
    if model is None:
        print(f"Loading Depth Anything 3 model ({model_name})...")
        model = DepthAnything3.from_pretrained(model_name)
        model = model.to(device=torch.device(device))
        model.eval()
        print(f"Model loaded on {device}")
    return model


def unload_model():
    """Unload model from memory."""
    global model
    if model is not None:
        model.cpu()
        del model
        model = None
        torch.cuda.empty_cache()
        print("Model unloaded")


@cache_output(func_name="_da3_inference", override=False)
def _da3_inference(
    image_names: list = None,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
) -> dict:
    """
    Cached Depth Anything 3 inference function.

    Args:
        image_names: List of image file paths
        process_res: Processing resolution
        process_res_method: Resize method for processing

    Returns:
        Dictionary containing:
            - depth: Depth maps (N, H, W)
            - conf: Confidence scores (N, H, W)
            - extrinsics: Camera extrinsics (N, 3, 4)
            - intrinsics: Camera intrinsics (N, 3, 3)
            - processed_images: Processed input images (N, H, W, 3)
    """
    # Load model if not already loaded
    if model is None:
        load_model()

    print(f"Running DA3 inference on {len(image_names)} images...")

    # Run inference
    prediction = model.inference(
        image_names,
        process_res=process_res,
        process_res_method=process_res_method,
    )

    # Prepare output dictionary
    predictions = {
        'depth': prediction.depth,  # (N, H, W)
        'conf': prediction.conf,  # (N, H, W)
        'extrinsics': prediction.extrinsics,  # (N, 3, 4)
        'intrinsics': prediction.intrinsics,  # (N, 3, 3)
        'processed_images': prediction.processed_images,  # (N, H, W, 3)
    }

    # Add optional fields if available
    if hasattr(prediction, 'gs') and prediction.gs is not None:
        predictions['gs'] = prediction.gs
    if hasattr(prediction, 'feat') and prediction.feat is not None:
        predictions['feat'] = prediction.feat

    print("Inference completed")
    return predictions


def da3_inference(
    image_folder: str = None,
    image_names: list = None,
    n_images: int = -1,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
) -> dict:
    """
    Run Depth Anything 3 inference on images.

    Args:
        image_folder: Path to image directory
        image_names: List of image file paths (overrides image_folder)
        n_images: Number of images to sample from the sequence (-1 for all)
        process_res: Processing resolution
        process_res_method: Resize method for processing

    Returns:
        Dictionary containing inference results
    """
    # Use the provided image folder path
    print(f"Loading images from {image_folder}...")
    if image_names is None:
        image_names = glob.glob(os.path.join(image_folder, "*"))
        try:
            image_names.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        except:
            image_names.sort(key=lambda p: os.path.splitext(os.path.basename(p))[0])

    if n_images > 0 and n_images < len(image_names):
        image_indices = np.linspace(0, len(image_names) - 1, n_images).astype(int)
        image_names = [image_names[i] for i in image_indices]

    print(f"Found {len(image_names)} images")
    prediction = _da3_inference(
        image_names,
        process_res=process_res,
        process_res_method=process_res_method,
    )
    # Run cached inference
    return EasyDict(prediction)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Depth Anything 3 inference")
    parser.add_argument("--image_folder", type=str, required=True,
                        help="Path to image directory")
    parser.add_argument("--n_images", type=int, default=-1,
                        help="Number of images to sample")
    parser.add_argument("--process_res", type=int, default=504,
                        help="Processing resolution")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on")

    args = parser.parse_args()

    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"

    # Run inference
    predictions = da3_inference(
        image_folder=args.image_folder,
        n_images=args.n_images,
        process_res=args.process_res,
    )

    print("Done!")
    print(f"Depth shape: {predictions['depth'].shape}")
    print(f"Extrinsics shape: {predictions['extrinsics'].shape}")
    print(f"Intrinsics shape: {predictions['intrinsics'].shape}")
