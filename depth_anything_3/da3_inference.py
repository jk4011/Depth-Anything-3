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


# def load_model(model_name: str = "depth-anything/DA3-Giant"):
def load_model(model_name: str = "depth-anything/DA3NESTED-GIANT-LARGE"):
    """Load Depth Anything 3 model. If model is already loaded, return it."""
    global model
    if model is None:
        print(f"Loading Depth Anything 3 ({model_name})...")
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


def load_colmap_poses(
    pose_path: str,
    image_names: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load camera poses from COLMAP binary files.

    Args:
        pose_path: Path to directory containing images.bin and cameras.bin
        image_names: Optional list of image paths to filter and order by

    Returns:
        extrinsics: (N, 4, 4) world-to-camera matrices
        intrinsics: (N, 3, 3) camera intrinsic matrices
        ordered_image_names: List of image names in the order of extrinsics/intrinsics
    """
    images_bin_path = os.path.join(pose_path, "images.bin")
    cameras_bin_path = os.path.join(pose_path, "cameras.bin")

    if not os.path.exists(images_bin_path):
        raise FileNotFoundError(f"images.bin not found at {images_bin_path}")
    if not os.path.exists(cameras_bin_path):
        raise FileNotFoundError(f"cameras.bin not found at {cameras_bin_path}")

    print(f"Loading COLMAP poses from {pose_path}...")

    # Read COLMAP binary files
    images = read_images_binary(images_bin_path)
    cameras = read_cameras_binary(cameras_bin_path)

    # Build a mapping from image name to pose data
    pose_data = {}
    for image_id, image_data in images.items():
        image_name = image_data.name
        camera = cameras[image_data.camera_id]

        # Convert quaternion to rotation matrix
        R = image_data.qvec2rotmat()
        t = image_data.tvec

        # Create extrinsic matrix (world to camera)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t

        # Create intrinsic matrix
        if camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        elif camera.model == "SIMPLE_PINHOLE":
            f, cx, cy = camera.params
            fx = fy = f
        elif camera.model in ["SIMPLE_RADIAL", "RADIAL"]:
            f, cx, cy = camera.params[:3]
            fx = fy = f
        elif camera.model == "OPENCV":
            fx, fy, cx, cy = camera.params[:4]
        else:
            # Fallback for other models
            fx = fy = camera.params[0] if len(camera.params) > 0 else 1000
            cx = camera.width / 2
            cy = camera.height / 2

        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        pose_data[image_name] = {
            'extrinsic': extrinsic,
            'intrinsic': intrinsic,
        }

    print(f"Loaded {len(pose_data)} poses from COLMAP")

    # If image_names provided, order by them and filter
    if image_names is not None:
        extrinsics = []
        intrinsics = []
        ordered_names = []

        for img_path in image_names:
            img_name = os.path.basename(img_path)
            if img_name in pose_data:
                extrinsics.append(pose_data[img_name]['extrinsic'])
                intrinsics.append(pose_data[img_name]['intrinsic'])
                ordered_names.append(img_path)
            else:
                print(f"Warning: No pose found for {img_name}, skipping...")

        if len(ordered_names) == 0:
            raise ValueError("No matching poses found for provided image names")

        return np.array(extrinsics), np.array(intrinsics), ordered_names
    else:
        # Return all poses in arbitrary order
        extrinsics = []
        intrinsics = []
        ordered_names = []

        for img_name, data in pose_data.items():
            extrinsics.append(data['extrinsic'])
            intrinsics.append(data['intrinsic'])
            ordered_names.append(img_name)

        return np.array(extrinsics), np.array(intrinsics), ordered_names


@cache_output(func_name="_da3_inference", override=False)
def _da3_inference(
    image_names: list = None,
    extrinsics: np.ndarray = None,
    intrinsics: np.ndarray = None,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
) -> dict:
    """
    Cached Depth Anything 3 inference function.

    Args:
        image_names: List of image file paths
        extrinsics: Optional camera extrinsics (N, 4, 4) for pose-conditioned depth
        intrinsics: Optional camera intrinsics (N, 3, 3) for pose-conditioned depth
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

    mode = "Pose-Conditioned" if extrinsics is not None else "Pose Estimation"
    print(f"Running DA3 inference ({mode}) on {len(image_names)} images...")

    # Run inference
    prediction = model.inference(
        image_names,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
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
    pose_path: str = None,
    colmap_dir: str = None,
    n_images: int = -1,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
) -> EasyDict:
    """
    Run Depth Anything 3 inference on images.

    Args:
        image_folder: Path to image directory
        image_names: List of image file paths (overrides image_folder)
        pose_path: Path to COLMAP sparse directory (containing images.bin, cameras.bin)
                   If provided, runs pose-conditioned depth estimation
        colmap_dir: Path to COLMAP directory (containing images/ and sparse/)
                    If provided, automatically sets image_folder and pose_path
        n_images: Number of images to sample from the sequence (-1 for all)
        process_res: Processing resolution
        process_res_method: Resize method for processing

    Returns:
        EasyDict containing inference results
    """
    # Handle colmap_dir: automatically set image_folder and pose_path
    if colmap_dir is not None:
        image_folder = os.path.join(colmap_dir, "images")
        # Try sparse/ first, then sparse/0/
        sparse_dir = os.path.join(colmap_dir, "sparse")
        if os.path.exists(os.path.join(sparse_dir, "images.bin")):
            pose_path = sparse_dir
        elif os.path.exists(os.path.join(sparse_dir, "0", "images.bin")):
            pose_path = os.path.join(sparse_dir, "0")
        else:
            raise FileNotFoundError(f"No COLMAP sparse data found in {sparse_dir}")
        print(f"Using COLMAP directory: {colmap_dir}")
        print(f"  - Images: {image_folder}")
        print(f"  - Poses: {pose_path}")

    # Load image names
    print(f"Loading images from {image_folder}...")
    if image_names is None:
        image_names = glob.glob(os.path.join(image_folder, "*"))
        # Filter only image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
        image_names = [p for p in image_names if os.path.splitext(p)[1].lower() in image_extensions]
        try:
            image_names.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        except:
            image_names.sort(key=lambda p: os.path.splitext(os.path.basename(p))[0])

    # Sample images if requested
    if n_images > 0 and n_images < len(image_names):
        image_indices = np.linspace(0, len(image_names) - 1, n_images).astype(int)
        image_names = [image_names[i] for i in image_indices]

    print(f"Found {len(image_names)} images")

    # Load poses if pose_path is provided
    extrinsics = None
    intrinsics = None

    if pose_path is not None:
        extrinsics, intrinsics, image_names = load_colmap_poses(pose_path, image_names)
        print(f"Loaded {len(image_names)} images with poses")

    # Run inference
    prediction = _da3_inference(
        image_names,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        process_res=process_res,
        process_res_method=process_res_method,
    )

    return EasyDict(prediction)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Depth Anything 3 inference")
    parser.add_argument("--image_folder", type=str, default=None,
                        help="Path to image directory")
    parser.add_argument("--pose_path", type=str, default=None,
                        help="Path to COLMAP sparse directory (images.bin, cameras.bin)")
    parser.add_argument("--colmap_dir", type=str, default=None,
                        help="Path to COLMAP directory (images/, sparse/)")
    parser.add_argument("--n_images", type=int, default=-1,
                        help="Number of images to sample")
    parser.add_argument("--process_res", type=int, default=504,
                        help="Processing resolution")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on")

    args = parser.parse_args()

    # Validate arguments
    if args.colmap_dir is None and args.image_folder is None:
        parser.error("Either --colmap_dir or --image_folder is required")

    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"

    # Run inference
    predictions = da3_inference(
        image_folder=args.image_folder,
        pose_path=args.pose_path,
        colmap_dir=args.colmap_dir,
        n_images=args.n_images,
        process_res=args.process_res,
    )

    print("Done!")
    print(f"Depth shape: {predictions['depth'].shape}")
    print(f"Extrinsics shape: {predictions['extrinsics'].shape}")
    print(f"Intrinsics shape: {predictions['intrinsics'].shape}")
