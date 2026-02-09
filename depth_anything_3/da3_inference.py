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
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import glob
import torch
import numpy as np
from typing import List, Optional, Tuple
import builtins

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.read_write_model import (
    read_images_binary,
    read_cameras_binary,
)
from jhutil import cache_output
from easydict import EasyDict
from depth_anything_3.specs import Gaussians



device = "cuda" if torch.cuda.is_available() else "cpu"


class DA3State:
    """Singleton to persist model state across module reloads (even with autoreload)."""

    def __new__(cls):
        # Store in builtins to survive module reloads
        if not hasattr(builtins, '_da3_state_instance'):
            instance = super().__new__(cls)
            instance.model = None
            instance.upsampler = None
            builtins._da3_state_instance = instance
        return builtins._da3_state_instance


_state = DA3State()


def load_upsampler():
    """Load AnyUp upsampler model for feature upsampling."""
    global _state
    if _state.upsampler is None:
        print("Loading AnyUp upsampler...")
        _state.upsampler = torch.hub.load("wimmerth/anyup", "anyup", verbose=False).to(device).eval()
        print("AnyUp loaded")


# def load_model(model_name: str = "depth-anything/DA3-Giant"):
def load_model(model_name):
    model_path = {
        "giant": "depth-anything/DA3-Giant-1.1",
        "nested": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    }

    """Load Depth Anything 3 model. If model is already loaded, return it."""
    global _state
    if _state.model is None or model_name not in _state.model.model_name:
        print(f"Loading Depth Anything 3 ({model_name})...")
        _state.model = DepthAnything3.from_pretrained(model_path[model_name])
        _state.model = _state.model.to(device=torch.device(device))
        _state.model.eval()
        print(f"Model loaded on {device}")


def unload_model():
    """Unload model from memory."""
    global _state
    if _state.model is not None:
        _state.model.cpu()
        del _state.model
        _state.model = None
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
    from PIL import Image as PILImage

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

        # Store COLMAP camera size for later scaling
        pose_data[image_name] = {
            'extrinsic': extrinsic,
            'intrinsic': intrinsic,
            'colmap_width': camera.width,
            'colmap_height': camera.height,
        }

    print(f"Loaded {len(pose_data)} poses from COLMAP")

    def scale_intrinsic_to_image(intrinsic, colmap_w, colmap_h, img_path):
        """Scale intrinsic matrix if actual image size differs from COLMAP calibration size."""
        try:
            actual_img = PILImage.open(img_path)
            actual_w, actual_h = actual_img.size
            actual_img.close()

            # Scale intrinsics if image size differs from COLMAP calibration size
            if actual_w != colmap_w or actual_h != colmap_h:
                scale_x = actual_w / colmap_w
                scale_y = actual_h / colmap_h
                scaled_intrinsic = intrinsic.copy()
                scaled_intrinsic[0, :] *= scale_x  # fx, cx
                scaled_intrinsic[1, :] *= scale_y  # fy, cy
                return scaled_intrinsic
        except Exception as e:
            print(f"Warning: Could not read image {img_path}: {e}")
        return intrinsic

    # If image_names provided, order by them and filter
    if image_names is not None:
        extrinsics = []
        intrinsics = []
        ordered_names = []

        for img_path in image_names:
            img_name = os.path.basename(img_path)
            if img_name in pose_data:
                data = pose_data[img_name]
                extrinsics.append(data['extrinsic'])
                # Scale intrinsic to actual image size
                scaled_intrinsic = scale_intrinsic_to_image(
                    data['intrinsic'],
                    data['colmap_width'],
                    data['colmap_height'],
                    img_path
                )
                intrinsics.append(scaled_intrinsic)
                ordered_names.append(img_path)
            else:
                print(f"Warning: No pose found for {img_name}, skipping...")

        if len(ordered_names) == 0:
            raise ValueError("No matching poses found for provided image names")

        return np.array(extrinsics), np.array(intrinsics), ordered_names
    else:
        # Return all poses in arbitrary order (no scaling since we don't have image paths)
        extrinsics = []
        intrinsics = []
        ordered_names = []

        for img_name, data in pose_data.items():
            extrinsics.append(data['extrinsic'])
            intrinsics.append(data['intrinsic'])
            ordered_names.append(img_name)

        return np.array(extrinsics), np.array(intrinsics), ordered_names


@cache_output(func_name="da3_inference_simple", override=False)
def da3_inference_simple(
    image_names: list = None,
    extrinsics: np.ndarray = None,
    intrinsics: np.ndarray = None,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    feat_layer: int = None,
    model_name: str = "giant",
    infer_gs: bool = True,
) -> dict:
    load_model(model_name)

    return _state.model.inference(
        image_names,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        process_res=process_res,
        process_res_method=process_res_method,
        export_feat_layers=None if feat_layer is None else [feat_layer],
        infer_gs=infer_gs,
    )


@cache_output(func_name="_da3_inference", override=False)
@torch.inference_mode()
def _da3_inference(
    image_names: list = None,
    extrinsics: np.ndarray = None,
    intrinsics: np.ndarray = None,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    feat_layer: int = None,
    model_name: str = "giant",
    infer_gs: bool = False,
) -> dict:
    """
    Cached Depth Anything 3 inference function.

    Args:
        image_names: List of image file paths
        extrinsics: Optional camera extrinsics (N, 4, 4) for pose-conditioned depth
        intrinsics: Optional camera intrinsics (N, 3, 3) for pose-conditioned depth
        process_res: Processing resolution
        process_res_method: Resize method for processing
        export_feat_layers: List of layer indices to export features from
        upsample: If True, upsample low-res features to high-res using AnyUp
        infer_gs: If True, infer Gaussian parameters

    Returns:
        Dictionary containing:
            - depth: Depth maps (N, H, W)
            - conf: Confidence scores (N, H, W)
            - extrinsics: Camera extrinsics (N, 3, 4)
            - intrinsics: Camera intrinsics (N, 3, 3)
            - processed_images: Processed input images (N, H, W, 3)
            - feat: (if export_feat_layers) Dict of features per layer
            - feat_hr: (if upsample) Dict of upsampled features per layer
            - gaussians: (if infer_gs) Gaussian parameters
    """
    load_model(model_name)

    mode = "Pose-Conditioned" if extrinsics is not None else "Pose Estimation"
    print(f"Running DA3 inference ({mode}) on {len(image_names)} images...")

    prediction = _state.model.inference(
        image_names,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        process_res=process_res,
        process_res_method=process_res_method,
        export_feat_layers=None if feat_layer is None else [feat_layer],
        infer_gs=infer_gs,
    )

    # Prepare output dictionary — all tensors on CUDA
    _dev = device
    predictions = {
        'depth': torch.as_tensor(prediction.depth, dtype=torch.float32, device=_dev),
        'conf': torch.as_tensor(prediction.conf, dtype=torch.float32, device=_dev),
        'extrinsics': torch.as_tensor(prediction.extrinsics, dtype=torch.float32, device=_dev),
        'intrinsics': torch.as_tensor(prediction.intrinsics, dtype=torch.float32, device=_dev),
        'processed_images': torch.as_tensor(prediction.processed_images, dtype=torch.float32, device=_dev).permute(0, 3, 1, 2) / 255.0,
    }

    # Add optional fields if available
    if hasattr(prediction, 'gaussians'):
        predictions['gaussians'] = prediction.gaussians

    # Extract features from aux if feat_layer was requested
    if feat_layer is not None:
        key = f"feat_layer_{feat_layer}"
        predictions['feat'] = torch.as_tensor(prediction.aux[key], dtype=torch.float32, device=_dev)

    print("Inference completed")
    return EasyDict(predictions)


@cache_output(func_name="upsample_features", override=False)
@torch.inference_mode()
def upsample_features(processed_images, feat, pca_dim=3, pca_subsamples=10000):
    """
    Apply PCA on low-res features first, then upsample to high-res using AnyUp.

    Args:
        processed_images: (N, 3, H, W) in [0, 1]
        feat: (N, h, w, C) low-res features
        pca_dim: Target dimension after PCA
        pca_subsamples: Number of samples for fitting PCA

    Returns:
        feat_hr: (N, H, W, pca_dim) PCA-reduced and upsampled features
    """
    load_upsampler()

    N = processed_images.shape[0]
    H, W = processed_images.shape[2], processed_images.shape[3]

    # Convert feat to tensor if numpy
    if isinstance(feat, np.ndarray):
        feat = torch.from_numpy(feat)

    h, w = feat.shape[1], feat.shape[2]
    C = feat.shape[-1]  # feature channels

    # === Step 1: Apply PCA on low-res features ===
    feat_flat = feat.reshape(-1, C)  # (N*h*w, C)
    total_pixels = feat_flat.shape[0]

    # Subsample for PCA fitting
    if pca_subsamples < total_pixels:
        indices = torch.randperm(total_pixels)[:pca_subsamples]
        feat_subsample = feat_flat[indices]
    else:
        feat_subsample = feat_flat

    print(f"Fitting PCA on low-res features: {C} -> {pca_dim} dims (using {feat_subsample.shape[0]} samples)...")

    # Center the data
    mean_feat = feat_subsample.mean(dim=0, keepdim=True)
    feat_centered = feat_subsample - mean_feat

    # Compute PCA via SVD
    U, S, Vh = torch.linalg.svd(feat_centered.to(feat_flat.device), full_matrices=False)
    components = Vh[:pca_dim]  # (pca_dim, C)

    # Apply PCA to all low-res features
    feat_flat_centered = feat_flat - mean_feat
    feat_pca_flat = feat_flat_centered @ components.T  # (N*h*w, pca_dim)

    # Reshape back to low-res spatial dimensions
    feat_pca = feat_pca_flat.reshape(N, h, w, pca_dim)
    print(f"PCA completed: (N, {h}, {w}, {C}) -> (N, {h}, {w}, {pca_dim})")

    # === Step 2: Upsample PCA-reduced features to high-res ===
    # ImageNet normalization constants
    mean_img = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_img = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    print(f"Upsampling PCA features with AnyUp ({N} images, {pca_dim} channels)...")

    # Batched: normalize all images and upsample features at once
    hr_imgs = processed_images.float().to(device)
    hr_imgs = (hr_imgs - mean_img) / std_img  # (N, 3, H, W)

    lr_feats = feat_pca.permute(0, 3, 1, 2).to(device)  # (N, pca_dim, h, w)

    hr_feats = _state.upsampler(hr_imgs, lr_feats, q_chunk_size=256).to(torch.float32)  # (N, pca_dim, H, W)

    feat_hr = hr_feats.permute(0, 2, 3, 1).cpu()  # (N, H, W, pca_dim)

    print(f"Upsampling completed: (N, {h}, {w}, {pca_dim}) -> {feat_hr.shape}")
    return feat_hr


def unproject_depth_to_points(
    depth,
    extrinsics,
    intrinsics,
) -> torch.Tensor:
    """
    Unproject depth maps to 3D world points (batched, on GPU).

    Args:
        depth: Depth maps (N, H, W)
        extrinsics: Camera extrinsics w2c (N, 3, 4) or (N, 4, 4)
        intrinsics: Camera intrinsics (N, 3, 3)

    Returns:
        World points (N, H, W, 3) as torch.Tensor
    """
    if isinstance(depth, np.ndarray):
        depth = torch.from_numpy(depth)
    if isinstance(extrinsics, np.ndarray):
        extrinsics = torch.from_numpy(extrinsics)
    if isinstance(intrinsics, np.ndarray):
        intrinsics = torch.from_numpy(intrinsics)

    depth = depth.float()
    extrinsics = extrinsics.float()
    intrinsics = intrinsics.float()

    if extrinsics.shape[1] == 4:
        extrinsics = extrinsics[:, :3, :]

    N, H, W = depth.shape

    _dev = depth.device

    # Pixel grid (H, W, 3) — shared across all views, same device
    v, u = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=_dev), torch.arange(W, dtype=torch.float32, device=_dev), indexing='ij')
    pixels = torch.stack([u, v, torch.ones_like(u)], dim=-1)  # (H, W, 3)

    # Batched K_inv: (N, 3, 3)
    K_inv = torch.linalg.inv(intrinsics)

    # Unproject to camera coords: K_inv @ pixels^T -> (N, 3, H*W)
    cam_dirs = K_inv @ pixels.reshape(-1, 3).T  # (N, 3, H*W)
    # Scale by depth: (N, 3, H*W) * (N, 1, H*W)
    cam_coords = cam_dirs * depth.reshape(N, 1, H * W)  # (N, 3, H*W)

    # c2w from w2c: R_inv = R^T, t_inv = -R^T @ t
    R = extrinsics[:, :3, :3]  # (N, 3, 3)
    t = extrinsics[:, :3, 3:]  # (N, 3, 1)
    R_inv = R.transpose(1, 2)  # (N, 3, 3)
    t_inv = -R_inv @ t  # (N, 3, 1)

    # Transform to world: R_inv @ cam_coords + t_inv
    world_coords = R_inv @ cam_coords + t_inv  # (N, 3, H*W)

    return world_coords.reshape(N, 3, H, W).permute(0, 2, 3, 1)  # (N, H, W, 3)

from jhutil import print_time

def da3_inference(
    image_folder: str = None,
    image_names: list = None,
    pose_path: str = None,
    colmap_dir: str = None,
    n_images: int = -1,
    process_res: int = 504,
    process_res_method: str = "upper_bound_resize",
    feat_layer: int = None,
    upsample: bool = False,
    pca_dim: int = 3,
    pca_subsamples: int = 10000,
    model_name: str = "giant",
    infer_gs: bool = False,
) -> EasyDict:
    if image_names is not None:
        image_names = [str(image_name) for image_name in image_names]
    
    if upsample:
        assert feat_layer is not None, "feat_layer must be provided if upsample is True"
    print("da3_viser \\\n" + \
        (f"--image_folder  {image_folder} \\\n"          if image_folder is not None else "") + \
        (f"--image_names   {' '.join(image_names)} \\\n" if image_names is not None else "") + \
        (f"--pose_path     {pose_path} \\\n"             if pose_path is not None else "") + \
        (f"--colmap_dir    {colmap_dir} \\\n"            if colmap_dir is not None else "") + \
        (f"--n_images      {n_images} \\\n"              if n_images != -1 else "") + \
        (f"--process_res   {process_res} \\\n"           if process_res != 504 else "") + \
        (f"--feat_layer    {feat_layer} \\\n"            if feat_layer is not None else "")
    )
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
        feat_layer: Layer index to export features from
        upsample: If True, upsample low-res features to high-res using AnyUp
        pca_dim: Target dimension after PCA (None to skip PCA)
        pca_subsamples: Number of samples for fitting PCA
        infer_gs: If True, infer Gaussian parameters

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

    # Run inference (may return CPU tensors from cache)
    with print_time("da3_inference"):
        prediction = _da3_inference(
            image_names,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            process_res=process_res,
            process_res_method=process_res_method,
            feat_layer=feat_layer,
            model_name=model_name,
        )

    # Move all tensors to CUDA
    for k, v in prediction.items():
        if isinstance(v, torch.Tensor) and not v.is_cuda:
            prediction[k] = v.cuda()

    # get points
    prediction.world_points = unproject_depth_to_points(
        prediction.depth,
        prediction.extrinsics,
        prediction.intrinsics,
    )

    # TODO: infer gaussian을 여기서 만들기 
    if infer_gs:
        prediction.gaussians = infer_gaussian(prediction)

    if hasattr(prediction, 'feat') and prediction.feat is not None and upsample:

        with print_time("upsample_features"):
            prediction.feat_hr = upsample_features(
                prediction.processed_images,
                prediction.feat,
                pca_dim=pca_dim,
                pca_subsamples=pca_subsamples,
            )
    
    return prediction

C0 = 0.28209479177387814


def get_gaussians(
    depths: torch.Tensor,  # b h w
    points: torch.Tensor,  # b h w 3
    images: torch.Tensor,  # b 3 h w
    conf: torch.Tensor,  # b h w
    device: torch.device,
    scale_multiplier: float = 0.01,
) -> Gaussians:
    b, h, w = depths.shape
    N = b * h * w

    # means: (1, N, 3)
    gs_means = points.reshape(N, 3).unsqueeze(0)

    # harmonics: inline SH0 = (rgb - 0.5) / C0, shape (1, N, 3, 1)
    gs_harmonics = ((images.permute(0, 2, 3, 1).reshape(N, 3) - 0.5) / C0).unsqueeze(-1).unsqueeze(0)

    # scales: isotropic depth * multiplier, (1, N, 3)
    gs_scales = (depths.reshape(N, 1) * scale_multiplier).expand(-1, 3).unsqueeze(0)

    # rotations: identity [1,0,0,0], (1, N, 4)
    gs_rotations = torch.zeros((1, N, 4), device=device, dtype=points.dtype)
    gs_rotations[..., 0] = 1.0

    # opacities: per-view 2nd percentile threshold, vectorized
    conf_flat = conf.reshape(b, h * w)
    thresholds = torch.quantile(conf_flat, 0.02, dim=1, keepdim=True)  # (b, 1)
    gs_opacities = (conf_flat >= thresholds).float().reshape(1, N)

    return Gaussians(
        means=gs_means,
        scales=gs_scales,
        rotations=gs_rotations,
        harmonics=gs_harmonics,
        opacities=gs_opacities,
    )


def infer_gaussian(prediction, scale_multiplier=1e-4):

    points = prediction.world_points
    if isinstance(points, np.ndarray):
        points = torch.as_tensor(points, dtype=torch.float32, device="cuda")
    else:
        points = points.to(dtype=torch.float32, device="cuda")

    return get_gaussians(
        depths=prediction.depth.to(dtype=torch.float32, device="cuda"),
        points=points,
        images=prediction.processed_images.to(dtype=torch.float32, device="cuda"),
        conf=prediction.conf.to(dtype=torch.float32, device="cuda"),
        device=torch.device("cuda"),
        scale_multiplier=scale_multiplier,
    )


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
    parser.add_argument("--feat_layer", type=int, default=None,
                        help="Layer index to export features from (e.g., --feat_layer 20)")
    parser.add_argument("--upsample", action='store_true',
                        help="Upsample features using AnyUp")
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
        feat_layer=args.feat_layer,
        upsample=args.upsample,
    )

    print("Done!")
    print(f"Depth shape: {predictions['depth'].shape}")
    print(f"Extrinsics shape: {predictions['extrinsics'].shape}")
    print(f"Intrinsics shape: {predictions['intrinsics'].shape}")
    if 'feat' in predictions:
        print(f"Features (layer {args.feat_layer}): {predictions['feat'].shape}")
    if 'feat_hr' in predictions:
        print(f"Features HR (layer {args.feat_layer}): {predictions['feat_hr'].shape}")
