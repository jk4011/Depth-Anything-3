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


def load_upsampler():
    """Load AnyUp upsampler model for feature upsampling."""
    global upsampler
    if upsampler is None:
        print("Loading AnyUp upsampler...")
        upsampler = torch.hub.load("wimmerth/anyup", "anyup", verbose=False).to(device).eval()
        print("AnyUp loaded")


# def load_model(model_name: str = "depth-anything/DA3-Giant"):
def load_model(model_name):
    model_path = {
        "giant": "depth-anything/DA3-Giant-1.1",
        "nested": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    }

    """Load Depth Anything 3 model. If model is already loaded, return it."""
    global model
    if model is None or model_name not in model.model_name:
        print(f"Loading Depth Anything 3 ({model_name})...")
        model = DepthAnything3.from_pretrained(model_path[model_name])
        model = model.to(device=torch.device(device))
        model.eval()
        print(f"Model loaded on {device}")


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

    Returns:
        Dictionary containing:
            - depth: Depth maps (N, H, W)
            - conf: Confidence scores (N, H, W)
            - extrinsics: Camera extrinsics (N, 3, 4)
            - intrinsics: Camera intrinsics (N, 3, 3)
            - processed_images: Processed input images (N, H, W, 3)
            - feat: (if export_feat_layers) Dict of features per layer
            - feat_hr: (if upsample) Dict of upsampled features per layer
    """
    load_model(model_name)

    mode = "Pose-Conditioned" if extrinsics is not None else "Pose Estimation"
    print(f"Running DA3 inference ({mode}) on {len(image_names)} images...")

    # Run inference
    from jhutil import color_log; color_log(1111, image_names)
    from jhutil import color_log; color_log(2222, extrinsics)
    from jhutil import color_log; color_log(3333, intrinsics)
    from jhutil import color_log; color_log(4444, process_res)
    from jhutil import color_log; color_log(5555, process_res_method)
    prediction = model.inference(
        image_names,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        process_res=process_res,
        process_res_method=process_res_method,
        export_feat_layers=None if feat_layer is None else [feat_layer],
    )

    # Prepare output dictionary
    predictions = {
        'depth': torch.tensor(prediction.depth),  # (N, H, W)
        'conf': torch.tensor(prediction.conf),  # (N, H, W)
        'extrinsics': torch.tensor(prediction.extrinsics),  # (N, 3, 4)
        'intrinsics': torch.tensor(prediction.intrinsics),  # (N, 3, 3)
        'processed_images': torch.tensor(prediction.processed_images).permute(0, 3, 1, 2) / 255.0,  # (N, H, W, 3)
    }

    # Add optional fields if available
    if hasattr(prediction, 'gaussians') and prediction.gaussians is not None:
        predictions['gs'] = prediction.gaussians

    # Extract features from aux if feat_layer was requested
    if feat_layer is not None:
        key = f"feat_layer_{feat_layer}"
        predictions['feat'] = torch.tensor(prediction.aux[key])

    print("Inference completed")
    return EasyDict(predictions)


@cache_output(func_name="upsample_features", override=False)
@torch.inference_mode()
def upsample_features(processed_images, feat, pca_dim=3, pca_subsamples=10000):
    """
    Upsample low-res features to high-res using AnyUp, then apply PCA.

    Args:
        processed_images: (N, H, W, 3) in [0, 1]
        feat: (N, h, w, C) low-res features
        pca_dim: Target dimension after PCA
        pca_subsamples: Number of samples for fitting PCA

    Returns:
        feat_hr: (N, H, W, pca_dim) upsampled and PCA-reduced features
    """
    load_upsampler()

    N = processed_images.shape[0]
    H, W = processed_images.shape[2], processed_images.shape[3]

    # ImageNet normalization constants
    mean_img = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std_img = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    print(f"Upsampling features with AnyUp ({N} images)...")

    # Convert feat to tensor if numpy
    if isinstance(feat, np.ndarray):
        feat = torch.from_numpy(feat)

    C = feat.shape[-1]  # feature channels

    # Process one image at a time to avoid 32-bit index overflow
    hr_features_list = []
    for i in range(N):
        # Prepare single image: (1, 3, H, W), already in [0, 1]
        hr_img = processed_images[i:i+1].float().to(device)
        hr_img = (hr_img - mean_img) / std_img

        # Get single feature map: (1, h, w, C) -> (1, C, h, w) for AnyUp
        lr_feat = feat[i:i+1].permute(0, 3, 1, 2).to(device)

        # Upsample: output is (1, C, H, W)
        hr_feat = upsampler(hr_img, lr_feat, q_chunk_size=256).to(torch.float32)

        # Convert back to (1, H, W, C)
        hr_feat = hr_feat.permute(0, 2, 3, 1).cpu()
        hr_features_list.append(hr_feat)

    feat_hr = torch.cat(hr_features_list, dim=0)  # (N, H, W, C)
    print("Upsampling completed")

    # Apply PCA
    feat_flat = feat_hr.reshape(-1, C)  # (N*H*W, C)
    total_pixels = feat_flat.shape[0]

    # Subsample for PCA fitting
    if pca_subsamples < total_pixels:
        indices = torch.randperm(total_pixels)[:pca_subsamples]
        feat_subsample = feat_flat[indices]
    else:
        feat_subsample = feat_flat

    print(f"Fitting PCA: {C} -> {pca_dim} dims (using {feat_subsample.shape[0]} samples)...")

    # Center the data
    mean_feat = feat_subsample.mean(dim=0, keepdim=True)
    feat_centered = feat_subsample - mean_feat

    # Compute PCA via SVD
    U, S, Vh = torch.linalg.svd(feat_centered.cuda(), full_matrices=False)
    components = Vh[:pca_dim].cpu()  # (pca_dim, C)

    # Apply PCA to all features
    feat_flat_centered = feat_flat - mean_feat
    feat_pca_flat = feat_flat_centered @ components.T  # (N*H*W, pca_dim)

    # Reshape back
    feat_hr = feat_pca_flat.reshape(N, H, W, pca_dim)

    print(f"PCA completed: (N, H, W, {C}) -> {feat_hr.shape}")
    return feat_hr


def unproject_depth_to_points(
    depth: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:

    # TODO: change this logic from numpy to torch
    if isinstance(depth, torch.Tensor):
        depth = depth.cpu().numpy()
    if isinstance(extrinsics, torch.Tensor):
        extrinsics = extrinsics.cpu().numpy()
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.cpu().numpy()
    """
    Unproject depth maps to 3D world points.

    Args:
        depth: Depth maps (N, H, W)
        extrinsics: Camera extrinsics w2c (N, 3, 4)
        intrinsics: Camera intrinsics (N, 3, 3)

    Returns:
        World points (N, H, W, 3)
    """
    N, H, W = depth.shape

    # Create pixel coordinates
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    ones = np.ones_like(u)
    pixels = np.stack([u, v, ones], axis=-1).astype(np.float32)  # (H, W, 3)

    world_points = []
    for i in range(N):
        # Get camera parameters
        K = intrinsics[i]  # (3, 3)
        ext = extrinsics[i]  # (3, 4)

        # Compute camera-to-world transform
        R = ext[:3, :3]
        t = ext[:3, 3:4]
        # c2w = inv(w2c)
        R_inv = R.T
        t_inv = -R_inv @ t
        c2w = np.concatenate([R_inv, t_inv], axis=1)  # (3, 4)

        # Unproject pixels to camera space
        K_inv = np.linalg.inv(K)
        d = depth[i]  # (H, W)

        # Camera coordinates
        cam_coords = (K_inv @ pixels.reshape(-1, 3).T).T.reshape(H, W, 3)
        cam_coords = cam_coords * d[..., None]  # (H, W, 3)

        # Transform to world coordinates
        cam_coords_homo = np.concatenate([cam_coords, np.ones((H, W, 1))], axis=-1)  # (H, W, 4)
        c2w_4x4 = np.eye(4)
        c2w_4x4[:3, :] = c2w
        world_coords = (c2w_4x4 @ cam_coords_homo.reshape(-1, 4).T).T.reshape(H, W, 4)
        world_points.append(world_coords[..., :3])

    return np.stack(world_points, axis=0)  # (N, H, W, 3)



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
) -> EasyDict:
    if image_names is not None:
        image_names = [str(image_name) for image_name in image_names]
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
        feat_layer=feat_layer,
        model_name=model_name,
    )

    # get points
    prediction.world_points = unproject_depth_to_points(
        prediction.depth,
        prediction.extrinsics,
        prediction.intrinsics,
    )

    if hasattr(prediction, 'feat') and prediction.feat is not None and upsample:
        prediction.feat_hr = upsample_features(
            prediction.processed_images,
            prediction.feat,
            pca_dim=pca_dim,
            pca_subsamples=pca_subsamples,
        )
    
    return prediction


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
