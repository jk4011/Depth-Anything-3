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
import time
import threading
import argparse
from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf

from depth_anything_3.da3_inference import da3_inference
from depth_anything_3.utils.geometry import affine_inverse


def unproject_depth_to_points(
    depth: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
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


def viser_wrapper(
    pred_dict: dict,
    port: int = 8081,
    init_conf_threshold: float = 50.0,
    background_mode: bool = False,
    image_folder: str = None,
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "processed_images": (N, H, W, 3) uint8,
                "depth": (N, H, W),
                "conf": (N, H, W),
                "extrinsics": (N, 3, 4),
                "intrinsics": (N, 3, 3),
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        background_mode (bool): Whether to run the server in background thread.
        image_folder (str): Path to the folder containing input images.
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["processed_images"]  # (N, H, W, 3) uint8
    depth_map = pred_dict["depth"]  # (N, H, W)
    conf = pred_dict["conf"]  # (N, H, W)
    extrinsics_cam = pred_dict["extrinsics"]  # (N, 3, 4)
    intrinsics_cam = pred_dict["intrinsics"]  # (N, 3, 3)

    # Compute world points from depth
    world_points = unproject_depth_to_points(depth_map, extrinsics_cam, intrinsics_cam)

    # Convert images from uint8 to float [0, 1] for colors
    colors = images.astype(np.float32) / 255.0  # (N, H, W, 3)
    N, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    # Compute camera-to-world transforms
    cam_to_world = []
    for i in range(N):
        ext = extrinsics_cam[i]  # (3, 4)
        R = ext[:3, :3]
        t = ext[:3, 3:4]
        R_inv = R.T
        t_inv = -R_inv @ t
        c2w = np.concatenate([R_inv, t_inv], axis=1)  # (3, 4)
        cam_to_world.append(c2w)
    cam_to_world = np.stack(cam_to_world, axis=0)  # (N, 3, 4)

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(N), H * W)

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)

    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )

    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames", options=["All"] + [str(i) for i in range(N)], initial_value="All"
    )

    # Create the main point cloud handle
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    # Store references to frames & frustums
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """Add camera frames and frustums to the scene."""
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        for img_id in tqdm(range(N)):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            img = images_[img_id]  # (H, W, 3) uint8
            h, w = img.shape[:2]

            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)

        print(f"Threshold absolute value: {threshold_val}, percentage: {current_percentage}%")

        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    # Add the camera frames to the scene
    visualize_frames(cam_to_world, images)

    print("Starting viser server...")
    if background_mode:
        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server


parser = argparse.ArgumentParser(description="Depth Anything 3 demo with viser for 3D visualization")
parser.add_argument("--image_folder", type=str, default=None, help="Path to folder containing images")
parser.add_argument("--image_names", nargs="+", default=None,
    help="List of image paths to use for inference")
parser.add_argument("--pose_path", type=str, default=None,
    help="Path to COLMAP sparse directory (images.bin, cameras.bin) for pose-conditioned depth")
parser.add_argument("--colmap_dir", type=str, default=None,
    help="Path to COLMAP directory (images/, sparse/) for pose-conditioned depth")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8081, help="Port number for the viser server")
parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
parser.add_argument("--n_images", type=int, default=-1, help="Number of images to use for visualization")
parser.add_argument("--process_res", type=int, default=504, help="Processing resolution")
parser.add_argument("--visualize_cache_file", type=str, default=None, help="Path to cached predictions")
parser.add_argument("--skip_visualization", action="store_true", help="Skip visualization and only save predictions")


def main():
    """
    Main function for the Depth Anything 3 demo with viser for 3D visualization.
    """
    args = parser.parse_args()

    # Validate arguments
    if args.visualize_cache_file is None:
        if args.colmap_dir is None and args.image_folder is None:
            parser.error("Either --colmap_dir or --image_folder is required")

    if args.visualize_cache_file:
        predictions = torch.load(args.visualize_cache_file)
    else:
        predictions = da3_inference(
            image_folder=args.image_folder,
            image_names=args.image_names,
            pose_path=args.pose_path,
            colmap_dir=args.colmap_dir,
            n_images=args.n_images,
            process_res=args.process_res,
        )

    if not args.skip_visualization:
        print("Processing model outputs...")
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy()

        print("Visualizing 3D points by unprojecting depth map by cameras")
        print("Starting viser visualization...")

        viser_server = viser_wrapper(
            predictions,
            port=args.port,
            init_conf_threshold=args.conf_threshold,
            background_mode=args.background_mode,
            image_folder=args.image_folder,
        )
        print("Visualization complete")


if __name__ == "__main__":
    main()
