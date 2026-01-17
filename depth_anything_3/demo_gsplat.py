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

"""
Gaussian Splatting Viewer for Depth Anything 3

This demo uses gsplat's rasterization to visualize Gaussian parameters
predicted by Depth Anything 3 from multi-view images.

Based on gsplat's simple_viewer.py:
https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_viewer.py
"""

import argparse
import math
import time
import sys
import torch
import numpy as np
from pathlib import Path

# Add gsplat examples to path for gsplat_viewer
sys.path.insert(0, str(Path("/root/data1/jinhyeok/gsplat/examples")))

from gsplat.rendering import rasterization
import viser
from nerfview import CameraState, RenderTabState, apply_float_colormap
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from depth_anything_3.da3_inference import da3_inference


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Gaussian Splatting viewer for Depth Anything 3"
    )

    # Input arguments
    parser.add_argument("--image_folder", type=str, default=None, help="Path to image directory")
    parser.add_argument("--image_names", nargs="+", default=None, help="List of image paths")
    parser.add_argument("--pose_path", type=str, default=None, help="Path to COLMAP sparse directory")
    parser.add_argument("--colmap_dir", type=str, default=None, help="Path to COLMAP directory")

    # DA3 inference arguments
    parser.add_argument("--n_images", type=int, default=-1, help="Number of images to sample")
    parser.add_argument("--process_res", type=int, default=504, help="Processing resolution")
    parser.add_argument("--model_name", type=str, default="giant", choices=["giant", "nested"],
                        help="DA3 model to use")

    # Viewer arguments
    parser.add_argument("--port", type=int, default=8082, help="Port for the viewer server")
    parser.add_argument("--output_dir", type=str, default="results/", help="Output directory")

    # Gaussian filtering arguments
    parser.add_argument("--conf_threshold", type=float, default=0.0,
                        help="Confidence threshold to filter Gaussians (0.0 to 1.0)")
    parser.add_argument("--opacity_threshold", type=float, default=0.005,
                        help="Opacity threshold to filter Gaussians (0.0 to 1.0)")
    parser.add_argument("--max_gaussians", type=int, default=None,
                        help="Maximum number of Gaussians to keep (for memory efficiency)")

    args = parser.parse_args()

    # Validate arguments
    if args.colmap_dir is None and args.image_folder is None and args.image_names is None:
        parser.error("Either --colmap_dir, --image_folder, or --image_names is required")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ================================================================================
    # Step 1: Run DA3 inference to get Gaussian parameters
    # ================================================================================
    print("\n" + "=" * 80)
    print("Step 1: Running Depth Anything 3 inference with Gaussian prediction")
    print("=" * 80)

    predictions = da3_inference(
        image_folder=args.image_folder,
        image_names=args.image_names,
        pose_path=args.pose_path,
        colmap_dir=args.colmap_dir,
        n_images=args.n_images,
        process_res=args.process_res,
        model_name=args.model_name,
        infer_gs=True,  # Enable Gaussian inference
    )

    # Check if Gaussians were predicted
    if not hasattr(predictions, 'gaussians') or predictions.gaussians is None:
        raise RuntimeError(
            "Gaussian parameters were not predicted. "
            "Make sure the model supports Gaussian inference."
        )

    # ================================================================================
    # Step 2: Prepare Gaussians for visualization
    # ================================================================================
    print("\n" + "=" * 80)
    print("Step 2: Preparing Gaussians for visualization")
    print("=" * 80)

    gaussians = predictions.gaussians

    # Extract Gaussian parameters (all should be in world space)
    # DA3 provides: means, scales, rotations (wxyz), opacities, harmonics (SH)
    means = gaussians.means.to(device)  # [B, N, 3] or [N, 3]
    scales = gaussians.scales.to(device)  # [B, N, 3] or [N, 3]
    quats = gaussians.rotations.to(device)  # [B, N, 4] or [N, 4] wxyz
    opacities = gaussians.opacities.to(device)  # [B, N] or [N]
    harmonics = gaussians.harmonics.to(device)  # [B, N, 3, d_sh] or [N, 3, d_sh]

    # Remove batch dimension if present (DA3 returns batch of 1 for single scene)
    # Check each tensor individually since they have different dimensions
    if means.dim() == 3 and means.shape[0] == 1:
        means = means.squeeze(0)  # [N, 3]
    if scales.dim() == 3 and scales.shape[0] == 1:
        scales = scales.squeeze(0)  # [N, 3]
    if quats.dim() == 3 and quats.shape[0] == 1:
        quats = quats.squeeze(0)  # [N, 4]
    if opacities.dim() == 2 and opacities.shape[0] == 1:
        opacities = opacities.squeeze(0)  # [N]
    if harmonics.dim() == 4 and harmonics.shape[0] == 1:
        harmonics = harmonics.squeeze(0)  # [N, 3, d_sh]

    # Rearrange harmonics to [N, K, 3] format for gsplat
    # DA3 format: [N, 3, d_sh] (N points, xyz channels, d_sh coefficients)
    # gsplat format: [N, K, 3] (N points, K SH bases, xyz channels)
    if harmonics.dim() == 3:
        # [N, 3, d_sh] -> [N, d_sh, 3]
        colors = harmonics.permute(0, 2, 1).contiguous()
    else:
        raise ValueError(f"Unexpected harmonics shape: {harmonics.shape}, dim: {harmonics.dim()}")

    # Compute SH degree
    sh_degree = int(math.sqrt(colors.shape[1]) - 1)

    # Normalize quaternions (DA3 should provide normalized quats, but ensure it)
    quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-6)

    # Store original number of Gaussians
    N_orig = len(means)

    print(f"Loaded Gaussians:")
    print(f"  - Number of Gaussians: {N_orig}")
    print(f"  - Means shape: {means.shape}")
    print(f"  - Scales shape: {scales.shape}")
    print(f"  - Quats shape: {quats.shape}")
    print(f"  - Opacities shape: {opacities.shape}")
    print(f"  - Colors shape: {colors.shape}")
    print(f"  - SH degree: {sh_degree}")

    # ================================================================================
    # Filter Gaussians for memory efficiency
    # ================================================================================
    print("\nFiltering Gaussians...")

    # Get confidence if available
    mask = torch.ones(N_orig, dtype=torch.bool, device=device)

    # Filter by confidence threshold
    if hasattr(predictions, 'conf') and args.conf_threshold > 0:
        conf = predictions.conf.to(device)  # [N, H, W]
        conf_flat = conf.reshape(-1)
        if len(conf_flat) == N_orig:
            conf_mask = conf_flat >= args.conf_threshold
            mask = mask & conf_mask
            print(f"  - After confidence filter ({args.conf_threshold}): {mask.sum().item()} Gaussians")

    # Filter by opacity threshold
    if args.opacity_threshold > 0:
        opacity_mask = opacities >= args.opacity_threshold
        mask = mask & opacity_mask
        print(f"  - After opacity filter ({args.opacity_threshold}): {mask.sum().item()} Gaussians")

    # Apply mask
    means = means[mask]
    scales = scales[mask]
    quats = quats[mask]
    opacities = opacities[mask]
    colors = colors[mask]

    # Limit maximum number of Gaussians
    N = len(means)
    if args.max_gaussians is not None and N > args.max_gaussians:
        # Keep top Gaussians by opacity
        _, top_indices = torch.topk(opacities, args.max_gaussians)
        means = means[top_indices]
        scales = scales[top_indices]
        quats = quats[top_indices]
        opacities = opacities[top_indices]
        colors = colors[top_indices]
        N = args.max_gaussians
        print(f"  - After max Gaussians limit ({args.max_gaussians}): {N} Gaussians")

    print(f"\nFinal Gaussian count: {N} / {N_orig} ({100*N/N_orig:.1f}%)")

    # Clean up predictions to free memory
    del predictions
    torch.cuda.empty_cache()

    # Print memory usage
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"Memory after cleanup: {allocated:.2f} GB")

    # ================================================================================
    # Step 3: Setup interactive viewer
    # ================================================================================
    print("\n" + "=" * 80)
    print("Step 3: Starting interactive viewer")
    print("=" * 80)

    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        """Render function called by the viewer for each frame"""
        assert isinstance(render_tab_state, GsplatRenderTabState)

        # Get render resolution
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height

        # Get camera parameters
        c2w = camera_state.c2w  # camera-to-world
        K = camera_state.get_K((width, height))  # intrinsics
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()  # world-to-camera

        # Map render mode
        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        # Render using gsplat
        render_colors, render_alphas, info = rasterization(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            colors,  # [N, K, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            sh_degree=(
                min(render_tab_state.max_sh_degree, sh_degree)
                if sh_degree is not None
                else None
            ),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor(render_tab_state.backgrounds, device=device) / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            packed=True,  # Use packed mode for memory efficiency
        )

        # Update stats
        render_tab_state.total_gs_count = N
        render_tab_state.rendered_gs_count = (info["radii"] > 0).sum().item()

        # Process rendered output based on mode
        if render_tab_state.render_mode == "rgb":
            # Colors represented with SH are not guaranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # Normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )

        return renders

    # Create viser server and viewer
    server = viser.ViserServer(port=args.port, verbose=False)
    _ = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
    )

    print(f"Viewer running at http://localhost:{args.port}")
    print("Ctrl+C to exit.")

    # Keep server running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
