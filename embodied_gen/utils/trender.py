# Project EmbodiedGen
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import os
import sys

import numpy as np
import spaces
import torch
from tqdm import tqdm

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_dir, "../.."))
from thirdparty.TRELLIS.trellis.renderers.mesh_renderer import MeshRenderer
from thirdparty.TRELLIS.trellis.representations import MeshExtractResult
from thirdparty.TRELLIS.trellis.utils.render_utils import (
    render_frames,
    yaw_pitch_r_fov_to_extrinsics_intrinsics,
)

__all__ = [
    "render_video",
]


@spaces.GPU
def render_mesh(sample, extrinsics, intrinsics, options={}, **kwargs):
    renderer = MeshRenderer()
    renderer.rendering_options.resolution = options.get("resolution", 512)
    renderer.rendering_options.near = options.get("near", 1)
    renderer.rendering_options.far = options.get("far", 100)
    renderer.rendering_options.ssaa = options.get("ssaa", 4)
    rets = {}
    for extr, intr in tqdm(zip(extrinsics, intrinsics), desc="Rendering"):
        res = renderer.render(sample, extr, intr)
        if "normal" not in rets:
            rets["normal"] = []
        normal = torch.lerp(
            torch.zeros_like(res["normal"]), res["normal"], res["mask"]
        )
        normal = np.clip(
            normal.detach().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255
        ).astype(np.uint8)
        rets["normal"].append(normal)

    return rets


@spaces.GPU
def render_video(
    sample,
    resolution=512,
    bg_color=(0, 0, 0),
    num_frames=300,
    r=2,
    fov=40,
    **kwargs,
):
    yaws = torch.linspace(0, 2 * 3.1415, num_frames)
    yaws = yaws.tolist()
    pitch = [0.5] * num_frames
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws, pitch, r, fov
    )
    render_fn = (
        render_mesh if isinstance(sample, MeshExtractResult) else render_frames
    )
    result = render_fn(
        sample,
        extrinsics,
        intrinsics,
        {"resolution": resolution, "bg_color": bg_color},
        **kwargs,
    )

    return result
