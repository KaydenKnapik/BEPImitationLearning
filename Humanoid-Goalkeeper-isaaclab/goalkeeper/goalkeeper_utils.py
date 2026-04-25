"""Utility functions and MotionLib for the Goalkeeper environment.

Ported from:
  Humanoid-Goalkeeper/legged_gym/legged_gym/envs/g1/g1_utils.py
  Humanoid-Goalkeeper/legged_gym/legged_gym/utils/math.py

Key change: quaternion convention is wxyz here (Isaac Lab), not xyzw (Isaac Gym).
"""
from __future__ import annotations

import math
import os
import random

import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Math utilities (wxyz convention — Isaac Lab)
# ---------------------------------------------------------------------------

def euler_from_quaternion_wxyz(quat_wxyz: torch.Tensor):
    """Convert wxyz quaternion to (roll, pitch, yaw)."""
    w = quat_wxyz[:, 0]
    x = quat_wxyz[:, 1]
    y = quat_wxyz[:, 2]
    z = quat_wxyz[:, 3]

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1.0, 1.0)
    pitch_y = torch.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z


def torch_rand_float(lower: float, upper: float, shape: tuple, device: str) -> torch.Tensor:
    return (upper - lower) * torch.rand(*shape, device=device) + lower


# ---------------------------------------------------------------------------
# Dataset loading (unchanged from original)
# ---------------------------------------------------------------------------

def load_imitation_dataset(folder: str, mapping: str, suffix: str = ".pt"):
    """Load all .pt motion files from folder and return (multidataset, joint_id_dict)."""
    filenames = [name for name in os.listdir(folder) if name.endswith(suffix)]

    multidataset = {}
    for filename in tqdm(filenames, desc="Loading motion dataset"):
        try:
            data = torch.load(os.path.join(folder, filename), weights_only=False)
            dataset_list = [data]
            random.shuffle(dataset_list)
            multidataset[filename[: -len(suffix)]] = dataset_list
        except Exception as e:
            print(f"[MotionLib] {filename} load failed: {e}")
            continue

    with open(mapping, "r") as f:
        lines = f.readlines()
    lines = [line.strip().split(" ") for line in lines]
    joint_id_dict = {k: int(v) for v, k in lines}

    return multidataset, joint_id_dict


# ---------------------------------------------------------------------------
# MotionLib (unchanged from original — pure PyTorch)
# ---------------------------------------------------------------------------

class MotionLib:
    def __init__(
        self,
        datasets,
        mapping,
        dof_names,
        keyframe_names,
        fps=30,
        min_dt=0.1,
        device="cpu",
        amp_obs_type="dof",
        num_steps=2,
    ):
        self.fps = fps
        self.device = device
        self.amp_obs_type = amp_obs_type
        self.num_steps = num_steps
        self.min_frames = int(min_dt * fps)

        self.dof_names = dof_names
        self.keyframe_names = keyframe_names
        self.mapping = mapping

        # Build list of valid (dataset_key, motion_tensor) pairs
        self.motion_list = []
        for key, dataset_list in datasets.items():
            for motion in dataset_list:
                if isinstance(motion, dict):
                    for sub_key, sub_motion in motion.items():
                        frames = self._load_motion(sub_motion)
                        if frames is not None and frames.shape[0] >= self.min_frames:
                            self.motion_list.append(frames)
                else:
                    frames = self._load_motion(motion)
                    if frames is not None and frames.shape[0] >= self.min_frames:
                        self.motion_list.append(frames)

        if len(self.motion_list) == 0:
            raise ValueError("No valid motions found in dataset!")

        print(f"[MotionLib] Loaded {len(self.motion_list)} motion clips.")

    def _load_motion(self, data):
        """Convert raw data dict/tensor to per-frame dof tensor."""
        try:
            if isinstance(data, dict):
                # Try to extract DOF data
                if "dof_pos" in data:
                    return data["dof_pos"].float()
                elif "joint_pos" in data:
                    return data["joint_pos"].float()
                # Try first value
                val = next(iter(data.values()))
                if isinstance(val, torch.Tensor):
                    return val.float()
            elif isinstance(data, torch.Tensor):
                return data.float()
        except Exception:
            pass
        return None

    def sample_motion(self, n_envs: int):
        """Sample a random starting frame from a random motion clip."""
        indices = torch.randint(0, len(self.motion_list), (n_envs,))
        frames = []
        for idx in indices:
            motion = self.motion_list[idx.item()]
            start = random.randint(0, max(0, motion.shape[0] - self.num_steps - 1))
            frames.append(motion[start: start + self.num_steps])
        # Stack: (n_envs, num_steps, num_dof)
        try:
            return torch.stack(frames, dim=0).to(self.device)
        except Exception:
            return None

    def get_amp_obs(self, n_envs: int):
        """Return AMP observation tensor (n_envs, num_steps * num_dof)."""
        frames = self.sample_motion(n_envs)
        if frames is None:
            return torch.zeros(n_envs, self.num_steps * len(self.dof_names), device=self.device)
        # Flatten last two dims
        return frames.view(n_envs, -1)
