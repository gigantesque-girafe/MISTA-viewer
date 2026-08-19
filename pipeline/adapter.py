"""MistaPoseAdapter: (72,) SMPL pose -> MISTA deformer camera."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dataset.zjumocap import ZJUMoCapDataset               # for _recompute_bone_transforms


def pose_to_camera_fields(smpl_thetas_72, Jtr_target, b02v_inv, device):
    """
    Convert a single ROMP SMPL pose (72-d axis-angle) into the tensors MISTA's
    deformer consumes: `rots` (1,24,9) and `bone_transforms` (24,4,4). Mirrors
    dataset/zjumocap.py getitem() exactly. Fixed camera => trans = 0.
    """
    thetas = np.asarray(smpl_thetas_72, dtype=np.float32).reshape(-1)
    root_orient = thetas[0:3]
    pose_body = thetas[3:66]     # 63 = 21 joints
    pose_hand = thetas[66:72]    # 6  = 2 joints (zero-padded by ROMP)

    pose_full = np.concatenate([root_orient, pose_body, pose_hand], axis=-1)
    pose_mat_full = Rotation.from_rotvec(pose_full.reshape([-1, 3])).as_matrix()  # (24,3,3)
    pose_mat = pose_mat_full[1:, ...].copy()                                      # (23,3,3)
    pose_rot = np.concatenate(
        [np.expand_dims(np.eye(3), axis=0), pose_mat], axis=0
    ).reshape([-1, 9])                                                            # (24,9)
    rots = torch.from_numpy(pose_rot).float().unsqueeze(0).to(device)            # (1,24,9)

    bt = ZJUMoCapDataset._recompute_bone_transforms(
        root_orient, pose_body, pose_hand, Jtr_target
    )                                                                            # (24,4,4)
    bt = (bt @ b02v_inv).astype(np.float32)
    bone_transforms = torch.from_numpy(bt).to(device)                            # (24,4,4)

    return rots, bone_transforms


class MistaPoseAdapter:
    """Turns a (72,) SMPL pose into the deformer camera MISTA consumes.

    Wraps `pose_to_camera_fields` and the per-identity template camera. Also
    hands out the canonical (rest-pose) camera for the "no pose yet" case.
    """

    def __init__(self, template_cam, Jtr_target, b02v_inv, device):
        self.template_cam = template_cam
        self.Jtr_target = Jtr_target
        self.b02v_inv = b02v_inv
        self.device = device

    def to_camera(self, pose, identity):
        rots, bone_transforms = pose_to_camera_fields(
            pose, self.Jtr_target, self.b02v_inv, self.device
        )
        cam = self.template_cam.copy()
        cam.update(rots=rots, bone_transforms=bone_transforms)
        cam.person_id = identity
        return cam

    def canonical(self):
        """Rest-pose camera (template as-is), for the first-frame fallback."""
        return self.template_cam
