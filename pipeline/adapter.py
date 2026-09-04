"""MistaPoseAdapter: (72,) SMPL pose -> MISTA deformer camera."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dataset.zjumocap import ZJUMoCapDataset               # for _recompute_bone_transforms


def pose_to_camera_fields(smpl_thetas_72, Jtr_target, b02v_inv, device,
                          trans=None, trans_xform=None):
    """
    Convert a single ROMP SMPL pose (72-d axis-angle) into the tensors MISTA's
    deformer consumes: `rots` (1,24,9) and `bone_transforms` (24,4,4). Mirrors
    dataset/zjumocap.py getitem() exactly.

    When `trans` (a (3,) world translation) is provided it is mapped into the
    canonical frame via `trans_xform` and added to the bone-transform
    translation column, mirroring dataset/zjumocap.py:465
    (`bone_transforms[:, :3, 3] += trans`). With `trans=None` the root stays
    pinned (trans = 0), the original fixed-camera behaviour.
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
    if trans is not None:
        t = trans_xform(trans) if trans_xform is not None else np.asarray(trans, dtype=np.float32)
        bt[:, :3, 3] += t.astype(np.float32)                                     # mirrors zjumocap.py:465
    bone_transforms = torch.from_numpy(bt).to(device)                            # (24,4,4)

    return rots, bone_transforms


def smpl_posed_joints(smpl_thetas_72, Jtr_target):
    """SMPL forward kinematics -> the 24 posed joint positions (24,3) in the SAME
    frame as MISTA's deformed Gaussians.

    Mirrors `ZJUMoCapDataset._recompute_bone_transforms` (dataset/zjumocap.py:178)
    but returns the global joint translations `G_posed[j][:3,3]` instead of the
    skinning bone transforms. Used to solve a per-frame image-aligned camera
    (PnP) against the estimator's 2D joints. Joint order is the standard SMPL-24
    kinematic order (same as ROMP's first 24 `pj2d_org` joints).
    """
    thetas = np.asarray(smpl_thetas_72, dtype=np.float32).reshape(-1)
    root_orient = thetas[0:3]
    pose_body = thetas[3:66]
    pose_hand = thetas[66:72]

    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
               12, 13, 14, 16, 17, 18, 19, 20, 21]
    pose_full = np.concatenate([root_orient, pose_body, pose_hand], axis=-1)
    pose_rots = Rotation.from_rotvec(pose_full.reshape([-1, 3])).as_matrix().astype(np.float64)

    Jtr = np.asarray(Jtr_target, dtype=np.float64)
    n_joints = len(parents)
    G_posed = np.zeros((n_joints, 4, 4), dtype=np.float64)
    for j in range(n_joints):
        p = parents[j]
        t_local = Jtr[j] if p < 0 else Jtr[j] - Jtr[p]
        T_posed = np.eye(4)
        T_posed[:3, :3] = pose_rots[j]
        T_posed[:3, 3] = t_local
        G_posed[j] = T_posed if p < 0 else G_posed[p] @ T_posed
    return G_posed[:, :3, 3].astype(np.float32)                                  # (24,3)


class MistaPoseAdapter:
    """Turns a (72,) SMPL pose into the deformer camera MISTA consumes.

    Wraps `pose_to_camera_fields` and the per-identity template camera. Also
    hands out the canonical (rest-pose) camera for the "no pose yet" case.
    """

    def __init__(self, template_cam, Jtr_target, b02v_inv, device, trans_xform=None,
                 retargeter=None):
        self.template_cam = template_cam
        self.Jtr_target = Jtr_target
        self.b02v_inv = b02v_inv
        self.device = device
        # Maps a raw estimator (3,) translation into the canonical frame. None
        # (default) pins the root, preserving the original in-place behaviour.
        self.trans_xform = trans_xform
        # Optional deterministic IK retargeter (--retarget). None => passthrough,
        # so with retargeting off the pose path is byte-for-byte the old one.
        self.retargeter = retargeter

    def retarget(self, pose, trans=None):
        """Filtered (72,) pose (+trans) -> (corrected pose, trans, correction).

        No-op passthrough when no retargeter is attached (returns the pose and
        trans unchanged and a None correction). Otherwise runs the IK retarget
        and returns the corrected pose plus a root-correction offset that
        `to_camera` applies regardless of the --root-motion gate.
        """
        if self.retargeter is None:
            return pose, trans, None
        new_pose, correction = self.retargeter.process(pose, trans, self.Jtr_target)
        return new_pose, trans, correction

    def to_camera(self, pose, identity, trans=None, extra_trans=None):
        # Only drive root motion when a mapping is configured (--root-motion).
        trans_in = trans if self.trans_xform is not None else None
        rots, bone_transforms = pose_to_camera_fields(
            pose, self.Jtr_target, self.b02v_inv, self.device,
            trans=trans_in, trans_xform=self.trans_xform,
        )
        # The retarget correction (e.g. ground-penetration lift) is always
        # applied, even with --root-motion off, mirroring zjumocap.py:465.
        if extra_trans is not None:
            et = torch.as_tensor(np.asarray(extra_trans, dtype=np.float32),
                                 device=bone_transforms.device)
            bone_transforms[:, :3, 3] += et
        cam = self.template_cam.copy()
        cam.update(rots=rots, bone_transforms=bone_transforms)
        cam.person_id = identity
        return cam

    def posed_joints(self, pose):
        """SMPL-24 posed joint positions (24,3) for `pose`, in the Gaussian frame."""
        return smpl_posed_joints(pose, self.Jtr_target)

    def canonical(self):
        """Rest-pose camera (template as-is), for the first-frame fallback."""
        return self.template_cam
