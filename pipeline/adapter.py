"""MistaPoseAdapter: (72,) SMPL pose -> MISTA deformer camera."""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dataset.zjumocap import ZJUMoCapDataset               # for _recompute_bone_transforms

# SMPL-24 kinematic chain (parent index per joint, -1 = root). Shared by the forward
# kinematics below and by pipeline/overlay.py's 2D skeleton bone-line drawing.
SMPL_24_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
                    12, 13, 14, 16, 17, 18, 19, 20, 21]


def pose_to_camera_fields(smpl_thetas_72, Jtr_target, b02v_inv, device,
                          trans=None, trans_xform=None):
    """Convert a single SMPL pose into the tensors MISTA's deformer consumes.

    @param smpl_thetas_72: array-like of 72 axis-angle values (24 joints x 3).
    @param Jtr_target: (24, 3) target joint positions for the identity being posed.
    @param b02v_inv: (24, 4, 4) inverse bind-to-volume bone transform, applied
        on the right of the recomputed bone transforms.
    @param device: torch device for the returned tensors.
    @param trans: optional (3,) world translation; if given, mapped via
        `trans_xform` (or used as-is if `trans_xform` is None) and added to the
        bone-transform translation column. If None, the root stays pinned
        (translation = 0).
    @param trans_xform: optional callable `(3,) -> (3,)` mapping `trans` into
        the canonical frame; ignored if `trans` is None.
    @return: tuple `(rots, bone_transforms)` — `rots` is a `(1, 24, 9)` float
        tensor, `bone_transforms` is a `(24, 4, 4)` float tensor, both on `device`.
    @note: Mirrors `dataset/zjumocap.py`'s `__getitem__` exactly, including the
        `bone_transforms[:, :3, 3] += trans` step at zjumocap.py:465.
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
    """Run SMPL forward kinematics to the 24 posed joint positions.

    @param smpl_thetas_72: array-like of 72 axis-angle values (24 joints x 3).
    @param Jtr_target: (24, 3) target (rest) joint positions for the identity.
    @return: np.ndarray of shape (24, 3), float32 — posed joint positions in
        the same frame as MISTA's deformed Gaussians, in standard SMPL-24
        kinematic order (matching ROMP's first 24 `pj2d_org` joints).
    @note: Mirrors `ZJUMoCapDataset._recompute_bone_transforms`
        (dataset/zjumocap.py:178) but returns the global joint translations
        `G_posed[j][:3, 3]` instead of the skinning bone transforms.
    """
    thetas = np.asarray(smpl_thetas_72, dtype=np.float32).reshape(-1)
    root_orient = thetas[0:3]
    pose_body = thetas[3:66]
    pose_hand = thetas[66:72]

    parents = SMPL_24_PARENTS
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
    """Wraps `pose_to_camera_fields` and the per-identity template camera to
    convert a (72,) SMPL pose into the deformer camera MISTA consumes."""

    def __init__(self, template_cam, Jtr_target, b02v_inv, device, trans_xform=None,
                 retargeter=None):
        """@brief Store the per-identity template camera and pose-conversion inputs.

        @param template_cam: camera object providing `.copy()` and `.update()`,
            reused as the base for every posed/canonical camera.
        @param Jtr_target: (24, 3) target joint positions for this identity.
        @param b02v_inv: (24, 4, 4) inverse bind-to-volume bone transform.
        @param device: torch device for pose tensors.
        @param trans_xform: optional callable `(3,) -> (3,)` mapping a raw
            translation into the canonical frame; None pins the root (original
            in-place behavior).
        @param retargeter: optional IK retarget solver (`--retarget`); None
            makes `retarget()` a passthrough.
        """
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
        """Apply the attached IK retarget solver to a filtered pose, if any.

        @param pose: (72,) SMPL axis-angle pose.
        @param trans: optional (3,) world translation, forwarded unchanged.
        @return: tuple `(pose, trans, correction)`. If no retargeter is
            attached, `pose`/`trans` are returned unchanged and `correction` is
            None. Otherwise `pose` is IK-corrected and `correction` is the
            root-correction offset that `to_camera` applies regardless of
            `--root-motion`.
        """
        if self.retargeter is None:
            return pose, trans, None
        new_pose, correction = self.retargeter.process(pose, trans, self.Jtr_target)
        return new_pose, trans, correction

    def to_camera(self, pose, identity, trans=None, extra_trans=None):
        """Build a posed camera for `pose` on top of the template camera.

        @param pose: (72,) SMPL axis-angle pose.
        @param identity: identity index assigned to `cam.person_id`.
        @param trans: optional (3,) world translation; only applied if
            `self.trans_xform` is set (i.e. `--root-motion`), otherwise ignored.
        @param extra_trans: optional (3,) retarget root-correction offset,
            always added to the bone-transform translation column regardless
            of `--root-motion`.
        @return: a copy of `self.template_cam` with `rots`, `bone_transforms`,
            and `person_id` set for this pose.
        """
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
        """@brief Return the (24, 3) SMPL-24 posed joint positions for `pose`, in the Gaussian frame."""
        return smpl_posed_joints(pose, self.Jtr_target)

    def canonical(self):
        """@brief Return the rest-pose (template) camera, for the first-frame fallback."""
        return self.template_cam
