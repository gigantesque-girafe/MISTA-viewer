"""
motion-drive-composite.py  —  ROMP/PARE -> MISTA -> deepfake-style video overlay

Offline sibling of `motion-drive-render-v43.py`. Where v43 streams deformed Gaussian
attributes to the C++/SIBR viewer (free camera, realtime), this script does everything in
Python and writes a single MP4: it drives the MISTA avatar with a source video's motion,
rasterizes it with the SAME offline renderer `render.py`/`test()` uses, and alpha-composites
the rendered avatar over the original frame — like putting a new body onto the original clip.

Alignment (--align pnp, default): for each frame it solves an image-aligned camera by PnP
between the avatar's posed 3D joints (SMPL forward kinematics) and the estimator's 2D joints
(ROMP pj2d_org / PARE smpl_joints2d, remapped to SMPL order), using ROMP's virtual-camera
focal. Rendering the Gaussians through that camera gives correct position, scale AND
orientation (the body facing comes from root_orient, not a 2D warp). Fallbacks: 'bbox'
(2D-warp onto the person's bbox) and 'none' (centered letterbox).

Reused verbatim from the live pipeline (pipeline/ + render.py):
  FrameSource     -> next BGR source frame
  build_estimator -> ROMP/PARE backend: frame -> (72,) pose (+ last_pj2d 2D joints)
  PoseProcessor   -> One-Euro smoothing + missing-detection reuse policy
  MistaPoseAdapter-> (72,) pose -> deformer camera + posed_joints() for PnP
  gaussian_renderer.render(..., return_opacity=True) -> RGB (3,H,W) + alpha mask (1,H,W)

Nothing in render.py or the C++/SIBR path changes; estimators.py gains only additive
2D-joint exposure (last_pj2d / pj2d_smpl_map), which v43 ignores.

Run:
  python motion-drive-composite.py --video clip.mp4 --identity 3 \
      --load-ckpt "<ckpt>.pth" --estimator romp --out overlay.mp4
"""

import os
import sys
import time
import math
import argparse

# Some of MISTA's import chain (dataset / geometry helpers) links a second OpenMP
# runtime (libomp) alongside torch's libiomp5md; on Windows that aborts at import
# with "OMP: Error #15" unless duplicates are allowed. The live v43 script only hits
# this on its PARE branch, but the offline render path here pulls it in for ROMP too,
# so set it up-front (before torch/cv2 load) unless the user already chose a value.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch

# Make the MISTA package root importable when run directly (repo-root modules + `pipeline`).
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from render import build_scene                       # checkpoint + scene loader (as v43)
from gaussian_renderer import render                 # offline rasterizer: RGB + opacity mask
from utils.graphics_utils import getWorld2View2, getProjectionMatrix  # 3dgs camera matrices
from pipeline import (
    load_mista_config,   # composes the MISTA Hydra config (test split, chosen identity)
    build_estimator,     # factory -> ROMP/PARE PoseEstimator: frame -> (72,) pose or None
    FrameSource,         # next BGR frame from the source video
    PoseProcessor,       # One-Euro smoothing + missing-detection reuse/reset policy
    MistaPoseAdapter,    # (72,) SMPL pose -> MISTA deformer camera (rots + bone transforms)
)
from pipeline.retarget import (add_retarget_args, build_retargeter,  # optional IK retarget (--retarget)
                               source_jtr_from_betas)


# --------------------------------------------------------------------------- #
# CLI  (a focused subset of v43's flags: driving + smoothing + estimator, plus
# the new video-output options. No webcam / port / VR / root-motion here.)
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="ROMP/PARE -> MISTA -> avatar-over-video MP4 (offline deepfake-style overlay)")
    # Source is always a video file for offline compositing; FrameSource keys off args.source.
    p.add_argument("--video", type=str, required=True, help="Source clip to drive + overlay onto.")
    p.add_argument("--identity", type=int, default=0, help="Target MISTA identity index (0-7).")
    p.add_argument("--load-ckpt", type=str, required=True,
                   help="Path to the MISTA 8-identity checkpoint (.pth).")
    p.add_argument("--estimator", choices=["romp", "pare"], default="romp",
                   help="Pose estimation backend ('romp' default, or vendored 'pare').")

    # Output.
    p.add_argument("--out", type=str, default=None,
                   help="Output MP4 path (default: '<video>_overlay.mp4' beside the input).")
    p.add_argument("--fps", type=float, default=None,
                   help="Output FPS (default: source video FPS, falling back to 25).")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after this many frames (0 = whole clip). Handy for quick tests.")
    p.add_argument("--bg", choices=["black", "white"], default="black",
                   help="Rasterizer clear color. Black (default) gives the cleanest alpha; "
                        "the composite uses the separate opacity mask, so it is never shown.")
    p.add_argument("--alpha-gamma", type=float, default=1.0,
                   help="Exponent applied to the alpha mask before compositing (>1 tightens "
                        "edges, <1 feathers). Default 1.0 = use the mask as-is.")

    # Alignment: place the rendered avatar onto the source person instead of centering it.
    p.add_argument("--align", choices=["none", "bbox", "pnp"], default="pnp",
                   help="'pnp' (default, ROMP only) renders the avatar through a per-frame "
                        "image-aligned camera solved from 3D<->2D joint correspondences — "
                        "correct position, scale AND orientation. 'bbox' 2D-warps the avatar "
                        "onto the person's bbox (position+scale only). 'none' centers it.")
    p.add_argument("--pnp-ransac", dest="pnp_ransac", action="store_true",
                   help="Use RANSAC PnP (default; robust to bad joints).")
    p.add_argument("--no-pnp-ransac", dest="pnp_ransac", action="store_false",
                   help="Use plain iterative PnP instead of RANSAC.")
    p.set_defaults(pnp_ransac=True)
    p.add_argument("--smooth-camera", dest="smooth_camera", action="store_true",
                   help="One-Euro smooth the solved PnP camera (rotation in quaternion "
                        "space + translation) across frames. Default ON; removes the "
                        "per-frame facing wobble from re-solving PnP on noisy 2D joints.")
    p.add_argument("--no-smooth-camera", dest="smooth_camera", action="store_false",
                   help="Disable PnP camera smoothing (raw per-frame solve).")
    p.set_defaults(smooth_camera=True)
    p.add_argument("--cam-min-cutoff", type=float, default=1.0,
                   help="One-Euro min cutoff for the PnP camera. Lower = steadier but "
                        "laggier (default 1.0).")
    p.add_argument("--cam-beta", type=float, default=0.3,
                   help="One-Euro speed coefficient for the PnP camera. Higher = more "
                        "responsive to fast motion (default 0.3).")
    p.add_argument("--align-scale", type=float, default=1.0,
                   help="Multiplies the fitted avatar scale (>1 bigger, <1 smaller).")
    p.add_argument("--align-pad", type=float, default=0.0,
                   help="Expand the person's target bbox by this fraction on every side.")
    p.add_argument("--flip-x", action="store_true",
                   help="Mirror the avatar horizontally (fixes a common left/right facing "
                        "mismatch between the render camera and the source camera).")
    p.add_argument("--debug-joints", action="store_true",
                   help="Draw the estimator's 2D joints (green) on the output — use to "
                        "verify they land on the person and match the avatar's joint order "
                        "before trusting PnP (esp. for the PARE backend).")

    # PARE backend options (only used when --estimator pare); defaults match v43.
    p.add_argument("--pare-ckpt", type=str,
                   default="submodules/PARE/data/pare/checkpoints/pare_w_3dpw_checkpoint.ckpt")
    p.add_argument("--pare-cfg", type=str,
                   default="submodules/PARE/data/pare/checkpoints/pare_w_3dpw_config.yaml")
    p.add_argument("--pare-crop-size", type=int, default=224)
    p.add_argument("--pare-scale", type=float, default=1.0)

    # ROMP inference backend (ONNX default; TensorRT A/B). Identical semantics to v43.
    p.add_argument("--onnx", dest="onnx", action="store_true",
                   help="Use ROMP's ONNX GPU backend (default; big speedup).")
    p.add_argument("--no-onnx", dest="onnx", action="store_false",
                   help="Force ROMP's plain PyTorch backbone instead of ONNX.")
    p.set_defaults(onnx=True)
    p.add_argument("--trt", dest="trt", action="store_true",
                   help="Run ROMP's ONNX model via the TensorRT EP (A/B vs CUDA).")
    p.add_argument("--no-trt", dest="trt", action="store_false")
    p.set_defaults(trt=False)
    p.add_argument("--trt-fp16", dest="trt_fp16", action="store_true")
    p.add_argument("--no-trt-fp16", dest="trt_fp16", action="store_false")
    p.set_defaults(trt_fp16=True)
    p.add_argument("--trt-cache-dir", type=str,
                   default=os.path.join(os.path.expanduser("~"), ".romp", "trt_cache"))
    p.add_argument("--trt-lib-dir", type=str, default=os.environ.get("TENSORRT_LIB_DIR"))

    # Run the estimator (and re-deform) every Nth frame, reusing the last pose otherwise.
    p.add_argument("--romp-every-n", type=int, default=1,
                   help="Estimate + re-pose every N frames (1 = every frame). Output stays "
                        "full-length; skipped frames reuse the last avatar pose.")

    # One-Euro temporal smoothing (enabled by default; same flags/defaults as v43).
    p.add_argument("--smooth", dest="smooth", action="store_true",
                   help="Enable One-Euro smoothing of the estimated pose (default).")
    p.add_argument("--no-smooth", dest="smooth", action="store_false")
    p.set_defaults(smooth=True)
    p.add_argument("--min-cutoff", type=float, default=0.8)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--derivative-cutoff", type=float, default=1.0)
    p.add_argument("--smooth-frequency", type=float, default=12.5)
    p.add_argument("--filter-space", choices=["axis-angle", "quat"], default="quat")
    p.add_argument("--reuse-frames", type=int, default=12)
    p.add_argument("--reset-after", type=int, default=30)
    p.add_argument("--debug", action="store_true")

    # Root motion: thread ROMP's cam_trans into the bone transforms so a walk/step
    # translates the avatar instead of rendering in place. Off by default; the
    # cam_trans -> canonical-frame mapping is tunable via the flags below (same as v43).
    p.add_argument("--root-motion", action="store_true",
                   help="Drive avatar translation from the estimator's cam_trans "
                        "(ROMP only). Default off = pinned root (in-place).")
    p.add_argument("--root-scale", type=float, default=1.0,
                   help="Scalar applied to cam_trans before adding to the root.")
    p.add_argument("--root-axis", type=str, default="x,y,z",
                   help="Axis permute+sign mapping cam_trans -> canonical frame, "
                        "e.g. 'x,y,z' (identity) or 'x,-z,y'.")
    p.add_argument("--root-horizontal", action="store_true",
                   help="Zero the depth (Z) component after mapping: keep only "
                        "ground-plane motion, drop noisy toward/away motion.")

    # Optional deterministic IK retarget (same flags as v43). For the no-floor
    # overlay use --retarget --retarget-mode principled --no-ground.
    add_retarget_args(p)

    args = p.parse_args()

    # FrameSource keys off these; compositing is video-only (no webcam path here).
    args.source = "video"
    args.camera_index = 0
    return args


def build_trans_xform(args):
    """Closure mapping a raw (3,) cam_trans into the canonical frame, or None.

    Returns None unless --root-motion is set (so the adapter pins the root).
    Applies, in order: axis permute+sign (--root-axis), scale (--root-scale),
    and optional depth zeroing (--root-horizontal). Ported from v43.
    """
    if not args.root_motion:
        return None

    idx, sign = [], []
    for tok in args.root_axis.split(","):
        tok = tok.strip().lower()
        s = -1.0 if tok.startswith("-") else 1.0
        axis = tok.lstrip("+-")
        if axis not in ("x", "y", "z"):
            raise SystemExit(f"--root-axis: bad token '{tok}' (use x/y/z with optional -)")
        idx.append({"x": 0, "y": 1, "z": 2}[axis])
        sign.append(s)
    if len(idx) != 3:
        raise SystemExit("--root-axis must have exactly 3 comma-separated axes")

    idx = np.asarray(idx, dtype=np.int64)
    sign = np.asarray(sign, dtype=np.float32)
    scale = float(args.root_scale)
    horizontal = args.root_horizontal

    def xform(trans):
        t = np.asarray(trans, dtype=np.float32).reshape(-1)
        out = (t[idx] * sign) * scale
        if horizontal:
            out[2] = 0.0
        return out.astype(np.float32)

    return xform


def _probe_video(path):
    """Return (fps, frame_count) for a video file (best-effort; 0/None on failure)."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    cap.release()
    return fps, n


def _tensor_to_bgr_u8(rgb_chw):
    """(3,H,W) float RGB in [0,1] -> (H,W,3) uint8 BGR for OpenCV."""
    img = rgb_chw.clamp(0.0, 1.0).mul(255.0).byte().permute(1, 2, 0).cpu().numpy()  # RGB HxWx3
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# Image-aligned camera helpers (intrinsics, PnP, rotation smoothing) now live in
# pipeline/camera.py so the live v43 2D-reprojection retarget solves the SAME
# camera the same way. Imported under the original private names to keep the call
# sites below unchanged.
from pipeline.camera import (romp_intrinsics as _romp_intrinsics,   # noqa: E402
                             solve_pnp as _solve_pnp,
                             mat_to_quat as _mat_to_quat,
                             quat_to_mat as _quat_to_mat,
                             CameraSmoother as _CameraSmoother)


def _build_pnp_cam_fields(R, t, W, H, f, znear, zfar, device):
    """Camera tensors (world_view / full_proj / center / FoV / size) for a PnP camera.

    Built exactly like scene/cameras.py from a world->camera (R,t): 3dgs's
    getWorld2View2 takes the camera-to-world rotation, i.e. R.T, and the translation t.
    """
    W2C = getWorld2View2(R.T.astype(np.float32), t.astype(np.float32))          # (4,4) world->cam
    world_view = torch.tensor(W2C).transpose(0, 1).to(device)
    FoVx = 2.0 * math.atan(W / (2.0 * f))
    FoVy = 2.0 * math.atan(H / (2.0 * f))
    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx, fovY=FoVy).transpose(0, 1).to(device)
    full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
    center = world_view.inverse()[3, :3]
    return dict(image_width=int(W), image_height=int(H),
                FoVx=float(FoVx), FoVy=float(FoVy),
                world_view_transform=world_view, projection_matrix=proj,
                full_proj_transform=full_proj, camera_center=center)


def _alpha_bbox(alpha, thr=0.5):
    """Tight bbox (x1,y1,x2,y2) of alpha>thr, or None if the mask is empty."""
    ys, xs = np.where(alpha > thr)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0


def _pj2d_bbox(pj2d, pad, W, H):
    """Person bbox (x1,y1,x2,y2) from 2D joints, padded and clamped to the frame."""
    x1, y1 = pj2d[:, 0].min(), pj2d[:, 1].min()
    x2, y2 = pj2d[:, 0].max(), pj2d[:, 1].max()
    bw, bh = x2 - x1, y2 - y1
    x1 -= pad * bw; x2 += pad * bw
    y1 -= pad * bh; y2 += pad * bh
    return (max(0.0, float(x1)), max(0.0, float(y1)),
            min(float(W), float(x2)), min(float(H), float(y2)))


def _fit_affine(avatar_bbox, target_bbox, extra_scale, flip_x):
    """2x3 affine mapping the avatar sprite onto the target bbox.

    Uniform (height-based) scale so the avatar is never distorted, translating the
    avatar-bbox center onto the target-bbox center. With flip_x the x axis is mirrored
    about the avatar-bbox center before translation.
    """
    ax1, ay1, ax2, ay2 = avatar_bbox
    tx1, ty1, tx2, ty2 = target_bbox
    acx, acy = (ax1 + ax2) * 0.5, (ay1 + ay2) * 0.5
    tcx, tcy = (tx1 + tx2) * 0.5, (ty1 + ty2) * 0.5
    a_h = max(1.0, ay2 - ay1)
    t_h = max(1.0, ty2 - ty1)
    s = (t_h / a_h) * extra_scale
    sx = -s if flip_x else s
    # out = S*(p - a_center) + t_center  ->  affine [[sx,0, tcx - sx*acx],[0,s, tcy - s*acy]]
    return np.array([[sx, 0.0, tcx - sx * acx],
                     [0.0, s,  tcy - s * acy]], dtype=np.float32)


def _draw_joints(img, pj2d, n=24):
    """Draw the first n 2D joints (green dots, index labels on a few) for debugging."""
    if pj2d is None:
        return img
    for i, (x, y) in enumerate(pj2d[:n]):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        cv2.circle(img, (int(round(x)), int(round(y))), 3, (0, 255, 0), -1)
        if i in (0, 1, 2, 15, 20, 21):     # pelvis, hips, head, hands — orientation cues
            cv2.putText(img, str(i), (int(x) + 4, int(y) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return img


def _place_avatar(avatar_bgr, alpha, M, out_w, out_h):
    """warpAffine the avatar sprite + alpha into an (out_h,out_w) canvas via M."""
    warp_rgb = cv2.warpAffine(avatar_bgr, M, (out_w, out_h),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    warp_a = cv2.warpAffine(alpha, M, (out_w, out_h),
                            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0.0)
    return warp_rgb, warp_a


def main():
    args = parse_args()
    if not (0 <= args.identity <= 7):
        raise SystemExit("--identity must be in 0..7")
    if args.romp_every_n < 1:
        raise SystemExit("--romp-every-n must be >= 1")
    if args.trt and args.estimator != "romp":
        print("[INIT] --trt only applies to the ROMP backend; ignored for PARE.")
    if args.trt and not args.onnx:
        print("[INIT] --trt requires the ONNX model; enabling ONNX (ignoring --no-onnx).")
        args.onnx = True
    if args.root_motion and args.estimator != "romp":
        print("[INIT] --root-motion needs cam_trans (ROMP only); PARE provides none, "
              "so the root stays pinned for this backend.")

    device = "cuda"

    # ---- MISTA scene (checkpoint + identity restriction), same as v43 ----
    print(f"[INIT] Loading MISTA scene for identity {args.identity} ...")
    config = load_mista_config(args.load_ckpt, args.identity)
    with torch.set_grad_enabled(False):
        scene, migs_type, _ = build_scene(config)
    print(f"[INIT] Scene ready (migs_type={migs_type}).")

    # Per-identity geometry + fixed template camera (R/T/FoV baked; only pose changes/frame).
    meta = scene.test_dataset.metadata
    Jtr_target = np.asarray(meta["Jtr"], dtype=np.float64)
    b02v_inv = np.linalg.inv(np.asarray(meta["bone_transforms_02v"], dtype=np.float64))
    template_cam = scene.test_dataset[0]

    # Decode the canonical Gaussians ONCE for this identity (only pose changes per frame).
    with torch.no_grad():
        scene.update_gaussians_from_migs(int(args.identity))
    print(f"[INIT] Canonical avatar ready: {scene.gaussians.get_xyz.shape[0]} Gaussians.")

    # Avatar render size = the template camera's image size (the sprite we warp/composite).
    RW = int(template_cam.image_width)
    RH = int(template_cam.image_height)
    bg_color = [1.0, 1.0, 1.0] if args.bg == "white" else [0.0, 0.0, 0.0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    # ---- Pose estimator (ROMP or PARE) ----
    estimator = build_estimator(args, device)

    # ---- Pipeline stages (FrameSource -> estimator -> PoseProcessor -> adapter) ----
    frame_source = FrameSource(args)
    processor = PoseProcessor(args)
    # trans_xform threads ROMP's cam_trans into the root when --root-motion is set;
    # None (default) pins the root (fixed placement), mirroring v43.
    trans_xform = build_trans_xform(args)
    if trans_xform is not None:
        print(f"[INIT] Root motion ON (scale={args.root_scale}, axis={args.root_axis}, "
              f"horizontal={args.root_horizontal}).")
    # Optional IK retarget. principled proportion mode needs the actor rest
    # skeleton (source_Jtr): from --source-jtr-npz if given, else locked from the
    # first N valid betas below (deferred attach, mirroring v43). Passthrough when
    # --retarget is off, so the current composite behaviour is unchanged.
    source_Jtr = None
    principled = args.retarget and args.retarget_mode == "principled"
    if principled and args.source_jtr_npz:
        source_Jtr = np.asarray(np.load(args.source_jtr_npz)["Jtr"], dtype=np.float64)
        print(f"[INIT] Loaded source_Jtr from {args.source_jtr_npz}.")
    defer_retarget = principled and source_Jtr is None
    retargeter = None if defer_retarget else build_retargeter(args, Jtr_target, source_Jtr)
    if retargeter is not None:
        print(f"[INIT] IK retarget ON (mode={args.retarget_mode}, "
              f"proportion={args.proportion}, "
              f"limb_scale={getattr(args, 'limb_scale', 1.0)}).")
    elif defer_retarget:
        print(f"[INIT] IK retarget (principled) pending: locking actor betas over "
              f"the first {args.source_betas_frames} valid frames ...")
    _betas_buf = []                        # accumulates estimator.last_betas until lock
    adapter = MistaPoseAdapter(template_cam, Jtr_target, b02v_inv, device,
                               trans_xform=trans_xform, retargeter=retargeter)

    # ---- Output video (at SOURCE resolution; the avatar sprite is placed onto it) ----
    src_fps, src_n = _probe_video(args.video)
    fps = args.fps or (src_fps if src_fps and src_fps > 0 else 25.0)
    out_path = args.out or os.path.splitext(args.video)[0] + "_overlay.mp4"
    # Peek the source resolution from the first frame the FrameSource yields.
    first_frame = frame_source.read()
    H_out, W_out = first_frame.shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_out, H_out))
    if not writer.isOpened():
        raise SystemExit(f"Failed to open VideoWriter for {out_path}")

    # PnP needs ROMP's 2D joints; fall back to centered placement otherwise.
    pnp_on = (args.align == "pnp")
    bbox_on = (args.align == "bbox")
    # Both ROMP and PARE now expose last_pj2d, so alignment works for either; only the
    # (unlikely) no-joints case falls back to centered placement, handled per-frame.
    K_full, f_full = _romp_intrinsics(W_out, H_out)
    znear = float(getattr(template_cam, "znear", 0.01))
    zfar = float(getattr(template_cam, "zfar", 100.0))

    cam_smooth_note = (f", cam-smooth min_cutoff={args.cam_min_cutoff} beta={args.cam_beta}"
                       if (pnp_on and args.smooth_camera) else
                       (", cam-smooth OFF" if pnp_on else ""))
    render_note = (f"full-frame {W_out}x{H_out} via PnP camera (f={f_full:.1f}{cam_smooth_note})"
                   if pnp_on else f"{RW}x{RH} template then warped")
    print(f"[RUN] Compositing '{args.video}' ({src_n or '?'} frames @ {src_fps or '?'} fps) "
          f"-> '{out_path}' at {W_out}x{H_out} @ {fps:.3f} fps (avatar rendered {render_note}).")
    print(f"[RUN] identity={args.identity}  estimator={estimator.name}  "
          f"smoothing={'ON' if args.smooth else 'OFF'}  romp_every_n={args.romp_every_n}  "
          f"bg={args.bg}  alpha_gamma={args.alpha_gamma}  align={args.align}  "
          f"pnp_ransac={args.pnp_ransac}.")

    last_cam = None            # reuse on skip / freeze on miss (carries pose + camera fields)
    last_M = None              # last 2D placement affine (bbox/none modes)
    prev_rvec = prev_tvec = None  # last PnP solution, seeds the next solve (temporal continuity)
    cam_smoother = (_CameraSmoother(args.smooth_frequency, args.cam_min_cutoff, args.cam_beta)
                    if (pnp_on and args.smooth_camera) else None)
    pnp_fail = 0
    pending_frame = first_frame  # the frame we already read to probe resolution
    romp_ctr = 0
    n_written = 0
    t_start = time.perf_counter()

    try:
        while True:
            if args.max_frames and n_written >= args.max_frames:
                break
            t0 = time.perf_counter()

            # 1) Next source frame (FrameSource raises KeyboardInterrupt at end of stream).
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
            else:
                try:
                    frame = frame_source.read()
                except KeyboardInterrupt:
                    break

            # 2) Estimate + smooth + build the deformer camera, every Nth frame; reuse otherwise.
            run_estimator = (romp_ctr % args.romp_every_n == 0) or (last_cam is None)
            romp_ctr += 1

            target_bbox = None
            dbg_proj = None
            if run_estimator:
                raw_pose, raw_trans = estimator.estimate(frame)
                pose, trans, status = processor.process((raw_pose, raw_trans), t0)
                pj2d = getattr(estimator, "last_pj2d", None)
                if bbox_on and pj2d is not None:
                    target_bbox = _pj2d_bbox(pj2d, args.align_pad, W_out, H_out)
                if pose is not None:
                    # Lock the actor rest skeleton from the first N valid betas, then
                    # attach the principled solver (no-op unless deferred).
                    if defer_retarget:
                        b = getattr(estimator, "last_betas", None)
                        if b is not None:
                            _betas_buf.append(np.asarray(b, dtype=np.float64).reshape(-1))
                        if len(_betas_buf) >= args.source_betas_frames:
                            source_Jtr = source_jtr_from_betas(
                                np.mean(np.stack(_betas_buf, 0), 0))
                            adapter.retargeter = build_retargeter(args, Jtr_target, source_Jtr)
                            defer_retarget = False
                            print(f"[INIT] IK retarget ON (mode={args.retarget_mode}, "
                                  f"ground={getattr(args, 'ground', True)}); source_Jtr "
                                  f"locked from {args.source_betas_frames} frames.", flush=True)
                    # Deterministic IK retarget (passthrough when no retargeter). Done
                    # BEFORE to_camera / posed_joints so PnP aligns the RETARGETED avatar.
                    pose, trans, correction = adapter.retarget(pose, trans)
                    cam = adapter.to_camera(pose, int(args.identity), trans,
                                            extra_trans=correction)
                    # PnP: solve an image-aligned camera from 3D<->2D joint pairs and bake
                    # its extrinsics/intrinsics onto this frame's (posed) deformation camera.
                    # The estimator's pj2d_smpl_map pairs each 2D row with the matching SMPL
                    # joint (identity for ROMP; OpenPose->SMPL remap for PARE's SPIN joints).
                    if pnp_on and pj2d is not None:
                        joints3d = adapter.posed_joints(pose)                    # (24,3)
                        pairs = getattr(estimator, "pj2d_smpl_map", None) \
                            or [(i, i) for i in range(min(24, len(pj2d)))]
                        pairs = [(r, s) for (r, s) in pairs
                                 if r < len(pj2d) and s < len(joints3d)]
                        obj3d = joints3d[[s for _, s in pairs]]
                        img2d = pj2d[[r for r, _ in pairs]]
                        R, t, rt = (None, None, None)
                        if len(pairs) >= 6:
                            R, t, rt = _solve_pnp(obj3d, img2d, K_full,
                                                  use_ransac=args.pnp_ransac,
                                                  prev_rvec=prev_rvec, prev_tvec=prev_tvec)
                        if R is not None:
                            prev_rvec, prev_tvec = rt   # seed next solve from the RAW solution
                            if cam_smoother is not None:
                                R, t = cam_smoother(R, t)   # temporal smoothing (post-solve)
                            fields = _build_pnp_cam_fields(R, t, W_out, H_out, f_full,
                                                           znear, zfar, device)
                            cam.update(**fields)
                            if args.debug_joints:
                                dbg_rvec, _ = cv2.Rodrigues(R)
                                proj, _ = cv2.projectPoints(obj3d, dbg_rvec, t.reshape(3, 1),
                                                            K_full, np.zeros((4, 1)))
                                dbg_proj = proj.reshape(-1, 2)   # avatar joints in image px
                                if n_written < 3:
                                    err = float(np.linalg.norm(dbg_proj - img2d, axis=1).mean())
                                    print(f"[PNP-DBG] pairs={len(pairs)}  "
                                          f"tvec={t.reshape(3).round(2)}  "
                                          f"reproj_err={err:.1f}px", flush=True)
                        else:
                            pnp_fail += 1
                            status = (status or "OK") + " PNPFAIL"
                    last_cam = cam
                elif last_cam is not None:
                    if adapter.retargeter is not None:
                        adapter.retargeter.reset()       # drop stale foot locks
                    cam = last_cam                       # freeze avatar on lost target
                else:
                    cam = adapter.canonical()            # no pose yet: canonical rest pose
                    last_cam = cam
            else:
                cam = last_cam
                status = f"SKIP {romp_ctr % args.romp_every_n}/{args.romp_every_n}"

            # 3) Rasterize the posed avatar. In PnP mode the camera is already the
            #    full-frame image-aligned one, so this renders straight into source space.
            with torch.no_grad():
                pkg = render(
                    cam, config.opt.iterations, scene, config.pipeline, background,
                    compute_loss=False, return_opacity=True,
                )
            avatar_bgr = _tensor_to_bgr_u8(pkg["render"])
            alpha = pkg["opacity_render"].clamp(0.0, 1.0)[0].cpu().numpy()
            if args.alpha_gamma != 1.0:
                alpha = np.power(alpha, args.alpha_gamma)

            # 4) Composite over the source frame.
            if pnp_on:
                # Avatar was rendered at source resolution through the aligned camera:
                # blend directly, no 2D warp. If this frame had no aligned camera yet
                # (first-frame canonical fallback, or a PnP failure), the render is at the
                # 512 template size — skip the overlay and pass the source through.
                if avatar_bgr.shape[:2] != (H_out, W_out) or _alpha_bbox(alpha) is None:
                    if args.debug_joints:
                        frame = _draw_joints(frame.copy(), getattr(estimator, "last_pj2d", None))
                    writer.write(frame); n_written += 1; continue
                a = alpha[..., None]
                out = a * avatar_bgr.astype(np.float32) + (1.0 - a) * frame.astype(np.float32)
            else:
                # bbox / none modes: avatar rendered at the 512 template size -> warp onto
                # the source-resolution canvas, then blend.
                av_bbox = _alpha_bbox(alpha)
                if av_bbox is None:
                    if args.debug_joints:
                        frame = _draw_joints(frame.copy(), getattr(estimator, "last_pj2d", None))
                    writer.write(frame); n_written += 1; continue
                if bbox_on and target_bbox is not None:
                    M = _fit_affine(av_bbox, target_bbox, args.align_scale, args.flip_x)
                    last_M = M
                elif bbox_on and last_M is not None:
                    M = last_M                           # skip/miss frame: reuse last placement
                else:
                    cx, cy = W_out * 0.5, H_out * 0.5    # 'none' / no-joints: scale to height, center
                    bh = max(1.0, av_bbox[3] - av_bbox[1])
                    s = (H_out * 0.9 / bh) * (args.align_scale if bbox_on else 1.0)
                    acx = (av_bbox[0] + av_bbox[2]) * 0.5
                    acy = (av_bbox[1] + av_bbox[3]) * 0.5
                    sx = -s if args.flip_x else s
                    M = np.array([[sx, 0.0, cx - sx * acx], [0.0, s, cy - s * acy]], dtype=np.float32)
                warp_rgb, warp_a = _place_avatar(avatar_bgr, alpha, M, W_out, H_out)
                a = np.clip(warp_a, 0.0, 1.0)[..., None]
                out = a * warp_rgb.astype(np.float32) + (1.0 - a) * frame.astype(np.float32)

            out = np.clip(out, 0, 255).astype(np.uint8)
            if args.debug_joints:
                _draw_joints(out, getattr(estimator, "last_pj2d", None))   # targets (green)
                if dbg_proj is not None:
                    for x, y in dbg_proj:                                   # avatar joints (red)
                        if np.isfinite(x) and np.isfinite(y):
                            cv2.circle(out, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1)
            writer.write(out)
            n_written += 1

            if n_written % 25 == 0:
                ms = (time.perf_counter() - t0) * 1000.0
                print(f"  frame {n_written}{('/' + str(src_n)) if src_n else ''}  "
                      f"[{status}]  {ms:5.1f} ms", flush=True)
    finally:
        writer.release()
        frame_source.release()

    dt = time.perf_counter() - t_start
    fail_note = f"  (PnP failed on {pnp_fail} frame(s))" if pnp_fail else ""
    print(f"\n[DONE] Wrote {n_written} frames to {out_path} "
          f"in {dt:.1f}s ({n_written / dt:.2f} fps).{fail_note}")


if __name__ == "__main__":
    main()
