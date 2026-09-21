"""
Pose estimator abstraction (Adapter pattern behind an abstract base class).

The downstream pipeline (One-Euro filter -> pose_to_camera_fields -> deform ->
IPC) only ever needs ONE thing from a pose backend: a (72,) axis-angle SMPL
body pose for the current frame, or None when no person is found. We express
exactly that as an abstract interface; each backend (ROMP, PARE, ...) is a
thin adapter that translates its native output into this contract. The live
source class depends only on PoseEstimator, never on romp/pare directly.

Pose estimators (ROMP / PARE) are imported lazily inside their adapters /
build_estimator so that a run using one backend does not require the other to
be installed.
"""

import os
import sys
import time
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch

# Repo root (parent of this `pipeline/` package) — used to locate the vendored
# PARE / HybrIK submodules and to keep their relative resource paths resolvable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _StageProfiler:
    """Opt-in per-stage timer for estimate(), enabled by MISTA_PROFILE=1.

    Purpose: settle whether mesh generation or the CNN backbone dominates a
    backend's frame time. Each estimate() brackets its phases (preprocess /
    forward / post) with `with prof.stage("forward"): ...`; we CUDA-synchronize
    around each so GPU async work is attributed to the right phase rather than
    bleeding into the next. A rolling mean (ms) per stage prints every
    `report_every` frames. Disabled (the default) it is a no-op: stage() yields
    immediately and adds no sync, so normal runs are unaffected.
    """

    _enabled = os.environ.get("MISTA_PROFILE", "") not in ("", "0", "false", "False")

    def __init__(self, label, report_every=60):
        self._label = label
        self._every = int(os.environ.get("MISTA_PROFILE_EVERY", report_every))
        self._sums = {}
        self._n = 0
        self._order = []
        self._cuda = torch.cuda.is_available()

    def _sync(self):
        if self._cuda:
            torch.cuda.synchronize()

    class _Timer:
        def __init__(self, prof, name):
            self._p, self._name = prof, name

        def __enter__(self):
            self._p._sync()
            self._t = time.perf_counter()
            return self

        def __exit__(self, *exc):
            self._p._sync()
            dt = (time.perf_counter() - self._t) * 1000.0
            p = self._p
            if self._name not in p._sums:
                p._sums[self._name] = 0.0
                p._order.append(self._name)
            p._sums[self._name] += dt
            return False

    def stage(self, name):
        if not _StageProfiler._enabled:
            return _NULL_CTX
        return self._Timer(self, name)

    def tick(self):
        """Call once per processed frame; prints a rolling mean every N frames."""
        if not _StageProfiler._enabled:
            return
        self._n += 1
        if self._n % self._every == 0:
            parts = [f"{k}={self._sums[k] / self._n:.2f}ms" for k in self._order]
            total = sum(self._sums.values()) / self._n
            print(f"[PROFILE:{self._label}] n={self._n} " + " ".join(parts)
                  + f" total={total:.2f}ms")


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL_CTX = _NullCtx()


def _root_fix_matrix(spec):
    """Parse a root-orientation-fix spec ('x-90','x90','x180','y90','z90','none') into
    a 3x3 rotation matrix, or None to disable. Used to reconcile HybrIK's SMPL root
    frame with the ROMP/ZJU frame the MISTA deformer expects."""
    if not spec or spec.lower() == "none":
        return None
    axis = spec[0].lower()
    try:
        deg = float(spec[1:])
    except ValueError:
        return None
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    return None


def _rotmats_to_axis_angle(rotmats_np):
    """(N,3,3) rotation matrices -> (N*3,) axis-angle vector.

    Uses cv2.Rodrigues (already a dependency) per joint — robust and exact,
    avoiding a hand-rolled rotmat->quat->axis-angle path. N is small (24 SMPL
    joints), so the Python loop is negligible. Returns float32, laid out to match
    the [root(3) | body(63) | hand(6)] SMPL axis-angle contract when N==24.
    """
    n = rotmats_np.shape[0]
    aa = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        vec, _ = cv2.Rodrigues(rotmats_np[i].astype(np.float64))
        aa[i] = vec.reshape(3)
    return aa.reshape(-1)


class PoseEstimator(ABC):
    """Backend-agnostic single-person SMPL pose estimator.

    Contract: given a BGR frame (np.ndarray, HxWx3), return a tuple
    `(pose, trans)` where `pose` is a (72,) float32 axis-angle vector
    [root(3) | body(63) | hand(6)] and `trans` is a (3,) float32 world
    translation (or None if the backend does not provide one). On no
    detection, return `(None, None)`.

    Side channels (set every estimate(), like `last_pj2d`): `last_betas` holds
    the current person's SMPL shape vector ((10,) float32) or None. The (pose,
    trans) return tuple is unchanged; consumers that want shape read the
    attribute. Used by the proportion retarget to build the actor's rest
    skeleton (see pipeline/retarget.py:source_jtr_from_betas).
    """

    @abstractmethod
    def estimate(self, frame_bgr):
        """(np.ndarray (72,) float32 axis-angle, np.ndarray (3,)|None), or (None, None)."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__


class RompEstimator(PoseEstimator):
    """Adapts romp.ROMP -> PoseEstimator. ROMP's output is already axis-angle."""

    # ROMP's pj2d_org first 24 rows ARE the SMPL-kinematic joints (its own code slices
    # [:24] as the SMPL joints), so the 2D row index equals the SMPL joint index.
    PJ2D_SMPL_MAP = [(i, i) for i in range(24)]

    def __init__(self, romp_model, name="ROMP"):
        self._model = romp_model
        self._name = name
        # Latest person's 2D SMPL joints in ORIGINAL image pixels (J,2), or None on
        # a miss. Set every estimate(); consumed by the offline overlay compositor to
        # place the avatar on the source person. v43 ignores it, so nothing changes there.
        self.last_pj2d = None
        # (pj2d_row, smpl_joint) correspondences for PnP alignment; see PJ2D_SMPL_MAP.
        self.pj2d_smpl_map = self.PJ2D_SMPL_MAP
        # Latest person's SMPL betas ((10,) float32) or None. ROMP returns these
        # under `smpl_betas` when built with --calc_smpl (see from_args).
        self.last_betas = None
        self._prof = _StageProfiler(self._name)

    @property
    def name(self):
        return self._name

    def estimate(self, frame_bgr):
        with self._prof.stage("forward"):
            with torch.no_grad():
                out = self._model(frame_bgr)
        with self._prof.stage("post"):
            if out is not None and out.get("smpl_thetas", None) is not None \
                    and len(out["smpl_thetas"]) > 0:
                thetas = np.asarray(out["smpl_thetas"][0], dtype=np.float32).reshape(-1)
                trans = None
                ct = out.get("cam_trans", None)
                if ct is not None and len(ct) > 0:
                    trans = np.asarray(ct[0], dtype=np.float32).reshape(-1)
                pj = out.get("pj2d_org", None)
                self.last_pj2d = (np.asarray(pj[0], dtype=np.float32)
                                  if pj is not None and len(pj) > 0 else None)
                bt = out.get("smpl_betas", None)
                self.last_betas = (np.asarray(bt[0], dtype=np.float32).reshape(-1)
                                   if bt is not None and len(bt) > 0 else None)
                result = (thetas, trans)
            else:
                self.last_pj2d = None
                self.last_betas = None
                result = (None, None)
        self._prof.tick()
        return result

    @classmethod
    def from_args(cls, args):
        """Construct a RompEstimator with the ONNX/CUDA/TensorRT backend selected by args.

        ONNX-GPU is ROMP's real-time path; TensorRT (--trt) routes that SAME ONNX
        graph through onnxruntime's TensorRT execution provider for a clean A/B. We
        set the providers EXPLICITLY after construction (ROMP hardcodes
        [TRT, CUDA, CPU] in romp/main.py) so the non-TRT baseline is pure CUDA and
        the TRT path carries FP16 + a persistent engine cache. We never fall back to
        the CPU-ONNX provider (slower than ROMP's PyTorch backbone).
        """
        import romp  # lazy: only needed for the ROMP backend

        use_onnx = args.onnx
        use_trt = bool(getattr(args, "trt", False))

        # Put TensorRT's (and cuDNN's) DLLs on PATH BEFORE onnxruntime is imported.
        # onnxruntime's TensorRT provider (onnxruntime_providers_tensorrt.dll) resolves
        # its dependencies (nvinfer*.dll + cuDNN) via the process PATH -- NOT via
        # os.add_dll_directory -- so a missing entry surfaces as LoadLibrary error 126.
        # cuDNN 8 ships inside torch/lib; ORT 1.15.x's TRT EP is happy with it (1.16.x
        # wants cuDNN 8.9 and clashes with torch's 8.7). Must run before `import
        # onnxruntime` so the provider sees the augmented PATH.
        if use_trt:
            _prepend_trt_dll_path(getattr(args, "trt_lib_dir", None))

        ort = None
        if use_onnx:
            try:
                import onnxruntime as ort
            except ImportError:
                print("[INIT] onnxruntime not installed; "
                      "falling back to ROMP PyTorch backend.")
                use_onnx = use_trt = False
        if use_onnx:
            available = ort.get_available_providers()
            if use_trt and "TensorrtExecutionProvider" not in available:
                print("[INIT] onnxruntime has no TensorrtExecutionProvider; "
                      "falling back to the CUDA ONNX backend.")
                use_trt = False
            if "CUDAExecutionProvider" not in available:
                print("[INIT] onnxruntime has no CUDAExecutionProvider; "
                      "falling back to ROMP PyTorch backend.")
                use_onnx = use_trt = False

        # Pre-seed the TensorRT EP env options BEFORE constructing ROMP, so the
        # session ROMP builds internally (its hardcoded [TRT, CUDA, CPU] list) shares
        # the same engine/timing cache we reuse below -- no wasted first engine build.
        if use_trt:
            os.makedirs(args.trt_cache_dir, exist_ok=True)
            os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1" if args.trt_fp16 else "0"
            os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
            os.environ["ORT_TENSORRT_CACHE_PATH"] = args.trt_cache_dir
            os.environ["ORT_TENSORRT_TIMING_CACHE_ENABLE"] = "1"

        romp_argv = ["--GPU", "0", "--show_largest", "--calc_smpl"]
        if use_onnx:
            romp_argv.append("--onnx")
        backend = "TensorRT" if use_trt else ("ONNX-GPU" if use_onnx else "PyTorch")
        print(f"[INIT] Initializing ROMP ({backend}) ...")
        model = romp.ROMP(romp.romp_settings(input_args=romp_argv))

        # Rebuild the onnxruntime session with an EXPLICIT provider list so the
        # backend is deterministic (ROMP's default list would silently prefer TRT).
        if use_onnx:
            if use_trt:
                trt_opts = {
                    "trt_fp16_enable": bool(args.trt_fp16),
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": args.trt_cache_dir,
                    "trt_timing_cache_enable": True,
                }
                providers = [("TensorrtExecutionProvider", trt_opts),
                             "CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            model.ort_session = ort.InferenceSession(
                model.settings.model_onnx_path, providers=providers)
            # get_available_providers() lists TensorRT even when its DLL can't load
            # (missing TensorRT/cuDNN on PATH -> LoadLibrary error 126), in which case
            # ORT silently falls back. Trust the SESSION's active provider, not the
            # requested list, so we never mislabel a CUDA run as TensorRT.
            active = model.ort_session.get_providers()[0]
            print(f"[INIT] ROMP session provider: {active}")
            if use_trt and active != "TensorrtExecutionProvider":
                print("[INIT] WARNING: --trt requested but the TensorRT EP did not "
                      "load (see the EP Error above); running on "
                      f"{active} instead. Install TensorRT 8.6.x (CUDA 11.8) and put "
                      "its libs on PATH to enable it. This run is NOT TensorRT.")
                use_trt = False

        label = "ROMP-TRT" if use_trt else "ROMP"
        print("[INIT] ROMP ready.")
        return cls(model, name=label)


class BevEstimator(PoseEstimator):
    """Adapts bev.BEV -> PoseEstimator. BEV is ROMP's depth-reasoning successor; its
    output dict uses the SAME keys as ROMP (smpl_thetas/cam_trans/smpl_betas/pj2d_org),
    so this mirrors RompEstimator. Two BEV-specific differences: smpl_betas is 11-d
    (SMPL-A: 10 shape + 1 kid/age offset), sliced to (10,) for the contract; and BEV
    returns batched multi-person arrays, so we index person 0 (--show_largest keeps it
    to the largest subject)."""

    # BEV projects the SMPL-kinematic joints (its pj2d_org rows), first 24 = SMPL joint
    # index, same convention as ROMP -> identity map for PnP alignment.
    PJ2D_SMPL_MAP = [(i, i) for i in range(24)]

    def __init__(self, bev_model, name="BEV"):
        self._model = bev_model
        self._name = name
        self.last_pj2d = None
        self.pj2d_smpl_map = self.PJ2D_SMPL_MAP
        self.last_betas = None
        self._prof = _StageProfiler(self._name)

    @property
    def name(self):
        return self._name

    def estimate(self, frame_bgr):
        with self._prof.stage("forward"):
            with torch.no_grad():
                out = self._model(frame_bgr)
        with self._prof.stage("post"):
            if out is not None and out.get("smpl_thetas", None) is not None \
                    and len(out["smpl_thetas"]) > 0:
                thetas = np.asarray(out["smpl_thetas"][0], dtype=np.float32).reshape(-1)
                trans = None
                ct = out.get("cam_trans", None)
                if ct is not None and len(ct) > 0:
                    trans = np.asarray(ct[0], dtype=np.float32).reshape(-1)
                pj = out.get("pj2d_org", None)
                self.last_pj2d = (np.asarray(pj[0], dtype=np.float32)
                                  if pj is not None and len(pj) > 0 else None)
                bt = out.get("smpl_betas", None)
                # BEV betas are 11-d (SMPL-A); the contract / retarget path expect (10,).
                self.last_betas = (np.asarray(bt[0], dtype=np.float32).reshape(-1)[:10]
                                   if bt is not None and len(bt) > 0 else None)
                result = (thetas, trans)
            else:
                self.last_pj2d = None
                self.last_betas = None
                result = (None, None)
        self._prof.tick()
        return result

    @classmethod
    def from_args(cls, args):
        """Construct a BevEstimator (PyTorch backend; backbone optionally ONNX/TensorRT).

        BEV shares ROMP's ~/.romp model directory. BEV.pth and SMPLA_NEUTRAL.pth are
        the real BEV weights (auto-downloaded / already present).

        SMIL (BEV's baby body model) is the one gap: it is NOT in the public ROMP release
        (its download is commented out in bev_settings) and requires manual registration
        with the SMIL project + packing via `bev.prepare_smil`. But SMPLA_parser only ever
        *invokes* the SMIL model for detections flagged as babies (betas[..,10] kid offset);
        for adult subjects -- the MISTA webcam/video use case -- it is constructed but never
        called. So when smil_packed_info.pth is absent we point smil_path at the standard
        SMPL_NEUTRAL.pth already in ~/.romp (same packed format, model_type='smpl'), which
        lets BEV construct and run with identical adult output. If a real SMIL file is
        present it is used as-is.

        BEV's rendering/vis stack (Sim3DR_Cython, vis_human) pulls in a second OpenMP
        runtime alongside torch's libiomp5md, which hard-aborts the process on Windows
        (native exit 0xC06D007F) unless duplicates are allowed -- same guard the PARE and
        HybrIK backends use. We also pass --render_mesh (a store_false flag: presence turns
        OFF mesh rendering) so BEV skips the per-frame vertex render we never consume; the
        2D joint projection (pj2d_org, used by --align) is gated by --calc_smpl, not
        --render_mesh, so it is still produced.

        --trt/--onnx accelerate BEV's HRNet backbone via onnxruntime (TensorRT/CUDA EP),
        the same stack ROMP's --trt uses; the dynamic heads stay in PyTorch.
        """
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        import bev  # lazy: only needed for the BEV backend

        settings = bev.bev_settings(
            input_args=["--GPU", "0", "--show_largest", "--calc_smpl", "--render_mesh"])
        if not os.path.exists(settings.smil_path):
            smpl_stub = os.path.join(os.path.dirname(settings.smpl_path), "SMPL_NEUTRAL.pth")
            if not os.path.exists(smpl_stub):
                raise FileNotFoundError(
                    f"BEV needs a SMIL model at {settings.smil_path} (not in the public "
                    f"ROMP release) or a fallback SMPL model at {smpl_stub}; neither exists. "
                    "Provide SMPL_NEUTRAL.pth in ~/.romp (ROMP ships it) or a real SMIL file.")
            print(f"[INIT] SMIL model not found at {settings.smil_path}; substituting "
                  f"{smpl_stub} (baby model unused for adult subjects).")
            settings.smil_path = smpl_stub

        print("[INIT] Initializing BEV ...")
        model = bev.BEV(settings)
        # Optionally replace the HRNet backbone (the heavy CNN) with an onnxruntime
        # TensorRT/CUDA session when --trt/--onnx is passed. BEV ships no ONNX graph, so
        # we export the backbone once (cached at ~/.romp/BEV_backbone.onnx) and route only
        # that static-shape CNN through TRT, keeping BEV's dynamic heads/parser in PyTorch.
        label = _accelerate_bev_backbone(model, args)
        print(f"[INIT] {label} ready.")
        return cls(model, name=label)


class PareEstimator(PoseEstimator):
    """Adapts a PARE model -> PoseEstimator.

    Uses a full-frame square bbox (single-user webcam / video: the subject is
    assumed to roughly fill the frame, so we skip a separate person detector).
    PARE's forward returns `pred_pose` as rotation matrices (1,24,3,3); we
    convert to (72,) axis-angle with PARE's own geometry utility to match the
    axis-angle SMPL pose ROMP produces.

    The crop + ImageNet normalization is inlined here (equivalent to PARE's
    get_single_image_crop_demo with rot=0/no-flip) so we depend only on
    pare.utils.geometry -- NOT pare.utils.vibe_image_utils, which imports
    scikit-image at module load (a helper we never use, and which crashes on
    import in this environment). We also convert BGR->RGB, which the ndarray
    path of get_single_image_crop_demo does NOT do but the ImageNet mean/std
    require (PARE's demo converts before calling it).
    """

    # ImageNet normalization (matches PARE get_default_transform).
    _IMAGENET_MEAN = (0.485, 0.456, 0.406)
    _IMAGENET_STD = (0.229, 0.224, 0.225)

    # PARE's smpl_joints2d are 49 SPIN-convention joints (pare/core/constants.py:
    # JOINT_NAMES), NOT SMPL-kinematic. Its first 25 are OpenPose joints; JOINT_MAP
    # gives each one's SMPL index. These are the OpenPose rows whose SMPL index is a
    # real kinematic joint (<24), as (pj2d_row, smpl_joint) pairs — the correspondences
    # PnP uses to fit the camera against the avatar's FK joints.
    PJ2D_SMPL_MAP = [
        (1, 12), (2, 17), (3, 19), (4, 21), (5, 16), (6, 18), (7, 20),
        (8, 0), (9, 2), (10, 5), (11, 8), (12, 1), (13, 4), (14, 7),
    ]

    def __init__(self, model, device, crop_size=224, scale=1.0):
        self._model = model
        self._device = device
        self._crop_size = int(crop_size)
        self._scale = float(scale)
        from pare.utils.geometry import rotation_matrix_to_angle_axis
        self._rotmat_to_aa = rotation_matrix_to_angle_axis
        self._mean = torch.tensor(self._IMAGENET_MEAN, device=device).view(3, 1, 1)
        self._std = torch.tensor(self._IMAGENET_STD, device=device).view(3, 1, 1)
        # Latest person's 2D joints in ORIGINAL image pixels (J,2), or None — same
        # contract as RompEstimator.last_pj2d, so --align pnp/bbox work with PARE too.
        self.last_pj2d = None
        self._last_crop_trans = None      # original->crop affine (2x3), to invert for pj2d
        self.pj2d_smpl_map = self.PJ2D_SMPL_MAP
        # Latest person's SMPL betas ((10,) float32) or None, from PARE's pred_shape.
        self.last_betas = None
        self._prof = _StageProfiler("PARE")

    @property
    def name(self):
        return "PARE"

    def _crop_and_normalize(self, frame_bgr):
        """Full-frame square crop -> crop_size, ImageNet-normalized (3,H,W) tensor.

        Reproduces PARE's generate_patch_image_cv (rot=0, no flip) + ToTensor +
        Normalize for a bbox centered on the frame covering max(w,h)*scale.
        """
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        src_half = float(max(w, h)) * self._scale * 0.5
        dst = float(self._crop_size)
        # 3 correspondences (center, down, right): scaled square bbox -> crop square.
        src = np.array([[cx, cy], [cx, cy + src_half], [cx + src_half, cy]], np.float32)
        dst_pts = np.array([[dst * 0.5, dst * 0.5],
                            [dst * 0.5, dst], [dst, dst * 0.5]], np.float32)
        trans = cv2.getAffineTransform(src, dst_pts)
        self._last_crop_trans = trans          # remember original->crop map for pj2d inverse
        patch = cv2.warpAffine(img, trans, (int(dst), int(dst)),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        t = torch.from_numpy(patch).to(self._device).float().permute(2, 0, 1) / 255.0
        return (t - self._mean) / self._std

    def _joints2d_to_original(self, joints2d_norm):
        """PARE's normalized crop joints (J,2) -> ORIGINAL-image pixels (J,2).

        PARE projects with camera_center=0 (crop-centered) and normalizes by
        img_res/2 (smpl_head.py), so a crop pixel is `j*(crop/2) + crop/2`. We then
        invert the original->crop affine we used to build the input patch.
        """
        half = self._crop_size / 2.0
        crop_px = joints2d_norm * half + half                    # (J,2) in crop pixels
        inv = cv2.invertAffineTransform(self._last_crop_trans)    # 2x3 crop->original
        return (crop_px @ inv[:, :2].T + inv[:, 2]).astype(np.float32)

    def estimate(self, frame_bgr):
        with self._prof.stage("preprocess"):
            inp = self._crop_and_normalize(frame_bgr).unsqueeze(0)   # (1,3,H,W)
        with self._prof.stage("forward"):
            with torch.no_grad():
                out = self._model(inp)
        with self._prof.stage("post"):
            rotmat = out["pred_pose"].reshape(-1, 3, 3)          # (24,3,3) rotmat
            aa = self._rotmat_to_aa(rotmat).reshape(-1)          # (72,) axis-angle
            # 2D joints (normalized, crop-centered) -> original pixels, for PnP/bbox align.
            j2d = out.get("smpl_joints2d", None)
            if j2d is not None and self._last_crop_trans is not None:
                j = j2d[0].detach().cpu().numpy().astype(np.float32)   # (J,2)
                self.last_pj2d = self._joints2d_to_original(j)
            else:
                self.last_pj2d = None
            shape = out.get("pred_shape", None)
            self.last_betas = (shape[0].detach().cpu().numpy().astype(np.float32).reshape(-1)
                               if shape is not None and len(shape) > 0 else None)
            result = aa.detach().cpu().numpy().astype(np.float32)
        self._prof.tick()
        # PARE's demo path exposes no consistent world translation here, so root
        # motion is simply disabled for this backend (trans = None).
        return result, None

    @classmethod
    def from_config(cls, pare_cfg, pare_ckpt, device, crop_size=224, scale=1.0):
        """Instantiate the PARE network from its config + checkpoint, in eval mode.

        Mirrors PARETester._build_model + _load_pretrained_model (pare/core/tester.py)
        but without the multi-person tracker / video pipeline: we only need the raw
        network forward on a pre-cropped frame. (--onnx/--no-onnx are ROMP-only and
        simply don't apply here.)
        """
        if "--onnx" in sys.argv or "--no-onnx" in sys.argv:
            print("[INIT] --onnx/--no-onnx only apply to the ROMP backend; ignored for PARE.")
        # PARE's dependency stack pulls in a second OpenMP runtime (libomp alongside
        # torch's libiomp5md), which aborts on Windows unless duplicates are allowed.
        # The ROMP path does not trigger this, so we only set it for PARE, and only
        # if the user hasn't already chosen a value.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        print(f"[INIT] Initializing PARE from {pare_ckpt} ...")

        # Ensure the vendored PARE package (submodules/PARE) is importable.
        pare_root = os.path.join(_ROOT, "submodules", "PARE")
        if os.path.isdir(pare_root) and pare_root not in sys.path:
            sys.path.insert(0, pare_root)

        # Compat shim: newer torchvision (matching torch 2.x) removed
        # torchvision.models.utils; load_state_dict_from_url now lives in torch.hub.
        # PARE's vendored backbones still import the old path.
        import types
        import torch.hub as _hub
        import torchvision.models as _tvm
        if not hasattr(_tvm, "utils"):
            _shim = types.ModuleType("torchvision.models.utils")
            _shim.load_state_dict_from_url = _hub.load_state_dict_from_url
            sys.modules["torchvision.models.utils"] = _shim

        from pare.core.config import update_hparams
        from pare.models import PARE

        # PARE resolves its SMPL body model / regressors via paths relative to CWD
        # (e.g. 'data/body_models/smpl', see pare/core/config.py). Those files live
        # under the submodule, so build the model with CWD switched to pare_root.
        # Resolve cfg/ckpt to absolute first so they survive the chdir.
        pare_cfg = os.path.abspath(pare_cfg)
        pare_ckpt = os.path.abspath(pare_ckpt)
        _prev_cwd = os.getcwd()
        os.chdir(pare_root)
        try:
            hparams = update_hparams(pare_cfg)
            model = cls._construct(PARE, hparams, device)
            ckpt = torch.load(pare_ckpt, map_location=device)["state_dict"]
            # Checkpoint keys are prefixed with "model." (LightningModule); strip.
            state = {k.replace("model.", "", 1): v for k, v in ckpt.items()
                     if k.startswith("model.")}
            model.load_state_dict(state, strict=False)
        finally:
            os.chdir(_prev_cwd)
        model.eval()
        print("[INIT] PARE ready.")
        return cls(model, device, crop_size=crop_size, scale=scale)

    @staticmethod
    def _construct(PARE, hparams, device):
        """Instantiate the PARE nn.Module from hparams (split out for readability)."""
        return PARE(
            backbone=hparams.PARE.BACKBONE,
            num_joints=hparams.PARE.NUM_JOINTS,
            softmax_temp=hparams.PARE.SOFTMAX_TEMP,
            num_features_smpl=hparams.PARE.NUM_FEATURES_SMPL,
            focal_length=hparams.DATASET.FOCAL_LENGTH,
            img_res=hparams.DATASET.IMG_RES,
            pretrained=hparams.TRAINING.PRETRAINED,
            iterative_regression=hparams.PARE.ITERATIVE_REGRESSION,
            num_iterations=hparams.PARE.NUM_ITERATIONS,
            iter_residual=hparams.PARE.ITER_RESIDUAL,
            shape_input_type=hparams.PARE.SHAPE_INPUT_TYPE,
            pose_input_type=hparams.PARE.POSE_INPUT_TYPE,
            pose_mlp_num_layers=hparams.PARE.POSE_MLP_NUM_LAYERS,
            shape_mlp_num_layers=hparams.PARE.SHAPE_MLP_NUM_LAYERS,
            pose_mlp_hidden_size=hparams.PARE.POSE_MLP_HIDDEN_SIZE,
            shape_mlp_hidden_size=hparams.PARE.SHAPE_MLP_HIDDEN_SIZE,
            use_keypoint_features_for_smpl_regression=hparams.PARE.USE_KEYPOINT_FEATURES_FOR_SMPL_REGRESSION,
            use_heatmaps=hparams.DATASET.USE_HEATMAPS,
            use_keypoint_attention=hparams.PARE.USE_KEYPOINT_ATTENTION,
            use_postconv_keypoint_attention=hparams.PARE.USE_POSTCONV_KEYPOINT_ATTENTION,
            use_coattention=hparams.PARE.USE_COATTENTION,
            num_coattention_iter=hparams.PARE.NUM_COATTENTION_ITER,
            coattention_conv=hparams.PARE.COATTENTION_CONV,
            use_upsampling=hparams.PARE.USE_UPSAMPLING,
            deconv_conv_kernel_size=hparams.PARE.DECONV_CONV_KERNEL_SIZE,
            use_soft_attention=hparams.PARE.USE_SOFT_ATTENTION,
            num_branch_iteration=hparams.PARE.NUM_BRANCH_ITERATION,
            branch_deeper=hparams.PARE.BRANCH_DEEPER,
            num_deconv_layers=hparams.PARE.NUM_DECONV_LAYERS,
            num_deconv_filters=hparams.PARE.NUM_DECONV_FILTERS,
            use_resnet_conv_hrnet=hparams.PARE.USE_RESNET_CONV_HRNET,
            use_position_encodings=hparams.PARE.USE_POS_ENC,
            use_mean_camshape=hparams.PARE.USE_MEAN_CAMSHAPE,
            use_mean_pose=hparams.PARE.USE_MEAN_POSE,
            init_xavier=hparams.PARE.INIT_XAVIER,
        ).to(device)


class HybrIKEstimator(PoseEstimator):
    """Adapts a HybrIK model -> PoseEstimator.

    HybrIK (analytical-neural IK) regresses SMPL parameters directly, so it fits
    the same (72,) axis-angle contract as ROMP/PARE. Its forward returns
    `pred_theta_mats` as SMPL joint rotation matrices (1, 24*9); we reshape to
    (24,3,3) and convert to (72,) axis-angle (via cv2.Rodrigues) to match ROMP.

    Detection strategy: full-frame bbox (no separate person detector), mirroring
    the PARE backend's assumption that the single subject roughly fills the frame.
    This keeps HybrIK light enough for the RTX 1080 when paired with the ResNet-34
    backbone config. A real detector (Faster R-CNN, per HybrIK's demo) can be added
    later behind a flag. We reuse HybrIK's own `SimpleTransform3DSMPLCam.test_transform`
    for preprocessing rather than reinventing its bbox/camera normalization, which
    the IK head depends on.
    """

    # HybrIK's pred_uvd_jts / pred_theta_mats use a 29-joint SMPL layout whose first
    # 24 are the SMPL kinematic joints, so the row index equals the SMPL joint index.
    PJ2D_SMPL_MAP = [(i, i) for i in range(24)]

    def __init__(self, model, transform, device):
        self._model = model
        self._transform = transform
        self._device = device
        # Same side-channel contract as the other backends (see RompEstimator).
        self.last_pj2d = None
        self.pj2d_smpl_map = self.PJ2D_SMPL_MAP
        self.last_betas = None
        # Keypoint-tracked person bbox (xyxy) reused across frames. None => cold start
        # on a full frame; after each forward it is retightened from the predicted 2D
        # joints (see below). HybrIK's analytical IK + camera regression are highly
        # sensitive to the crop: a full-frame bbox yields a wrong GLOBAL ORIENTATION
        # (avatar tipped over) and degraded body pose. Detector-free tracking keeps it
        # real-time on the RTX 1080 while giving the network the tight crop it expects.
        self._bbox = None
        self._prof = _StageProfiler("HybrIK")

    @property
    def name(self):
        return "HybrIK"

    @staticmethod
    def _bbox_from_pts(pts, w, h, pad=0.3):
        """Tight square xyxy bbox around 2D joints `pts` (J,2), padded, clamped to
        the image. Returns None if the points are degenerate (so the caller resets
        to a full-frame crop rather than tracking a collapsed box)."""
        x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
        side = max(x1 - x0, y1 - y0) * (1.0 + pad)
        if not np.isfinite(side) or side < 8.0 or side > 4.0 * max(w, h):
            return None
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        return np.array([cx - side / 2, cy - side / 2,
                         cx + side / 2, cy + side / 2], dtype=np.float32)

    def estimate(self, frame_bgr):
        with self._prof.stage("preprocess"):
            img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            # Person bbox (xyxy): tracked from the previous frame's predicted 2D joints;
            # full frame only on the first frame (cold start, corrected within one frame).
            tight_bbox = (self._bbox if self._bbox is not None
                          else np.array([0.0, 0.0, float(w), float(h)], dtype=np.float32))
            pose_input, bbox, img_center = self._transform.test_transform(img, tight_bbox)
            pose_input = pose_input.to(self._device)[None, :, :, :]
        with self._prof.stage("forward"):
            with torch.no_grad():
                # flip_test=False: HybrIK's flip branch has an upstream bug (it feeds the
                # raw pooled features `flip_x0` into self.decsigma instead of the fc-processed
                # `flip_xc`, causing a 512-vs-1024 matmul error). Flip test is only a small
                # test-time-augmentation accuracy bump and doubles the forward cost, so we
                # disable it — better for real-time and sidesteps the bug.
                out = self._model(
                    pose_input, flip_test=False,
                    bboxes=torch.from_numpy(np.asarray(bbox)).to(self._device).unsqueeze(0).float(),
                    img_center=torch.from_numpy(np.asarray(img_center)).to(self._device).unsqueeze(0).float(),
                )
        # pred_theta_mats: (1, 24*9) SMPL joint rotation matrices -> (24,3,3) -> (72,) aa.
        rotmats = out.pred_theta_mats.reshape(-1, 3, 3)[:24].detach().cpu().numpy()
        aa = _rotmats_to_axis_angle(rotmats)
        # Optional global-orientation fix. With the tracked TIGHT bbox above, HybrIK's
        # raw root already matches the ROMP/ZJU frame the deformer expects to within
        # ~10 deg (verified against ROMP), so the default is "none". The earlier
        # apparent ~90 deg tip was an artefact of the full-frame crop, not a frame
        # offset. Kept configurable via MISTA_HYBRIK_ROOTFIX ("x-90","x90","x180","z90",
        # "none") in case a different camera/rig needs a real axis correction:
        # R_root' = R_fix @ R_root.
        _spec = os.environ.get("MISTA_HYBRIK_ROOTFIX", "none")
        _fix = _root_fix_matrix(_spec)
        if _fix is not None:
            R_root, _ = cv2.Rodrigues(aa[:3].astype(np.float64))
            aa[:3] = cv2.Rodrigues(_fix @ R_root)[0].reshape(3).astype(np.float32)
        shape = getattr(out, "pred_shape", None)
        self.last_betas = (shape.reshape(-1)[:10].detach().cpu().numpy().astype(np.float32)
                           if shape is not None else None)
        # HybrIK's pred_uvd_jts (xy) are normalized center-relative to the crop bbox;
        # map them back to ORIGINAL image pixels (mirrors HybrIK demo_video.py:283-285,
        # with the square bbox from test_transform). These drive two things: (1) the
        # tight person bbox tracked into the NEXT frame, and (2) last_pj2d for the
        # PnP/overlay align path (24 SMPL-kinematic joints, per PJ2D_SMPL_MAP).
        uv = out.pred_uvd_jts.reshape(-1, 3)[:24, :2].detach().cpu().numpy()
        bx0, by0, bx1, by1 = [float(v) for v in bbox]
        bw = bx1 - bx0
        pj2d = uv * bw
        pj2d[:, 0] += (bx0 + bx1) * 0.5
        pj2d[:, 1] += (by0 + by1) * 0.5
        self.last_pj2d = pj2d.astype(np.float32)
        # Retighten the tracked bbox for the next frame; reset to full-frame if the
        # predicted joints degenerate (person lost), so tracking self-recovers.
        self._bbox = self._bbox_from_pts(pj2d, w, h)
        # One-shot diagnostic dump to a file (real-app env is stable, unlike the offline
        # probe). Captures the evidence needed to pin the pose-convention / orientation.
        if not getattr(self, "_dbg_done", False):
            self._dbg_done = True
            try:
                _raw = _rotmats_to_axis_angle(rotmats)
                n = np.linalg.norm(_raw.reshape(24, 3), axis=1)
                L = []
                L.append(f"rootfix_spec={_spec}")
                L.append(f"raw_root_aa={np.round(_raw[:3],3).tolist()}  fixed_root_aa={np.round(aa[:3],3).tolist()}")
                L.append(f"aa_norm max={float(n.max()):.2f} mean={float(n.mean()):.2f}")
                L.append(f"betas={None if self.last_betas is None else np.round(self.last_betas,3).tolist()}")
                L.append(f"L_shoulder(16) R=\n{np.round(rotmats[16],3)}")
                L.append(f"R_shoulder(17) R=\n{np.round(rotmats[17],3)}")
                xyz = getattr(out, "pred_xyz_jts_29", None)
                if xyz is not None:
                    x = xyz.detach().cpu().numpy().reshape(-1, 3)
                    L.append(f"xyz j0pelvis={np.round(x[0],3).tolist()} j15head={np.round(x[15],3).tolist()} "
                             f"j16Lsh={np.round(x[16],3).tolist()} j20Lwri={np.round(x[20],3).tolist()}")
                with open(os.path.join(_ROOT, "hybrik_dbg.txt"), "w") as f:
                    f.write("\n".join(L) + "\n")
                print("[HYBRIK-DBG] wrote", os.path.join(_ROOT, "hybrik_dbg.txt"))
            except Exception as e:
                print("[HYBRIK-DBG] dump failed:", e)
        # World translation for --root-motion. HybrIK's `transl` (== cam_root, the SMPL
        # root in camera metres) is returned in ROMP's cam_trans convention: measured
        # against ROMP on the same frames, HybrIK's X and Y (vertical) match ROMP ~1:1
        # (Y corr 0.99, slope ~1.0, no sign flip), so the same default --root-axis /
        # --root-scale drive both backends. CAVEAT: HybrIK's Z (depth) is monocularly
        # unreliable (large offset + several-metre jitter), so `--root-horizontal`
        # (zero depth) is recommended for HybrIK — it keeps the clean vertical jump and
        # lateral motion and drops the noisy toward/away component.
        transl = getattr(out, "transl", None)
        if transl is not None:
            transl = transl.reshape(-1)[:3].detach().cpu().numpy().astype(np.float32)
        self._prof.tick()
        return aa, transl

    @classmethod
    def from_config(cls, hybrik_cfg, hybrik_ckpt, device):
        """Instantiate the HybrIK network + its preprocessing transform, in eval mode.

        Mirrors HybrIK's scripts/demo_video.py setup (builder.build_sppe + the
        SimpleTransform3DSMPLCam test transform) minus the Faster R-CNN detector and
        the video/tracking loop: we only need the raw network forward on a full-frame
        crop.
        """
        # HybrIK's stack (like PARE's) can pull in a second OpenMP runtime; allow the
        # duplicate on Windows unless the user already chose a value. --onnx/--trt are
        # ROMP-only and simply don't apply here.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        print(f"[INIT] Initializing HybrIK from {hybrik_ckpt} ...")

        # Ensure the vendored HybrIK package (submodules/HybrIK) is importable.
        hybrik_root = os.path.join(_ROOT, "submodules", "HybrIK")
        if os.path.isdir(hybrik_root) and hybrik_root not in sys.path:
            sys.path.insert(0, hybrik_root)

        # HybrIK's SMPL layer imports two pure-torch helpers from pytorch3d
        # (axis_angle_to_matrix / matrix_to_axis_angle). Building the real pytorch3d on
        # Windows is painful and we never use its CUDA renderer, so fall back to the
        # bundled shim (pipeline/_p3d_shim) only when pytorch3d isn't already installed.
        try:
            import pytorch3d.transforms.rotation_conversions  # noqa: F401
        except Exception:
            shim_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_p3d_shim")
            if shim_dir not in sys.path:
                sys.path.insert(0, shim_dir)
            print("[INIT] pytorch3d not found; using bundled rotation-conversion shim.")

        # hybrik.utils.presets.__init__ eagerly imports the SMPL-X transform, which
        # instantiates SMPLXLayer objects at module load (needing model_files/smplx/*.npz).
        # Our res34 path uses only the SMPL transform (SimpleTransform3DSMPLCam), so stub
        # the SMPL-X submodule to avoid requiring SMPL-X model files we never use.
        import types as _types
        _smplx_mod = "hybrik.utils.presets.simple_transform_3d_smplx"
        if _smplx_mod not in sys.modules:
            _stub = _types.ModuleType(_smplx_mod)

            class _UnavailSMPLX:  # only reached if someone actually uses the SMPL-X path
                def __init__(self, *a, **k):
                    raise RuntimeError("HybrIK SMPL-X transform is not enabled in this "
                                       "integration (SMPL-only res34 path).")

            _stub.SimpleTransform3DSMPLX = _UnavailSMPLX
            sys.modules[_smplx_mod] = _stub

        from easydict import EasyDict as edict
        from hybrik.models import builder
        from hybrik.utils.config import update_config
        from hybrik.utils.presets import SimpleTransform3DSMPLCam

        # HybrIK resolves its SMPL body model / joint regressors via paths relative to
        # CWD (model_files/..., see its config), which live under the submodule, so build
        # with CWD switched to hybrik_root. Resolve cfg/ckpt to absolute first.
        hybrik_cfg = os.path.abspath(hybrik_cfg)
        hybrik_ckpt = os.path.abspath(hybrik_ckpt)
        _prev_cwd = os.getcwd()
        os.chdir(hybrik_root)
        try:
            cfg = update_config(hybrik_cfg)
            # Follow demo_video.py exactly: read the 3D bbox shape (mm) and convert to
            # metres. Must match the chosen config's DATA/MODEL preset.
            bbox_3d_shape = getattr(cfg.MODEL, "BBOX_3D_SHAPE", (2000, 2000, 2000))
            bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
            # Dummy dataset: the transform only needs bbox_3d_shape here; joint_pairs are
            # for flip augmentation, which flip_test handles inside the model, not here.
            dummy_set = edict({
                "joint_pairs_17": None,
                "joint_pairs_24": None,
                "joint_pairs_29": None,
                "bbox_3d_shape": bbox_3d_shape,
            })
            transform = SimpleTransform3DSMPLCam(
                dummy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
                color_factor=cfg.DATASET.COLOR_FACTOR,
                occlusion=cfg.DATASET.OCCLUSION,
                input_size=cfg.MODEL.IMAGE_SIZE,
                output_size=cfg.MODEL.HEATMAP_SIZE,
                depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
                bbox_3d_shape=bbox_3d_shape,
                rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
                train=False, add_dpg=False,
                loss_type=cfg.LOSS["TYPE"])
            model = builder.build_sppe(cfg.MODEL)
            save_dict = torch.load(hybrik_ckpt, map_location="cpu")
            model_dict = (save_dict["model"]
                          if isinstance(save_dict, dict) and "model" in save_dict
                          else save_dict)
            model.load_state_dict(model_dict, strict=False)
        finally:
            os.chdir(_prev_cwd)
        model = model.to(device)
        model.eval()
        print("[INIT] HybrIK ready.")
        return cls(model, transform, device)


def _prepend_trt_dll_path(trt_lib_dir):
    """Prepend the TensorRT lib dir (+ torch/lib for cuDNN) to the process PATH.

    onnxruntime's TensorRT provider loads its dependent DLLs through the PATH
    search order, so this must run before `import onnxruntime`. torch/lib supplies
    cuDNN 8 (and cuBLAS/cudart) that TensorRT needs.
    """
    parts = []
    if trt_lib_dir and os.path.isdir(trt_lib_dir):
        parts.append(os.path.abspath(trt_lib_dir))
        print(f"[INIT] TensorRT lib dir on PATH: {trt_lib_dir}")
    elif trt_lib_dir:
        print(f"[INIT] WARNING: --trt-lib-dir '{trt_lib_dir}' does not exist.")
    else:
        print("[INIT] WARNING: no TensorRT lib dir given (--trt-lib-dir / "
              "$TENSORRT_LIB_DIR); the TensorRT EP will likely fail to load.")
    try:
        import torch as _t
        torch_lib = os.path.join(os.path.dirname(_t.__file__), "lib")
        if os.path.isdir(torch_lib):
            parts.append(torch_lib)  # cuDNN 8 / cuBLAS / cudart for TensorRT
    except Exception:
        pass
    if parts:
        os.environ["PATH"] = os.pathsep.join(parts) + os.pathsep + os.environ.get("PATH", "")


class _OrtBackbone(torch.nn.Module):
    """Drop-in replacement for BEV's HRNet backbone that runs an ONNX graph through
    onnxruntime (CUDA or TensorRT EP) instead of PyTorch.

    BEVv1.forward does `x = self.backbone(x)` and feeds that single feature map to
    both the localization and param heads (bev/model.py:233-245). The backbone is a
    pure static-shape CNN -- input NHWC (1,512,512,3) from romp.img_preprocess, output
    (1,32,128,128) -- so it exports cleanly to ONNX and is the heavy part worth
    accelerating. We keep the rest of BEV (dynamic center parsing, 3D transformer,
    SMPL-A parser) in PyTorch. I/O crosses the GPU<->host boundary as numpy; the
    copies (~3MB in, ~2MB out) are negligible next to the conv backbone cost.
    """

    def __init__(self, session, device, input_name, output_name):
        super().__init__()
        self._sess = session
        self._device = device
        self._in = input_name
        self._out = output_name

    def forward(self, x):
        feats = self._sess.run([self._out],
                               {self._in: x.detach().cpu().numpy().astype(np.float32)})[0]
        return torch.from_numpy(feats).to(self._device)


def _bev_backbone_onnx_path(settings):
    """Cache the exported backbone next to BEV.pth (~/.romp/BEV_backbone.onnx)."""
    return os.path.join(os.path.dirname(settings.model_path), "BEV_backbone.onnx")


def _export_bev_backbone_onnx(backbone, onnx_path):
    """Export the HRNet backbone to a static-shape ONNX graph (once; cached)."""
    device = next(backbone.parameters()).device
    # img_preprocess yields NHWC (1,512,512,3); the backbone permutes internally.
    dummy = torch.zeros(1, 512, 512, 3, dtype=torch.float32, device=device)
    print(f"[INIT] Exporting BEV backbone to ONNX at {onnx_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            backbone, dummy, onnx_path,
            input_names=["image"], output_names=["feat"],
            export_params=True, opset_version=12, do_constant_folding=True)


def _accelerate_bev_backbone(model, args):
    """Splice an onnxruntime (TensorRT/CUDA EP) backbone into a constructed BEV.

    Reuses the SAME TRT plumbing as the ROMP path (_prepend_trt_dll_path + the
    ORT_TENSORRT_* env + an explicit provider list) so `--estimator bev --trt
    --trt-fp16 --trt-lib-dir ...` behaves like ROMP's --trt. Returns a backend label
    ('BEV-TRT' / 'BEV-ONNX' / 'BEV'). Any failure falls back to the PyTorch backbone
    so BEV still runs.
    """
    use_onnx = bool(getattr(args, "onnx", False))
    use_trt = bool(getattr(args, "trt", False))
    if not (use_onnx or use_trt):
        return "BEV"

    if use_trt:
        _prepend_trt_dll_path(getattr(args, "trt_lib_dir", None))
    try:
        import onnxruntime as ort
    except ImportError:
        print("[INIT] onnxruntime not installed; BEV backbone stays on PyTorch.")
        return "BEV"

    available = ort.get_available_providers()
    if use_trt and "TensorrtExecutionProvider" not in available:
        print("[INIT] onnxruntime has no TensorrtExecutionProvider; using CUDA EP for "
              "the BEV backbone.")
        use_trt = False
    if "CUDAExecutionProvider" not in available:
        print("[INIT] onnxruntime has no CUDAExecutionProvider; BEV backbone stays on "
              "PyTorch.")
        return "BEV"

    net = model.model.module  # BEVv1 (unwrap DataParallel)
    onnx_path = _bev_backbone_onnx_path(model.settings)
    if not os.path.exists(onnx_path):
        _export_bev_backbone_onnx(net.backbone, onnx_path)

    if use_trt:
        os.makedirs(args.trt_cache_dir, exist_ok=True)
        os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1" if args.trt_fp16 else "0"
        os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
        os.environ["ORT_TENSORRT_CACHE_PATH"] = args.trt_cache_dir
        os.environ["ORT_TENSORRT_TIMING_CACHE_ENABLE"] = "1"
        trt_opts = {
            "trt_fp16_enable": bool(args.trt_fp16),
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": args.trt_cache_dir,
            "trt_timing_cache_enable": True,
        }
        providers = [("TensorrtExecutionProvider", trt_opts),
                     "CUDAExecutionProvider", "CPUExecutionProvider"]
        print("[INIT] Building BEV backbone TensorRT engine (first run may take a few "
              "minutes; cached afterwards) ...")
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path, providers=providers)
    active = sess.get_providers()[0]
    print(f"[INIT] BEV backbone session provider: {active}")
    if use_trt and active != "TensorrtExecutionProvider":
        print("[INIT] WARNING: --trt requested but the TensorRT EP did not load (see the "
              f"EP Error above); running the BEV backbone on {active} instead. Put "
              "TensorRT 8.6.x (CUDA 11.8) libs on PATH (--trt-lib-dir) to enable it.")
        use_trt = False

    device = next(net.parameters()).device
    inp = sess.get_inputs()[0].name
    outp = sess.get_outputs()[0].name
    net.backbone = _OrtBackbone(sess, device, inp, outp)
    return "BEV-TRT" if use_trt else "BEV-ONNX"


def build_estimator(args, device) -> PoseEstimator:
    """Factory: construct the selected PoseEstimator backend. Keeps main() and the
    source class free of any backend-specific imports/branching. Each backend's
    construction (imports, OpenMP guard, ONNX/TensorRT setup, logging) lives in its
    own class's from_args/from_config classmethod; this dispatcher just selects one."""
    if args.estimator == "romp":
        return RompEstimator.from_args(args)
    if args.estimator == "bev":
        return BevEstimator.from_args(args)
    if args.estimator == "hybrik":
        return HybrIKEstimator.from_config(args.hybrik_cfg, args.hybrik_ckpt, device)
    return PareEstimator.from_config(args.pare_cfg, args.pare_ckpt, device,
                                     crop_size=args.pare_crop_size, scale=args.pare_scale)
