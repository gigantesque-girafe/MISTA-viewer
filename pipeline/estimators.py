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
from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch

# Repo root (parent of this `pipeline/` package) — used to locate the vendored
# PARE submodule and to keep its relative resource paths resolvable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PoseEstimator(ABC):
    """Backend-agnostic single-person SMPL pose estimator.

    Contract: given a BGR frame (np.ndarray, HxWx3), return the SMPL body pose
    as a (72,) float32 axis-angle vector [root(3) | body(63) | hand(6)], or
    None if no person is detected.
    """

    @abstractmethod
    def estimate(self, frame_bgr):
        """np.ndarray (72,) float32 axis-angle, or None."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__


class RompEstimator(PoseEstimator):
    """Adapts romp.ROMP -> PoseEstimator. ROMP's output is already axis-angle."""

    def __init__(self, romp_model, name="ROMP"):
        self._model = romp_model
        self._name = name

    @property
    def name(self):
        return self._name

    def estimate(self, frame_bgr):
        with torch.no_grad():
            out = self._model(frame_bgr)
        if out is not None and out.get("smpl_thetas", None) is not None \
                and len(out["smpl_thetas"]) > 0:
            return np.asarray(out["smpl_thetas"][0], dtype=np.float32).reshape(-1)
        return None


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

    def __init__(self, model, device, crop_size=224, scale=1.0):
        self._model = model
        self._device = device
        self._crop_size = int(crop_size)
        self._scale = float(scale)
        from pare.utils.geometry import rotation_matrix_to_angle_axis
        self._rotmat_to_aa = rotation_matrix_to_angle_axis
        self._mean = torch.tensor(self._IMAGENET_MEAN, device=device).view(3, 1, 1)
        self._std = torch.tensor(self._IMAGENET_STD, device=device).view(3, 1, 1)

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
        patch = cv2.warpAffine(img, trans, (int(dst), int(dst)),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        t = torch.from_numpy(patch).to(self._device).float().permute(2, 0, 1) / 255.0
        return (t - self._mean) / self._std

    def estimate(self, frame_bgr):
        inp = self._crop_and_normalize(frame_bgr).unsqueeze(0)   # (1,3,H,W)
        with torch.no_grad():
            out = self._model(inp)
            rotmat = out["pred_pose"].reshape(-1, 3, 3)          # (24,3,3) rotmat
            aa = self._rotmat_to_aa(rotmat).reshape(-1)          # (72,) axis-angle
        return aa.detach().cpu().numpy().astype(np.float32)


def _build_pare_model(pare_cfg: str, pare_ckpt: str, device):
    """Instantiate the PARE network from its config + checkpoint, in eval mode.

    Mirrors PARETester._build_model + _load_pretrained_model (pare/core/tester.py)
    but without the multi-person tracker / video pipeline: we only need the raw
    network forward on a pre-cropped frame.
    """
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
        model = _construct_pare(PARE, hparams, device)
        ckpt = torch.load(pare_ckpt, map_location=device)["state_dict"]
        # Checkpoint keys are prefixed with "model." (LightningModule); strip.
        state = {k.replace("model.", "", 1): v for k, v in ckpt.items()
                 if k.startswith("model.")}
        model.load_state_dict(state, strict=False)
    finally:
        os.chdir(_prev_cwd)
    model.eval()
    return model


def _construct_pare(PARE, hparams, device):
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


def _build_romp_model(args):
    """Construct romp.ROMP with the ONNX/CUDA/TensorRT backend selected by args.

    Returns (model, label) where label describes the active backend for logging.

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
    return model, label


def build_estimator(args, device) -> PoseEstimator:
    """Factory: construct the selected PoseEstimator backend. Keeps main() and
    the source class free of any backend-specific imports/branching."""
    if args.estimator == "romp":
        model, label = _build_romp_model(args)
        return RompEstimator(model, name=label)

    # PARE backend. (--onnx/--no-onnx are ROMP-only and simply don't apply here.)
    if "--onnx" in sys.argv or "--no-onnx" in sys.argv:
        print("[INIT] --onnx/--no-onnx only apply to the ROMP backend; ignored for PARE.")
    # PARE's dependency stack pulls in a second OpenMP runtime (libomp alongside
    # torch's libiomp5md), which aborts on Windows unless duplicates are allowed.
    # The ROMP path does not trigger this, so we only set it for PARE, and only
    # if the user hasn't already chosen a value.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    print(f"[INIT] Initializing PARE from {args.pare_ckpt} ...")
    model = _build_pare_model(args.pare_cfg, args.pare_ckpt, device)
    print("[INIT] PARE ready.")
    return PareEstimator(model, device, crop_size=args.pare_crop_size, scale=args.pare_scale)
