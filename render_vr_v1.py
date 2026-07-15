"""
render_vr_v1.py  —  MISTA VR streaming server (CUDA IPC zero-copy transport).

Talks to the SIBR remote OpenXR application v4.2 (SIBR_remoteGaussianOpenXRv4_2_app.exe)
using the SAME wire protocol as 3dgs-v2's render_vr_v4_2.py. The later stages of the
pipeline (per-frame deformation, Color-MLP feature extraction, CUDA-IPC packing and the
double-buffered handshake) are preserved verbatim from that reference; the ONLY thing that
changes for MISTA is the *start* of the pipeline:

  3dgs-v2 : canonical Gaussians live directly in `scene.gaussians` and are loaded with a
            single `scene.load_checkpoint(...)`.
  MISTA   : the canonical Gaussians are stored *factorized* (Tensor-Train / CP MIGS). We
            therefore reproduce render.py's checkpoint-loading logic (shape extraction +
            core allocation + weight load + converter load), then DECODE the canonical
            Gaussians for the chosen identity ONCE via `scene.update_gaussians_from_migs`.
            After that, `scene.gaussians` holds an ordinary canonical avatar and the rest
            of the loop is identical to render_vr_v4_2.py.

Because a `predict` run fixes a single identity, the MIGS decode is done once up-front
(the factorization output does not change per frame — only the deformation does).

Startup handshake (port 6012) and per-frame protocol: see render_vr_v4_2.py docstring —
unchanged here.

Usage:
  conda activate <mista-env>
  python render_vr_v1.py \\
      mode=predict dataset.predict_seq=0 \\
      dataset=zju_377_mono wandb_disable=True \\
      load_ckpt="./exp/<run>/ckpt<iter>.pth" \\
      appearance_identity=0

  Then launch:
      SIBR_remoteGaussianOpenXRv4_2_app.exe --port 6012
"""

import ctypes
import io
import os
import socket
import struct
import sys
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general_utils import build_rotation, build_scaling_rotation, strip_symmetric

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mista_vr_v1")

_PORT         = 6012
_LOG_INTERVAL = 60
_HANDSHAKE    = b"V42E"   # wire v2 (cov3D[6] tail); matches SIBR_remoteGaussianOpenXRv4_2_app
_N_MAX        = 50_000   # IPC buffer capacity; must be >= actual Gaussian count.
                          # MISTA avatars are typically larger than 3dgs-v2's, so this is
                          # bumped up; increase further if you hit the capacity error.

# See render_vr_v4_2.py — pace Python to a steady rate so the two processes don't both
# continuously submit CUDA work and thrash the WDDM scheduler.
_TARGET_FPS = 30
_TARGET_FRAME_SECONDS = 1.0 / _TARGET_FPS

# Axis-convention correction for the predict-sequence source data (chest-to-back depth
# comes out along world Y instead of Z). See render_vr_v4_2.py for the full derivation.
# If the avatar renders upside down, negate row 1 (height/Y) entries.
_AXIS_FIX = [[-0.9601507782936096,  0.15131592750549316, -0.2349766492843628],
             [-0.27107205986976624, -0.29949113726615906,  0.914781391620636],
             [ 0.06804757565259933,  0.9420236349105835,   0.32857415080070496]]


# ─────────────────────────────────────────────────────────────────────────────
# Inline CUDA IPC utilities 
# ─────────────────────────────────────────────────────────────────────────────

class _IpcMem(ctypes.Structure):
    _fields_ = [("raw", ctypes.c_byte * 64)]

class _IpcEvt(ctypes.Structure):
    _fields_ = [("raw", ctypes.c_byte * 64)]


def _load_cuda_runtime() -> ctypes.CDLL:
    """Load CUDA runtime DLL/SO. Tries CUDA 11.8 first (this project's version)."""
    if sys.platform == "win32":
        for name in ["cudart64_118.dll", "cudart64_110.dll", "cudart64_12.dll"]:
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if cuda_path:
            for name in ["cudart64_118.dll", "cudart64_110.dll", "cudart64_12.dll"]:
                p = os.path.join(cuda_path, "bin", name)
                if os.path.isfile(p):
                    try:
                        return ctypes.CDLL(p)
                    except OSError:
                        continue
        raise OSError(
            "Cannot load CUDA runtime DLL. Tried cudart64_118/110/12.dll. "
            "Ensure CUDA_PATH is set or the DLL is on PATH."
        )
    else:
        for name in ["libcudart.so", "libcudart.so.11", "libcudart.so.12"]:
            try:
                return ctypes.CDLL(name)
            except OSError:
                continue
        raise OSError("Cannot load libcudart.so")


def _get_ipc_offset(tensor: torch.Tensor) -> int:
    """Byte offset of tensor.data_ptr() from the base of its cudaMalloc block."""
    try:
        return tensor.untyped_storage()._share_cuda_()[3]
    except (RuntimeError, AttributeError):
        pass
    if sys.platform == "win32":
        try:
            nvcuda = ctypes.CDLL("nvcuda.dll")
        except OSError:
            nvcuda = ctypes.WinDLL("nvcuda")
    else:
        nvcuda = ctypes.CDLL("libcuda.so.1")
    base = ctypes.c_uint64(0)
    size = ctypes.c_size_t(0)
    ret = nvcuda.cuMemGetAddressRange(
        ctypes.byref(base), ctypes.byref(size), ctypes.c_uint64(tensor.data_ptr())
    )
    if ret != 0:
        raise RuntimeError(f"cuMemGetAddressRange failed: {ret}")
    return tensor.data_ptr() - base.value


# ─────────────────────────────────────────────────────────────────────────────
# Pre-allocated CUDA buffer for all Gaussian attributes  (identical to v4_2)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianAttrBuffer:
    """
    Fixed-capacity contiguous CUDA float32 buffer holding all per-Gaussian
    attributes needed by the C++ rasterizer.

    Layout (each field occupies N_max rows; only rows [0, N_live) are valid):
      xyz:     [N_max, 3]
      feat:    [N_max, K]
      R_bwd:   [N_max, 9]   (backward rotation, flat 3x3)
      opacity: [N_max]
      scale:   [N_max, 3]
      rot:     [N_max, 4]   (unit quaternion)
    """

    def __init__(self, N_max: int, K: int, device: torch.device):
        self.N_max  = N_max
        self.K      = K
        self.device = device

        stride = 3 + K + 9 + 1 + 6   # floats per Gaussian (K + 19): xyz|feat|R_bwd|opa|cov3D
        total  = N_max * stride

        torch.cuda.init()
        torch.cuda.synchronize()
        self.buffer = torch.empty(total, dtype=torch.float32, device=device).contiguous()

        self.ptr        = self.buffer.data_ptr()
        self.ipc_offset = _get_ipc_offset(self.buffer)
        self.device_idx = self.buffer.device.index or torch.cuda.current_device()

        s = [0]
        def _next(n):
            start = s[0]; s[0] += n; return start, s[0]

        s0, s1  = _next(N_max * 3)
        s1_, s2 = _next(N_max * K)
        s2_, s3 = _next(N_max * 9)
        s3_, s4 = _next(N_max * 1)
        s4_, s5 = _next(N_max * 6)

        self._xyz_v   = self.buffer[s0:s1].view(N_max, 3)
        self._feat_v  = self.buffer[s1:s2].view(N_max, K)
        self._rbwd_v  = self.buffer[s2:s3].view(N_max, 9)
        self._opa_v   = self.buffer[s3:s4]              # flat [N_max]
        self._cov_v   = self.buffer[s4:s5].view(N_max, 6)   # cov3D_precomp (Sigma upper-tri)

        log.info(
            "GaussianAttrBuffer: N_max=%d  K=%d  total=%.2f MB  ptr=0x%x  ipc_offset=%d",
            N_max, K, total * 4 / 1024 / 1024, self.ptr, self.ipc_offset,
        )

    def write(self, xyz, feat, R_bwd, opacity, cov):
        N = xyz.shape[0]
        if N > self.N_max:
            raise RuntimeError(
                f"N={N} exceeds IPC buffer capacity N_max={self.N_max}. "
                "Increase _N_MAX in render_vr_v1.py."
            )
        self._xyz_v[:N].copy_(xyz, non_blocking=True)
        self._feat_v[:N].copy_(feat, non_blocking=True)
        self._rbwd_v[:N].copy_(R_bwd, non_blocking=True)
        self._opa_v[:N].copy_(opacity.squeeze(-1), non_blocking=True)
        self._cov_v[:N].copy_(cov, non_blocking=True)


# ─────────────────────────────────────────────────────────────────────────────
# CUDA IPC handle manager  (identical to v4_2)
# ─────────────────────────────────────────────────────────────────────────────

class GaussianIPCManager:
    def __init__(self, attr_buf: GaussianAttrBuffer):
        self._cuda = _load_cuda_runtime()
        self._buf  = attr_buf
        self.mem_handle = self._create_mem_handle()
        self.data_ready_evt_handle, self._data_ready_evt_ptr = self._create_evt_handle()
        self.read_complete_evt_handle, self._read_complete_evt_ptr = self._create_evt_handle()
        log.info("CUDA IPC handles created.")

    def _create_mem_handle(self) -> _IpcMem:
        mh  = _IpcMem()
        ret = self._cuda.cudaIpcGetMemHandle(
            ctypes.byref(mh), ctypes.c_void_p(self._buf.ptr)
        )
        if ret != 0:
            raise RuntimeError(
                f"cudaIpcGetMemHandle failed (error {ret}). "
                "On Windows this requires CUDA 11.2+, driver 452.39+, and Windows 10 v2004+."
            )
        return mh

    def _create_evt_handle(self):
        cudaEventInterprocess  = 0x04
        cudaEventDisableTiming = 0x02
        flags = cudaEventInterprocess | cudaEventDisableTiming

        evt_ptr = ctypes.c_void_p()
        ret = self._cuda.cudaEventCreateWithFlags(
            ctypes.byref(evt_ptr), ctypes.c_uint(flags)
        )
        if ret != 0:
            raise RuntimeError(f"cudaEventCreateWithFlags failed (error {ret})")
        if not evt_ptr.value:
            raise RuntimeError("cudaEventCreateWithFlags returned NULL")

        eh  = _IpcEvt()
        ret = self._cuda.cudaIpcGetEventHandle(ctypes.byref(eh), evt_ptr)
        if ret != 0:
            error_names = {
                400: "cudaErrorInvalidResourceHandle (event does not support IPC)",
                801: "cudaErrorNotSupported (IPC events not supported on this system)",
            }
            raise RuntimeError(
                f"cudaIpcGetEventHandle failed: {error_names.get(ret, f'error {ret}')}"
            )
        return eh, evt_ptr

    def record_data_ready(self):
        torch.cuda.synchronize()
        self._cuda.cudaEventRecord(self._data_ready_evt_ptr, ctypes.c_void_p(0))

    def wait_read_complete(self):
        self._cuda.cudaEventSynchronize(self._read_complete_evt_ptr)


# ─────────────────────────────────────────────────────────────────────────────
# SH basis (trace-friendly) — identical to render_vr_v4_2.py
# ─────────────────────────────────────────────────────────────────────────────

def _sh_bases(deg: int, dirs: torch.Tensor) -> torch.Tensor:
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    cols = [torch.full((dirs.shape[0],), C0, dtype=dirs.dtype, device=dirs.device)]
    if deg >= 1:
        x = dirs[:, 0]; y = dirs[:, 1]; z = dirs[:, 2]
        cols.extend([-C1 * y, C1 * z, -C1 * x])
        if deg >= 2:
            C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
                  -1.0925484305920792, 0.5462742152960396]
            xx, yy, zz = x*x, y*y, z*z
            xy, yz, xz = x*y, y*z, x*z
            cols.extend([C2[0]*xy, C2[1]*yz, C2[2]*(2*zz-xx-yy), C2[3]*xz, C2[4]*(xx-yy)])
            if deg >= 3:
                C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
                       0.3731763325901154, -0.4570457994644658, 1.445305721320277,
                      -0.5900435899266435]
                cols.extend([
                    C3[0]*y*(3*xx-yy), C3[1]*xy*z, C3[2]*y*(4*zz-xx-yy),
                    C3[3]*z*(2*zz-3*xx-3*yy), C3[4]*x*(4*zz-xx-yy),
                    C3[5]*z*(xx-yy), C3[6]*x*(xx-3*yy),
                ])
    return torch.stack(cols, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# TorchScript wrapper — mirrors MISTA's ColorMLP.compose_input ordering.
#
# MISTA compose_input concatenation order (models/texture/texture.py):
#   base_features | xyz_norm? | cov? | normal? | dir_embed(SH) | non_rigid?
# The SH dir_embed is inserted after the geometric features and BEFORE the
# non-rigid feature block, so pre_dim = feature_dim + xyz? + cov? + normal?.
# (MISTA's ColorMLP has no latent branch — unlike 3dgs-v2 — so none is emitted.)
# ─────────────────────────────────────────────────────────────────────────────

class ColorMLPV4Wrapper(nn.Module):
    def __init__(self, color_mlp, pre_dim: int):
        super().__init__()
        self._mlp        = color_mlp.mlp
        self._activation = color_mlp.color_activation
        self._sh_degree  = int(color_mlp.sh_degree)
        self._cano       = bool(getattr(color_mlp, 'cano_view_dir', False))
        self._pre_dim    = int(pre_dim)

    def forward(self, feat_no_view, xyz, cam_center, R_bwd):
        if self._sh_degree > 0:
            dir_pp = xyz - cam_center.unsqueeze(0)
            if self._cano:
                dir_pp = torch.bmm(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
            dir_pp_norm = F.normalize(dir_pp, dim=-1)
            basis       = _sh_bases(self._sh_degree, dir_pp_norm)
            dir_embed   = basis[:, 1:]
            feat_pre    = feat_no_view[:, :self._pre_dim]
            feat_post   = feat_no_view[:, self._pre_dim:]
            features    = torch.cat([feat_pre, dir_embed, feat_post], dim=1)
        else:
            features = feat_no_view
        return self._activation(self._mlp(features))


def _pre_sh_feat_dim(color_mlp) -> int:
    """Column count before the SH dir_embed insertion point (base + xyz? + cov? + normal?)."""
    d = int(color_mlp.cfg.feature_dim)
    if getattr(color_mlp, 'use_xyz',    False): d += 3
    if getattr(color_mlp, 'use_cov',    False): d += 6
    if getattr(color_mlp, 'use_normal', False): d += 3
    return d


def _view_indep_feat_dim(color_mlp) -> int:
    from models.texture.texture import ColorMLP
    if not isinstance(color_mlp, ColorMLP):
        raise TypeError("render_vr_v1 requires a ColorMLP texture model (cfg.model.texture.name=mlp).")
    K = int(color_mlp.cfg.feature_dim)
    if getattr(color_mlp, 'use_xyz',    False): K += 3
    if getattr(color_mlp, 'use_cov',    False): K += 6
    if getattr(color_mlp, 'use_normal', False): K += 3
    if getattr(color_mlp, 'non_rigid_dim', 0) > 0: K += color_mlp.non_rigid_dim
    return K


def export_color_mlp(color_mlp, K: int, device: torch.device) -> bytes:
    pre_dim = _pre_sh_feat_dim(color_mlp)
    wrapper = ColorMLPV4Wrapper(color_mlp, pre_dim).to(device).eval()
    N_ex   = 64
    opts   = dict(dtype=torch.float32, device=device)
    f_ex   = torch.zeros(N_ex, K, **opts)
    xyz_ex = torch.zeros(N_ex, 3, **opts)
    cam_ex = torch.zeros(3,       **opts)
    r_ex   = torch.eye(3, **opts).unsqueeze(0).expand(N_ex, -1, -1).contiguous()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (f_ex, xyz_ex, cam_ex, r_ex),
                                 strict=False, check_trace=False)
    buf = io.BytesIO()
    torch.jit.save(traced, buf)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — mirrors MISTA ColorMLP.compose_input (view-independent part)
# ─────────────────────────────────────────────────────────────────────────────

def extract_view_indep_features(color_mlp, gaussians, camera) -> torch.Tensor:
    features = gaussians.get_features.squeeze(-1)
    if getattr(color_mlp, 'use_xyz', False):
        aabb     = color_mlp.metadata["aabb"]
        xyz_norm = aabb.normalize(gaussians.get_xyz, sym=True)
        features = torch.cat([features, xyz_norm], dim=1)
    if getattr(color_mlp, 'use_cov', False):
        features = torch.cat([features, gaussians.get_covariance()], dim=1)
    if getattr(color_mlp, 'use_normal', False):
        scale  = gaussians._scaling
        rot    = build_rotation(gaussians._rotation)
        normal = torch.gather(rot, dim=2,
            index=scale.argmin(1).reshape(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
        features = torch.cat([features, normal], dim=1)
    if getattr(color_mlp, 'non_rigid_dim', 0) > 0:
        features = torch.cat([features, gaussians.non_rigid_feature], dim=1)
    return features   # [N, K]


def extract_R_bwd(gaussians, cano_view_dir: bool) -> torch.Tensor:
    if cano_view_dir and hasattr(gaussians, 'fwd_transform'):
        T_fwd = gaussians.fwd_transform
        R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
        return R_bwd.reshape(-1, 9).contiguous()
    N   = gaussians.get_xyz.shape[0]
    dev = gaussians.get_xyz.device
    return torch.eye(3, dtype=torch.float32, device=dev)\
               .unsqueeze(0).expand(N, -1, -1).reshape(N, 9).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Socket helpers
# ─────────────────────────────────────────────────────────────────────────────

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("viewer disconnected")
        buf.extend(chunk)
    return bytes(buf)


# ─────────────────────────────────────────────────────────────────────────────
# Startup handshake  (identical wire format to v4_2)
# ─────────────────────────────────────────────────────────────────────────────

def do_handshake(conn, ipc_mgrs, attr_bufs, K, model_bytes):
    magic = _recv_exact(conn, 4)
    if magic != _HANDSHAKE:
        raise ValueError(f"Unexpected handshake magic: {magic!r} (expected {_HANDSHAKE!r})")

    log.info("Handshake OK — sending N_max=%d  K=%d  model=%d bytes (double-buffered).",
             attr_bufs[0].N_max, K, len(model_bytes))

    conn.sendall(struct.pack("<III", attr_bufs[0].N_max, K, len(model_bytes)))
    conn.sendall(model_bytes)
    conn.sendall(struct.pack("<i", attr_bufs[0].device_idx))  # int32

    for buf, mgr in zip(attr_bufs, ipc_mgrs):
        conn.sendall(struct.pack("<Q", buf.ipc_offset))            # uint64
        conn.sendall(bytes(mgr.mem_handle.raw))                    # 64 bytes
        conn.sendall(bytes(mgr.data_ready_evt_handle.raw))         # 64 bytes
        conn.sendall(bytes(mgr.read_complete_evt_handle.raw))      # 64 bytes

    log.info("IPC handles sent for both buffers. device=%d", attr_bufs[0].device_idx)


# ─────────────────────────────────────────────────────────────────────────────
# Server loop  (per-frame body identical to v4_2; canonical Gaussians already
# decoded from MIGS in main())
# ─────────────────────────────────────────────────────────────────────────────

def serve(port, scene, config, smpl_cams, iteration, model_bytes, K, attr_bufs, ipc_mgrs):
    converter  = scene.converter
    gaussians  = scene.gaussians            # canonical avatar, decoded from MIGS in main()
    color_mlp  = converter.texture
    cano       = bool(getattr(color_mlp, 'cano_view_dir', False))
    n_frames   = len(smpl_cams)
    anim_frame = 0

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    log.info("MISTA VR CUDA-IPC Gaussian server on 0.0.0.0:%d", port)
    log.info("Launch: SIBR_remoteGaussianOpenXRv4_2_app.exe --port %d", port)

    try:
        while True:
            conn, addr = srv.accept()
            log.info("C++ viewer connected from %s:%d", *addr)
            try:
                do_handshake(conn, ipc_mgrs, attr_bufs, K, model_bytes)

                pipeline_frame_id = 0
                buf_idx = 0
                axis_fix = torch.tensor(_AXIS_FIX, dtype=torch.float32, device=attr_bufs[0].device)
                avatar_center0 = None   # captured frame 0; pins avatar start to world origin

                while True:
                    t0 = time.perf_counter()

                    # Don't overwrite this buffer until C++ is done reading it.
                    tw = time.perf_counter()
                    ipc_mgrs[buf_idx].wait_read_complete()
                    wait_ms = (time.perf_counter() - tw) * 1000.0

                    cam = smpl_cams[anim_frame % n_frames]

                    # ── Phase 1: Deformation ──────────────────────────────────
                    t1 = time.perf_counter()
                    with torch.no_grad():
                        cam_c, _ = converter.pose_correction(cam, iteration)
                        deformed_pc, _ = converter.deformer(
                            gaussians, cam_c, iteration, compute_loss=False
                        )
                    torch.cuda.synchronize()
                    deform_ms = (time.perf_counter() - t1) * 1000.0

                    # ── Phase 2: View-independent feature extraction ──────────
                    t2 = time.perf_counter()
                    with torch.no_grad():
                        feat    = extract_view_indep_features(color_mlp, deformed_pc, cam_c)
                        R_bwd_f = extract_R_bwd(deformed_pc, cano)
                    feat_ms = (time.perf_counter() - t2) * 1000.0

                    # ── Phase 3: Pack into IPC buffer (GPU-to-GPU, non-blocking)
                    t3 = time.perf_counter()
                    xyz = deformed_pc.get_xyz @ axis_fix.T
                    if avatar_center0 is None:
                        avatar_center0 = xyz.mean(dim=0, keepdim=True).clone()
                    xyz     = xyz - avatar_center0
                    opacity = deformed_pc.get_opacity.squeeze(-1)
                    scale   = deformed_pc.get_scaling
                    # Build the full posed covariance (cov3D_precomp), matching render.py's
                    # get_covariance() path. rotation_precomp is the LBS forward transform
                    # (T_fwd[:3,:3] @ canonical R) — a general non-orthonormal affine that
                    # carries the bone-blend scale/shear. Converting it to a quaternion (as
                    # the old wire v1 did) discarded that shear; feeding Sigma directly
                    # preserves it. axis_fix is baked into the transform so the covariance
                    # matches the axis-fixed xyz (Sigma_world = fix @ A diag(s^2) A^T @ fix^T).
                    if hasattr(deformed_pc, 'rotation_precomp'):
                        rot_matrix = deformed_pc.rotation_precomp
                    else:
                        rot_matrix = build_rotation(deformed_pc.get_rotation)
                    A   = axis_fix.unsqueeze(0) @ rot_matrix          # [N,3,3] full affine
                    L   = build_scaling_rotation(scale, A)            # A @ diag(scale)
                    cov = strip_symmetric(L @ L.transpose(1, 2))      # [N,6] Sigma upper-tri
                    N   = xyz.shape[0]

                    attr_bufs[buf_idx].write(xyz, feat, R_bwd_f, opacity, cov)
                    pack_ms = (time.perf_counter() - t3) * 1000.0

                    if anim_frame == 0:
                        lo = xyz.amin(dim=0).tolist()
                        hi = xyz.amax(dim=0).tolist()
                        ctr = xyz.mean(dim=0).tolist()
                        log.info(
                            "[PLACEMENT] center=(%.3f, %.3f, %.3f)  "
                            "bbox_min=(%.3f, %.3f, %.3f)  bbox_max=(%.3f, %.3f, %.3f)  "
                            "extent=(%.3f, %.3f, %.3f)",
                            ctr[0], ctr[1], ctr[2],
                            lo[0], lo[1], lo[2], hi[0], hi[1], hi[2],
                            hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2],
                        )

                    # ── Phase 4: Sync + record data-ready event, announce frame
                    t4 = time.perf_counter()
                    ipc_mgrs[buf_idx].record_data_ready()
                    conn.sendall(struct.pack("<QII", pipeline_frame_id, buf_idx, N))
                    signal_ms = (time.perf_counter() - t4) * 1000.0

                    total_ms = (time.perf_counter() - t0) * 1000.0

                    remaining = _TARGET_FRAME_SECONDS - (time.perf_counter() - t0)
                    if remaining > 0:
                        time.sleep(remaining)

                    anim_frame += 1
                    pipeline_frame_id += 1
                    buf_idx = 1 - buf_idx

                    if anim_frame % _LOG_INTERVAL == 0 or anim_frame <= 3:
                        log.info(
                            "frame %4d  N=%d  K=%d  total=%.1fms"
                            "  wait=%.1f  deform=%.1f  feat=%.1f  pack=%.1f  signal=%.1f",
                            anim_frame, N, K, total_ms,
                            wait_ms, deform_ms, feat_ms, pack_ms, signal_ms,
                        )

            except (ConnectionError, BrokenPipeError, OSError) as e:
                log.info("Disconnected (%s) — waiting for reconnect.", e)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    except KeyboardInterrupt:
        log.info("Server stopped.")
    finally:
        srv.close()


# ─────────────────────────────────────────────────────────────────────────────
# MISTA checkpoint loading — reproduces render.py predict()'s "extra start steps".
# Builds the Scene, allocates TT/CP cores from the checkpoint shapes, loads weights
# and the converter state. Returns (scene, migs_type, appearance_id).
# ─────────────────────────────────────────────────────────────────────────────

def build_scene_from_checkpoint(config):
    from scene import GaussianModel, Scene

    load_ckpt = config.get("load_ckpt", None)
    if load_ckpt is None:
        raise ValueError("Please provide load_ckpt.")

    log.info("Loading checkpoint: %s", load_ckpt)
    tmp = torch.load(load_ckpt, map_location="cpu")
    sd  = tmp["migs_module_state_dict"]

    migs_type = tmp.get("migs_type", config.migs.type)
    log.info("Detected migs_type = %s", migs_type)
    config.migs.type = migs_type

    appearance_id = getattr(config, "appearance_identity", None)

    # CASE 1: CP / Tucker — plain checkpoint loading.
    if migs_type in ("cp", "tucker"):
        config.migs.skip_init_from_tensor = False
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.appearance_identity = appearance_id
        scene.eval()
        scene.load_checkpoint(load_ckpt)
        log.info("CP/Tucker checkpoint loaded")
        return scene, migs_type, appearance_id

    # CASE 2: TT (tt5d / tt / MARS-wrapped) — allocate cores from checkpoint shapes.
    has_mars_prefix = any(k.startswith("tensorized_model.tt.") for k in sd.keys())
    prefix = "tensorized_model.tt." if has_mars_prefix else ""
    if has_mars_prefix:
        log.info("Detected MARS-wrapped checkpoint")

    r1 = sd[f"{prefix}tt_tensor_gpu.0"].shape[-1]
    r2 = sd[f"{prefix}tt_tensor_gpu.1"].shape[-1]
    r3 = sd[f"{prefix}tt_tensor_gpu.2"].shape[-1]
    r4 = sd[f"{prefix}tt_tensor_gpu.3"].shape[-1]
    config.migs.rank = [1, r1, r2, r3, r4, 1]

    n_id = sd[f"{prefix}tt_tensor_gpu.0"].shape[1]
    n1   = sd[f"{prefix}tt_tensor_gpu.1"].shape[1]
    n2   = sd[f"{prefix}tt_tensor_gpu.2"].shape[1]
    n3   = sd[f"{prefix}tt_tensor_gpu.3"].shape[1]

    try:
        M_xyz = sd[f"{prefix}core4_xyz"].shape[1]
        M_scl = sd[f"{prefix}core4_scaling"].shape[1]
        M_rot = sd[f"{prefix}core4_rotation"].shape[1]
        M_dc  = sd[f"{prefix}core4_dc"].shape[1]
        M_rst = sd[f"{prefix}core4_rest"].shape[1]
        M_opa = sd[f"{prefix}core4_opacity"].shape[1]
        M = M_xyz + M_scl + M_rot + M_dc + M_rst + M_opa
    except KeyError:
        M_xyz, M_scl, M_rot, M_dc, M_rst, M_opa = 3, 3, 4, 1, 31, 1
        M = 43

    config.migs.tt_shape = [n_id, n1, n2, n3, M]
    config.migs.n_identities_ckpt = int(n_id)
    config.migs.use_mars = False
    config.migs.skip_init_from_tensor = True

    log.info("rank     = %s", config.migs.rank)
    log.info("tt_shape = %s", config.migs.tt_shape)

    if has_mars_prefix:
        sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}

    gaussians = GaussianModel(config.model.gaussian)
    scene = Scene(config, gaussians, config.exp_dir)
    scene.appearance_identity = appearance_id
    scene.eval()

    tt_module = scene.migs_module
    tt_module.tt_shape = tuple(config.migs.tt_shape)
    tt_module.tt_rank  = config.migs.rank

    device = "cuda"
    tt_module.tt_tensor_gpu = nn.ParameterList([
        nn.Parameter(torch.zeros(1,  n_id, r1, device=device)),
        nn.Parameter(torch.zeros(r1, n1,   r2, device=device)),
        nn.Parameter(torch.zeros(r2, n2,   r3, device=device)),
        nn.Parameter(torch.zeros(r3, n3,   r4, device=device)),
    ])
    tt_module.core4_xyz      = nn.Parameter(torch.zeros(r4, M_xyz, 1, device=device))
    tt_module.core4_scaling  = nn.Parameter(torch.zeros(r4, M_scl, 1, device=device))
    tt_module.core4_rotation = nn.Parameter(torch.zeros(r4, M_rot, 1, device=device))
    tt_module.core4_dc       = nn.Parameter(torch.zeros(r4, M_dc,  1, device=device))
    tt_module.core4_rest     = nn.Parameter(torch.zeros(r4, M_rst, 1, device=device))
    tt_module.core4_opacity  = nn.Parameter(torch.zeros(r4, M_opa, 1, device=device))

    G = n1 * n2 * n3
    tt_module.register_buffer("perm",     torch.arange(G, dtype=torch.long, device=device))
    tt_module.register_buffer("inv_perm", torch.arange(G, dtype=torch.long, device=device))
    log.info("TT cores allocated (G=%d)", G)

    missing, unexpected = tt_module.load_state_dict(sd, strict=False)
    if missing:
        log.info("Missing keys   : %s%s", missing[:5], '...' if len(missing) > 5 else '')
    if unexpected:
        log.info("Unexpected keys: %s%s", unexpected[:5], '...' if len(unexpected) > 5 else '')
    log.info("TT weights loaded")

    scene.converter.load_state_dict(tmp["converter_state"])
    log.info("Converter loaded")

    return scene, migs_type, appearance_id


# ─────────────────────────────────────────────────────────────────────────────
# Hydra entry point
# ─────────────────────────────────────────────────────────────────────────────

import hydra
from omegaconf import OmegaConf, DictConfig
import wandb


def collect_smpl_frames(dataset) -> list:
    cams = []
    for i in range(len(dataset)):
        cam = dataset[i]
        cam.data.pop('original_image', None)
        cam.data.pop('original_mask', None)
        cams.append(cam)
    log.info("Loaded %d animation frames.", len(cams))
    return cams


@hydra.main(version_base=None, config_path="configs", config_name="config_5d")
def main(config: DictConfig):
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False
    config.exp_dir = config.get("exp_dir") or os.path.join("./exp", config.exp_name)
    os.makedirs(config.exp_dir, exist_ok=True)

    if config.mode not in ("predict", "test"):
        config.mode = "predict"

    predict_dict = {0: "sameperson", 1: "TransferMotion", 2: "dance0",
                    3: "dance1", 4: "flipping", 5: "canonical"}
    config.suffix = "predict-" + predict_dict.get(config.dataset.get("predict_seq", 0), "dance0")

    wandb.init(
        mode="disabled" if config.get("wandb_disable", True) else None,
        name=f"{config.exp_name}-mista-vr-v1",
        project="mista-gaussians-vr",
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method="spawn"),
    )

    from utils.general_utils import fix_random
    fix_random(config.seed)

    # ── Extra MISTA start steps: load factorized checkpoint & build scene ─────
    scene, migs_type, appearance_id = build_scene_from_checkpoint(config)

    iteration = config.opt.iterations

    # ── Decode the canonical Gaussians from the MIGS factorization ONCE ───────
    # (predict fixes one identity; only the deformation changes per frame).
    smpl_cams = collect_smpl_frames(scene.test_dataset)
    if appearance_id is not None:
        identity_id = int(appearance_id)
    else:
        identity_id = int(getattr(smpl_cams[0], "person_id", 0))
    log.info("Decoding canonical Gaussians for identity %d ...", identity_id)
    with torch.no_grad():
        scene.update_gaussians_from_migs(identity_id)
    log.info("Canonical avatar ready: %d Gaussians.", scene.gaussians.get_xyz.shape[0])

    # ── Preserved 3dgs-v2 pipeline: Color-MLP export + IPC buffers ────────────
    color_mlp = scene.converter.texture
    device    = next(color_mlp.parameters()).device

    K = _view_indep_feat_dim(color_mlp)
    log.info("Feature dim K=%d. Tracing Color MLP...", K)
    model_bytes = export_color_mlp(color_mlp, K, device)
    log.info("TorchScript model: %d bytes.", len(model_bytes))

    N_max = int(config.get("gaussians_vr", {}).get("n_max", _N_MAX))
    log.info("Allocating double-buffered IPC buffers: N_max=%d  K=%d", N_max, K)
    attr_bufs = [GaussianAttrBuffer(N_max, K, device) for _ in range(2)]

    log.info("Creating CUDA IPC handles...")
    ipc_mgrs = [GaussianIPCManager(buf) for buf in attr_bufs]

    port = int(config.get("gaussians_vr", {}).get("port", _PORT))
    serve(port, scene, config, smpl_cams, iteration, model_bytes, K, attr_bufs, ipc_mgrs)


if __name__ == "__main__":
    main()
