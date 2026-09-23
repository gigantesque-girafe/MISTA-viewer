"""
vr_viewer.ipc  —  Pipeline-agnostic CUDA-IPC transport primitives.

Extracted verbatim (behaviour-preserving) from render_vr_v1.py. Nothing in this
module knows anything about MISTA, Gaussians-from-MIGS, deformation or Hydra — it
is pure CUDA-IPC plumbing shared by every pipeline that wants to feed the SIBR
remote OpenXR viewer.

  * GaussianAttrBuffer  — fixed-capacity contiguous CUDA float32 buffer.
  * GaussianIPCManager  — cudaIpc mem/event handle owner + frame signalling.
"""

import ctypes
import os
import sys
import logging

import torch

log = logging.getLogger("vr_viewer.ipc")


# ─────────────────────────────────────────────────────────────────────────────
# Inline CUDA IPC utilities
# ─────────────────────────────────────────────────────────────────────────────

class _IpcMem(ctypes.Structure):
    """ctypes mirror of CUDA's `cudaIpcMemHandle_t` (64 opaque bytes)."""
    _fields_ = [("raw", ctypes.c_byte * 64)]


class _IpcEvt(ctypes.Structure):
    """ctypes mirror of CUDA's `cudaIpcEventHandle_t` (64 opaque bytes)."""
    _fields_ = [("raw", ctypes.c_byte * 64)]


def load_cuda_runtime() -> ctypes.CDLL:
    """Load the CUDA runtime shared library, preferring CUDA 11.8.

    @return: loaded ctypes.CDLL for the CUDA runtime.
    @throws OSError: if no matching CUDA runtime library can be found/loaded.
    """
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


def get_ipc_offset(tensor: torch.Tensor) -> int:
    """Compute the byte offset of a CUDA tensor's data pointer from its cudaMalloc block base.

    @param tensor: CUDA torch.Tensor.
    @return: int byte offset.
    @throws RuntimeError: if the CUDA driver call `cuMemGetAddressRange` fails.
    @note: Tries `tensor.untyped_storage()._share_cuda_()` first; falls back
        to a direct `cuMemGetAddressRange` driver call via ctypes.
    """
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
# Pre-allocated CUDA buffer for all Gaussian attributes
# ─────────────────────────────────────────────────────────────────────────────

class GaussianAttrBuffer:
    """Fixed-capacity contiguous CUDA float32 buffer holding all per-Gaussian
    attributes needed by the C++ rasterizer.

    Layout (each field occupies N_max rows; only rows [0, N_live) are valid):
      xyz:     [N_max, 3]
      feat:    [N_max, K]
      R_bwd:   [N_max, 9]   (backward rotation, flat 3x3)
      opacity: [N_max]
      cov3D:   [N_max, 6]   (posed 3D covariance, upper triangle
                             [00,01,02,11,12,22] — the cov3D_precomp path,
                             matching render.py's compute_cov3D_python=True)

    @note: Wire v2 (magic "V42E") sends the full covariance instead of the
        previous scale[3]+rot[4] tail, preserving the non-orthonormal
        scale/shear that LBS blending puts into rotation_precomp (which a
        quaternion would discard), so the VR splats match render.py.
    """

    def __init__(self, N_max: int, K: int, device: torch.device):
        """Allocate the contiguous CUDA buffer and its per-field views.

        @param N_max: buffer capacity in Gaussians.
        @param K: per-Gaussian view-independent feature width.
        @param device: CUDA torch device to allocate on.
        """
        self.N_max  = N_max
        self.K      = K
        self.device = device

        stride = 3 + K + 9 + 1 + 6   # floats per Gaussian (K + 19)
        total  = N_max * stride

        torch.cuda.init()
        torch.cuda.synchronize()
        self.buffer = torch.empty(total, dtype=torch.float32, device=device).contiguous()

        self.ptr        = self.buffer.data_ptr()
        self.ipc_offset = get_ipc_offset(self.buffer)
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
        self._cov_v   = self.buffer[s4:s5].view(N_max, 6)

        log.info(
            "GaussianAttrBuffer: N_max=%d  K=%d  total=%.2f MB  ptr=0x%x  ipc_offset=%d",
            N_max, K, total * 4 / 1024 / 1024, self.ptr, self.ipc_offset,
        )

    def write(self, xyz, feat, R_bwd, opacity, cov3D):
        """Copy one frame's per-Gaussian attribute tensors into the buffer's leading rows.

        @param xyz: [N, 3] CUDA tensor.
        @param feat: [N, K] CUDA tensor.
        @param R_bwd: [N, 9] CUDA tensor.
        @param opacity: [N, 1] (or [N]) CUDA tensor.
        @param cov3D: [N, 6] CUDA tensor.
        @throws RuntimeError: if N exceeds `self.N_max`.
        """
        N = xyz.shape[0]
        if N > self.N_max:
            raise RuntimeError(
                f"N={N} exceeds IPC buffer capacity N_max={self.N_max}. "
                "Increase N_max passed to the VR source."
            )
        self._xyz_v[:N].copy_(xyz, non_blocking=True)
        self._feat_v[:N].copy_(feat, non_blocking=True)
        self._rbwd_v[:N].copy_(R_bwd, non_blocking=True)
        self._opa_v[:N].copy_(opacity.squeeze(-1), non_blocking=True)
        self._cov_v[:N].copy_(cov3D, non_blocking=True)


# ─────────────────────────────────────────────────────────────────────────────
# CUDA IPC handle manager
# ─────────────────────────────────────────────────────────────────────────────

class GaussianIPCManager:
    """Owns the CUDA IPC memory/event handles for one `GaussianAttrBuffer` and
    signals frame readiness to the C++ viewer."""

    def __init__(self, attr_buf: GaussianAttrBuffer):
        """@brief Create the IPC memory handle and the data-ready/read-complete event handles.
        @param attr_buf: the GaussianAttrBuffer whose CUDA memory is shared.
        @throws RuntimeError: if any underlying CUDA IPC call fails.
        """
        self._cuda = load_cuda_runtime()
        self._buf  = attr_buf
        self.mem_handle = self._create_mem_handle()
        self.data_ready_evt_handle, self._data_ready_evt_ptr = self._create_evt_handle()
        self.read_complete_evt_handle, self._read_complete_evt_ptr = self._create_evt_handle()
        log.info("CUDA IPC handles created.")

    def _create_mem_handle(self) -> _IpcMem:
        """@brief Create the CUDA IPC memory handle for the buffer's device pointer.
        @return: _IpcMem handle.
        @throws RuntimeError: if `cudaIpcGetMemHandle` fails.
        """
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
        """@brief Create an interprocess, non-timing CUDA event and its IPC handle.
        @return: tuple `(_IpcEvt handle, ctypes.c_void_p event pointer)`.
        @throws RuntimeError: if event creation or `cudaIpcGetEventHandle` fails.
        """
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
        """@brief Synchronize the CUDA stream and record the data-ready IPC event."""
        torch.cuda.synchronize()
        self._cuda.cudaEventRecord(self._data_ready_evt_ptr, ctypes.c_void_p(0))

    def wait_read_complete(self):
        """@brief Block until the C++ viewer signals it has finished reading this buffer."""
        self._cuda.cudaEventSynchronize(self._read_complete_evt_ptr)
