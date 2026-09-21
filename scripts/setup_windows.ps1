<#
  setup_windows.ps1  —  One-shot environment build for the MISTA real-time VR demo.

  Prereqs (install manually first — these need admin rights):
    * NVIDIA driver + CUDA Toolkit 12.8
    * Visual Studio 2019 Build Tools (MSVC + CMake) AND VS Community 2022
    * Miniconda (on PATH)
    * git

  Run from an "x64 Native Tools Command Prompt for VS 2019" so `cl` is on PATH,
  OR let this script source vcvars64.bat (edit $VcVars below if your path differs).

  Usage:
    powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
    powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1 -EnvName mista -CudaArch 120
#>
param(
  [string]$EnvName  = "mista",
  # SM architecture for tiny-cuda-nn: RTX 50-series = 120, RTX 40 = 89, RTX 30 = 86, RTX 1080 = 61.
  [string]$CudaArch = "120",
  [string]$VcVars   = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

Step "Sanity: nvcc + cl must be present"
try { nvcc --version } catch { throw "nvcc not found. Install CUDA Toolkit 12.8 and re-open the shell." }
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
  if (Test-Path $VcVars) {
    Step "Initializing MSVC via vcvars64.bat"
    cmd /c "`"$VcVars`" && set" | ForEach-Object {
      if ($_ -match "^(.*?)=(.*)$") { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
    }
  } else {
    throw "cl (MSVC) not found and vcvars64.bat missing at $VcVars. Open an x64 Native Tools prompt, or pass -VcVars."
  }
}
$env:DISTUTILS_USE_SDK = "1"

Step "Create/refresh conda env '$EnvName' (python 3.10)"
conda create -y -n $EnvName python=3.10 ipython
# Activate within this process
conda activate $EnvName

Step "Install PyTorch (CUDA 12.8 wheels) — pinned, do not change"
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

Step "Install Python requirements"
pip install -r requirements-vr.txt

Step "Build CUDA submodules (--no-build-isolation)"
git submodule update --init submodules/diff-gaussian-rasterization submodules/simple-knn
pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/simple-knn --no-build-isolation

Step "Install tiny-cuda-nn (arch $CudaArch)"
$env:TCNN_CUDA_ARCHITECTURES = $CudaArch
pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch" --no-build-isolation

Step "Persist runtime env var (avoids OpenMP runtime clash)"
[Environment]::SetEnvironmentVariable("KMP_DUPLICATE_LIB_OK", "TRUE", "User")

Write-Host "`nSetup complete. Next:" -ForegroundColor Green
Write-Host "  1) copy .env.example to .env and edit paths"
Write-Host "  2) scripts\fetch_assets.ps1   (download weights + SMPL)"
Write-Host "  3) scripts\run_demo.ps1       (launch producer + VR viewer)"
