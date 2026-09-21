<#
  run_demo.ps1  —  One-command launcher for the MISTA real-time VR avatar demo.

  Starts the C++ viewer (prebuilt SIBR exe) and the Python producer together,
  wired to the same port. Loads paths from .env.

  VR (Meta Quest Link must be running, Link set as default OpenXR):
    powershell -ExecutionPolicy Bypass -File scripts/run_demo.ps1 -Identity 2
  Desktop mirror (no headset):
    powershell -ExecutionPolicy Bypass -File scripts/run_demo.ps1 -Desktop
  Drive from a video instead of webcam:
    powershell -ExecutionPolicy Bypass -File scripts/run_demo.ps1 -Source video -Video D:\clips\taichi.mp4
#>
param(
  [int]$Identity   = 2,
  [int]$Port       = 6012,
  [ValidateSet("romp","bev","pare","hybrik")][string]$Estimator = "romp",
  [ValidateSet("webcam","video")][string]$Source = "webcam",
  [string]$Video   = $null,
  [switch]$Desktop,          # launch the desktop-mirror viewer instead of the OpenXR one
  [switch]$NoTrt,            # disable TensorRT/FP16 acceleration
  [string]$EnvName = "mista"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# --- load .env ---
$envFile = Join-Path $RepoRoot ".env"
if (Test-Path $envFile) {
  Get-Content $envFile | Where-Object { $_ -match "^\s*[^#].*=" } | ForEach-Object {
    $k,$v = $_ -split "=", 2
    Set-Item -Path "env:$($k.Trim())" -Value $v.Trim()
  }
}
if (-not $env:MISTA_ROOT)          { $env:MISTA_ROOT = $RepoRoot }
if (-not $env:KMP_DUPLICATE_LIB_OK){ $env:KMP_DUPLICATE_LIB_OK = "TRUE" }
if (-not $env:MISTA_CKPT) { throw "MISTA_CKPT is not set. Copy .env.example to .env and set it (or run fetch_assets.ps1)." }

conda activate $EnvName

$bin = Join-Path $RepoRoot "submodules/sibr-core/install/bin"
$viewer = if ($Desktop) {
  Join-Path $bin "SIBR_remoteGaussianDesktopV42_app_rwdi.exe"
} else {
  Join-Path $bin "SIBR_remoteGaussianOpenXRv4_2_app_rwdi.exe"
}
if (-not (Test-Path $viewer)) { throw "Viewer not found: $viewer  (build sibr-core or unpack the prebuilt viewer bundle)." }

# --- producer args ---
$py = @(
  "motion-drive-render-v43.py",
  "--source", $Source,
  "--identity", $Identity,
  "--estimator", $Estimator,
  "--load-ckpt", $env:MISTA_CKPT,
  "--port", $Port
)
if ($Source -eq "video") {
  if (-not $Video) { throw "-Source video requires -Video <path>" }
  $py += @("--video", $Video)
}
if (-not $NoTrt) { $py += @("--trt", "--trt-fp16") }

Write-Host "Launching viewer: $viewer --ip 127.0.0.1 --port $Port" -ForegroundColor Cyan
Start-Process -FilePath $viewer -ArgumentList @("--ip","127.0.0.1","--port","$Port")

Start-Sleep -Seconds 2
Write-Host "Launching producer: python $($py -join ' ')" -ForegroundColor Cyan
python @py
