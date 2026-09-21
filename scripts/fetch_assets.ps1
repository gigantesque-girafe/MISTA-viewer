<#
  fetch_assets.ps1  —  Download model weights for the MISTA VR demo.

  Downloads what is publicly fetchable (MISTA checkpoints, HybrIK weights) via
  gdown, into the locations the code/.env expect. SMPL body models and PARE
  weights require registration/their own tooling — this script prints those steps.

  Run AFTER setup_windows.ps1, with the conda env active and .env filled in.
    powershell -ExecutionPolicy Bypass -File scripts/fetch_assets.ps1
    powershell -ExecutionPolicy Bypass -File scripts/fetch_assets.ps1 -What mista,hybrik
#>
param(
  [string[]]$What = @("mista", "hybrik")
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# --- load .env (KEY=VALUE lines) ---
$envFile = Join-Path $RepoRoot ".env"
if (Test-Path $envFile) {
  Get-Content $envFile | Where-Object { $_ -match "^\s*[^#].*=" } | ForEach-Object {
    $k,$v = $_ -split "=", 2
    Set-Item -Path "env:$($k.Trim())" -Value $v.Trim()
  }
}
if (-not (Get-Command gdown -ErrorAction SilentlyContinue)) { pip install gdown }

function Get-Drive($id, $out) {
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $out) | Out-Null
  Write-Host "downloading $out" -ForegroundColor Cyan
  gdown "https://drive.google.com/uc?id=$id" -O $out
}

# Google Drive ids (from README.md release table)
$MISTA_CKPT_ID   = "1j2c7ZdcAfkJSGyjlgxKfHi7q6fhVR3m1"   # MISTA pretrained (TT5D)
$HYBRIK_FILES_ID = "1un9yAGlGjDooPwlnwFpJrbGHRiLaBNzV"   # HybrIK model_files.zip
$HYBRIK_R34_ID   = "19ktHbERz0Un5EzJYZBdzdzTrFyd9gLCx"   # HybrIK hybrik_res34.pth

if ($What -contains "mista") {
  $ckpt = if ($env:MISTA_CKPT) { $env:MISTA_CKPT } else { Join-Path $RepoRoot "checkpoints/ckpt50000.pth" }
  Get-Drive $MISTA_CKPT_ID $ckpt
  Write-Host "MISTA checkpoint -> $ckpt (set MISTA_CKPT in .env to this)" -ForegroundColor Green
}

if ($What -contains "hybrik") {
  $hz = Join-Path $RepoRoot "submodules/HybrIK/model_files.zip"
  Get-Drive $HYBRIK_FILES_ID $hz
  Expand-Archive -Force $hz (Join-Path $RepoRoot "submodules/HybrIK/")
  Get-Drive $HYBRIK_R34_ID (Join-Path $RepoRoot "submodules/HybrIK/pretrained_models/hybrik_res34.pth")
  Write-Host "HybrIK ready. Still need the neutral SMPL .pkl in model_files/ (see below)." -ForegroundColor Green
}

Write-Host "`n--- MANUAL steps (license-gated, cannot auto-download) ---" -ForegroundColor Yellow
Write-Host "SMPL body models: register at https://smpl.is.tue.mpg.de and place male/female/neutral"
Write-Host "  model.pkl under `$MISTA_BODY_MODELS`/smpl/{male,female,neutral}/ then run:"
Write-Host "     python extract_smpl_parameters.py"
Write-Host "HybrIK also needs the neutral SMPL .pkl at submodules/HybrIK/model_files/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
Write-Host "PARE weights: run submodules/PARE's own fetch script (get_pare_weights) if using --estimator pare."
