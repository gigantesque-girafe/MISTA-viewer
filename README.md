# MISTA: Compact Multi-Identity Structure-Aware Tensorized Avatars

## Overview

This repository contains the official implementation of **MISTA** and **MISTA-AR**, two tensorized Gaussian avatar representations designed for efficient multi-identity human modeling, animation, and motion transfer.

MISTA introduces a structure-aware Tensor Train factorization of 3D Gaussian Splatting parameters, enabling compact multi-identity representations while preserving rendering quality.

MISTA-AR extends MISTA through adaptive rank selection using MARS, allowing automatic identification and pruning of less important tensor components during training.

The framework supports:

* Multi-identity avatar modeling
* Motion transfer between subjects
* Novel pose synthesis
* Adaptive rank selection
* Tensor Train and CP-based representations

---

## Installation

### Clone Repository

```bash
git clone https://github.com/OumaimaBadi/MISTA.git
cd MISTA
```

---

## Environment

A ready-to-use Singularity image is provided.

Download links:

| Resource                   | Link        |
| -------------------------- | ----------- |
| Singularity image          | [Download](https://drive.google.com/file/d/1qlvbzLRj3HTjnJF7AG4D-blOPADvj9g-/view?usp=drive_link) |
| MISTA pretrained model    | [Download](https://drive.google.com/file/d/1j2c7ZdcAfkJSGyjlgxKfHi7q6fhVR3m1/view?usp=drive_link) |
| MISTA no Hilbert pretrained model    |[Download](https://drive.google.com/file/d/1MQIz64GXHD_JzIzi8bGdXZkqZeNF8NxS/view?usp=drive_link) |
| MISTA-AR pretrained model | [Download](https://drive.google.com/file/d/1fgibV4yhadyJQmyRdX6qZKsL_t8yD8eR/view?usp=drive_link) |
| MIGS Rank-10 checkpoint   | [Download](https://drive.google.com/file/d/1bCRP93p9esL6pheGFOYEogQhkRoLnDLQ/view?usp=drive_link) |
| MIGS Rank-100 checkpoint  | [Download](https://drive.google.com/file/d/1FtSojegVHJCrx5OJbCoxA_uqyVL-y9Xc/view?usp=drive_link) |

---

## SMPL Setup

Download SMPL models from the official SMPL website and place them under:

```text
body_models/
└── smpl/
    ├── male/
    ├── female/
    └── neutral/
```

Then run:

```bash
python extract_smpl_parameters.py
```

---

## Dataset Preparation

Due to dataset licensing restrictions, we cannot publicly distribute the preprocessed ZJU-MoCap data.

Users should download the original dataset and follow the preprocessing procedure described in the ARAH repository:

https://github.com/taconite/arah-release

The resulting processed data can then be used directly with the MISTA framework.

Please prepare datasets according to the configuration files located in:

```text
configs/dataset/
```
---

## Training

Before training, modify the following variables inside the provided SLURM script:

```bash
SIF=${MISTA_SIF}

BIND_DATA=${MISTA_DATA_ROOT}:/data/

BIND_SRC=${MISTA_ROOT}:/src/

WANDB_API_KEY=<your_wandb_key>
```

---

### MISTA Training

```bash
python train5d_mars.py \
dataset=migs_multi_zju_5d \
migs.type=tt5d \
migs.use_mars=false
```

---

### MISTA-AR Training

```bash
python train5d_mars.py \
dataset=migs_multi_zju_5d_mars \
migs.type=tt5d \
migs.use_mars=true
```

---

### MIGS (CP) Training

```bash
python train5d_mars.py \
migs.type=cp
```

---

## Rendering

### Evaluation

```shell
python render.py mode=test

# On my Windows
python render.py mode=test wandb_disable=True appearance_identity=2 load_ckpt="%MISTA_CKPT%"

```

### Novel View Synthesis

```bash
python render.py \
mode=test \
dataset.test_mode=view \
dataset=migs_multi_zju_5d_mars \
opt.iterations=50000 \
migs.type=tt5d \
migs.use_mars=false \
appearance_identity=0 // 0:386, 1:387, 2:377, 3:392, 4:315, 5:394, 6:393, 7:390
load_ckpt=%MISTA_CKPT%
```

### Novel Pose Synthesis

```bash
python render.py mode=predict \
dataset=migs_multi_zju_5d_mars \
opt.iterations=50000 \
migs.type=tt5d \
migs.use_mars=false \
dataset.predict_seq= 0 // 0,1,2,3, to try differnt dances
appearance_identity=0 // 0:386, 1:387, 2:377, 3:392, 4:315, 5:394, 6:393, 7:390
load_ckpt=%MISTA_CKPT%

#on my rtx1080
python render.py mode=predict dataset=migs_multi_zju_5d_mars opt.iterations=50000 migs.use_mars=false dataset.predict_seq=0 appearance_identity=2 load_ckpt="%MISTA_CKPT%" wandb_disable=True
```

---

## Motion Transfer

Motion transfer can be performed by applying the motion sequence of a source identity to a target identity while preserving the target appearance.

Example:

```bash
python render.py mode=predict
```

The target identity is reconstructed using its learned appearance and animated using the source motion sequence.

---

## Pretrained Models

The following checkpoints are provided:

| Model     | Description                                     |
| --------- | ----------------------------------------------- |
| MIGS-R10  | CP decomposition with rank 10                   |
| MIGS-R100 | CP decomposition with rank 100                  |
| MISTA     | Tensor Train representation                     |
| MISTA-AR  | Tensor Train representation with adaptive ranks |

Download links will be added upon release.

---

## Citation

If you use this repository, pretrained models, datasets, or any part of this work in your research, please cite:

```bibtex
@inproceedings{badi2026mista,
  title={MISTA: Compact Multi-Identity Structure-Aware Tensorized Avatars},
  author={Badi, Oumaima and Jiang, Xiaoran and Morin, Luce and Sjöström, Mårten},
  booktitle={Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}

@inproceedings{badi2026mista_ar,
  title={Une représentation 3DGS de rangs faibles auto-sélectionnés pour les avatars multi-identités},
  author={Badi, Oumaima and Jiang, Xiaoran and Morin, Luce and Sjöström, Mårten},
  booktitle={Actes de CORESA},
  address={Nantes, France},
  year={2026}
}
```

---

## License

This repository is intended for research and academic purposes.

Please refer to the LICENSE file for additional details.

---

## Acknowledgements

This project builds upon ideas, datasets, and open-source implementations from:

* 3DGS-Avatar
* 3D Gaussian Splatting
* MIGS
* 4D-Humans
* AIST++
* ZJU-MoCap

We sincerely thank the authors of these works for making their research and resources publicly available.


---
## Run VR Application

> **Architecture note.** This is a **PC-VR** demo, not a standalone Quest app. A Python
> process (pose estimation → MISTA Gaussian deform) hands GPU memory to a native C++
> SIBR OpenXR viewer over **CUDA-IPC** (same machine, same GPU); the viewer renders to
> the headset through **Meta Quest Link's Windows OpenXR runtime**. It therefore runs on
> **Windows + NVIDIA only** — it cannot be shipped to the Meta Store or run from a Linux
> container.

### Quick start (portable demo kit)

1. Install the admin-level prereqs once: NVIDIA driver + **CUDA Toolkit 12.8**, **VS 2019
   Build Tools** + **VS Community 2022**, **Miniconda**, **git**, and **Meta Quest Link**.
2. Build the Python environment (automates the whole recipe below):
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
   ```
3. Configure paths — copy `.env.example` to `.env` and edit it (see **Path configuration**).
4. Download weights: `powershell -ExecutionPolicy Bypass -File scripts\fetch_assets.ps1`
   (then place SMPL models and run `python extract_smpl_parameters.py` as instructed).
5. Launch producer + viewer together:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1 -Identity 2          # VR
   powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1 -Desktop             # desktop mirror
   ```

### Path configuration

All machine-specific paths are read from environment variables — **no source edits needed**.
Copy `.env.example` to `.env` and set:

| Variable | Meaning |
| --- | --- |
| `MISTA_ROOT` | repo root (auto-detected if unset) |
| `MISTA_DATA_ROOT` | dataset root (ZJU / AIST / Neuman / PeopleSnapshot) |
| `MISTA_BODY_MODELS` | SMPL body-model dir (default `<MISTA_ROOT>/body_models`) |
| `MISTA_CKPT` | trained avatar checkpoint (`.pth`) used by the examples below |

Hydra configs resolve the same vars via `${oc.env:VAR}`. In the example commands below,
`%MISTA_CKPT%` / `%MISTA_DATA_ROOT%` are those environment variables (cmd.exe syntax; use
`$env:MISTA_CKPT` in PowerShell).

### Setup the headset
- Open Meta Horizon Link software, then goes into Settings > General > Set Meta Link as default OpenXR. You may need administrator right for this
-


## Setup the PC

There are two entry points for the interactive viewer:

* **`render_vr.py`** — plays back a *pre-recorded* ZJU/AIST pose sequence (Hydra-based)
  and streams the deformed Gaussians to the SIBR viewer (VR or desktop mirror).
* **`render_webcam.py`** — drives the avatar *live* from a **video file or webcam** using
  a pose estimator (ROMP / BEV / PARE / HybrIK), then streams to the same SIBR viewer.

First activate the environment (once per terminal):

```shell
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" "%USERPROFILE%\miniconda3"   REM adjust to your Miniconda install
conda activate mista
set KMP_DUPLICATE_LIB_OK=TRUE
```

### `render_vr.py` — pre-recorded sequence playback

General formula:

```shell
python render_vr.py mode=predict \
  dataset=<DATASET> migs.type=<TYPE> migs.use_mars=false \
  dataset.predict_seq=<SEQ> appearance_identity=<ID> \
  load_ckpt="<CKPT>" wandb_disable=True [+gaussians_vr.port=<PORT>]
```

Working examples (from my machine — adjust the checkpoint):

```shell
# MISTA (Tensor Train), identity 2, first dance sequence
python render_vr.py mode=predict dataset=migs_multi_zju_5d_mars migs.type=tt5d migs.use_mars=false dataset.predict_seq=0 appearance_identity=2 wandb_disable=True load_ckpt="%MISTA_CKPT%"

# MIGS (CP decomposition)
python render_vr.py mode=predict dataset=migs opt.iterations=50000 migs.type=cp migs.use_mars=false dataset.predict_seq=0 appearance_identity=2 load_ckpt="%MISTA_CKPT%" wandb_disable=True
```

Key Hydra overrides:

| Override | Choices / example | Meaning |
| --- | --- | --- |
| `mode` | `predict` / `test` | `predict` = novel-pose playback; `test` = eval view |
| `dataset` | `migs_multi_zju_5d_mars`, `migs` | dataset/config group; `migs` for CP checkpoints |
| `migs.type` | `tt5d`, `tt5d_color_split`, `cp` | representation (TT for MISTA, `cp` for MIGS) |
| `migs.use_mars` | `false` / `true` | adaptive-rank (MISTA-AR) variant |
| `dataset.predict_seq` | `0,1,2,3` | which motion/dance sequence to play back |
| `appearance_identity` | `0`–`7` | which learned identity to render (see note below) |
| `load_ckpt` | path to `.pth` | trained avatar checkpoint |
| `wandb_disable` | `True` | skip Weights & Biases logging |
| `+gaussians_vr.port` | `6012` (default) | TCP port the SIBR viewer connects to |

### `render_webcam.py` — live video / webcam drive

General formula:

```shell
python render_webcam.py --source <video|webcam> [--video <PATH> | --camera-index <N>] \
  --identity <ID> --load-ckpt "<CKPT>" --estimator <romp|bev|pare|hybrik> \
  --port <PORT> [--trt --trt-fp16 --trt-lib-dir "<TensorRT lib>"] [flags...]
```

Working examples (from my machine — adjust the checkpoint):

```shell
# video file, ROMP estimator (default smoothing on)
python render_webcam.py --source video --video "%MISTA_DATA_ROOT%/taichi-cut.mp4" --identity 2 --load-ckpt "%MISTA_CKPT%" --port 6012

# raw pose, no One-Euro smoothing
python render_webcam.py --source video --video "%MISTA_DATA_ROOT%/taichi-cut.mp4" --identity 2 --load-ckpt "%MISTA_CKPT%" --port 6012 --no-smooth

# run ROMP every 2nd frame (cheaper) with a color-split model
python render_webcam.py --source video --video "%MISTA_DATA_ROOT%/taichi-cut.mp4" --identity 3 --load-ckpt "%MISTA_CKPT%" --port 6012 --romp-every-n 2

# TensorRT (FP16) + root motion, identity 3
python render_webcam.py --source video --video "%MISTA_DATA_ROOT%\video\hiit-spider.mp4" --identity 3 --load-ckpt "%MISTA_CKPT%" --estimator romp --trt --trt-fp16 --trt-lib-dir "C:\Users\travu\TensorRT\TensorRT-8.6.1.6\lib" --root-motion --root-scale 1.0

# live webcam
python render_webcam.py --source webcam --camera-index 0 --identity 3 --load-ckpt "%MISTA_CKPT%" --estimator romp --trt --no-trt-fp16 --trt-lib-dir "C:\Users\travu\TensorRT\TensorRT-8.6.1.6\lib" --realtime off
```

Key CLI flags (see `python render_webcam.py --help` for the full list, incl. head-pose and retarget options):

| Flag | Choices / default | Meaning |
| --- | --- | --- |
| `--source` | `webcam` / `video` | live camera or a video file |
| `--video` | path | input video (with `--source video`) |
| `--camera-index` | `0` | webcam index (with `--source webcam`) |
| `--identity` | `0`–`7` | target MISTA identity (see note below) |
| `--load-ckpt` | path (required) | MISTA 8-identity checkpoint `.pth` |
| `--estimator` | `romp` / `bev` / `pare` / `hybrik` | pose-estimation backend |
| `--port` | `6012` | TCP port the SIBR viewer connects to |
| `--romp-every-n` | `1` | run the estimator every N frames, reuse in between |
| `--trt` / `--no-trt` | off | ROMP via TensorRT EP vs plain CUDA |
| `--trt-fp16` / `--no-trt-fp16` | on | TensorRT FP16 (faster) vs FP32 (safer) |
| `--trt-lib-dir` | path | TensorRT 8.6 `lib` folder |
| `--smooth` / `--no-smooth` | on | One-Euro pose smoothing |
| `--realtime` | `auto` / `on` / `off` | drop stale frames to keep a live source realtime |
| `--root-motion` | off | apply root translation from the source |
| `--no-window` | off | stream to the C++ viewer only (no source-frame window) |

Note:
- Appearance / identity index: 0:386, 1:387, 2:377, 3:392, 4:315, 5:394, 6:393, 7:390

In second terminal: OpenXR Application for VR Viewer
```shell
# Options (in normal shell)
$env:V42_TIMING=1
$env:V42_INTEROP=1

# Run VR App
submodules\sibr-core\install\bin\SIBR_remoteGaussianOpenXRv4_2_app_rwdi.exe --ip 127.0.0.1 --port 6012

# Run ViewerUI on computer screen
submodules\sibr-core\install\bin\SIBR_remoteGaussianDesktopV42_app_rwdi.exe --ip 127.0.0.1 --port 6012
```

Desktop viewer camera (mouse trackball, hover the rendered image — not the window chrome):

| input | action |
| --- | --- |
| Left-drag | orbit the avatar (roll if you start in the outer border) |
| Right-drag | pan (dolly if you start in the outer border) |
| Scroll | zoom in / out — no keyboard key may be held |
| `Y` | toggle FPS/WASD mode (the old keyboard navigation) and back |
| `P` / `Left` / `Right` | pause-resume the animation / step +/-1 frame (camera stays live) |

The `Camera ...` ImGui panel has the same mode dropdown plus FoV, near/far and camera save/load.

* Surveille le GPU
```
nvidia-smi dmon
```

### Running double viewer
Both tiles share one camera: drag inside either one and both viewpoints move together
(same controls as the single-tile table above).
```shell
# Low resolution
& "submodules\sibr-core\install\bin\SIBR_remoteGaussianDesktopV42_app_rwdi.exe" --port 6012 --port2 6013 --width 512 --height 512 --label1 TT5D --label2 CP

# high resolution
& "submodules\sibr-core\install\bin\SIBR_remoteGaussianDesktopV42_app_rwdi.exe" --port 6012 --port2 6013 --width 512 --height 512 --label1 TT5D --label2 CP

#Terminal 1 — TT5D
python render_vr.py mode=predict dataset=migs_multi_zju_5d_mars migs.type=tt5d migs.use_mars=false dataset.predict_seq=0 appearance_identity=2 wandb_disable=True load_ckpt="%MISTA_CKPT%" +gaussians_vr.port=6012

#Terminal 2 — CP-R100
python render_vr.py mode=predict dataset.predict_seq=0 dataset=migs opt.iterations=50000 migs.type=cp migs.use_mars=false appearance_identity=2 load_ckpt="%MISTA_CKPT%" wandb_disable=True +gaussians_vr.port=6013
```