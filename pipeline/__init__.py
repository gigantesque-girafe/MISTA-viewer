"""
pipeline: single-responsibility stages for the ROMP/PARE -> MISTA -> C++/SIBR
VR streaming server, re-exported below. The orchestrator (`LiveVRSource`)
lives in render_webcam.py and wires these together:

    FrameSource       -> next BGR frame (webcam/video), end-of-stream signal
    PoseEstimator     -> raw (72,) SMPL pose or None            (ROMP/PARE)
    PoseProcessor     -> One-Euro smoothing + missing handling  (72,) or None
    MistaPoseAdapter  -> (72,) pose -> MISTA deformer camera
    MistaRenderer     -> camera -> deformed 5-tuple attribute tensors
    SourceWindow      -> the Python OpenCV preview of the driving frame
"""

from pipeline.filters import OneEuroFilter, RotationOneEuroFilter
from pipeline.frame_source import FrameSource
from pipeline.estimators import (
    PoseEstimator,
    RompEstimator,
    BevEstimator,
    PareEstimator,
    build_estimator,
)
from pipeline.processor import PoseProcessor
from pipeline.adapter import MistaPoseAdapter, pose_to_camera_fields
from pipeline.renderer import MistaRenderer, RENDER_ITER
from pipeline.window import SourceWindow
from pipeline.config import load_mista_config

__all__ = [
    "OneEuroFilter",
    "RotationOneEuroFilter",
    "FrameSource",
    "PoseEstimator",
    "RompEstimator",
    "BevEstimator",
    "PareEstimator",
    "build_estimator",
    "PoseProcessor",
    "MistaPoseAdapter",
    "pose_to_camera_fields",
    "MistaRenderer",
    "RENDER_ITER",
    "SourceWindow",
    "load_mista_config",
]
