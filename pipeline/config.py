"""MISTA config loading (same compose pattern as motion-driven-render.py)."""

import os

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Repo root (parent of this `pipeline/` package) — where `configs/` lives.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_mista_config(load_ckpt: str, identity: int):
    configs_dir = os.path.join(_ROOT, "configs")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=configs_dir):
        config = compose(config_name="config_5d")

    OmegaConf.set_struct(config, False)
    config.mode = "test"                 # test split -> real ZJU frames + per-identity Jtr
    config.appearance_identity = int(identity)
    config.load_ckpt = load_ckpt
    config.dataset.preload = False
    config.wandb_disable = True
    if config.get("export", None) is not None:
        config.export.enable = False     # never write PLY snapshots
    return config
