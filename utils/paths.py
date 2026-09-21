"""Central, machine-independent path resolution for MISTA.

Everything that used to be a hardcoded absolute path (``C:/Users/travu/...``,
``H:/dataMISTA/...``, ``/src/...``, ``/data/...``) or a fragile cwd-relative
literal (``body_models/...``, ``submodules/...``) should route through here so a
fresh machine only needs a few environment variables set (see ``.env.example``).

Environment variables honoured:
    MISTA_ROOT         repo root (defaults to this file's repo, auto-detected)
    MISTA_DATA_ROOT    dataset root (ZJU / AIST / Neuman / PeopleSnapshot)
    MISTA_BODY_MODELS  SMPL body-model dir (defaults to <MISTA_ROOT>/body_models)

Hydra/OmegaConf configs use the built-in ``${oc.env:VAR,default}`` resolver for
the same variables; this module is the Python-side twin.
"""

import os

# Repo root: honour MISTA_ROOT, else derive from this file (utils/ -> repo root).
# Mirrors the pattern already used in pipeline/config.py.
REPO_ROOT = os.environ.get("MISTA_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


def repo_path(*parts):
    """Join ``parts`` under the repo root, giving an absolute, cwd-independent path."""
    return os.path.join(REPO_ROOT, *parts)


def body_models_dir():
    """SMPL body-model directory (MISTA_BODY_MODELS, else <repo>/body_models)."""
    return os.environ.get("MISTA_BODY_MODELS") or repo_path("body_models")


def body_models_path(*parts):
    """Join ``parts`` under the body-model directory."""
    return os.path.join(body_models_dir(), *parts)


def data_root():
    """Dataset root (MISTA_DATA_ROOT). Returns None if unset — callers should
    either require it or fall back to config-provided paths."""
    return os.environ.get("MISTA_DATA_ROOT")
