#!/bin/bash
#SBATCH --job-name=mista
#SBATCH --partition=insa-gpu
#SBATCH --gres=gpu:1
#SBATCH --output=mista/cout_%x_%A.txt
#SBATCH --error=mista/cerr_%x_%A.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
# Singularity & bind
# All host-side paths come from the environment (see .env.example). Override by
# exporting these before sbatch, e.g.
#   export MISTA_SIF=/path/to/mista.sif MISTA_DATA_ROOT=/data/zju MISTA_ROOT=$PWD
SIF="${MISTA_SIF:?Set MISTA_SIF to your Singularity image path}"

# Bind mount the host dataset dir (MISTA_DATA_ROOT) as /data/ inside the container.
BIND_DATA="${MISTA_DATA_ROOT:?Set MISTA_DATA_ROOT to your dataset dir}:/data/"

# Bind mount the host source dir (MISTA_ROOT, default: this repo) as /src/.
BIND_SRC="${MISTA_ROOT:-$(cd "$(dirname "$0")" && pwd)}:/src/"

# Paramètres uniques
WANDB_PROJECT="${WANDB_PROJECT:-mista}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
mkdir -p out

echo "===> WANDB project: ${WANDB_PROJECT}"


if [[ -z "${WANDB_API_KEY}" ]]; then
  echo "!! WANDB_API_KEY non défini dans l'environnement. Abandon."
  exit 1
fi


srun singularity run --nv \
  --bind "${BIND_DATA}" \
  --bind "${BIND_SRC}" \
  "${SIF}" \
  /bin/bash -lc "
    set -e
    export WANDB_API_KEY='${WANDB_API_KEY}'
    export WANDB_PROJECT='${WANDB_PROJECT}'

    # Conda env
    . /opt/conda/bin/activate 3dgs-avatar

    python3 -m pip install --no-cache-dir matplotlib seaborn scikit-learn
    # python3 -m pip uninstall -y hilbert hilbertcurve || true
    # python3 -m pip install --no-cache-dir hilbertcurve==2.0.5

    # Sanity check GPU
    command -v nvidia-smi && nvidia-smi -L || true

    echo 'Starting training (TT)...'
    cd /src/ && python3 train5d_mars.py \
      dataset=migs_multi_zju_5d_mars \
      option=iter50k \
      pose_correction=none \
      migs.type=tt5d \
      migs.use_mars=false \
      exp_root='${WANDB_PROJECT}' \
      wandb_project='${WANDB_PROJECT}'

    echo 'End training.'
  "
