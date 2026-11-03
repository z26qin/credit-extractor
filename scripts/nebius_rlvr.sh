#!/usr/bin/env bash
# Step 3, RLVR on the teacher's pre-labels, on one Nebius GPU VM. The user approved one run on an
# H100 on 2026-11-02, with a cap of $18. Run on the VM as its default user:
#
#     HF_TOKEN=... MAX_RLVR_HOURS=2 bash nebius_rlvr.sh
#
# HF_TOKEN comes from the environment, never from this file. The VM is created, stopped and
# deleted from outside, with the Nebius CLI (docs/next_steps.md), because Nebius may restart a VM
# that shuts itself down. Every stage resumes from the Hub, so running this again continues.
set -euo pipefail
: "${HF_TOKEN:?set HF_TOKEN}"
BRANCH="${BRANCH:-m4-small-loop}"
REPO="$HOME/credit-extractor"

if [ ! -d "$REPO" ]; then
  git clone -q -b "$BRANCH" https://github.com/z26qin/credit-extractor.git "$REPO"
fi
cd "$REPO"
git fetch -q origin "$BRANCH" && git checkout -q "$BRANCH" && git pull -q --ff-only
git log -1 --oneline

# uv builds both venvs, as the notebooks do: vLLM's torch pin stays out of training's venv
if [ ! -x /tmp/uv-env/bin/uv ]; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv > /dev/null
  python3 -m venv /tmp/uv-env && /tmp/uv-env/bin/pip install -q uv
fi
UV=/tmp/uv-env/bin/uv
if [ ! -x /tmp/vllm-env/bin/python ]; then
  $UV venv -q /tmp/vllm-env --python 3.12
  $UV pip install -q --python /tmp/vllm-env/bin/python --torch-backend=auto \
    -r requirements-gpu.txt -r requirements.txt wrapt
fi
if [ ! -x /tmp/train-env/bin/python ]; then
  $UV venv -q /tmp/train-env --python 3.12
  $UV pip install -q --python /tmp/train-env/bin/python --torch-backend=auto \
    -r requirements-train.txt -r requirements.txt
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/tmp/vllm-env/bin/python -c "import torch, vllm; print('vllm', vllm.__version__, '| torch',
torch.__version__, 'gpu', torch.cuda.is_available())"
/tmp/train-env/bin/python -c "import torch, transformers, peft, trl; print('train: torch',
torch.__version__, 'gpu', torch.cuda.is_available(), '| trl', trl.__version__)"

# the training venv runs the stages too: it has everything they import
exec /tmp/train-env/bin/python -m src.real_rlvr --max-rlvr-hours "${MAX_RLVR_HOURS:-2}"
