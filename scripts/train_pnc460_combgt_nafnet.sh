#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
GPU_ID="${1:-0}"
python3 trainer_PNNP_SID.py -f runfiles/PNC460/PNNP_on_CombGT_NAFNet_stage1.yml --mode train --gpu "${GPU_ID}"
python3 trainer_PNNP_SID.py -f runfiles/PNC460/PNNP_on_CombGT_NAFNet_stage2.yml --mode train --gpu "${GPU_ID}"
