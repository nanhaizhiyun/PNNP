#!/usr/bin/env bash
# PNNP 本体训练：从暗帧学习噪声代理（需先准备 SonyA7S2/bias 暗帧目录）
set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${1:-0}"
python3 trainer_PNNP_proxy.py \
  -f runfiles/SonyA7S2/PNNP_proxy_train.yml \
  --gpu "${GPU_ID}"
