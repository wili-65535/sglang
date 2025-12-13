#!/bin/bash

# Wan2.2-T2V-A14B启动脚本（对标cache-dit示例配置）
# 参考：cache-dit/examples/pipeline/run_wan_2.2.py

echo "清理旧进程..."
pkill -9 -f sglang
sleep 3

echo "启动sglang服务（Cache-DiT配置对标cache-dit示例）..."

# Cache-DiT配置说明：
# 主Transformer（高噪声专家，处理前30%步数）：
#   - WARMUP=4: 预热4步
#   - MC=8: 最多连续缓存8步
# 次Transformer（低噪声专家，处理后70%步数）：
#   - SECONDARY_WARMUP=2: 预热2步
#   - SECONDARY_MC=20: 最多连续缓存20步

SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_WARMUP=4 \
SGLANG_CACHE_DIT_MC=8 \
SGLANG_CACHE_DIT_RDT=0.24 \
SGLANG_CACHE_DIT_FN=1 \
SGLANG_CACHE_DIT_BN=0 \
SGLANG_CACHE_DIT_SECONDARY_WARMUP=2 \
SGLANG_CACHE_DIT_SECONDARY_MC=20 \
SGLANG_CACHE_DIT_SECONDARY_RDT=0.24 \
SGLANG_CACHE_DIT_SECONDARY_FN=1 \
SGLANG_CACHE_DIT_SECONDARY_BN=0 \
SGLANG_CACHE_DIT_SCM_PRESET=medium \
SGLANG_CACHE_DIT_SCM_POLICY=dynamic \
sglang serve \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --text-encoder-cpu-offload \
  --pin-cpu-memory \
  --num-gpus 4 \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --port 8000

echo "服务已启动在 http://localhost:8000"
