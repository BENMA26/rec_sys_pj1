#!/bin/bash
# 推荐系统推理示例
# 用法: bash scripts/run_inference.sh

set -e

python -m scripts.inference \
    --recall_model_path  experiments/two_tower_feature/checkpoints/last.ckpt \
    --ranking_model_path experiments/rank/mmoe_jd/checkpoints/last.ckpt \
    --annoy_index_path   experiments/annoy/item_index.ann \
    --annoy_mapping_path experiments/annoy/item_index_mapping.pkl \
    --embedding_dim 128 \
    --top_k_recall 100 \
    --top_k_rank 10 \
    --metric angular
