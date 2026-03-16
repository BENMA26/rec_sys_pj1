#!/bin/bash
# 构建 Annoy 向量索引示例
# 用法: bash scripts/run_build_index.sh

set -e

# ---------- 构建索引 ----------
python -m scripts.build_index \
    --mode build \
    --model_path experiments/two_tower_feature/checkpoints/last.ckpt \
    --embedding_dim 128 \
    --n_trees 50 \
    --metric angular \
    --user_index_path experiments/annoy/user_index.ann \
    --item_index_path experiments/annoy/item_index.ann \
    --item_mapping_path experiments/annoy/item_index_mapping.pkl \
    --batch_size 256

# ---------- 测试检索 ----------
# python -m scripts.build_index \
#     --mode test \
#     --item_index_path experiments/annoy/item_index.ann \
#     --item_mapping_path experiments/annoy/item_index_mapping.pkl \
#     --embedding_dim 128 \
#     --metric angular \
#     --top_k 100

# ---------- 构建 + 测试 ----------
# python -m scripts.build_index \
#     --mode both \
#     --model_path experiments/two_tower_feature/checkpoints/last.ckpt \
#     --embedding_dim 128 \
#     --n_trees 50 \
#     --top_k 100
