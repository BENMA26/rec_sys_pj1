"""
构建 Annoy 向量索引脚本
"""
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.retriever import IndexBuilder
from src.models.features import SparseFeature, DenseFeature, SequenceFeature

def build_index():
    """构建 Annoy 向量索引"""

    # ========== 1. 配置 ==========
    model_path = 'experiments/two_tower/checkpoints/last.ckpt'
    embedding_dim = 128  # 模型输出向量维度
    n_trees = 50  # Annoy 树的数量，越多越精确

    # ========== 2. 准备数据 ==========
    print("准备数据...")

    # 用户数据（示例）
    user_data = pd.DataFrame({
        'user_id': range(10000),
        'age': np.random.randint(0, 10, 10000),
        'gender': np.random.randint(0, 2, 10000),
        'city': np.random.randint(0, 100, 10000),
        'user_score': np.random.rand(10000),
        'click_history': [str(list(np.random.randint(0, 5000, 20))) for _ in range(10000)],
    })

    # 物品数据（示例）
    item_data = pd.DataFrame({
        'item_id': range(50000),
        'category': np.random.randint(0, 20, 50000),
        'brand': np.random.randint(0, 500, 50000),
        'price': np.random.rand(50000) * 1000,
        'item_score': np.random.rand(50000),
    })

    # ========== 3. 特征配置 ==========
    # 用户特征
    user_features_cols = {
        'sparse': ['user_id', 'age', 'gender', 'city'],
        'dense': ['user_score'],
        'sequence': ['click_history'],
    }

    # 物品特征
    item_features_cols = {
        'sparse': ['item_id', 'category', 'brand'],
        'dense': ['price', 'item_score'],
        'sequence': [],
    }

    # ========== 4. 初始化构建器 ==========
    builder = IndexBuilder(
        model_path=model_path,
        embedding_dim=embedding_dim,
        metric='angular',  # 余弦相似度
        n_trees=n_trees,
    )

    # ========== 5. 构建用户索引 ==========
    builder.build_user_index(
        user_data=user_data,
        user_features_cols=user_features_cols,
        batch_size=256,
        save_path='experiments/annoy/user_index.ann',
    )

    # ========== 6. 构建物品索引 ==========
    builder.build_item_index(
        item_data=item_data,
        item_features_cols=item_features_cols,
        batch_size=256,
        save_path='experiments/annoy/item_index.ann',
    )

    print("\n✅ Annoy 索引构建完成！")
    print(f"用户索引: experiments/annoy/user_index.ann")
    print(f"物品索引: experiments/annoy/item_index.ann")

def test_index():
    """测试 Annoy 检索"""
    from src.utils.retriever import Retriever

    print("\n" + "="*60)
    print("测试 Annoy 检索")
    print("="*60)

    # 加载检索器
    retriever = Retriever(
        item_index_path='experiments/annoy/item_index.ann',
        item_mapping_path='experiments/annoy/item_index_mapping.pkl',
        embedding_dim=128,
        metric='angular',
    )

    # 生成随机查询向量（实际使用时应该是用户向量）
    query_vector = np.random.randn(128).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)  # 归一化

    # 检索 top-100 物品
    results = retriever.search(
        query_vector=query_vector,
        top_k=100,
        search_k=-1,  # 自动设置
    )

    # 打印结果
    print(f"\n检索到 {len(results)} 个物品:")
    for i, result in enumerate(results[:10], 1):
        print(f"{i}. Item {result['item_id']}: "
              f"distance={result['distance']:.4f}, "
              f"similarity={result['similarity']:.4f}")

    # 批量检索测试
    print("\n批量检索测试...")
    query_vectors = np.random.randn(10, 128).astype(np.float32)
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)

    batch_results = retriever.batch_search(
        query_vectors=query_vectors,
        top_k=50,
    )

    print(f"批量检索完成，共 {len(batch_results)} 个查询")
    print(f"每个查询返回 {len(batch_results[0])} 个结果")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        type=str,
        default='build',
        choices=['build', 'test', 'both'],
        help='运行模式'
    )
    args = parser.parse_args()

    if args.mode == 'build':
        build_index()
    elif args.mode == 'test':
        test_index()
    elif args.mode == 'both':
        build_index()
        test_index()
