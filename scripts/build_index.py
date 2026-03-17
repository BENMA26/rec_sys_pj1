"""
构建 Annoy 向量索引脚本
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.retriever import IndexBuilder, Retriever
from src.models.features import SparseFeature, DenseFeature, SequenceFeature


def parse_args():
    parser = argparse.ArgumentParser(description='构建/测试 Annoy 向量索引')

    parser.add_argument('--mode', type=str, default='build',
                        choices=['build', 'test', 'both'],
                        help='运行模式: build=构建索引, test=测试检索, both=两者都做')

    # 模型与索引配置
    parser.add_argument('--model_path', type=str,
                        default='experiments/two_tower/checkpoints/last.ckpt')
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--n_trees', type=int, default=50,
                        help='Annoy 树数量，越多越精确但构建越慢')
    parser.add_argument('--metric', type=str, default='angular',
                        choices=['angular', 'euclidean', 'manhattan', 'hamming', 'dot'],
                        help='Annoy 距离度量')

    # 索引保存路径
    parser.add_argument('--user_index_path', type=str,
                        default='experiments/annoy/user_index.ann')
    parser.add_argument('--item_index_path', type=str,
                        default='experiments/annoy/item_index.ann')
    parser.add_argument('--item_mapping_path', type=str,
                        default='experiments/annoy/item_index_mapping.pkl')

    # 构建时的 batch size
    parser.add_argument('--batch_size', type=int, default=256)

    # 测试检索参数
    parser.add_argument('--top_k', type=int, default=100,
                        help='检索返回的 top-K 数量')

    return parser.parse_args()


def build_index(args):
    """构建 Annoy 向量索引"""
    print("准备数据...")

    user_data = pd.DataFrame({
        'user_id': range(10000),
        'age': np.random.randint(0, 10, 10000),
        'gender': np.random.randint(0, 2, 10000),
        'city': np.random.randint(0, 100, 10000),
        'user_score': np.random.rand(10000),
        'click_history': [str(list(np.random.randint(0, 5000, 20))) for _ in range(10000)],
    })

    item_data = pd.DataFrame({
        'item_id': range(50000),
        'category': np.random.randint(0, 20, 50000),
        'brand': np.random.randint(0, 500, 50000),
        'price': np.random.rand(50000) * 1000,
        'item_score': np.random.rand(50000),
    })

    user_features_cols = {
        'sparse': ['user_id', 'age', 'gender', 'city'],
        'dense': ['user_score'],
        'sequence': ['click_history'],
    }
    item_features_cols = {
        'sparse': ['item_id', 'category', 'brand'],
        'dense': ['price', 'item_score'],
        'sequence': [],
    }

    # 确保输出目录存在
    Path(args.user_index_path).parent.mkdir(parents=True, exist_ok=True)

    builder = IndexBuilder(
        model_path=args.model_path,
        embedding_dim=args.embedding_dim,
        metric=args.metric,
        n_trees=args.n_trees,
    )

    builder.build_user_index(
        user_data=user_data,
        user_features_cols=user_features_cols,
        batch_size=args.batch_size,
        save_path=args.user_index_path,
    )

    builder.build_item_index(
        item_data=item_data,
        item_features_cols=item_features_cols,
        batch_size=args.batch_size,
        save_path=args.item_index_path,
    )

    print("\nAnnoy 索引构建完成！")
    print(f"用户索引: {args.user_index_path}")
    print(f"物品索引: {args.item_index_path}")


def test_index(args):
    """测试 Annoy 检索"""
    print("\n" + "=" * 60)
    print("测试 Annoy 检索")
    print("=" * 60)

    retriever = Retriever(
        item_index_path=args.item_index_path,
        item_mapping_path=args.item_mapping_path,
        embedding_dim=args.embedding_dim,
        metric=args.metric,
    )

    query_vector = np.random.randn(args.embedding_dim).astype(np.float32)
    query_vector = query_vector / np.linalg.norm(query_vector)

    results = retriever.search(
        query_vector=query_vector,
        top_k=args.top_k,
        search_k=-1,
    )

    print(f"\n检索到 {len(results)} 个物品:")
    for i, result in enumerate(results[:10], 1):
        print(f"{i}. Item {result['item_id']}: "
              f"distance={result['distance']:.4f}, "
              f"similarity={result['similarity']:.4f}")

    print("\n批量检索测试...")
    query_vectors = np.random.randn(10, args.embedding_dim).astype(np.float32)
    query_vectors = query_vectors / np.linalg.norm(query_vectors, axis=1, keepdims=True)

    batch_results = retriever.batch_search(query_vectors=query_vectors, top_k=50)
    print(f"批量检索完成，共 {len(batch_results)} 个查询")
    print(f"每个查询返回 {len(batch_results[0])} 个结果")


if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'build':
        build_index(args)
    elif args.mode == 'test':
        test_index(args)
    elif args.mode == 'both':
        build_index(args)
        test_index(args)
