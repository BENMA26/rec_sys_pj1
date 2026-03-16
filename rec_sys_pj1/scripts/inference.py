"""
完整推荐系统推理脚本
"""
import argparse
import torch
import numpy as np
import pandas as pd
from typing import List, Dict

from src.models.recall import TwoTowerModel
from src.models.ranking import MMOEModel
from src.utils.retriever import Retriever


def parse_args():
    parser = argparse.ArgumentParser(description='推荐系统推理脚本')

    # 模型路径
    parser.add_argument('--recall_model_path', type=str,
                        default='experiments/two_tower_feature/checkpoints/last.ckpt')
    parser.add_argument('--ranking_model_path', type=str,
                        default='experiments/rank/mmoe_jd/checkpoints/last.ckpt')
    parser.add_argument('--annoy_index_path', type=str,
                        default='experiments/annoy/item_index.ann')
    parser.add_argument('--annoy_mapping_path', type=str,
                        default='experiments/annoy/item_index_mapping.pkl')

    # 推理配置
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--top_k_recall', type=int, default=100,
                        help='召回阶段返回的候选数量')
    parser.add_argument('--top_k_rank', type=int, default=10,
                        help='排序阶段最终返回的推荐数量')
    parser.add_argument('--metric', type=str, default='angular',
                        choices=['angular', 'euclidean', 'manhattan', 'hamming', 'dot'])

    return parser.parse_args()


class RecommendationSystem:
    """完整推荐系统：召回 + 排序"""

    def __init__(self, args):
        self.top_k_recall = args.top_k_recall
        self.top_k_rank = args.top_k_rank

        print(f"加载召回模型: {args.recall_model_path}")
        self.recall_model = TwoTowerModel.load_from_checkpoint(args.recall_model_path)
        self.recall_model.eval()
        self.recall_model.freeze()

        print(f"加载排序模型: {args.ranking_model_path}")
        self.ranking_model = MMOEModel.load_from_checkpoint(args.ranking_model_path)
        self.ranking_model.eval()
        self.ranking_model.freeze()

        print(f"加载 Annoy 索引: {args.annoy_index_path}")
        self.retriever = Retriever(
            item_index_path=args.annoy_index_path,
            item_mapping_path=args.annoy_mapping_path,
            embedding_dim=args.embedding_dim,
            metric=args.metric,
        )

    def recall(self, user_features: Dict[str, torch.Tensor]) -> List[Dict]:
        with torch.no_grad():
            user_vec = self.recall_model.encode_user(user_features)
            user_vec = user_vec.cpu().numpy()[0]
            candidates = self.retriever.search(
                query_vector=user_vec,
                top_k=self.top_k_recall,
                search_k=-1,
            )
        return candidates

    def rank(
        self,
        user_features: Dict[str, torch.Tensor],
        candidate_items: List[Dict],
        item_data: pd.DataFrame,
    ) -> List[Dict]:
        item_ids = [item['item_id'] for item in candidate_items]
        candidate_data = item_data[item_data['item_id'].isin(item_ids)]

        if len(candidate_data) == 0:
            return []

        batch_size = len(candidate_data)
        inputs = {}

        for key, value in user_features.items():
            inputs[key] = value.repeat(batch_size)

        for col in candidate_data.columns:
            if col != 'item_id' and col in self.ranking_model.feature_names:
                inputs[col] = torch.tensor(candidate_data[col].values, dtype=torch.long)

        with torch.no_grad():
            ctr_logits, cvr_logits = self.ranking_model(inputs)
            ctr_scores = torch.sigmoid(ctr_logits).cpu().numpy()
            cvr_scores = torch.sigmoid(cvr_logits).cpu().numpy()

        results = []
        for i, item_id in enumerate(candidate_data['item_id'].values):
            results.append({
                'item_id': int(item_id),
                'ctr_score': float(ctr_scores[i]),
                'cvr_score': float(cvr_scores[i]),
                'combined_score': float(ctr_scores[i] * cvr_scores[i]),
            })

        results = sorted(results, key=lambda x: x['combined_score'], reverse=True)
        return results[:self.top_k_rank]

    def recommend(
        self,
        user_features: Dict[str, torch.Tensor],
        item_data: pd.DataFrame,
    ) -> List[Dict]:
        print(f"召回阶段: 检索 top-{self.top_k_recall} 候选物品...")
        candidates = self.recall(user_features)
        print(f"召回 {len(candidates)} 个候选物品")

        print(f"排序阶段: 预测 CTR 和 CVR...")
        ranked_items = self.rank(user_features, candidates, item_data)
        print(f"返回 top-{len(ranked_items)} 推荐结果")

        return ranked_items


def inference(args):
    print("\n" + "=" * 60)
    print("推荐系统推理示例")
    print("=" * 60 + "\n")

    print("准备物品数据...")
    item_data = pd.DataFrame({
        'item_id': range(1000),
        'category': np.random.randint(0, 20, 1000),
        'brand': np.random.randint(0, 100, 1000),
        'price': np.random.rand(1000) * 1000,
    })

    rec_system = RecommendationSystem(args)

    user_features = {
        'user_id': torch.tensor([123], dtype=torch.long),
        'age': torch.tensor([5], dtype=torch.long),
        'gender': torch.tensor([1], dtype=torch.long),
        'city': torch.tensor([10], dtype=torch.long),
        'user_score': torch.tensor([0.8], dtype=torch.float),
    }

    recommendations = rec_system.recommend(user_features, item_data)

    print("\n" + "=" * 60)
    print("推荐结果：")
    print("=" * 60)
    for i, item in enumerate(recommendations, 1):
        print(f"{i}. Item {item['item_id']}: "
              f"CTR={item['ctr_score']:.4f}, "
              f"CVR={item['cvr_score']:.4f}, "
              f"Score={item['combined_score']:.4f}")


if __name__ == '__main__':
    inference(parse_args())
