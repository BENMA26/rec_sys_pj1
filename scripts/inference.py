"""
完整推荐系统
"""
import torch
import numpy as np
import pandas as pd
from typing import List, Dict

from src.models.two_tower import TwoTowerModel
from src.models.mmoe import MMOEModel
from src.utils.retriever import Retriever

class RecommendationSystem:
    """完整推荐系统：召回 + 排序"""

    def __init__(
        self,
        recall_model_path: str,
        ranking_model_path: str,
        annoy_index_path: str,
        annoy_mapping_path: str,
        embedding_dim: int = 128,
        top_k_recall: int = 100,
        top_k_rank: int = 10,
    ):
        self.top_k_recall = top_k_recall
        self.top_k_rank = top_k_rank

        # 加载召回模型
        print(f"加载召回模型: {recall_model_path}")
        self.recall_model = TwoTowerModel.load_from_checkpoint(recall_model_path)
        self.recall_model.eval()
        self.recall_model.freeze()

        # 加载排序模型
        print(f"加载排序模型: {ranking_model_path}")
        self.ranking_model = MMOEModel.load_from_checkpoint(ranking_model_path)
        self.ranking_model.eval()
        self.ranking_model.freeze()

        # 加载 Annoy 索引
        print(f"加载 Annoy 索引: {annoy_index_path}")
        self.retriever = Retriever(
            item_index_path=annoy_index_path,
            item_mapping_path=annoy_mapping_path,
            embedding_dim=embedding_dim,
            metric='angular',
        )

    def recall(self, user_features: Dict[str, torch.Tensor]) -> List[Dict]:
        """
        召回阶段：使用双塔模型 + Annoy 检索 top-K 物品

        Args:
            user_features: 用户特征字典
        Returns:
            候选物品列表
        """
        with torch.no_grad():
            # 编码用户向量
            user_vec = self.recall_model.encode_user(user_features)
            user_vec = user_vec.cpu().numpy()[0]  # (embedding_dim,)

            # 使用 Annoy 检索
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
        """
        排序阶段：使用 MMOE 模型预测 CTR 和 CVR

        Args:
            user_features: 用户特征字典
            candidate_items: 候选物品列表
            item_data: 物品数据
        Returns:
            排序后的推荐结果
        """
        # 获取候选物品的 item_id
        item_ids = [item['item_id'] for item in candidate_items]

        # 从物品数据中获取特征
        candidate_data = item_data[item_data['item_id'].isin(item_ids)]

        if len(candidate_data) == 0:
            return []

        # 构造输入特征
        batch_size = len(candidate_data)
        inputs = {}

        # 用户特征：重复 batch_size 次
        for key, value in user_features.items():
            inputs[key] = value.repeat(batch_size)

        # 物品特征：从候选数据中获取
        for col in candidate_data.columns:
            if col != 'item_id' and col in self.ranking_model.feature_names:
                inputs[col] = torch.tensor(candidate_data[col].values, dtype=torch.long)

        # 预测 CTR 和 CVR
        with torch.no_grad():
            ctr_logits, cvr_logits = self.ranking_model(inputs)
            ctr_scores = torch.sigmoid(ctr_logits).cpu().numpy()
            cvr_scores = torch.sigmoid(cvr_logits).cpu().numpy()

        # 组合结果
        results = []
        for i, item_id in enumerate(candidate_data['item_id'].values):
            results.append({
                'item_id': int(item_id),
                'ctr_score': float(ctr_scores[i]),
                'cvr_score': float(cvr_scores[i]),
                'combined_score': float(ctr_scores[i] * cvr_scores[i]),
            })

        # 按组合分数排序
        results = sorted(results, key=lambda x: x['combined_score'], reverse=True)

        return results[:self.top_k_rank]

    def recommend(
        self,
        user_features: Dict[str, torch.Tensor],
        item_data: pd.DataFrame,
    ) -> List[Dict]:
        """
        完整推荐流程：召回 + 排序

        Args:
            user_features: 用户特征字典
            item_data: 物品数据
        Returns:
            推荐结果列表
        """
        # 1. 召回阶段
        print(f"召回阶段: 检索 top-{self.top_k_recall} 候选物品...")
        candidates = self.recall(user_features)
        print(f"召回 {len(candidates)} 个候选物品")

        # 2. 排序阶段
        print(f"排序阶段: 预测 CTR 和 CVR...")
        ranked_items = self.rank(user_features, candidates, item_data)
        print(f"返回 top-{len(ranked_items)} 推荐结果")

        return ranked_items

def inference():
    """推理示例"""

    print("\n" + "="*60)
    print("推荐系统推理示例")
    print("="*60 + "\n")

    # ========== 1. 准备物品数据 ==========
    print("准备物品数据...")
    item_data = pd.DataFrame({
        'item_id': range(1000),
        'category': np.random.randint(0, 20, 1000),
        'brand': np.random.randint(0, 100, 1000),
        'price': np.random.rand(1000) * 1000,
    })

    # ========== 2. 初始化推荐系统 ==========
    rec_system = RecommendationSystem(
        recall_model_path='experiments/two_tower_feature/checkpoints/last.ckpt',
        ranking_model_path='experiments/mmoe/checkpoints/last.ckpt',
        annoy_index_path='experiments/annoy/item_index.ann',
        annoy_mapping_path='experiments/annoy/item_index_mapping.pkl',
        embedding_dim=128,
        top_k_recall=100,
        top_k_rank=10,
    )

    # ========== 3. 用户特征 ==========
    user_features = {
        'user_id': torch.tensor([123], dtype=torch.long),
        'age': torch.tensor([5], dtype=torch.long),
        'gender': torch.tensor([1], dtype=torch.long),
        'city': torch.tensor([10], dtype=torch.long),
        'user_score': torch.tensor([0.8], dtype=torch.float),
    }

    # ========== 4. 推荐 ==========
    recommendations = rec_system.recommend(user_features, item_data)

    # ========== 5. 输出结果 ==========
    print("\n" + "="*60)
    print("推荐结果：")
    print("="*60)
    for i, item in enumerate(recommendations, 1):
        print(f"{i}. Item {item['item_id']}: "
              f"CTR={item['ctr_score']:.4f}, "
              f"CVR={item['cvr_score']:.4f}, "
              f"Score={item['combined_score']:.4f}")

if __name__ == '__main__':
    inference()
