"""
构建 Annoy 向量索引
基于训练好的双塔模型和离线数据构建向量数据库
"""
import torch
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from typing import Dict, List, Optional
from pathlib import Path
from tqdm import tqdm
import pickle

from src.models.recall import TwoTowerModel


class IndexBuilder:
    """Annoy 向量索引构建器"""

    def __init__(
        self,
        model_path: str,
        embedding_dim: int,
        metric: str = 'angular',  # 'angular', 'euclidean', 'manhattan', 'hamming', 'dot'
        n_trees: int = 10,
    ):
        """
        Args:
            model_path: 训练好的双塔模型路径
            embedding_dim: 向量维度
            metric: 距离度量方式
                - 'angular': 余弦相似度（推荐）
                - 'euclidean': 欧氏距离
                - 'dot': 点积
            n_trees: Annoy 树的数量，越多越精确但构建越慢
        """
        self.model_path = model_path
        self.embedding_dim = embedding_dim
        self.metric = metric
        self.n_trees = n_trees

        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = TwoTowerFeatureModel.load_from_checkpoint(model_path)
        self.model.eval()
        self.model.freeze()

        # 初始化 Annoy 索引
        self.user_index = AnnoyIndex(embedding_dim, metric)
        self.item_index = AnnoyIndex(embedding_dim, metric)

        # ID 映射
        self.user_id_map = {}  # annoy_id -> user_id
        self.item_id_map = {}  # annoy_id -> item_id
        self.user_id_reverse_map = {}  # user_id -> annoy_id
        self.item_id_reverse_map = {}  # item_id -> annoy_id

    def build_user_index(
        self,
        user_data: pd.DataFrame,
        user_features_cols: Dict[str, List[str]],
        batch_size: int = 256,
        save_path: str = 'experiments/annoy/user_index.ann',
    ):
        """
        构建用户向量索引

        Args:
            user_data: 用户数据，每行一个用户
            user_features_cols: 用户特征列名
                {
                    'sparse': ['user_id', 'age', 'gender'],
                    'dense': ['user_score'],
                    'sequence': ['click_history']
                }
            batch_size: 批次大小
            save_path: 索引保存路径
        """
        print(f"\n构建用户向量索引...")
        print(f"用户数量: {len(user_data)}")

        user_vectors = []
        user_ids = []

        # 批量编码用户向量
        with torch.no_grad():
            for i in tqdm(range(0, len(user_data), batch_size), desc="编码用户向量"):
                batch_data = user_data.iloc[i:i + batch_size]

                # 构造输入特征
                user_features = self._prepare_features(batch_data, user_features_cols)

                # 编码
                user_vec = self.model.encode_user(user_features)
                user_vectors.append(user_vec.cpu().numpy())

                # 记录 user_id
                user_ids.extend(batch_data['user_id'].tolist())

        # 合并所有向量
        user_vectors = np.vstack(user_vectors)
        print(f"用户向量维度: {user_vectors.shape}")

        # 添加到 Annoy 索引
        for annoy_id, (user_id, vec) in enumerate(zip(user_ids, user_vectors)):
            self.user_index.add_item(annoy_id, vec)
            self.user_id_map[annoy_id] = user_id
            self.user_id_reverse_map[user_id] = annoy_id

        # 构建索引
        print(f"构建 Annoy 索引 (n_trees={self.n_trees})...")
        self.user_index.build(self.n_trees)

        # 保存索引
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.user_index.save(save_path)
        print(f"用户索引已保存: {save_path}")

        # 保存 ID 映射
        mapping_path = save_path.replace('.ann', '_mapping.pkl')
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'id_map': self.user_id_map,
                'reverse_map': self.user_id_reverse_map,
            }, f)
        print(f"ID 映射已保存: {mapping_path}")

    def build_item_index(
        self,
        item_data: pd.DataFrame,
        item_features_cols: Dict[str, List[str]],
        batch_size: int = 256,
        save_path: str = 'experiments/annoy/item_index.ann',
    ):
        """
        构建物品向量索引

        Args:
            item_data: 物品数据，每行一个物品
            item_features_cols: 物品特征列名
            batch_size: 批次大小
            save_path: 索引保存路径
        """
        print(f"\n构建物品向量索引...")
        print(f"物品数量: {len(item_data)}")

        item_vectors = []
        item_ids = []

        # 批量编码物品向量
        with torch.no_grad():
            for i in tqdm(range(0, len(item_data), batch_size), desc="编码物品向量"):
                batch_data = item_data.iloc[i:i + batch_size]

                # 构造输入特征
                item_features = self._prepare_features(batch_data, item_features_cols)

                # 编码
                item_vec = self.model.encode_item(item_features)
                item_vectors.append(item_vec.cpu().numpy())

                # 记录 item_id
                item_ids.extend(batch_data['item_id'].tolist())

        # 合并所有向量
        item_vectors = np.vstack(item_vectors)
        print(f"物品向量维度: {item_vectors.shape}")

        # 添加到 Annoy 索引
        for annoy_id, (item_id, vec) in enumerate(zip(item_ids, item_vectors)):
            self.item_index.add_item(annoy_id, vec)
            self.item_id_map[annoy_id] = item_id
            self.item_id_reverse_map[item_id] = annoy_id

        # 构建索引
        print(f"构建 Annoy 索引 (n_trees={self.n_trees})...")
        self.item_index.build(self.n_trees)

        # 保存索引
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.item_index.save(save_path)
        print(f"物品索引已保存: {save_path}")

        # 保存 ID 映射
        mapping_path = save_path.replace('.ann', '_mapping.pkl')
        with open(mapping_path, 'wb') as f:
            pickle.dump({
                'id_map': self.item_id_map,
                'reverse_map': self.item_id_reverse_map,
            }, f)
        print(f"ID 映射已保存: {mapping_path}")

    def _prepare_features(
        self,
        data: pd.DataFrame,
        features_cols: Dict[str, List[str]],
    ) -> Dict[str, torch.Tensor]:
        """准备输入特征"""
        features = {}

        # 稀疏特征
        if 'sparse' in features_cols:
            for col in features_cols['sparse']:
                features[col] = torch.tensor(data[col].values, dtype=torch.long)

        # 稠密特征
        if 'dense' in features_cols:
            for col in features_cols['dense']:
                features[col] = torch.tensor(data[col].values, dtype=torch.float)

        # 序列特征
        if 'sequence' in features_cols:
            for col in features_cols['sequence']:
                # 假设序列已经是固定长度的列表
                seq_data = data[col].apply(lambda x: eval(x) if isinstance(x, str) else x)
                features[col] = torch.tensor(np.stack(seq_data.values), dtype=torch.long)

        return features


class Retriever:
    """Annoy 向量检索器"""

    def __init__(
        self,
        item_index_path: str,
        item_mapping_path: str,
        embedding_dim: int,
        metric: str = 'angular',
    ):
        """
        Args:
            item_index_path: 物品索引路径
            item_mapping_path: 物品 ID 映射路径
            embedding_dim: 向量维度
            metric: 距离度量方式
        """
        # 加载索引
        self.item_index = AnnoyIndex(embedding_dim, metric)
        self.item_index.load(item_index_path)
        print(f"加载物品索引: {item_index_path}")

        # 加载 ID 映射
        with open(item_mapping_path, 'rb') as f:
            mapping = pickle.load(f)
            self.item_id_map = mapping['id_map']
            self.item_id_reverse_map = mapping['reverse_map']

        print(f"物品数量: {len(self.item_id_map)}")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
        search_k: int = -1,
    ) -> List[Dict]:
        """
        向量检索

        Args:
            query_vector: 查询向量 (embedding_dim,)
            top_k: 返回 top-K 个结果
            search_k: 搜索的节点数，-1 表示 n_trees * top_k
        Returns:
            检索结果列表，每个元素包含 item_id 和 distance
        """
        # 检索
        annoy_ids, distances = self.item_index.get_nns_by_vector(
            query_vector, top_k, search_k=search_k, include_distances=True
        )

        # 转换为 item_id
        results = []
        for annoy_id, distance in zip(annoy_ids, distances):
            results.append({
                'item_id': self.item_id_map[annoy_id],
                'distance': distance,
                'similarity': 1 - distance if distance <= 1 else 0,  # 余弦相似度
            })

        return results

    def batch_search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 100,
        search_k: int = -1,
    ) -> List[List[Dict]]:
        """批量检索"""
        results = []
        for query_vec in query_vectors:
            results.append(self.search(query_vec, top_k, search_k))
        return results
