"""
数据集 - 支持 Sparse, Dense, Sequence 三种特征类型
参考 torch-rechub 的数据组织方式
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from src.models.features import SparseFeature, DenseFeature, SequenceFeature

'''
召回数据集
'''

def build_recall_tables(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    item_col: str = 'item_id',
    label_col: str = 'click',
    hard_neg_col: Optional[str] = None,
) -> Dict:
    """
    从原始交互表提取召回训练所需的各类数据结构。

    Returns:
        pos_df       - 正样本表 (click==1)
        hard_neg_df  - 困难负样本表 (click==0，或由 hard_neg_col 指定)
        item_pool    - np.ndarray，全量物品ID
        popularity   - dict {item_id: float}，归一化热度
        user_history - dict {user_id: set}，用户交互过的物品集合
    """
    pos_df = df[df[label_col] == 1].reset_index(drop=True)

    if hard_neg_col is not None:
        hard_neg_df = df[[user_col, hard_neg_col]].rename(columns={hard_neg_col: item_col}).dropna().reset_index(drop=True)
    else:
        hard_neg_df = df[df[label_col] == 0].reset_index(drop=True)

    item_pool = df[item_col].unique()

    counts = df[df[label_col] == 1][item_col].value_counts()
    total = counts.sum()
    popularity = {item: cnt / total for item, cnt in counts.items()}

    user_history = df[df[label_col] == 1].groupby(user_col)[item_col].apply(set).to_dict()

    return dict(pos_df=pos_df, hard_neg_df=hard_neg_df, item_pool=item_pool, popularity=popularity, user_history=user_history)

class RecallPosDataset(Dataset):
    """
    召回正样本 Dataset。
    每条样本只返回一个正样本行（user_id + item_id + 可选特征列）。
    负样本构建推迟到 RecallCollator，以支持 in-batch 策略。
    """

    def __init__(
        self,
        pos_df: pd.DataFrame,
        feature_cols: List[str],
        user_col: str = 'user_id',
        item_col: str = 'item_id',
    ):
        self.feature_cols = feature_cols
        self.user_col = user_col
        self.item_col = item_col

        # ---- 一次性转成 tensor，之后再也不碰 DataFrame ----
        self.user_ids = torch.from_numpy(
            pos_df[user_col].values.astype(np.int64)
        ).long()
        
        self.item_ids = torch.from_numpy(
            pos_df[item_col].values.astype(np.int64)
        ).long()

        # 按类型批量转换特征列
        self.feature_tensors = {}
        for col in feature_cols:
            values = pos_df[col].values
            if values.dtype in (np.int32, np.int64, int):
                self.feature_tensors[col] = torch.from_numpy(
                    values.astype(np.int64)
                ).long()
            elif isinstance(values[0], (list, np.ndarray)):
                # 变长序列特征：先pad再转tensor
                self.feature_tensors[col] = torch.from_numpy(
                    np.stack(values).astype(np.float32)
                ).float()
            else:
                self.feature_tensors[col] = torch.from_numpy(
                    values.astype(np.float32)
                ).float()

        self.size = len(pos_df)
        
        # DataFrame 已经不需要了，释放内存
        del pos_df

    def __len__(self):
        return self.size

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # 纯索引操作，每次调用 < 1微秒
        sample = {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx],
        }
        for col in self.feature_cols:
            sample[col] = self.feature_tensors[col][idx]
        return sample

class RecallCollator:
    """
    召回模型通用 Collator，支持训练和评估。
    全局采样 K 个负样本，batch 内所有用户共享（shared negatives）。

    输出结构：
        user_features        : Dict[str, Tensor(B, ...)]        用户特征
        pos_item_features    : Dict[str, Tensor(B, ...)]        正样本物品特征
        neg_item_features    : Dict[str, Tensor(K, ...)]        共享负样本物品特征

    训练时：K = num_neg（通常较少，如 8），配合 in-batch 负样本
    评估时：K = num_neg（通常较多，如 99），用于计算 Recall@K 等指标
    """

    def __init__(
        self,
        items_df: pd.DataFrame,
        item_sparse_features: List[SparseFeature],
        item_dense_features: List[DenseFeature],
        item_sequence_features: List[SequenceFeature],
        item_pool: np.ndarray,
        user_history: Dict,
        num_neg: int = 4,
        user_col: str = 'user_id',
        item_col: str = 'item_id',
    ):
        self.items_df = items_df.set_index(item_col) if item_col in items_df.columns else items_df
        self.item_sparse_features = item_sparse_features
        self.item_dense_features = item_dense_features
        self.item_sequence_features = item_sequence_features
        self.item_pool = item_pool
        self.user_history = user_history
        self.num_neg = num_neg
        self.user_col = user_col
        self.item_col = item_col

    def _sample_negs(self, pos_item_ids: set[int]) -> List[int]:
        """全局采样 K 个负样本，排除当前 batch 的正样本。"""
        mask = np.array([i not in pos_item_ids for i in self.item_pool], dtype=bool)
        candidates = self.item_pool[mask]
        if len(candidates) == 0:
            candidates = self.item_pool
        return np.random.choice(
            candidates, size=self.num_neg, replace=len(candidates) < self.num_neg
        ).tolist()

    def _extract_item_features(self, item_id: int) -> Dict[str, torch.Tensor]:
        """从 items_df 中提取单个 item 的完整特征。"""
        if item_id in self.items_df.index:
            row = self.items_df.loc[item_id]
        else:
            row = None

        features = {}
        for feat in self.item_sparse_features:
            val = int(row[feat.name]) if row is not None else 0
            features[feat.name] = torch.tensor(val, dtype=torch.long)
        for feat in self.item_dense_features:
            if row is not None:
                raw = row[feat.name]
                val = eval(raw) if isinstance(raw, str) else raw
            else:
                val = [0.0] * feat.dim if feat.dim > 1 else 0.0
            features[feat.name] = torch.tensor(val, dtype=torch.float)
        for feat in self.item_sequence_features:
            if row is not None:
                seq = row[feat.name]
                if isinstance(seq, str):
                    seq = eval(seq)
                seq = list(seq)
            else:
                seq = []
            seq_len = len(seq)
            if seq_len < feat.max_len:
                seq = seq + [0] * (feat.max_len - seq_len)
            else:
                seq = seq[:feat.max_len]
            features[feat.name] = torch.tensor(seq, dtype=torch.long)
        return features

    def _stack_item_features(self, feat_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """将多个 item 特征字典 stack 成一个 batch 维度的字典。"""
        return {k: torch.stack([f[k] for f in feat_list]) for k in feat_list[0]}

    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        item_keys = {feat.name for feat in self.item_sparse_features} \
                  | {feat.name for feat in self.item_dense_features} \
                  | {feat.name for feat in self.item_sequence_features}
        user_keys = set(samples[0].keys()) - item_keys - {self.item_col}

        user_features = {k: torch.stack([s[k] for s in samples]) for k in user_keys}
        pos_item_ids = torch.stack([s[self.item_col] for s in samples])

        # 正样本特征 (B, ...)
        pos_feats_list = [self._extract_item_features(int(iid)) for iid in pos_item_ids]
        pos_item_features = self._stack_item_features(pos_feats_list)

        # 共享负样本特征 (K, ...)，全局采样一次，所有用户共享
        pos_id_set = set(pos_item_ids.tolist())
        neg_ids = self._sample_negs(pos_id_set)
        neg_feats_list = [self._extract_item_features(nid) for nid in neg_ids]
        neg_item_features = self._stack_item_features(neg_feats_list)

        return user_features, pos_item_features, neg_item_features

'''
class RecallCollator:
    """
    召回模型通用 Collator，支持训练和评估。
    每个用户独立从物料池中采样负样本（排除用户历史）。

    输出结构：
        user_features        : Dict[str, Tensor(B, ...)]        用户特征
        pos_item_features    : Dict[str, Tensor(B, ...)]        正样本物品特征
        neg_item_features    : Dict[str, Tensor(B, K, ...)]     负样本物品特征

    训练时：K = num_neg（通常较少，如 8），配合 in-batch 负样本
    评估时：K = num_neg（通常较多，如 99），用于计算 Recall@K 等指标
    """

    def __init__(
        self,
        items_df: pd.DataFrame,
        item_sparse_features: List[SparseFeature],
        item_dense_features: List[DenseFeature],
        item_sequence_features: List[SequenceFeature],
        item_pool: np.ndarray,
        user_history: Dict,
        num_neg: int = 4,
        user_col: str = 'user_id',
        item_col: str = 'item_id',
    ):
        # 以 item_id 为索引，方便 O(1) 查找
        self.items_df = items_df.set_index(item_col) if item_col in items_df.columns else items_df
        self.item_sparse_features = item_sparse_features
        self.item_dense_features = item_dense_features
        self.item_sequence_features = item_sequence_features
        self.item_pool = item_pool
        self.user_history = user_history
        self.num_neg = num_neg
        self.user_col = user_col
        self.item_col = item_col

    def _sample_negs(self, user_id: int) -> List[int]:
        """为指定用户采样负样本，排除用户历史交互"""
        history = self.user_history.get(int(user_id), set())
        mask = np.array([i not in history for i in self.item_pool], dtype=bool)
        candidates = self.item_pool[mask]
        if len(candidates) == 0:
            candidates = self.item_pool
        return np.random.choice(candidates, size=self.num_neg, replace=len(candidates) < self.num_neg).tolist()

    def _extract_item_features(self, item_id: int) -> Dict[str, torch.Tensor]:
        """从 items_df 中提取单个 item 的完整特征。"""
        if item_id in self.items_df.index:
            row = self.items_df.loc[item_id]
        else:
            row = None

        features = {}
        for feat in self.item_sparse_features:
            val = int(row[feat.name]) if row is not None else 0
            features[feat.name] = torch.tensor(val, dtype=torch.long)
        for feat in self.item_dense_features:
            if row is not None:
                raw = row[feat.name]
                val = eval(raw) if isinstance(raw, str) else raw
            else:
                val = [0.0] * feat.dim if feat.dim > 1 else 0.0
            features[feat.name] = torch.tensor(val, dtype=torch.float)
        for feat in self.item_sequence_features:
            if row is not None:
                seq = row[feat.name]
                if isinstance(seq, str):
                    seq = eval(seq)
                seq = list(seq)
            else:
                seq = []
            seq_len = len(seq)
            if seq_len < feat.max_len:
                seq = seq + [0] * (feat.max_len - seq_len)
            else:
                seq = seq[:feat.max_len]
            features[feat.name] = torch.tensor(seq, dtype=torch.long)
        return features

    def _stack_item_features(self, feat_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """将多个 item 特征字典 stack 成一个 batch 维度的字典。"""
        return {k: torch.stack([f[k] for f in feat_list]) for k in feat_list[0]}

    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> Tuple[
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        # 分离 user / item 字段
        item_keys = {feat.name for feat in self.item_sparse_features} \
                  | {feat.name for feat in self.item_dense_features} \
                  | {feat.name for feat in self.item_sequence_features}
        user_keys = set(samples[0].keys()) - item_keys - {self.item_col}

        user_features = {k: torch.stack([s[k] for s in samples]) for k in user_keys}
        user_ids = user_features[self.user_col]  # (B,)
        pos_item_ids = torch.stack([s[self.item_col] for s in samples])  # (B,)

        # 正样本特征 (B, ...)
        pos_feats_list = [self._extract_item_features(int(iid)) for iid in pos_item_ids]
        pos_item_features = self._stack_item_features(pos_feats_list)

        # 负样本特征 (B, K, ...)，每个用户独立采样
        neg_feats_per_user = []
        for uid in user_ids:
            neg_ids = self._sample_negs(int(uid))
            neg_feats = [self._extract_item_features(nid) for nid in neg_ids]
            neg_feats_per_user.append(self._stack_item_features(neg_feats))  # Dict[str, (K,...)]

        # stack across batch: Dict[str, (B, K, ...)]
        neg_item_features = {
            k: torch.stack([neg_feats_per_user[i][k] for i in range(len(samples))])
            for k in neg_feats_per_user[0]
        }

        return user_features, pos_item_features, neg_item_features
'''

class TwoTowerDataModule(pl.LightningDataModule):
    """
    双塔召回 DataModule。

    训练阶段：RecallCollator（num_easy_neg 个负样本 + in-batch 负样本）。
    val/test 阶段：RecallCollator（num_eval_neg 个负样本，用于计算 Recall@K 等指标）。
                   构成 (1正 + K负) 的评估组，适合计算 Recall@K / NDCG@K 等离线指标。

    item_pool 和 user_history 只从 train_df 统计，避免数据泄漏。
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        items_df: pd.DataFrame,
        user_feature_cols: List[str],
        item_sparse_features: List[SparseFeature],
        item_dense_features: List[DenseFeature],
        item_sequence_features: List[SequenceFeature],
        num_easy_neg: int = 4,
        num_eval_neg: int = 99,
        batch_size: int = 256,
        eval_batch_size: Optional[int] = None,
        num_workers: int = 4,
        user_col: str = 'user_id',
        item_col: str = 'item_id',
        label_col: str = 'click',
        hard_neg_col: Optional[str] = None,
        test_df: Optional[pd.DataFrame] = None,
    ):
        super().__init__()
        self.items_df = items_df
        self.user_feature_cols = user_feature_cols
        self.item_sparse_features = item_sparse_features
        self.item_dense_features = item_dense_features
        self.item_sequence_features = item_sequence_features
        self.num_easy_neg = num_easy_neg
        self.num_eval_neg = num_eval_neg
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.num_workers = num_workers
        self.user_col = user_col
        self.item_col = item_col
        self.label_col = label_col
        self.hard_neg_col = hard_neg_col

        # item_pool / user_history 只从 train_df 统计，避免数据泄漏
        tables = build_recall_tables(train_df, user_col, item_col, label_col, hard_neg_col)
        self.item_pool = tables['item_pool']
        self.user_history = tables['user_history']
        self.train_pos_df = tables['pos_df']

        self.val_pos_df = val_df[val_df[label_col] == 1].reset_index(drop=True)
        self.test_pos_df = (
            test_df[test_df[label_col] == 1].reset_index(drop=True)
            if test_df is not None else None
        )

    def setup(self, stage: Optional[str] = None):
        collator_kwargs = dict(
            items_df=self.items_df,
            item_sparse_features=self.item_sparse_features,
            item_dense_features=self.item_dense_features,
            item_sequence_features=self.item_sequence_features,
            item_pool=self.item_pool,
            user_history=self.user_history,
        )

        if stage in ('fit', None):
            self.train_dataset = RecallPosDataset(
                self.train_pos_df, self.user_feature_cols, self.user_col, self.item_col
            )
            self.val_dataset = RecallPosDataset(
                self.val_pos_df, self.user_feature_cols, self.user_col, self.item_col
            )

        self.train_collator = RecallCollator(**collator_kwargs, num_neg=self.num_easy_neg)
        self.eval_collator = RecallCollator(**collator_kwargs, num_neg=self.num_eval_neg)

        if stage in ('test', None) and self.test_pos_df is not None:
            self.test_dataset = RecallPosDataset(
                self.test_pos_df, self.user_feature_cols, self.user_col, self.item_col
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            collate_fn=self.train_collator, pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.eval_batch_size,
            shuffle=False, num_workers=self.num_workers,
            collate_fn=self.eval_collator, pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.eval_batch_size,
            shuffle=False, num_workers=self.num_workers,
            collate_fn=self.eval_collator, pin_memory=True,
        )

'''
排序数据集
'''
class RankDataset(Dataset):

    def __init__(
        self,
        df,
        feature_cols: List[str],
        label_cols: List[str],
        user_col: str = 'user_id',
        item_col: str = 'item_id',
    ):
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.user_col = user_col
        self.item_col = item_col

        # 预先转换为 Tensor，避免 __getitem__ 中重复转换
        self.user_ids = torch.tensor(df[user_col].values, dtype=torch.long)
        self.item_ids = torch.tensor(df[item_col].values, dtype=torch.long)

        self.feature_tensors = {}
        for col in feature_cols:
            raw = df[col].values
            if df[col].dtype == 'object':
                # 序列特征：'[1,2,3]' -> list -> Tensor
                parsed = [eval(x) if isinstance(x, str) else x for x in raw]
                self.feature_tensors[col] = torch.tensor(parsed, dtype=torch.long)
            elif df[col].dtype in ['int64', 'int32']:
                self.feature_tensors[col] = torch.tensor(raw, dtype=torch.long)
            else:
                self.feature_tensors[col] = torch.tensor(raw, dtype=torch.float)

        self.label_tensors = {}
        for col in label_cols:
            self.label_tensors[col] = torch.tensor(df[col].values, dtype=torch.float)

    def __len__(self):
        return self.user_ids.size(0)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        # 纯索引操作，每次调用 < 1微秒
        sample = {
            'user_id': self.user_ids[idx],
            'item_id': self.item_ids[idx],
        }
        for col in self.feature_cols:
            sample[col] = self.feature_tensors[col][idx]

        labels = {}
        for col in self.label_cols:
            labels[col] = self.label_tensors[col][idx]

        return sample, labels

class RankDataModule(pl.LightningDataModule):
    """
    排序模型 DataModule。
    """
 
    def __init__(
        self,
        feature_cols: List[str],
        label_cols: List[str],
        user_col: str = 'user_id',
        item_col: str = 'item_id',
        # ---- 数据来源（二选一） ----
        train_df: Optional[pd.DataFrame] = None,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        # ---- 训练参数 ----
        batch_size: int = 2048,
        val_batch_size: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
 
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.user_col = user_col
        self.item_col = item_col
 
        # 数据来源
        self._train_df = train_df
        self._val_df = val_df
        self._test_df = test_df
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
 
        # 训练参数
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size or batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_ratio = val_ratio
        self.seed = seed
 
        # Dataset 占位
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
 
    def _read_file(self, path: str) -> pd.DataFrame:
        """根据文件后缀自动选择读取方式"""
        path = Path(path)
        if path.suffix == '.parquet':
            # 只读需要的列，加速读取
            cols = [self.user_col, self.item_col] + self.feature_cols + self.label_cols
            return pd.read_parquet(path, columns=cols)
        elif path.suffix in ('.csv', '.tsv'):
            sep = '\t' if path.suffix == '.tsv' else ','
            cols = [self.user_col, self.item_col] + self.feature_cols + self.label_cols
            return pd.read_csv(path, usecols=cols, sep=sep)
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")
 
    def setup(self, stage: Optional[str] = None):
        """
        加载数据并创建Dataset。
        PyTorch Lightning 会在训练/验证/测试前自动调用此方法。
        """
        if stage in ('fit', None):
            # ---- 获取训练数据 ----
            if self._train_df is not None:
                train_df = self._train_df
            elif self.train_path is not None:
                train_df = self._read_file(self.train_path)
            else:
                raise ValueError("必须提供 train_df 或 train_path")
 
            # ---- 获取验证数据 ----
            if self._val_df is not None:
                val_df = self._val_df
            elif self.val_path is not None:
                val_df = self._read_file(self.val_path)
            else:
                # 没有单独的验证集，从训练集中划分
                val_df = train_df.sample(
                    frac=self.val_ratio,
                    random_state=self.seed,
                )
                train_df = train_df.drop(val_df.index)
 
            print(f"📊 训练集: {len(train_df):,} 条")
            print(f"📊 验证集: {len(val_df):,} 条")
 
            self.train_dataset = RankDataset(
                df=train_df,
                feature_cols=self.feature_cols,
                label_cols=self.label_cols,
                user_col=self.user_col,
                item_col=self.item_col,
            )
            self.val_dataset = RankDataset(
                df=val_df,
                feature_cols=self.feature_cols,
                label_cols=self.label_cols,
                user_col=self.user_col,
                item_col=self.item_col,
            )
 
            # 释放 DataFrame 引用
            self._train_df = None
            self._val_df = None
 
        if stage in ('test', None):
            if self._test_df is not None:
                test_df = self._test_df
            elif self.test_path is not None:
                test_df = self._read_file(self.test_path)
            else:
                return  # 没有测试集，跳过
 
            print(f"📊 测试集: {len(test_df):,} 条")
 
            self.test_dataset = RankDataset(
                df=test_df,
                feature_cols=self.feature_cols,
                label_cols=self.label_cols,
                user_col=self.user_col,
                item_col=self.item_col,
            )
            self._test_df = None
 
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )
 
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            drop_last=False,
        )
 
    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("没有测试数据集，请提供 test_df 或 test_path")
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )


'''
全量物料测试 Dataset / DataModule
用于测试阶段：对每个用户计算其与所有测试物料的得分，再计算 topk recall
'''

class ItemDataset(Dataset):
    """
    全量物料 Dataset。
    每条样本返回一个物料的特征字典（item_id + item 特征列）。
    """

    def __init__(
        self,
        items_df: pd.DataFrame,
        item_sparse_features: List[SparseFeature],
        item_dense_features: List[DenseFeature],
        item_sequence_features: List[SequenceFeature],
        item_col: str = 'item_id',
    ):
        self.item_col = item_col
        self.item_ids = torch.from_numpy(
            items_df[item_col].values.astype(np.int64)
        ).long()

        feat_cols = (
            [f.name for f in item_sparse_features]
            + [f.name for f in item_dense_features]
            + [f.name for f in item_sequence_features]
        )
        sparse_names = {f.name for f in item_sparse_features}
        seq_names    = {f.name for f in item_sequence_features}

        self.feature_tensors: Dict[str, torch.Tensor] = {}
        for col in feat_cols:
            values = items_df[col].values
            if col in sparse_names:
                self.feature_tensors[col] = torch.from_numpy(
                    values.astype(np.int64)
                ).long()
            elif col in seq_names:
                self.feature_tensors[col] = torch.from_numpy(
                    np.stack(values).astype(np.int64)
                ).long()
            else:
                self.feature_tensors[col] = torch.from_numpy(
                    values.astype(np.float32)
                ).float()

        self.size = len(items_df)

    def __len__(self):
        return self.size

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = {'item_id': self.item_ids[idx]}
        for col, tensor in self.feature_tensors.items():
            sample[col] = tensor[idx]
        return sample


class UserTestDataset(Dataset):
    """
    测试用户 Dataset。
    每条样本返回 (user_features_dict, gt_item_id)，
    gt_item_id 是该用户在测试集中的正样本物料 ID。
    """

    def __init__(
        self,
        test_pos_df: pd.DataFrame,
        user_feature_cols: List[str],
        user_col: str = 'user_id',
        item_col: str = 'item_id',
    ):
        self.user_col = user_col
        self.item_col = item_col
        self.user_feature_cols = user_feature_cols

        self.user_ids = torch.from_numpy(
            test_pos_df[user_col].values.astype(np.int64)
        ).long()
        self.gt_item_ids = torch.from_numpy(
            test_pos_df[item_col].values.astype(np.int64)
        ).long()

        self.feature_tensors: Dict[str, torch.Tensor] = {}
        for col in user_feature_cols:
            values = test_pos_df[col].values
            if values.dtype in (np.int32, np.int64, int):
                self.feature_tensors[col] = torch.from_numpy(
                    values.astype(np.int64)
                ).long()
            elif isinstance(values[0], (list, np.ndarray)):
                self.feature_tensors[col] = torch.from_numpy(
                    np.stack(values).astype(np.float32)
                ).float()
            else:
                self.feature_tensors[col] = torch.from_numpy(
                    values.astype(np.float32)
                ).float()

        self.size = len(test_pos_df)

    def __len__(self):
        return self.size

    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        user_feat = {'user_id': self.user_ids[idx]}
        for col in self.user_feature_cols:
            user_feat[col] = self.feature_tensors[col][idx]
        gt = self.gt_item_ids[idx]
        return user_feat, gt


class FullItemTestDataModule:
    """
    全量物料测试 DataModule（非 LightningDataModule）。

    提供两个 DataLoader：
      - user_dataloader()  : 测试集正样本用户，每条 (user_features, gt_item_id)
      - item_dataloader()  : 测试集全量物料，每条 item_features

    使用方式：
        dm = FullItemTestDataModule(test_df, items_df, ...)
        # 先编码所有物料
        for item_batch in dm.item_dataloader():
            ...
        # 再逐 batch 用户计算全量得分
        for user_batch, gt_ids in dm.user_dataloader():
            ...
    """

    def __init__(
        self,
        test_df: pd.DataFrame,
        items_df: pd.DataFrame,
        user_feature_cols: List[str],
        item_sparse_features: List[SparseFeature],
        item_dense_features: List[DenseFeature],
        item_sequence_features: List[SequenceFeature],
        label_col: str = 'click',
        user_col: str = 'user_id',
        item_col: str = 'item_id',
        batch_size: int = 256,
        item_batch_size: int = 1024,
        num_workers: int = 0,
    ):
        test_pos_df = test_df[test_df[label_col] == 1].reset_index(drop=True)

        # 去重物料，保留每个 item_id 的第一条记录
        unique_items_df = items_df.drop_duplicates(subset=[item_col]).reset_index(drop=True)

        self._user_dataset = UserTestDataset(
            test_pos_df, user_feature_cols, user_col, item_col
        )
        self._item_dataset = ItemDataset(
            unique_items_df, item_sparse_features, item_dense_features,
            item_sequence_features, item_col
        )

        self.batch_size = batch_size
        self.item_batch_size = item_batch_size
        self.num_workers = num_workers

    def user_dataloader(self) -> DataLoader:
        return DataLoader(
            self._user_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def item_dataloader(self) -> DataLoader:
        return DataLoader(
            self._item_dataset,
            batch_size=self.item_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @property
    def num_users(self) -> int:
        return len(self._user_dataset)

    @property
    def num_items(self) -> int:
        return len(self._item_dataset)

    @property
    def item_ids(self) -> torch.Tensor:
        """返回全量物料 ID tensor，shape (M,)"""
        return self._item_dataset.item_ids
