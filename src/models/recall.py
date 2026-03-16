"""
双塔召回模型 - Lightning Module
支持 in-batch 负样本 + 简单负样本（完整物品塔编码）的 Softmax 损失
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Dict, Optional

from src.models.features import SparseFeature, DenseFeature, SequenceFeature, FeatureEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import List, Dict, Optional


class TwoTowerModel(pl.LightningModule):
    """
    双塔召回模型。
 
    训练阶段：softmax loss = 正样本 + easy负样本 + in-batch负样本
    验证阶段：正样本 vs easy负样本排序，计算 Recall@K / MRR（不含in-batch负样本）
    """
 
    def __init__(
        self,
        user_sparse_features,
        user_dense_features,
        user_sequence_features,
        item_sparse_features,
        item_dense_features,
        item_sequence_features,
        hidden_dims: List[int] = [128, 32],
        dropout: float = 0.2,
        temperature: float = 0.05,
        learning_rate: float = 1e-3,
        topk_list: List[int] = [50, 100],
        user_col: str = 'user_id',
        item_col: str = 'item_id',
    ):
        super().__init__()
        self.save_hyperparameters()
 
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.topk_list = topk_list
        self.user_col = user_col
        self.item_col = item_col
 
        # 验证阶段收集正样本排名
        self._eval_pos_ranks: List[torch.Tensor] = []
 
        # 编码器
        self.user_encoder = FeatureEncoder(
            user_sparse_features,
            user_dense_features,
            user_sequence_features,
            hidden_dims,
            dropout,
        )
 
        self.item_encoder = FeatureEncoder(
            item_sparse_features,
            item_dense_features,
            item_sequence_features,
            hidden_dims,
            dropout,
        )
 
    # ===================== 编码 =====================
 
    def encode_user(self, user_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.normalize(self.user_encoder(user_features), p=2, dim=1)
 
    def encode_item(self, item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        return F.normalize(self.item_encoder(item_features), p=2, dim=1)
 
    def _encode_neg_items(self, neg_item_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        neg_item_features: Dict[str, Tensor(B, K, ...)]
        Returns: (B, K, D)
        """
        first = next(iter(neg_item_features.values()))
        B, K = first.shape[0], first.shape[1]
        flat = {k: v.reshape(B * K, *v.shape[2:]) for k, v in neg_item_features.items()}
        vec = self.encode_item(flat)
        return vec.view(B, K, -1)
 
    # ===================== Loss =====================
 
    def _softmax_loss(
        self,
        user_vec: torch.Tensor,       # (B, D)
        pos_item_vec: torch.Tensor,    # (B, D)
        neg_item_vec: torch.Tensor,    # (B, K, D)
    ) -> torch.Tensor:
        """
        训练用 loss：正样本 + easy负样本 + in-batch负样本，组成 softmax cross entropy。
        """
        B = user_vec.size(0)
 
        # 正样本得分 (B, 1)
        pos_scores = (user_vec * pos_item_vec).sum(dim=1, keepdim=True) / self.temperature
 
        # easy 负样本得分 (B, K)
        easy_neg_scores = torch.bmm(
            neg_item_vec, user_vec.unsqueeze(-1)
        ).squeeze(-1) / self.temperature
 
        # in-batch 负样本得分 (B, B-1)
        inbatch_scores = torch.matmul(user_vec, pos_item_vec.T) / self.temperature
        eye = torch.eye(B, device=user_vec.device, dtype=torch.bool)
        inbatch_neg_scores = inbatch_scores[~eye].view(B, B - 1)
 
        # 拼接 [pos | easy_neg | inbatch_neg] → (B, 1+K+B-1)
        logits = torch.cat([pos_scores, easy_neg_scores, inbatch_neg_scores], dim=1)
 
        # 正样本在第0列
        labels = torch.zeros(B, dtype=torch.long, device=user_vec.device)
        return F.cross_entropy(logits, labels)
 
    # ===================== 验证用排名计算 =====================
 
    def _compute_pos_ranks(
        self,
        user_vec: torch.Tensor,       # (B, D)
        pos_item_vec: torch.Tensor,    # (B, D)
        neg_item_vec: torch.Tensor,    # (B, K, D)
    ) -> torch.Tensor:
        """
        计算正样本在 [正样本 + easy负样本] 中的排名。
        不使用 in-batch 负样本，保证评估结果固定、可复现。
 
        Returns: (B,) 每个样本的正样本排名（0-indexed，排名越小越好）
        """
        # 正样本得分 (B,)
        pos_scores = (user_vec * pos_item_vec).sum(dim=1)
 
        # easy 负样本得分 (B, K)
        neg_scores = torch.bmm(
            neg_item_vec, user_vec.unsqueeze(-1)
        ).squeeze(-1)
 
        # 正样本排名 = 有多少个负样本得分 > 正样本得分
        ranks = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)  # (B,)
 
        return ranks
 
    # ===================== 训练 =====================
 
    def training_step(self, batch, batch_idx):
        user_features, pos_item_features, neg_item_features = batch
 
        user_vec = self.encode_user(user_features)
        pos_item_vec = self.encode_item(pos_item_features)
        neg_item_vec = self._encode_neg_items(neg_item_features)
        loss = self._softmax_loss(user_vec, pos_item_vec, neg_item_vec)
 
        self.log('train_loss', loss, prog_bar=True)
        return loss
 
    # ===================== 验证 =====================
 
    def on_validation_epoch_start(self):
        self._eval_pos_ranks.clear()
 
    def validation_step(self, batch, batch_idx):
        user_features, pos_item_features, neg_item_features = batch
 
        user_vec = self.encode_user(user_features)
        pos_item_vec = self.encode_item(pos_item_features)
        neg_item_vec = self._encode_neg_items(neg_item_features)
 
        # 1. 计算 loss（训练用的完整 loss，含 in-batch）
        loss = self._softmax_loss(user_vec, pos_item_vec, neg_item_vec)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
 
        # 2. 计算正样本排名（不含 in-batch，只用 easy 负样本）
        with torch.no_grad():
            ranks = self._compute_pos_ranks(user_vec, pos_item_vec, neg_item_vec)
            self._eval_pos_ranks.append(ranks.detach().cpu())
 
        return loss
 
    def on_validation_epoch_end(self):
        all_ranks = torch.cat(self._eval_pos_ranks, dim=0)  # (N,)
 
        # Recall@K / HitRate@K
        for k in self.topk_list:
            recall = (all_ranks < k).float().mean().item()
            self.log(
                f'val_recall@{k}',
                recall,
                prog_bar=(k == self.topk_list[0]),
                sync_dist=True,
            )
 
        # MRR: 正样本排名倒数的均值
        mrr = (1.0 / (all_ranks.float() + 1)).mean().item()
        self.log('val_mrr', mrr, prog_bar=True, sync_dist=True)
 
        # Mean Rank: 正样本平均排名
        mean_rank = all_ranks.float().mean().item()
        self.log('val_mean_rank', mean_rank, sync_dist=True)
 
        self._eval_pos_ranks.clear()
 
    # ===================== 测试 =====================
 
    def on_test_epoch_start(self):
        self._eval_pos_ranks.clear()
 
    def test_step(self, batch, batch_idx):
        user_features, pos_item_features, neg_item_features = batch
 
        user_vec = self.encode_user(user_features)
        pos_item_vec = self.encode_item(pos_item_features)
        neg_item_vec = self._encode_neg_items(neg_item_features)
 
        with torch.no_grad():
            ranks = self._compute_pos_ranks(user_vec, pos_item_vec, neg_item_vec)
            self._eval_pos_ranks.append(ranks.detach().cpu())
 
    def on_test_epoch_end(self):
        all_ranks = torch.cat(self._eval_pos_ranks, dim=0)
 
        for k in self.topk_list:
            recall = (all_ranks < k).float().mean().item()
            self.log(f'test_recall@{k}', recall)
 
        mrr = (1.0 / (all_ranks.float() + 1)).mean().item()
        self.log('test_mrr', mrr)
 
        mean_rank = all_ranks.float().mean().item()
        self.log('test_mean_rank', mean_rank)
 
        self._eval_pos_ranks.clear()
 
    # ===================== 优化器 =====================
 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': f'val_recall@{self.topk_list[0]}',
        }
