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
        training_mode: str = 'listwise',
        use_inbatch_neg: bool = True,
        margin: float = 0.5,
        user_col: str = 'user_id',
        item_col: str = 'item_id',
    ):
        super().__init__()
        self.save_hyperparameters()
 
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.topk_list = topk_list
        self.training_mode = training_mode
        self.use_inbatch_neg = use_inbatch_neg
        self.margin = margin
        self.user_col = user_col
        self.item_col = item_col
 
        self._eval_pos_ranks: List[torch.Tensor] = []
 
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
        neg_item_features: Dict[str, Tensor(K, ...)]  共享负样本
        Returns: (K, D)
        """
        return self.encode_item(neg_item_features)
 
    # ===================== Loss =====================

    def _pointwise_loss(
        self,
        user_vec: torch.Tensor,       # (B, D)
        pos_item_vec: torch.Tensor,    # (B, D)
        neg_item_vec: torch.Tensor,    # (K, D)
    ) -> torch.Tensor:
        B = user_vec.size(0)
        K = neg_item_vec.size(0)

        pos_scores = (user_vec * pos_item_vec).sum(dim=1)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )

        if K > 0:
            neg_scores = user_vec @ neg_item_vec.t()  # (B, K)
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )
            return (pos_loss + neg_loss) / 2
        else:
            return pos_loss

    def _pairwise_loss(
        self,
        user_vec: torch.Tensor,       # (B, D)
        pos_item_vec: torch.Tensor,    # (B, D)
        neg_item_vec: torch.Tensor,    # (K, D)
    ) -> torch.Tensor:
        B = user_vec.size(0)
        K = neg_item_vec.size(0)

        if K == 0:
            return self._pointwise_loss(user_vec, pos_item_vec, neg_item_vec)

        pos_scores = (user_vec * pos_item_vec).sum(dim=1, keepdim=True)  # (B, 1)
        neg_scores = user_vec @ neg_item_vec.t()  # (B, K)

        loss = F.relu(self.margin - pos_scores + neg_scores).mean()
        return loss

    def _softmax_loss(
        self,
        user_vec: torch.Tensor,       # (B, D)
        pos_item_vec: torch.Tensor,    # (B, D)
        neg_item_vec: torch.Tensor,    # (K, D)
    ) -> torch.Tensor:

        B = user_vec.size(0)
        K = neg_item_vec.size(0)

        if self.use_inbatch_neg and B > 1:
            inbatch_scores = user_vec @ pos_item_vec.t() / self.temperature  # (B, B)
            labels = torch.arange(B, device=user_vec.device)
        else:
            inbatch_scores = (user_vec * pos_item_vec).sum(dim=1, keepdim=True) / self.temperature  # (B, 1)
            labels = torch.zeros(B, dtype=torch.long, device=user_vec.device)

        if K > 0:
            easy_neg_scores = user_vec @ neg_item_vec.t() / self.temperature  # (B, K)
            logits = torch.cat([inbatch_scores, easy_neg_scores], dim=1)
        else:
            logits = inbatch_scores

        return F.cross_entropy(logits, labels)

    # ===================== 验证用排名计算 =====================
 
    def _compute_pos_ranks(
        self,
        user_vec: torch.Tensor,       # (B, D)
        pos_item_vec: torch.Tensor,    # (B, D)
        neg_item_vec: torch.Tensor,    # (K, D)
    ) -> torch.Tensor:
        """
        计算正样本在 [正样本 + easy负样本] 中的排名。
        不使用 in-batch 负样本，保证评估结果固定、可复现。
 
        Returns: (B,) 每个样本的正样本排名（0-indexed，排名越小越好）
        """
        pos_scores = (user_vec * pos_item_vec).sum(dim=1)       # (B,)
        neg_scores = user_vec @ neg_item_vec.t()                # (B, K)
        ranks = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)  # (B,)
        return ranks
 
    # ===================== 训练 =====================

    def training_step(self, batch, batch_idx):
        user_features, pos_item_features, neg_item_features = batch

        user_vec = self.encode_user(user_features)
        pos_item_vec = self.encode_item(pos_item_features)
        neg_item_vec = self._encode_neg_items(neg_item_features)

        if self.training_mode == 'pointwise':
            loss = self._pointwise_loss(user_vec, pos_item_vec, neg_item_vec)
        elif self.training_mode == 'pairwise':
            loss = self._pairwise_loss(user_vec, pos_item_vec, neg_item_vec)
        elif self.training_mode == 'listwise':
            loss = self._softmax_loss(user_vec, pos_item_vec, neg_item_vec)
        else:
            raise ValueError(f"Unknown training_mode: {self.training_mode}")

        self.log('train_loss', loss, prog_bar=True)
        return loss
 
    # ===================== 验证 =====================

    def on_validation_epoch_start(self):
        self._eval_pos_ranks.clear()
        self._eval_pos_scores = []
        self._eval_neg_scores = []

    def validation_step(self, batch, batch_idx):
        user_features, pos_item_features, neg_item_features = batch

        user_vec = self.encode_user(user_features)
        pos_item_vec = self.encode_item(pos_item_features)
        neg_item_vec = self._encode_neg_items(neg_item_features)  # (K, D)

        with torch.no_grad():
            pos_scores = (user_vec * pos_item_vec).sum(dim=1)    # (B,)
            neg_scores = user_vec @ neg_item_vec.t()             # (B, K)
            ranks = self._compute_pos_ranks(user_vec, pos_item_vec, neg_item_vec)

            self._eval_pos_ranks.append(ranks.detach().cpu())
            self._eval_pos_scores.append(pos_scores.detach().cpu())
            self._eval_neg_scores.append(neg_scores.detach().cpu())

    def on_validation_epoch_end(self):
        all_ranks = torch.cat(self._eval_pos_ranks, dim=0)
        all_pos_scores = torch.cat(self._eval_pos_scores, dim=0)
        all_neg_scores = torch.cat(self._eval_neg_scores, dim=0)

        for k in self.topk_list:
            recall = (all_ranks < k).float().mean().item()
            self.log(
                f'val_recall@{k}',
                recall,
                prog_bar=(k == self.topk_list[0]),
                sync_dist=True,
            )

        mrr = (1.0 / (all_ranks.float() + 1)).mean().item()
        self.log('val_mrr', mrr, prog_bar=True, sync_dist=True)

        N, K = all_neg_scores.shape
        labels = torch.cat([
            torch.ones(N),
            torch.zeros(N * K)
        ])
        scores = torch.cat([
            all_pos_scores,
            all_neg_scores.flatten()
        ])

        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(labels.numpy(), scores.numpy())
            self.log('val_auc', auc, prog_bar=True, sync_dist=True)
        except:
            pass

        N = len(all_ranks)
        for k in self.topk_list:
            hit_count = (all_ranks < k).float().sum().item()
            precision = hit_count / (N * k)
            self.log(f'val_precision@{k}', precision, sync_dist=True)

        mean_rank = all_ranks.float().mean().item()
        self.log('val_mean_rank', mean_rank, sync_dist=True)

        self._eval_pos_ranks.clear()
        self._eval_pos_scores.clear()
        self._eval_neg_scores.clear()

    # ===================== 测试 =====================

    def on_test_epoch_start(self):
        self._eval_pos_ranks.clear()
        self._eval_pos_scores = []
        self._eval_neg_scores = []

    def test_step(self, batch, batch_idx):
        user_features, pos_item_features, neg_item_features = batch

        user_vec = self.encode_user(user_features)
        pos_item_vec = self.encode_item(pos_item_features)
        neg_item_vec = self._encode_neg_items(neg_item_features)  # (K, D)

        with torch.no_grad():
            pos_scores = (user_vec * pos_item_vec).sum(dim=1)    # (B,)
            neg_scores = user_vec @ neg_item_vec.t()             # (B, K)
            ranks = self._compute_pos_ranks(user_vec, pos_item_vec, neg_item_vec)

            self._eval_pos_ranks.append(ranks.detach().cpu())
            self._eval_pos_scores.append(pos_scores.detach().cpu())
            self._eval_neg_scores.append(neg_scores.detach().cpu())

    def on_test_epoch_end(self):
        all_ranks = torch.cat(self._eval_pos_ranks, dim=0)
        all_pos_scores = torch.cat(self._eval_pos_scores, dim=0)
        all_neg_scores = torch.cat(self._eval_neg_scores, dim=0)

        for k in self.topk_list:
            recall = (all_ranks < k).float().mean().item()
            self.log(f'test_recall@{k}', recall)

        mrr = (1.0 / (all_ranks.float() + 1)).mean().item()
        self.log('test_mrr', mrr)

        N, K = all_neg_scores.shape
        labels = torch.cat([
            torch.ones(N),
            torch.zeros(N * K)
        ])
        scores = torch.cat([
            all_pos_scores,
            all_neg_scores.flatten()
        ])

        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(labels.numpy(), scores.numpy())
            self.log('test_auc', auc)
        except:
            pass

        N = len(all_ranks)
        for k in self.topk_list:
            hit_count = (all_ranks < k).float().sum().item()
            precision = hit_count / (N * k)
            self.log(f'test_precision@{k}', precision)

        mean_rank = all_ranks.float().mean().item()
        self.log('test_mean_rank', mean_rank)

        self._eval_pos_ranks.clear()
        self._eval_pos_scores.clear()
        self._eval_neg_scores.clear()
 
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

    # ===================== 全量物料测试 =====================

    @torch.no_grad()
    def evaluate_full_item(self, test_dm, topk_list=None, device=None):
        """
        全量物料测试：
          1. 编码测试集中所有物料 -> item_vecs (M, D)
          2. 逐 batch 用户计算与所有物料的得分，取 topk
          3. 判断 gt_item_id 是否在 topk 中，统计 Recall@K

        Args:
            test_dm : FullItemTestDataModule 实例
            topk_list: 要计算的 K 列表，默认使用 self.topk_list
            device   : 推理设备，默认使用模型当前设备

        Returns:
            dict: {f'test_recall@{k}': float, ...}
        """
        if topk_list is None:
            topk_list = self.topk_list
        if device is None:
            device = next(self.parameters()).device

        self.eval()

        # ---- Step 1: 编码所有物料 ----
        all_item_vecs = []
        for item_batch in test_dm.item_dataloader():
            # item_batch: Dict[str, Tensor]，包含 item_id 和物料特征
            item_feats = {k: v.to(device) for k, v in item_batch.items() if k != 'item_id'}
            vecs = self.encode_item(item_feats)   # (batch, D)
            all_item_vecs.append(vecs.cpu())
        item_vecs = torch.cat(all_item_vecs, dim=0)  # (M, D)
        item_ids  = test_dm.item_ids                  # (M,) cpu tensor

        # ---- Step 2: 逐 batch 用户计算全量得分，统计命中 ----
        hits = {k: 0 for k in topk_list}
        total = 0

        for user_batch, gt_ids in test_dm.user_dataloader():
            # user_batch: Dict[str, Tensor]
            user_feats = {k: v.to(device) for k, v in user_batch.items()}
            user_vecs  = self.encode_user(user_feats)   # (B, D)

            # 得分矩阵 (B, M)
            scores = user_vecs @ item_vecs.to(device).t()

            gt_ids = gt_ids.to(device)  # (B,)
            B = scores.size(0)
            total += B

            max_k = max(topk_list)
            # topk 索引 (B, max_k)
            _, topk_indices = scores.topk(max_k, dim=1, largest=True, sorted=True)
            # 转换为 item_id
            topk_item_ids = item_ids.to(device)[topk_indices]  # (B, max_k)

            for k in topk_list:
                # 判断 gt 是否在 top-k 中
                hit = (topk_item_ids[:, :k] == gt_ids.unsqueeze(1)).any(dim=1).sum().item()
                hits[k] += hit

        results = {f'test_recall@{k}': hits[k] / total for k in topk_list}
        for k, v in results.items():
            print(f"  {k}: {v:.4f}  ({hits[int(k.split('@')[1])]}/{total})")
        return results

'''
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
        training_mode: str = 'listwise',
        use_inbatch_neg: bool = True,
        margin: float = 0.5,
        user_col: str = 'user_id',
        item_col: str = 'item_id',
    ):
        """
        Args:
            training_mode: 训练模式，可选 'pointwise', 'pairwise', 'listwise'
            use_inbatch_neg: listwise 模式下是否使用 in-batch 负样本
            margin: pairwise 模式下的 margin 参数
        """
        super().__init__()
        self.save_hyperparameters()
 
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.topk_list = topk_list
        self.training_mode = training_mode
        self.use_inbatch_neg = use_inbatch_neg
        self.margin = margin
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

    def _pointwise_loss(
        self,
        user_vec: torch.Tensor,       # (B, D)
        pos_item_vec: torch.Tensor,    # (B, D)
        neg_item_vec: torch.Tensor,    # (B, K, D)
    ) -> torch.Tensor:
        """
        Pointwise loss：每个 (user, item) 对独立预测，BCE loss。
        正样本 label=1，负样本 label=0。
        """
        B, K = neg_item_vec.shape[0], neg_item_vec.shape[1]

        # 正样本得分 (B,)
        pos_scores = (user_vec * pos_item_vec).sum(dim=1)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )

        # 负样本得分 (B, K)
        if K > 0:
            neg_scores = torch.bmm(
                neg_item_vec, user_vec.unsqueeze(-1)
            ).squeeze(-1)  # (B, K)
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_scores, torch.zeros_like(neg_scores)
            )
            return (pos_loss + neg_loss) / 2
        else:
            return pos_loss

    def _pairwise_loss(
        self,
        user_vec: torch.Tensor,       # (B, D)
        pos_item_vec: torch.Tensor,    # (B, D)
        neg_item_vec: torch.Tensor,    # (B, K, D)
    ) -> torch.Tensor:
        """
        Pairwise loss：每个正样本与每个负样本配对，hinge loss。
        loss = max(0, margin - pos_score + neg_score)
        """
        B, K = neg_item_vec.shape[0], neg_item_vec.shape[1]

        if K == 0:
            # 没有负样本，退化为 pointwise
            return self._pointwise_loss(user_vec, pos_item_vec, neg_item_vec)

        # 正样本得分 (B, 1)
        pos_scores = (user_vec * pos_item_vec).sum(dim=1, keepdim=True)

        # 负样本得分 (B, K)
        neg_scores = torch.bmm(
            neg_item_vec, user_vec.unsqueeze(-1)
        ).squeeze(-1)

        # BPR loss: -log(sigmoid(pos - neg))，等价于 log(1 + exp(neg - pos))
        # 或使用 margin ranking loss: max(0, margin - pos + neg)
        loss = F.relu(self.margin - pos_scores + neg_scores).mean()
        return loss

    def _softmax_loss(
        self,
        user_vec: torch.Tensor,
        pos_item_vec: torch.Tensor,
        neg_item_vec: torch.Tensor,
    ) -> torch.Tensor:
    
        B = user_vec.size(0)
        K = neg_item_vec.size(1)
    
        if self.use_inbatch_neg and B > 1:
            inbatch_scores = torch.matmul(user_vec, pos_item_vec.t()) / self.temperature
            labels = torch.arange(B, device=user_vec.device)
        else:
            inbatch_scores = (user_vec * pos_item_vec).sum(dim=1, keepdim=True) / self.temperature
            labels = torch.zeros(B, dtype=torch.long, device=user_vec.device)
    
        if K > 0:
            easy_neg_scores = torch.einsum('bkd,bd->bk', neg_item_vec, user_vec) / self.temperature
            logits = torch.cat([inbatch_scores, easy_neg_scores], dim=1)
        else:
            logits = inbatch_scores
    
        return F.cross_entropy(logits, labels)
    
    def _softmax_loss(
        self,
        user_vec: torch.Tensor,       # (B, D)
        pos_item_vec: torch.Tensor,    # (B, D)
        neg_item_vec: torch.Tensor,    # (B, K, D)，K=0 时表示不使用 easy neg
    ) -> torch.Tensor:
        """
        训练用 loss，支持三种负样本模式：
          - use_inbatch_neg=True,  K>0：in-batch + easy neg（默认）
          - use_inbatch_neg=True,  K=0：仅 in-batch neg
          - use_inbatch_neg=False, K>0：仅 easy neg
        """
        B = user_vec.size(0)

        # 正样本得分 (B, 1)
        pos_scores = (user_vec * pos_item_vec).sum(dim=1, keepdim=True) / self.temperature

        parts = [pos_scores]

        # easy 负样本得分 (B, K)
        if neg_item_vec.size(1) > 0:
            easy_neg_scores = torch.bmm(
                neg_item_vec, user_vec.unsqueeze(-1)
            ).squeeze(-1) / self.temperature
            parts.append(easy_neg_scores)

        # in-batch 负样本得分 (B, B-1)
        if self.use_inbatch_neg and B > 1:
            inbatch_scores = torch.matmul(user_vec, pos_item_vec.T) / self.temperature
            eye = torch.eye(B, device=user_vec.device, dtype=torch.bool)
            inbatch_neg_scores = inbatch_scores[~eye].view(B, B - 1)
            parts.append(inbatch_neg_scores)

        # 拼接 logits，正样本始终在第 0 列
        logits = torch.cat(parts, dim=1)
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

        if self.training_mode == 'pointwise':
            loss = self._pointwise_loss(user_vec, pos_item_vec, neg_item_vec)
        elif self.training_mode == 'pairwise':
            loss = self._pairwise_loss(user_vec, pos_item_vec, neg_item_vec)
        elif self.training_mode == 'listwise':
            loss = self._softmax_loss(user_vec, pos_item_vec, neg_item_vec)
        else:
            raise ValueError(f"Unknown training_mode: {self.training_mode}")

        self.log('train_loss', loss, prog_bar=True)
        return loss
 
    # ===================== 验证 =====================

    def on_validation_epoch_start(self):
        self._eval_pos_ranks.clear()
        self._eval_pos_scores = []
        self._eval_neg_scores = []


    def validation_step(self, batch, batch_idx):
        user_features, pos_item_features, neg_item_features = batch

        user_vec = self.encode_user(user_features)
        pos_item_vec = self.encode_item(pos_item_features)
        neg_item_vec = self._encode_neg_items(neg_item_features)

        with torch.no_grad():
            # 正样本得分 (B,)
            pos_scores = (user_vec * pos_item_vec).sum(dim=1)

            # 负样本得分 (B, K)
            neg_scores = torch.bmm(
                neg_item_vec, user_vec.unsqueeze(-1)
            ).squeeze(-1)

            # 计算排名
            ranks = self._compute_pos_ranks(user_vec, pos_item_vec, neg_item_vec)

            self._eval_pos_ranks.append(ranks.detach().cpu())
            self._eval_pos_scores.append(pos_scores.detach().cpu())
            self._eval_neg_scores.append(neg_scores.detach().cpu())


    def on_validation_epoch_end(self):
        all_ranks = torch.cat(self._eval_pos_ranks, dim=0)  # (N,)
        all_pos_scores = torch.cat(self._eval_pos_scores, dim=0)  # (N,)
        all_neg_scores = torch.cat(self._eval_neg_scores, dim=0)  # (N, K)

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

        # AUC: 正样本 vs 所有负样本的二分类 AUC
        N, K = all_neg_scores.shape
        labels = torch.cat([
            torch.ones(N),  # 正样本 label=1
            torch.zeros(N * K)  # 负样本 label=0
        ])
        scores = torch.cat([
            all_pos_scores,
            all_neg_scores.flatten()
        ])

        # 计算 AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(labels.numpy(), scores.numpy())
            self.log('val_auc', auc, prog_bar=True, sync_dist=True)
        except:
            pass

        # Precision@K: 推荐 K 个物品中正样本的平均比例
        N = len(all_ranks)
        for k in self.topk_list:
            # 分子：正样本在 top-K 中的样本数
            # 分母：N × K（总共推荐了 N×K 个物品）
            hit_count = (all_ranks < k).float().sum().item()
            precision = hit_count / (N * k)
            self.log(f'val_precision@{k}', precision, sync_dist=True)

        # 平均排名
        mean_rank = all_ranks.float().mean().item()
        self.log('val_mean_rank', mean_rank, sync_dist=True)

        self._eval_pos_ranks.clear()
        self._eval_pos_scores.clear()
        self._eval_neg_scores.clear()

    # ===================== 测试 =====================

    def on_test_epoch_start(self):
        self._eval_pos_ranks.clear()
        self._eval_pos_scores = []
        self._eval_neg_scores = []


    def test_step(self, batch, batch_idx):
        user_features, pos_item_features, neg_item_features = batch

        user_vec = self.encode_user(user_features)
        pos_item_vec = self.encode_item(pos_item_features)
        neg_item_vec = self._encode_neg_items(neg_item_features)

        with torch.no_grad():
            # 正样本得分 (B,)
            pos_scores = (user_vec * pos_item_vec).sum(dim=1)

            # 负样本得分 (B, K)
            neg_scores = torch.bmm(
                neg_item_vec, user_vec.unsqueeze(-1)
            ).squeeze(-1)

            # 计算排名
            ranks = self._compute_pos_ranks(user_vec, pos_item_vec, neg_item_vec)

            self._eval_pos_ranks.append(ranks.detach().cpu())
            self._eval_pos_scores.append(pos_scores.detach().cpu())
            self._eval_neg_scores.append(neg_scores.detach().cpu())


    def on_test_epoch_end(self):
        all_ranks = torch.cat(self._eval_pos_ranks, dim=0)
        all_pos_scores = torch.cat(self._eval_pos_scores, dim=0)
        all_neg_scores = torch.cat(self._eval_neg_scores, dim=0)

        # Recall@K
        for k in self.topk_list:
            recall = (all_ranks < k).float().mean().item()
            self.log(f'test_recall@{k}', recall)

        # MRR
        mrr = (1.0 / (all_ranks.float() + 1)).mean().item()
        self.log('test_mrr', mrr)

        # AUC
        N, K = all_neg_scores.shape
        labels = torch.cat([
            torch.ones(N),
            torch.zeros(N * K)
        ])
        scores = torch.cat([
            all_pos_scores,
            all_neg_scores.flatten()
        ])

        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(labels.numpy(), scores.numpy())
            self.log('test_auc', auc)
        except:
            pass

        # Precision@K: 推荐 K 个物品中正样本的平均比例
        N = len(all_ranks)
        for k in self.topk_list:
            # 分子：正样本在 top-K 中的样本数
            # 分母：N × K（总共推荐了 N×K 个物品）
            hit_count = (all_ranks < k).float().sum().item()
            precision = hit_count / (N * k)
            self.log(f'test_precision@{k}', precision)

        # Mean Rank
        mean_rank = all_ranks.float().mean().item()
        self.log('test_mean_rank', mean_rank)

        self._eval_pos_ranks.clear()
        self._eval_pos_scores.clear()
        self._eval_neg_scores.clear()
 
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
'
'''