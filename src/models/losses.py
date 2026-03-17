"""
双塔模型损失函数
包含 Pointwise, Pairwise, Listwise 三种损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointwiseLoss(nn.Module):
    """
    Pointwise Loss - 逐点损失
    将每个 user-item 对作为独立样本，预测是否为正样本
    适用于有明确正负样本标签的场景
    """

    def __init__(self):
        super().__init__()

    def forward(self, user_vec: torch.Tensor, item_vec: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_vec: (batch_size, embedding_dim) 用户向量
            item_vec: (batch_size, embedding_dim) 物品向量
            labels: (batch_size,) 标签，1 表示正样本，0 表示负样本
        Returns:
            loss: 标量损失值
        """
        # 计算相似度得分
        scores = torch.sum(user_vec * item_vec, dim=1)  # (batch_size,)

        # 二分类交叉熵损失
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())

        return loss

class PairwiseLoss(nn.Module):
    """
    Pairwise Loss - 成对损失 (BPR Loss)
    对比正样本和负样本的相对顺序
    目标：正样本得分 > 负样本得分
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        user_vec: torch.Tensor,
        pos_item_vec: torch.Tensor,
        neg_item_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            user_vec: (batch_size, embedding_dim) 用户向量
            pos_item_vec: (batch_size, embedding_dim) 正样本物品向量
            neg_item_vec: (batch_size, embedding_dim) 负样本物品向量
        Returns:
            loss: 标量损失值
        """
        # 计算正样本得分
        pos_scores = torch.sum(user_vec * pos_item_vec, dim=1)  # (batch_size,)

        # 计算负样本得分
        neg_scores = torch.sum(user_vec * neg_item_vec, dim=1)  # (batch_size,)

        # BPR Loss: -log(sigmoid(pos_score - neg_score))
        # 等价于: log(1 + exp(neg_score - pos_score))
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        return loss

class PairwiseHingeLoss(nn.Module):
    """
    Pairwise Hinge Loss - 成对铰链损失
    使用 margin-based 损失，更稳定
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        user_vec: torch.Tensor,
        pos_item_vec: torch.Tensor,
        neg_item_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            user_vec: (batch_size, embedding_dim) 用户向量
            pos_item_vec: (batch_size, embedding_dim) 正样本物品向量
            neg_item_vec: (batch_size, embedding_dim) 负样本物品向量
        Returns:
            loss: 标量损失值
        """
        pos_scores = torch.sum(user_vec * pos_item_vec, dim=1)
        neg_scores = torch.sum(user_vec * neg_item_vec, dim=1)

        # Hinge Loss: max(0, margin - (pos_score - neg_score))
        loss = F.relu(self.margin - (pos_scores - neg_scores)).mean()

        return loss

class ListwiseLoss(nn.Module):
    """
    Listwise Loss - 列表损失 (InfoNCE / Contrastive Loss)
    同时考虑一个正样本和多个负样本
    使用 in-batch negatives 策略
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(self, user_vec: torch.Tensor, item_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_vec: (batch_size, embedding_dim) 用户向量
            item_vec: (batch_size, embedding_dim) 物品向量
            假设 batch 中第 i 个 user 和第 i 个 item 是正样本对
        Returns:
            loss: 标量损失值
        """
        # 计算相似度矩阵 (batch_size, batch_size)
        logits = torch.matmul(user_vec, item_vec.T) / self.temperature

        # 对角线为正样本，其余为负样本
        labels = torch.arange(logits.size(0), device=logits.device)

        # 交叉熵损失
        loss = F.cross_entropy(logits, labels)

        return loss

class SampledSoftmaxLoss(nn.Module):
    """
    Sampled Softmax Loss - 采样 Softmax 损失
    适用于物品数量巨大的场景，通过负采样降低计算复杂度
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        user_vec: torch.Tensor,
        pos_item_vec: torch.Tensor,
        neg_item_vecs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            user_vec: (batch_size, embedding_dim) 用户向量
            pos_item_vec: (batch_size, embedding_dim) 正样本物品向量
            neg_item_vecs: (batch_size, num_negatives, embedding_dim) 负样本物品向量
        Returns:
            loss: 标量损失值
        """
        batch_size = user_vec.size(0)

        # 正样本得分 (batch_size, 1)
        pos_scores = torch.sum(user_vec * pos_item_vec, dim=1, keepdim=True) / self.temperature

        # 负样本得分 (batch_size, num_negatives)
        neg_scores = torch.bmm(
            neg_item_vecs,
            user_vec.unsqueeze(-1)
        ).squeeze(-1) / self.temperature

        # 拼接正负样本得分 (batch_size, 1 + num_negatives)
        logits = torch.cat([pos_scores, neg_scores], dim=1)

        # 正样本的标签是 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        # 交叉熵损失
        loss = F.cross_entropy(logits, labels)

        return loss

class TripletLoss(nn.Module):
    """
    Triplet Loss - 三元组损失
    确保 d(anchor, positive) + margin < d(anchor, negative)
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        user_vec: torch.Tensor,
        pos_item_vec: torch.Tensor,
        neg_item_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            user_vec: (batch_size, embedding_dim) 用户向量 (anchor)
            pos_item_vec: (batch_size, embedding_dim) 正样本物品向量
            neg_item_vec: (batch_size, embedding_dim) 负样本物品向量
        Returns:
            loss: 标量损失值
        """
        # 计算距离（使用欧氏距离）
        pos_dist = torch.sum((user_vec - pos_item_vec) ** 2, dim=1)
        neg_dist = torch.sum((user_vec - neg_item_vec) ** 2, dim=1)

        # Triplet Loss: max(0, pos_dist - neg_dist + margin)
        loss = F.relu(pos_dist - neg_dist + self.margin).mean()

        return loss
