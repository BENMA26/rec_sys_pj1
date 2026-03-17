"""
序列特征建模 - Multi-Head Self-Attention
用于用户行为历史序列的建模
"""
import torch
import torch.nn as nn
import math
from typing import Optional


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q, K, V 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: (batch_size, seq_len) 1 表示有效位置，0 表示 padding
        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.size()

        # 线性投影并分割成多头
        # (batch_size, seq_len, embed_dim) -> (batch_size, num_heads, seq_len, head_dim)
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        # (batch_size, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用 mask
        if mask is not None:
            # (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        # (batch_size, num_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, V)

        # 合并多头
        # (batch_size, seq_len, embed_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)

        # 输出投影
        output = self.out_proj(output)

        return output


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Transformer 编码器层"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Multi-Head Self-Attention
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)

        # Feed-Forward Network
        self.ffn = PositionwiseFeedForward(embed_dim, ffn_dim, dropout)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            mask: (batch_size, seq_len)
        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        # Self-Attention + Residual + LayerNorm
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-Forward + Residual + LayerNorm
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)

        return x


class SequenceAttentionEncoder(nn.Module):
    """
    序列特征编码器 - 基于 Multi-Head Self-Attention
    用于用户行为历史序列的建模
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        ffn_dim: int = 256,
        max_len: int = 50,
        dropout: float = 0.1,
        pooling: str = 'mean',  # 'mean', 'max', 'last', 'cls'
    ):
        """
        Args:
            vocab_size: 词表大小
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            num_layers: Transformer 层数
            ffn_dim: 前馈网络维度
            max_len: 最大序列长度
            dropout: Dropout 比例
            pooling: 池化方式
                - 'mean': 平均池化
                - 'max': 最大池化
                - 'last': 取最后一个有效位置
                - 'cls': 使用 CLS token
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.pooling = pooling

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)

        # CLS Token (如果使用 cls pooling)
        if pooling == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer Encoder Layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) 序列 ID
            mask: (batch_size, seq_len) 1 表示有效位置，0 表示 padding
        Returns:
            output: (batch_size, embed_dim) 序列表示
        """
        batch_size, seq_len = x.size()

        # Token Embedding
        x = self.token_embedding(x)  # (batch_size, seq_len, embed_dim)

        # 添加 CLS Token
        if self.pooling == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, seq_len+1, embed_dim)

            # 更新 mask
            if mask is not None:
                cls_mask = torch.ones(batch_size, 1, device=mask.device)
                mask = torch.cat([cls_mask, mask], dim=1)

        # Positional Encoding
        x = self.pos_encoding(x)

        # Transformer Encoder Layers
        for layer in self.layers:
            x = layer(x, mask)

        # Pooling
        if self.pooling == 'cls':
            # 取 CLS token
            output = x[:, 0, :]  # (batch_size, embed_dim)

        elif self.pooling == 'mean':
            # 平均池化（考虑 mask）
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand_as(x)
                sum_embeddings = torch.sum(x * mask_expanded, dim=1)
                sum_mask = torch.sum(mask, dim=1, keepdim=True).clamp(min=1)
                output = sum_embeddings / sum_mask
            else:
                output = torch.mean(x, dim=1)

        elif self.pooling == 'max':
            # 最大池化
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand_as(x)
                x = x.masked_fill(mask_expanded == 0, float('-inf'))
            output = torch.max(x, dim=1)[0]

        elif self.pooling == 'last':
            # 取最后一个有效位置
            if mask is not None:
                seq_lengths = mask.sum(dim=1) - 1  # (batch_size,)
                seq_lengths = seq_lengths.clamp(min=0)
                output = x[torch.arange(batch_size), seq_lengths]
            else:
                output = x[:, -1, :]

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        return output


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, embed_dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        Returns:
            output: (batch_size, seq_len, embed_dim)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
