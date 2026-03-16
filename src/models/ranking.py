"""
排序模型集合
包含 Share Bottom, MOE, MMOE 及其 TorchJD 版本
用于 CTR 和 CVR 多任务学习
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, List, Optional
from torchjd.aggregation import UPGrad, MGDA, PCGrad, GradDrop
from torchjd.autojac import backward
from torchmetrics import AUROC

# ============================================================================
# 基础组件
# ============================================================================
class Expert(nn.Module):
    """专家网络"""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Gate(nn.Module):
    """门控网络"""

    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.gate(x)

class Tower(nn.Module):
    """任务塔"""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ============================================================================
# Share Bottom 模型
# ============================================================================

class ShareBottomModel(pl.LightningModule):
    """Share Bottom 多任务学习模型"""

    def __init__(
        self,
        feature_names: List[str],
        feature_dims: Dict[str, int],
        embedding_dim: int = 32,
        shared_hidden_dims: List[int] = [256, 128],
        tower_hidden_dims: List[int] = [64],
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        ctr_weight: float = 0.5,
        cvr_weight: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_names = feature_names
        self.learning_rate = learning_rate
        self.ctr_weight = ctr_weight
        self.cvr_weight = cvr_weight

        # 特征嵌入层
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(feature_dims[feat], embedding_dim)
            for feat in feature_names
        })

        # 输入维度
        input_dim = len(feature_names) * embedding_dim

        # 共享底层网络
        self.shared_bottom = Expert(input_dim, shared_hidden_dims, dropout)

        # CTR 任务塔
        self.ctr_tower = Tower(shared_hidden_dims[-1], tower_hidden_dims, dropout)

        # CVR 任务塔
        self.cvr_tower = Tower(shared_hidden_dims[-1], tower_hidden_dims, dropout)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """前向传播"""
        # 特征嵌入
        embeddings = [self.embeddings[feat](inputs[feat]) for feat in self.feature_names]
        x = torch.cat(embeddings, dim=-1)

        # 共享底层
        shared_output = self.shared_bottom(x)

        # 两个任务塔
        ctr_logit = self.ctr_tower(shared_output)
        cvr_logit = self.cvr_tower(shared_output)

        return ctr_logit.squeeze(-1), cvr_logit.squeeze(-1)

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        inputs, ctr_labels, cvr_labels = batch
        ctr_logits, cvr_logits = self(inputs)

        ctr_loss = nn.functional.binary_cross_entropy_with_logits(ctr_logits, ctr_labels.float())
        cvr_loss = nn.functional.binary_cross_entropy_with_logits(cvr_logits, cvr_labels.float())
        loss = self.ctr_weight * ctr_loss + self.cvr_weight * cvr_loss

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_ctr_loss', ctr_loss)
        self.log('train_cvr_loss', cvr_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        inputs, ctr_labels, cvr_labels = batch
        ctr_logits, cvr_logits = self(inputs)

        ctr_loss = nn.functional.binary_cross_entropy_with_logits(ctr_logits, ctr_labels.float())
        cvr_loss = nn.functional.binary_cross_entropy_with_logits(cvr_logits, cvr_labels.float())
        loss = self.ctr_weight * ctr_loss + self.cvr_weight * cvr_loss

        ctr_preds = torch.sigmoid(ctr_logits)
        cvr_preds = torch.sigmoid(cvr_logits)
        ctr_acc = ((ctr_preds > 0.5) == ctr_labels).float().mean()
        cvr_acc = ((cvr_preds > 0.5) == cvr_labels).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_ctr_loss', ctr_loss)
        self.log('val_cvr_loss', cvr_loss)
        self.log('val_ctr_acc', ctr_acc, prog_bar=True)
        self.log('val_cvr_acc', cvr_acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

# ============================================================================
# Share Bottom with TorchJD
# ============================================================================

class ShareBottomModelWithTorchJD(pl.LightningModule):
    """Share Bottom 多任务学习模型 - 使用 TorchJD 梯度聚合"""

    def __init__(
        self,
        feature_names: List[str],
        feature_dims: Dict[str, int],
        embedding_dim: int = 32,
        shared_hidden_dims: List[int] = [256, 128],
        tower_hidden_dims: List[int] = [64],
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        aggregation_method: str = 'upgrad',
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_names = feature_names
        self.learning_rate = learning_rate
        self.aggregation_method = aggregation_method

        # 特征嵌入层
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(feature_dims[feat], embedding_dim)
            for feat in feature_names
        })

        input_dim = len(feature_names) * embedding_dim

        # 共享底层网络
        self.shared_bottom = Expert(input_dim, shared_hidden_dims, dropout)

        # CTR 任务塔
        self.ctr_tower = Tower(shared_hidden_dims[-1], tower_hidden_dims, dropout)

        # CVR 任务塔
        self.cvr_tower = Tower(shared_hidden_dims[-1], tower_hidden_dims, dropout)

        # 梯度聚合器
        self.aggregator = self._create_aggregator(aggregation_method)

        # 使用手动优化
        self.automatic_optimization = False

    def _create_aggregator(self, method: str):
        """创建梯度聚合器"""
        if method == 'upgrad':
            return UPGrad()
        elif method == 'mgda':
            return MGDA()
        elif method == 'pcgrad':
            return PCGrad()
        elif method == 'cagrad':
            return CAGrad(c=0.5)
        elif method == 'graddrop':
            return GradDrop(leak=0.0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """前向传播"""
        embeddings = [self.embeddings[feat](inputs[feat]) for feat in self.feature_names]
        x = torch.cat(embeddings, dim=-1)

        shared_output = self.shared_bottom(x)

        ctr_logit = self.ctr_tower(shared_output)
        cvr_logit = self.cvr_tower(shared_output)

        return ctr_logit.squeeze(-1), cvr_logit.squeeze(-1)

    def training_step(self, batch, batch_idx):
        """训练步骤 - 使用 TorchJD 梯度聚合"""
        optimizer = self.optimizers()
        optimizer.zero_grad()

        inputs, ctr_labels, cvr_labels = batch
        ctr_logits, cvr_logits = self(inputs)

        ctr_loss = nn.functional.binary_cross_entropy_with_logits(ctr_logits, ctr_labels.float())
        cvr_loss = nn.functional.binary_cross_entropy_with_logits(cvr_logits, cvr_labels.float())

        losses = [ctr_loss, cvr_loss]
        self.aggregator.backward(losses, self.parameters())

        optimizer.step()

        loss = ctr_loss + cvr_loss

        self.log('train_total_loss', loss, prog_bar=True)
        self.log('train_ctr_loss', ctr_loss)
        self.log('train_cvr_loss', cvr_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        inputs, ctr_labels, cvr_labels = batch
        ctr_logits, cvr_logits = self(inputs)

        ctr_loss = nn.functional.binary_cross_entropy_with_logits(ctr_logits, ctr_labels.float())
        cvr_loss = nn.functional.binary_cross_entropy_with_logits(cvr_logits, cvr_labels.float())
        loss = ctr_loss + cvr_loss

        ctr_preds = torch.sigmoid(ctr_logits)
        cvr_preds = torch.sigmoid(cvr_logits)
        ctr_acc = ((ctr_preds > 0.5) == ctr_labels).float().mean()
        cvr_acc = ((cvr_preds > 0.5) == cvr_labels).float().mean()

        self.log('val_total_loss', loss, prog_bar=True)
        self.log('val_ctr_loss', ctr_loss)
        self.log('val_cvr_loss', cvr_loss)
        self.log('val_ctr_acc', ctr_acc, prog_bar=True)
        self.log('val_cvr_acc', cvr_acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_total_loss'
        }

# ============================================================================
# MOE 模型
# ============================================================================

class MOEModel(pl.LightningModule):
    """MMOE (Multi-gate Mixture-of-Experts) 多任务学习模型"""

    def __init__(
        self,
        feature_names: List[str],
        feature_dims: Dict[str, int],
        embedding_dim: int = 32,
        num_experts: int = 3,
        expert_hidden_dims: List[int] = [256, 128],
        tower_hidden_dims: List[int] = [64],
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        ctr_weight: float = 0.5,
        cvr_weight: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_names = feature_names
        self.learning_rate = learning_rate
        self.ctr_weight = ctr_weight
        self.cvr_weight = cvr_weight

        # 特征嵌入层
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(feature_dims[feat], embedding_dim)
            for feat in feature_names
        })

        input_dim = len(feature_names) * embedding_dim

        # 专家网络
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dims, dropout)
            for _ in range(num_experts)
        ])

        # CTR 门控网络
        self.ctr_gate = Gate(input_dim, num_experts)

        # CVR 门控网络
        self.cvr_gate = Gate(input_dim, num_experts)

        # CTR 任务塔
        self.ctr_tower = Tower(expert_hidden_dims[-1], tower_hidden_dims, dropout)

        # CVR 任务塔
        self.cvr_tower = Tower(expert_hidden_dims[-1], tower_hidden_dims, dropout)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """前向传播"""
        # 特征嵌入
        embeddings = [self.embeddings[feat](inputs[feat]) for feat in self.feature_names]
        x = torch.cat(embeddings, dim=-1)

        # 专家输出
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # CTR 任务
        ctr_gate_weights = self.ctr_gate(x).unsqueeze(-1)
        ctr_expert_output = torch.sum(expert_outputs * ctr_gate_weights, dim=1)
        ctr_logit = self.ctr_tower(ctr_expert_output)

        # CVR 任务
        cvr_gate_weights = self.cvr_gate(x).unsqueeze(-1)
        cvr_expert_output = torch.sum(expert_outputs * cvr_gate_weights, dim=1)
        cvr_logit = self.cvr_tower(cvr_expert_output)

        return ctr_logit.squeeze(-1), cvr_logit.squeeze(-1)

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        inputs, labels = batch
        ctr_labels = labels["click"]
        cvr_labels = labels["purchase"]
        ctr_logits, cvr_logits = self(inputs)

        ctr_loss = nn.functional.binary_cross_entropy_with_logits(ctr_logits, ctr_labels.float())
        cvr_loss = nn.functional.binary_cross_entropy_with_logits(cvr_logits, cvr_labels.float())
        loss = self.ctr_weight * ctr_loss + self.cvr_weight * cvr_loss

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_ctr_loss', ctr_loss)
        self.log('train_cvr_loss', cvr_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        inputs, labels = batch
        ctr_labels = labels["click"]
        cvr_labels = labels["purchase"]
        ctr_logits, cvr_logits = self(inputs)

        ctr_loss = nn.functional.binary_cross_entropy_with_logits(ctr_logits, ctr_labels.float())
        cvr_loss = nn.functional.binary_cross_entropy_with_logits(cvr_logits, cvr_labels.float())
        loss = self.ctr_weight * ctr_loss + self.cvr_weight * cvr_loss

        ctr_preds = torch.sigmoid(ctr_logits)
        cvr_preds = torch.sigmoid(cvr_logits)
        ctr_acc = ((ctr_preds > 0.5) == ctr_labels).float().mean()
        cvr_acc = ((cvr_preds > 0.5) == cvr_labels).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_ctr_loss', ctr_loss)
        self.log('val_cvr_loss', cvr_loss)
        self.log('val_ctr_acc', ctr_acc, prog_bar=True)
        self.log('val_cvr_acc', cvr_acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

# ============================================================================
# MOE with TorchJD
# ============================================================================

class MOEModelWithTorchJD(pl.LightningModule):
    """MOE 多任务学习模型 - 使用 TorchJD 梯度聚合"""

    def __init__(
        self,
        feature_names: List[str],
        feature_dims: Dict[str, int],
        embedding_dim: int = 32,
        num_experts: int = 3,
        expert_hidden_dims: List[int] = [256, 128],
        tower_hidden_dims: List[int] = [64],
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        aggregation_method: str = 'upgrad',
    ):
        super().__init__()
        self.save_hyperparameters()

        self.feature_names = feature_names
        self.learning_rate = learning_rate
        self.aggregation_method = aggregation_method

        # 特征嵌入层
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(feature_dims[feat], embedding_dim)
            for feat in feature_names
        })

        input_dim = len(feature_names) * embedding_dim

        # 专家网络
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dims, dropout)
            for _ in range(num_experts)
        ])

        # 共享门控网络
        self.gate = Gate(input_dim, num_experts)

        # CTR 任务塔
        self.ctr_tower = Tower(expert_hidden_dims[-1], tower_hidden_dims, dropout)

        # CVR 任务塔
        self.cvr_tower = Tower(expert_hidden_dims[-1], tower_hidden_dims, dropout)

        # 梯度聚合器
        self.aggregator = self._create_aggregator(aggregation_method)

        # 使用手动优化
        self.automatic_optimization = False

    def _create_aggregator(self, method: str):
        """创建梯度聚合器"""
        if method == 'upgrad':
            return UPGrad()
        elif method == 'mgda':
            return MGDA()
        elif method == 'pcgrad':
            return PCGrad()
        elif method == 'cagrad':
            return CAGrad(c=0.5)
        elif method == 'graddrop':
            return GradDrop(leak=0.0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """前向传播"""
        embeddings = [self.embeddings[feat](inputs[feat]) for feat in self.feature_names]
        x = torch.cat(embeddings, dim=-1)

        # 专家输出
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # 门控权重
        gate_weights = self.gate(x).unsqueeze(-1)
        expert_output = torch.sum(expert_outputs * gate_weights, dim=1)

        # 两个任务塔
        ctr_logit = self.ctr_tower(expert_output)
        cvr_logit = self.cvr_tower(expert_output)

        return ctr_logit.squeeze(-1), cvr_logit.squeeze(-1)

    def training_step(self, batch, batch_idx):
        """训练步骤 - 使用 TorchJD 梯度聚合"""
        optimizer = self.optimizers()
        optimizer.zero_grad()

        inputs, ctr_labels, cvr_labels = batch
        ctr_logits, cvr_logits = self(inputs)

        ctr_loss = nn.functional.binary_cross_entropy_with_logits(ctr_logits, ctr_labels.float())
        cvr_loss = nn.functional.binary_cross_entropy_with_logits(cvr_logits, cvr_labels.float())

        losses = [ctr_loss, cvr_loss]
        self.aggregator.backward(losses, self.parameters())

        optimizer.step()

        loss = ctr_loss + cvr_loss

        self.log('train_total_loss', loss, prog_bar=True)
        self.log('train_ctr_loss', ctr_loss)
        self.log('train_cvr_loss', cvr_loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        inputs, ctr_labels, cvr_labels = batch
        ctr_logits, cvr_logits = self(inputs)

        ctr_loss = nn.functional.binary_cross_entropy_with_logits(ctr_logits, ctr_labels.float())
        cvr_loss = nn.functional.binary_cross_entropy_with_logits(cvr_logits, cvr_labels.float())
        loss = ctr_loss + cvr_loss

        ctr_preds = torch.sigmoid(ctr_logits)
        cvr_preds = torch.sigmoid(cvr_logits)
        ctr_acc = ((ctr_preds > 0.5) == ctr_labels).float().mean()
        cvr_acc = ((cvr_preds > 0.5) == cvr_labels).float().mean()

        self.log('val_total_loss', loss, prog_bar=True)
        self.log('val_ctr_loss', ctr_loss)
        self.log('val_cvr_loss', cvr_loss)
        self.log('val_ctr_acc', ctr_acc, prog_bar=True)
        self.log('val_cvr_acc', cvr_acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_total_loss'
        }

# ============================================================================
# MMOE 模型
# ============================================================================

class MMOEModel(pl.LightningModule):
    """MMOE (Multi-gate Mixture-of-Experts) 多任务学习模型"""
 
    def __init__(
        self,
        feature_names: List[str],
        feature_dims: Dict[str, int],
        embedding_dim: int = 32,
        num_experts: int = 3,
        expert_hidden_dims: List[int] = [256, 128],
        tower_hidden_dims: List[int] = [64],
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        ctr_weight: float = 0.5,
        cvr_weight: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
 
        self.feature_names = feature_names
        self.learning_rate = learning_rate
        self.ctr_weight = ctr_weight
        self.cvr_weight = cvr_weight
 
        # 特征嵌入层
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(feature_dims[feat], embedding_dim)
            for feat in feature_names
        })
 
        input_dim = len(feature_names) * embedding_dim
 
        # 专家网络
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dims, dropout)
            for _ in range(num_experts)
        ])
 
        # CTR 门控网络
        self.ctr_gate = Gate(input_dim, num_experts)
 
        # CVR 门控网络
        self.cvr_gate = Gate(input_dim, num_experts)
 
        # CTR 任务塔
        self.ctr_tower = Tower(expert_hidden_dims[-1], tower_hidden_dims, dropout)
 
        # CVR 任务塔
        self.cvr_tower = Tower(expert_hidden_dims[-1], tower_hidden_dims, dropout)
 
        # ---- AUC 指标 ----
        # torchmetrics 会自动处理多卡同步（sync_dist）
        self.val_ctr_auc = AUROC(task='binary')
        self.val_cvr_auc = AUROC(task='binary')
 
    def forward(self, inputs: Dict[str, torch.Tensor]):
        """前向传播"""
        embeddings = [self.embeddings[feat](inputs[feat]) for feat in self.feature_names]
        x = torch.cat(embeddings, dim=-1)
 
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)
 
        ctr_gate_weights = self.ctr_gate(x).unsqueeze(-1)
        ctr_expert_output = torch.sum(expert_outputs * ctr_gate_weights, dim=1)
        ctr_logit = self.ctr_tower(ctr_expert_output)
 
        cvr_gate_weights = self.cvr_gate(x).unsqueeze(-1)
        cvr_expert_output = torch.sum(expert_outputs * cvr_gate_weights, dim=1)
        cvr_logit = self.cvr_tower(cvr_expert_output)
 
        return ctr_logit.squeeze(-1), cvr_logit.squeeze(-1)
 
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        inputs, labels = batch
        ctr_labels = labels["click"]
        cvr_labels = labels["purchase"]
        ctr_logits, cvr_logits = self(inputs)
 
        ctr_loss = nn.functional.binary_cross_entropy_with_logits(ctr_logits, ctr_labels.float())
        cvr_loss = nn.functional.binary_cross_entropy_with_logits(cvr_logits, cvr_labels.float())
        loss = self.ctr_weight * ctr_loss + self.cvr_weight * cvr_loss
 
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_ctr_loss', ctr_loss)
        self.log('train_cvr_loss', cvr_loss)
 
        return loss
 
    def validation_step(self, batch, batch_idx):
        """验证步骤：每步更新 AUC 状态"""
        inputs, labels = batch
        ctr_labels = labels["click"]
        cvr_labels = labels["purchase"]
        ctr_logits, cvr_logits = self(inputs)
 
        # loss
        ctr_loss = nn.functional.binary_cross_entropy_with_logits(ctr_logits, ctr_labels.float())
        cvr_loss = nn.functional.binary_cross_entropy_with_logits(cvr_logits, cvr_labels.float())
        loss = self.ctr_weight * ctr_loss + self.cvr_weight * cvr_loss
 
        # 转成概率值
        ctr_preds = torch.sigmoid(ctr_logits)
        cvr_preds = torch.sigmoid(cvr_logits)
 
        # 更新 AUC 状态（torchmetrics 内部会累积所有 batch 的预测值和标签）
        self.val_ctr_auc.update(ctr_preds, ctr_labels.long())
        self.val_cvr_auc.update(cvr_preds, cvr_labels.long())
 
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_ctr_loss', ctr_loss, sync_dist=True)
        self.log('val_cvr_loss', cvr_loss, sync_dist=True)
 
        return loss
 
    def on_validation_epoch_end(self):
        """验证 epoch 结束时：计算并记录 AUC"""
        ctr_auc = self.val_ctr_auc.compute()
        cvr_auc = self.val_cvr_auc.compute()
 
        self.log('val_ctr_auc', ctr_auc, prog_bar=True, sync_dist=True)
        self.log('val_cvr_auc', cvr_auc, prog_bar=True, sync_dist=True)
 
        # 重置状态，为下一个 epoch 做准备
        self.val_ctr_auc.reset()
        self.val_cvr_auc.reset()
 
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
 
# ============================================================================
# MMOE with TorchJD
# ============================================================================

class MMOEModelWithTorchJD(pl.LightningModule):
    """MMOE (Multi-gate Mixture-of-Experts) 多任务学习模型 - 使用 TorchJD 梯度聚合
 
    相比普通 MMOE：用 TorchJD 梯度聚合替代手动 loss 加权，自动平衡 CTR/CVR 任务梯度。
    相比单门控 MOE+TorchJD：每个任务有独立的门控网络，能学到任务特异性的专家组合。
    """
 
    def __init__(
        self,
        feature_names: List[str],
        feature_dims: Dict[str, int],
        embedding_dim: int = 32,
        num_experts: int = 3,
        expert_hidden_dims: List[int] = [256, 128],
        tower_hidden_dims: List[int] = [64],
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        aggregation_method: str = 'upgrad',
    ):
        super().__init__()
        self.save_hyperparameters()
 
        self.feature_names = feature_names
        self.learning_rate = learning_rate
        self.aggregation_method = aggregation_method
 
        # ---- 特征嵌入层 ----
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(feature_dims[feat], embedding_dim)
            for feat in feature_names
        })
 
        input_dim = len(feature_names) * embedding_dim
 
        # ---- 专家网络（共享） ----
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_hidden_dims, dropout)
            for _ in range(num_experts)
        ])
 
        # ---- MMOE 核心：每个任务独立的门控网络 ----
        self.ctr_gate = Gate(input_dim, num_experts)
        self.cvr_gate = Gate(input_dim, num_experts)
 
        # ---- 任务塔 ----
        self.ctr_tower = Tower(expert_hidden_dims[-1], tower_hidden_dims, dropout)
        self.cvr_tower = Tower(expert_hidden_dims[-1], tower_hidden_dims, dropout)
 
        # ---- TorchJD 梯度聚合器 ----
        self.aggregator = self._create_aggregator(aggregation_method)
 
        # ---- 使用手动优化（TorchJD 需要） ----
        self.automatic_optimization = False
 
        # ---- AUC 指标 ----
        self.val_ctr_auc = AUROC(task='binary')
        self.val_cvr_auc = AUROC(task='binary')
 
    def _create_aggregator(self, method: str):
        """创建梯度聚合器"""
        if method == 'upgrad':
            return UPGrad()
        elif method == 'mgda':
            return MGDA()
        elif method == 'pcgrad':
            return PCGrad()
        elif method == 'cagrad':
            return CAGrad(c=0.5)
        elif method == 'graddrop':
            return GradDrop(leak=0.0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
 
    def forward(self, inputs: Dict[str, torch.Tensor]):
        """前向传播 - MMOE 双门控结构"""
        # 特征嵌入拼接
        embeddings = [self.embeddings[feat](inputs[feat]) for feat in self.feature_names]
        x = torch.cat(embeddings, dim=-1)
 
        # 所有专家的输出
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch, num_experts, expert_dim)
 
        # CTR 门控 —— 独立的专家权重
        ctr_gate_weights = self.ctr_gate(x).unsqueeze(-1)  # (batch, num_experts, 1)
        ctr_expert_output = torch.sum(expert_outputs * ctr_gate_weights, dim=1)
        ctr_logit = self.ctr_tower(ctr_expert_output)
 
        # CVR 门控 —— 独立的专家权重
        cvr_gate_weights = self.cvr_gate(x).unsqueeze(-1)
        cvr_expert_output = torch.sum(expert_outputs * cvr_gate_weights, dim=1)
        cvr_logit = self.cvr_tower(cvr_expert_output)
 
        return ctr_logit.squeeze(-1), cvr_logit.squeeze(-1)
 
    def training_step(self, batch, batch_idx):
        """训练步骤 - 使用 TorchJD 梯度聚合替代手动 loss 加权"""
        optimizer = self.optimizers()
        optimizer.zero_grad()
 
        inputs, labels = batch
        ctr_labels = labels["click"]
        cvr_labels = labels["purchase"]
        ctr_logits, cvr_logits = self(inputs)
 
        ctr_loss = nn.functional.binary_cross_entropy_with_logits(ctr_logits, ctr_labels.float())
        cvr_loss = nn.functional.binary_cross_entropy_with_logits(cvr_logits, cvr_labels.float())
 
        # TorchJD 梯度聚合：自动平衡两个任务的梯度方向和大小
        losses = [ctr_loss, cvr_loss]
        backward(losses,aggregator=self.aggregator)
        optimizer.step()
 
        # 记录（仅用于监控，不参与梯度计算）
        total_loss = ctr_loss + cvr_loss
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('train_ctr_loss', ctr_loss)
        self.log('train_cvr_loss', cvr_loss)
 
        return total_loss
 
    def validation_step(self, batch, batch_idx):
        """验证步骤：计算 loss 并更新 AUC 状态"""
        inputs, labels = batch
        ctr_labels = labels["click"]
        cvr_labels = labels["purchase"]
        ctr_logits, cvr_logits = self(inputs)
 
        # loss
        ctr_loss = nn.functional.binary_cross_entropy_with_logits(ctr_logits, ctr_labels.float())
        cvr_loss = nn.functional.binary_cross_entropy_with_logits(cvr_logits, cvr_labels.float())
        loss = ctr_loss + cvr_loss
 
        # 转成概率值
        ctr_preds = torch.sigmoid(ctr_logits)
        cvr_preds = torch.sigmoid(cvr_logits)
 
        # 更新 AUC 状态
        self.val_ctr_auc.update(ctr_preds, ctr_labels.long())
        self.val_cvr_auc.update(cvr_preds, cvr_labels.long())
 
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_ctr_loss', ctr_loss, sync_dist=True)
        self.log('val_cvr_loss', cvr_loss, sync_dist=True)
 
        return loss
 
    def on_validation_epoch_end(self):
        """验证 epoch 结束时：计算并记录 AUC"""
        ctr_auc = self.val_ctr_auc.compute()
        cvr_auc = self.val_cvr_auc.compute()
 
        self.log('val_ctr_auc', ctr_auc, prog_bar=True, sync_dist=True)
        self.log('val_cvr_auc', cvr_auc, prog_bar=True, sync_dist=True)
 
        self.val_ctr_auc.reset()
        self.val_cvr_auc.reset()
 
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }