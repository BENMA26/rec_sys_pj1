# 推荐系统项目

基于 PyTorch Lightning 的完整推荐系统实现，包含召回和排序两个阶段。

## 项目结构

```
rec_sys_pj1/
├── src/
│   ├── models/
│   │   ├── two_tower.py       # 双塔召回模型
│   │   ├── two_tower_attention.py  # 基于 Attention 的双塔模型
│   │   ├── mmoe.py            # MMOE 多任务学习模型
│   │   ├── mmoe_torchjd.py    # MMOE + TorchJD 梯度聚合
│   │   ├── attention.py       # Multi-Head Self-Attention 模块
│   │   ├── losses.py          # 损失函数集合
│   │   └── features.py        # 特征工程模块
│   ├── data/
│   │   └── dataset.py         # 数据集和数据模块
│   └── utils/
│       ├── metrics.py         # 评估指标
│       └── retriever.py       # 向量索引构建和检索
├── scripts/
│   ├── train_recall.py        # 训练召回模型
│   ├── train_attention.py     # 训练 Attention 召回模型
│   ├── train_rank.py          # 训练排序模型
│   ├── train_mmoe_comparison.py  # 对比 MMOE 和 MMOE+TorchJD
│   ├── build_index.py         # 构建向量索引
│   └── inference.py           # 推理脚本
├── docs/
│   ├── loss_functions.md      # 损失函数文档
│   ├── attention.md           # Attention 序列建模文档
│   ├── mmoe_torchjd.md        # MMOE + TorchJD 文档
│   └── feature_engineering_and_annoy.md  # 特征工程文档
├── experiments/               # 实验结果和模型保存
├── data/                      # 数据目录
└── requirements.txt           # 依赖包
```

## 核心功能

### 1. 双塔召回模型
- 用户塔和物品塔分别编码
- 支持 6 种损失函数：Pointwise, Pairwise (BPR/Hinge), Listwise (InfoNCE), Sampled Softmax, Triplet
- 支持 Sparse/Dense/Sequence 三种特征类型
- 向量检索实现高效召回

### 2. MMOE 多任务学习
- 同时预测 CTR 和 CVR
- 多个专家网络共享底层表示
- 门控机制自适应选择专家
- 独立的任务塔处理不同目标
- **支持 TorchJD 梯度聚合**：UPGrad, PCGrad, MGDA, CAGrad, GradDrop

### 3. Multi-Head Self-Attention 序列建模
- 使用 Transformer 编码用户行为序列
- 捕捉序列中的顺序信息和长距离依赖
- 支持多种池化方式：mean/max/last/cls
- 显著提升序列特征建模效果

### 4. 特征工程（参考 torch-rechub）
- **Sparse Feature**: 类别特征，支持共享嵌入
- **Dense Feature**: 数值特征，自动归一化
- **Sequence Feature**: 序列特征，支持多种池化方式（mean/sum/max）

### 5. 向量索引
- 高效的近似最近邻搜索（基于 Annoy）
- 支持百万、千万级向量检索
- 毫秒级检索速度
- 支持多种距离度量（余弦/欧氏/点积）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 训练双塔召回模型

```bash
# 基础双塔模型
python scripts/train_recall.py

# 使用 Attention 的双塔模型
python scripts/train_attention.py
```

### 2. 构建向量索引

```bash
# 构建索引
python scripts/build_index.py --mode build

# 测试检索
python scripts/build_index.py --mode test
```

### 3. 训练 MMOE 排序模型

```bash
# 普通 MMOE
python scripts/train_rank.py

# 对比 MMOE 和 MMOE+TorchJD
python scripts/train_mmoe_comparison.py --mode all --aggregation upgrad
```

### 4. 推理

```bash
python scripts/inference.py
```

## 特征定义示例

```python
from src.models.features import SparseFeature, DenseFeature, SequenceFeature

# 稀疏特征
user_id = SparseFeature(name='user_id', vocab_size=10000, embed_dim=32)
category = SparseFeature(name='category', vocab_size=100, embed_dim=16)

# 稠密特征
price = DenseFeature(name='price', dim=1)
score = DenseFeature(name='score', dim=1)

# 序列特征（支持共享嵌入）
click_history = SequenceFeature(
    name='click_history',
    vocab_size=50000,
    embed_dim=32,
    max_len=20,
    pooling='mean',
    shared_embed='item_id',  # 与 item_id 共享嵌入
)
```

## 损失函数

支持 6 种损失函数，详见 [docs/loss_functions.md](docs/loss_functions.md)

| 损失类型 | 适用场景 | 优点 |
|---------|---------|------|
| Pointwise | 有明确正负标签 | 简单稳定 |
| Pairwise (BPR) | 隐式反馈 | 符合排序目标 |
| Pairwise Hinge | 小数据集 | 更稳定 |
| Listwise (InfoNCE) | 大规模召回 | 效率高，效果好 ⭐ |
| Sampled Softmax | 海量物品 | 降低计算复杂度 |
| Triplet | 对比学习 | 基于距离度量 |

**推荐使用 Listwise (InfoNCE)** - 训练效率高，效果好，适合大规模召回任务。

## 完整推荐流程

```python
from src.models import TwoTowerModel, MMOEModel
from src.utils import Retriever

# 1. 加载召回模型
recall_model = TwoTowerModel.load_from_checkpoint('model.ckpt')

# 2. 编码用户向量
user_vec = recall_model.encode_user(user_features).cpu().numpy()

# 3. 使用向量索引检索候选物品
retriever = Retriever(...)
candidates = retriever.search(user_vec, top_k=100)

# 4. 使用 MMOE 模型精排
ranking_model = MMOEModel.load_from_checkpoint('mmoe.ckpt')
ctr_scores, cvr_scores = ranking_model(features)

# 5. 返回推荐结果
recommendations = sorted(candidates, key=lambda x: x['score'], reverse=True)[:10]
```

## 性能指标

### 向量索引性能

| 数据规模 | n_trees | 构建时间 | 检索时间 | Recall@100 |
|---------|---------|---------|---------|-----------|
| 10万    | 10      | 10s     | 1ms     | 95%       |
| 100万   | 50      | 5min    | 2ms     | 98%       |
| 1000万  | 100     | 1h      | 5ms     | 99%       |

## 文档

- [损失函数详解](docs/loss_functions.md) - 6 种损失函数的原理、适用场景和使用方法
- [特征工程和向量索引](docs/feature_engineering_and_annoy.md) - 特征工程和向量索引的完整指南
- [Multi-Head Self-Attention 序列建模](docs/attention.md) - 使用 Attention 建模用户行为序列
- [MMOE + TorchJD 多任务学习](docs/mmoe_torchjd.md) - 使用梯度聚合提升多任务学习效果

## 主要依赖

- PyTorch Lightning >= 2.0.0
- PyTorch >= 2.0.0
- TorchJD >= 0.2.0 (用于多任务学习梯度聚合)
- Annoy >= 1.17.0
- Pandas, NumPy, scikit-learn

## 参考

本项目参考了 [torch-rechub/FunRec](https://github.com/datawhalechina/torch-rechub) 的实现思路。

## License

MIT
category = SparseFeature(name='category', vocab_size=100, embed_dim=16)

# 稠密特征
price = DenseFeature(name='price', dim=1)
score = DenseFeature(name='score', dim=1)

# 序列特征（支持共享嵌入）
click_history = SequenceFeature(
    name='click_history',
    vocab_size=50000,
    embed_dim=32,
    max_len=20,
    pooling='mean',
    shared_embed='item_id',  # 与 item_id 共享嵌入
)
```

## 损失函数

支持 6 种损失函数，详见 [docs/loss_functions.md](docs/loss_functions.md)

| 损失类型 | 适用场景 | 优点 |
|---------|---------|------|
| Pointwise | 有明确正负标签 | 简单稳定 |
| Pairwise (BPR) | 隐式反馈 | 符合排序目标 |
| Pairwise Hinge | 小数据集 | 更稳定 |
| Listwise (InfoNCE) | 大规模召回 | 效率高，效果好 ⭐ |
| Sampled Softmax | 海量物品 | 降低计算复杂度 |
| Triplet | 对比学习 | 基于距离度量 |

**推荐使用 Listwise (InfoNCE)** - 训练效率高，效果好，适合大规模召回任务。

## 完整推荐流程

```python
from src.models.two_tower_feature import TwoTowerFeatureModel
from src.models.mmoe import MMOEModel
from src.utils.annoy_builder import AnnoyRetriever

# 1. 加载召回模型
recall_model = TwoTowerFeatureModel.load_from_checkpoint('model.ckpt')

# 2. 编码用户向量
user_vec = recall_model.encode_user(user_features).cpu().numpy()

# 3. 使用 Annoy 检索候选物品
retriever = AnnoyRetriever(...)
candidates = retriever.search(user_vec, top_k=100)

# 4. 使用 MMOE 模型精排
ranking_model = MMOEModel.load_from_checkpoint('mmoe.ckpt')
ctr_scores, cvr_scores = ranking_model(features)

# 5. 返回推荐结果
recommendations = sorted(candidates, key=lambda x: x['score'], reverse=True)[:10]
```

## 性能指标

### Annoy 索引性能

| 数据规模 | n_trees | 构建时间 | 检索时间 | Recall@100 |
|---------|---------|---------|---------|-----------|
| 10万    | 10      | 10s     | 1ms     | 95%       |
| 100万   | 50      | 5min    | 2ms     | 98%       |
| 1000万  | 100     | 1h      | 5ms     | 99%       |

## 文档

- [损失函数详解](docs/loss_functions.md) - 6 种损失函数的原理、适用场景和使用方法
- [特征工程和 Annoy 索引](docs/feature_engineering_and_annoy.md) - 特征工程和向量索引的完整指南

## 主要依赖

- PyTorch Lightning >= 2.0.0
- PyTorch >= 2.0.0
- TorchJD >= 0.2.0 (用于多任务学习梯度聚合)
- Annoy >= 1.17.0
- Pandas, NumPy, scikit-learn

## 参考

本项目参考了 [torch-rechub/FunRec](https://github.com/datawhalechina/torch-rechub) 的实现思路。

## License

MIT
