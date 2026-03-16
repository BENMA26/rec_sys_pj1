"""
双塔召回模型
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
import numpy as np

from src.models.recall import TwoTowerModel
from src.models.features import SparseFeature, DenseFeature, SequenceFeature
from src.data.dataset import RecallPosDataset, TwoTowerDataModule
from src.utils.constants import ITEM_DENSE,ITEM_SPARSE,USER_DENSE,USER_SPARSE,vocabulary_size
from torch.utils.data import DataLoader

def train_recall():
    """训练基于特征工程的双塔模型"""

    print("\n" + "="*60)
    print("训练双塔模型 - 支持 Sparse/Dense/Sequence 特征")
    print("="*60 + "\n")

    # ========== 1. 数据准备 ==========
    print("准备数据...")
    np.random.seed(42)

    # 生成示例数据
    train_data = pd.read_parquet("/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_train.parquet")
    val_data = pd.read_parquet("/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_val.parquet")
    test_data = pd.read_parquet("/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_test.parquet")
    # ========== 2. 特征定义 ==========
    print("定义特征...")

    user_sparse_features = [
        SparseFeature(name=name,vocab_size=vocabulary_size[name],embed_dim=16) for name in USER_SPARSE
    ]
    user_dense_features = [
        DenseFeature(name=name, dim=1) for name in USER_DENSE
    ]
    user_sequence_features = [
    ]
    item_sparse_features = [
        SparseFeature(name=name,vocab_size=vocabulary_size[name],embed_dim=16) for name in ITEM_SPARSE
    ]
    item_dense_features = [
        DenseFeature(name=name, dim=1) for name in ITEM_DENSE
    ]
    item_sequence_features = []

    # ========== 3. 数据集 ==========
    print("创建数据集...")
    datamodule = TwoTowerDataModule(train_df=train_data,
    val_df=val_data,
    items_df=train_data,
    user_feature_cols=USER_DENSE+USER_SPARSE,
    item_sparse_features=item_sparse_features,
    item_dense_features=item_dense_features,
    item_sequence_features=item_sequence_features,
    num_easy_neg=8,
    batch_size=1024,
    num_workers=8,
    user_col="101",
    item_col="205"
    )

    # ========== 4. 模型初始化 ==========
    print("初始化模型...")
    model = TwoTowerModel(
        user_sparse_features=user_sparse_features,
        user_dense_features=user_dense_features,
        user_sequence_features=user_sequence_features,
        item_sparse_features=item_sparse_features,
        item_dense_features=item_dense_features,
        item_sequence_features=item_sequence_features,
        hidden_dims=[128, 32],
        dropout=0.2,
        temperature=0.05,
        learning_rate=1e-3,
    )

    # ========== 5. 回调函数 ==========
    checkpoint_callback = ModelCheckpoint(
        dirpath='experiments/two_tower_feature/checkpoints',
        filename='two_tower_feature-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min',
        verbose=True,
    )

    # ========== 6. Logger ==========
    logger = TensorBoardLogger(
        save_dir='experiments/two_tower_feature',
        name='logs',
    )

    # ========== 7. Trainer ==========
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator='gpu',
        strategy="ddp_find_unused_parameters_false",
        profiler="simple",
        devices=4,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=100,
        val_check_interval=10,
        limit_val_batches=100,
        gradient_clip_val=1.0,
    )

    # ========== 8. 训练 ==========
    print("\n开始训练...")
    trainer.fit(model, datamodule)

    print(f"\n✅ 训练完成！")
    print(f"最佳模型: {checkpoint_callback.best_model_path}")
    print(f"TensorBoard: tensorboard --logdir experiments/two_tower_feature/logs")

if __name__ == '__main__':
    train_recall()
