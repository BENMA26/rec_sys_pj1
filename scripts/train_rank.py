"""
MMOE 多任务学习模型
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
from pathlib import Path

from src.models.ranking import MMOEModel,MMOEModelWithTorchJD
from src.data.dataset import RankDataModule
from src.utils.constants import USER_SPARSE, USER_DENSE, ITEM_SPARSE, ITEM_DENSE, vocabulary_size

def train_rank():
    """训练 MMOE 多任务学习模型"""

    train_data = pd.read_parquet("/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_train.parquet")
    val_data = pd.read_parquet("/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_val.parquet")
    test_data = pd.read_parquet("/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_test.parquet")

    # ========== 2. 特征配置 ==========
    feature_names = USER_SPARSE + USER_DENSE + ITEM_SPARSE + ITEM_DENSE

    # 特征维度（词表大小）
    feature_dims = vocabulary_size

    # ========== 3. 数据模块 ==========
    data_module = RankDataModule(
        train_df=train_data,
        val_df=val_data,
        test_df=test_data,
        feature_cols=feature_names,
        label_cols=['click','purchase'],
        user_col=['101'],
        item_col=['205'],
        batch_size=256,
        num_workers=4,
    )

    # ========== 4. 模型初始化 ==========
    model = MMOEModelWithTorchJD(
        feature_names=feature_names,
        feature_dims=feature_dims,
        embedding_dim=32,
        num_experts=6,
        expert_hidden_dims=[256, 128],
        tower_hidden_dims=[64],
        dropout=0.2,
        learning_rate=1e-3,
        #ctr_weight=0.5,
        #cvr_weight=0.5,
    )

    # ========== 5. 回调函数 ==========
    checkpoint_callback = ModelCheckpoint(
        dirpath='experiments/mmoe/checkpoints',
        filename='mmoe-{epoch:02d}-{val_loss:.4f}',
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
        save_dir='experiments/mmoe',
        name='logs',
    )

    # ========== 7. Trainer ==========
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu',
        strategy="ddp_find_unused_parameters_false",
        profiler="simple",
        devices=4,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=100,
        val_check_interval=10000,
        limit_val_batches=100,
        #gradient_clip_val=1.0,
    )

    # ========== 8. 训练 ==========
    trainer.fit(model, data_module)

    # ========== 9. 测试 ==========
    if test_data is not None:
        trainer.test(model, data_module)

    print(f"训练完成！最佳模型保存在: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    train_rank()
