"""
双塔召回模型训练脚本
"""
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
import numpy as np

from src.models.recall import TwoTowerModel
from src.models.features import SparseFeature, DenseFeature, SequenceFeature
from src.data.dataset import TwoTowerDataModule
from src.utils.constants import ITEM_DENSE, ITEM_SPARSE, USER_DENSE, USER_SPARSE, vocabulary_size


def parse_args():
    parser = argparse.ArgumentParser(description='双塔召回模型训练脚本')

    # 数据路径
    parser.add_argument('--train_path', type=str,
                        default='/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_train.parquet')
    parser.add_argument('--val_path', type=str,
                        default='/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_val.parquet')
    parser.add_argument('--test_path', type=str,
                        default='/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_test.parquet')

    # 模型超参数
    parser.add_argument('--embed_dim', type=int, default=16,
                        help='Sparse 特征 embedding 维度')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 32],
                        help='塔网络隐层维度')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--temperature', type=float, default=0.05,
                        help='对比学习温度系数')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--training_mode', type=str, default='listwise',
                        choices=['pointwise', 'pairwise', 'listwise'],
                        help='训练模式：pointwise/pairwise/listwise')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='pairwise 模式的 margin 参数')

    # 数据加载
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_easy_neg', type=int, default=8,
                        help='每条样本的简单负样本数')
    parser.add_argument('--use_inbatch_neg', type=lambda x: x.lower() == 'true', default=True,
                        help='是否使用 in-batch 负样本（true/false）')
    parser.add_argument('--user_col', type=str, default='101')
    parser.add_argument('--item_col', type=str, default='205')

    # 训练配置
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=4)
    parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_false')
    parser.add_argument('--log_every_n_steps', type=int, default=100)
    parser.add_argument('--val_check_interval', type=int, default=10)
    parser.add_argument('--limit_val_batches', type=int, default=100)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--early_stop_patience', type=int, default=5)

    # 输出路径
    parser.add_argument('--exp_dir', type=str, default='experiments/two_tower_feature')

    return parser.parse_args()


def train_recall(args):
    print("\n" + "=" * 60)
    print("训练双塔模型 - 支持 Sparse/Dense/Sequence 特征")
    print("=" * 60 + "\n")

    print("准备数据...")
    np.random.seed(42)
    train_data = pd.read_parquet(args.train_path)
    val_data   = pd.read_parquet(args.val_path)

    print("定义特征...")
    user_sparse_features   = [SparseFeature(name=n, vocab_size=vocabulary_size[n], embed_dim=args.embed_dim) for n in USER_SPARSE]
    user_dense_features    = [DenseFeature(name=n, dim=1) for n in USER_DENSE]
    user_sequence_features = []
    item_sparse_features   = [SparseFeature(name=n, vocab_size=vocabulary_size[n], embed_dim=args.embed_dim) for n in ITEM_SPARSE]
    item_dense_features    = [DenseFeature(name=n, dim=1) for n in ITEM_DENSE]
    item_sequence_features = []

    print("创建数据集...")
    datamodule = TwoTowerDataModule(
        train_df=train_data,
        val_df=val_data,
        items_df=train_data,
        user_feature_cols=USER_DENSE + USER_SPARSE,
        item_sparse_features=item_sparse_features,
        item_dense_features=item_dense_features,
        item_sequence_features=item_sequence_features,
        num_easy_neg=args.num_easy_neg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        user_col=args.user_col,
        item_col=args.item_col,
    )

    print("初始化模型...")
    model = TwoTowerModel(
        user_sparse_features=user_sparse_features,
        user_dense_features=user_dense_features,
        user_sequence_features=user_sequence_features,
        item_sparse_features=item_sparse_features,
        item_dense_features=item_dense_features,
        item_sequence_features=item_sequence_features,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        training_mode=args.training_mode,
        use_inbatch_neg=args.use_inbatch_neg,
        margin=args.margin,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{args.exp_dir}/checkpoints',
        filename='two_tower_feature-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.early_stop_patience,
        mode='min',
        verbose=True,
    )
    logger = TensorBoardLogger(save_dir=args.exp_dir, name='logs')

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        strategy=args.strategy,
        profiler='simple',
        devices=args.devices,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        gradient_clip_val=args.gradient_clip_val,
    )

    print("\n开始训练...")
    trainer.fit(model, datamodule)

    print(f"\n训练完成！")
    print(f"最佳模型: {checkpoint_callback.best_model_path}")
    print(f"TensorBoard: tensorboard --logdir {args.exp_dir}/logs")


if __name__ == '__main__':
    train_recall(parse_args())
