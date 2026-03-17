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
from src.data.dataset import TwoTowerDataModule, FullItemTestDataModule
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
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--early_stop_patience', type=int, default=5)

    # 输出路径
    parser.add_argument('--exp_dir', type=str, default='experiments/two_tower_feature')

    # 测试参数
    parser.add_argument('--do_test', action='store_true', help='运行测试阶段全量物料评估')
    parser.add_argument('--checkpoint_path', type=str, default='', help='测试时加载的模型 checkpoint 路径')
    parser.add_argument('--topk_list', type=int, nargs='+', default=[50, 100], help='Recall@K 的 K 列表')

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
        monitor='val_recall@50', mode='max',
        save_top_k=3,
        save_last=True,
    )
    '''
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.early_stop_patience,
        mode='min',
        verbose=True,
    )
    '''
    logger = TensorBoardLogger(save_dir=args.exp_dir, name='logs')

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        strategy=args.strategy,
        profiler='simple',
        devices=args.devices,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=args.gradient_clip_val,
    )

    print("\n开始训练...")
    trainer.fit(model, datamodule)
    #trainer.test(model, datamodule)
    print(f"\n训练完成！")
    print(f"最佳模型: {checkpoint_callback.best_model_path}")
    print(f"TensorBoard: tensorboard --logdir {args.exp_dir}/logs")


def test_recall(args):
    """
    测试阶段召回评估：
      1. 获取测试集中所有用户（正样本行）及所有测试物料
      2. 对每个用户计算其与所有测试物料的得分
      3. 计算 topk Recall@K
    """
    print("\n" + "=" * 60)
    print("测试双塔召回模型 - 全量物料评估")
    print("=" * 60 + "\n")

    import torch

    # ---- 加载测试数据 ----
    print("加载测试数据...")
    test_data  = pd.read_parquet(args.test_path)
    # 用测试集本身作为物料池（所有出现过的物料）
    items_df   = test_data.drop_duplicates(subset=[args.item_col]).reset_index(drop=True)

    # ---- 定义特征 ----
    item_sparse_features = [
        SparseFeature(name=n, vocab_size=vocabulary_size[n], embed_dim=args.embed_dim)
        for n in ITEM_SPARSE
    ]
    item_dense_features    = [DenseFeature(name=n, dim=1) for n in ITEM_DENSE]
    item_sequence_features = []

    # ---- 构建全量物料测试 DataModule ----
    test_dm = FullItemTestDataModule(
        test_df=test_data,
        items_df=items_df,
        user_feature_cols=USER_DENSE + USER_SPARSE,
        item_sparse_features=item_sparse_features,
        item_dense_features=item_dense_features,
        item_sequence_features=item_sequence_features,
        label_col='click',
        user_col=args.user_col,
        item_col=args.item_col,
        batch_size=args.batch_size,
        item_batch_size=args.batch_size * 4,
        num_workers=args.num_workers,
    )
    print(f"测试用户数: {test_dm.num_users}, 测试物料数: {test_dm.num_items}")

    # ---- 加载模型 ----
    print(f"加载模型: {args.checkpoint_path}")
    user_sparse_features = [
        SparseFeature(name=n, vocab_size=vocabulary_size[n], embed_dim=args.embed_dim)
        for n in USER_SPARSE
    ]
    user_dense_features    = [DenseFeature(name=n, dim=1) for n in USER_DENSE]
    user_sequence_features = []

    model = TwoTowerModel.load_from_checkpoint(
        args.checkpoint_path,
        user_sparse_features=user_sparse_features,
        user_dense_features=user_dense_features,
        user_sequence_features=user_sequence_features,
        item_sparse_features=item_sparse_features,
        item_dense_features=item_dense_features,
        item_sequence_features=item_sequence_features,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # ---- 全量物料评估 ----
    print("\n开始全量物料评估...")
    results = model.evaluate_full_item(test_dm, topk_list=args.topk_list, device=device)
    print("\n评估完成！")
    return results


if __name__ == '__main__':
    args = parse_args()
    if getattr(args, 'do_test', False):
        test_recall(args)
    else:
        train_recall(args)
