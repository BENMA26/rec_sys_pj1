"""
MMOE 多任务学习模型训练脚本
"""
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
from pathlib import Path

from src.models.ranking import (
    ShareBottomModel, ShareBottomModelWithTorchJD,
    MOEModel, MOEModelWithTorchJD,
    MMOEModel, MMOEModelWithTorchJD,
)
from src.data.dataset import RankDataModule
from src.utils.constants import USER_SPARSE, USER_DENSE, ITEM_SPARSE, ITEM_DENSE, vocabulary_size

MODEL_MAP = {
    'share_bottom': ShareBottomModel,
    'share_bottom_jd': ShareBottomModelWithTorchJD,
    'moe': MOEModel,
    'moe_jd': MOEModelWithTorchJD,
    'mmoe': MMOEModel,
    'mmoe_jd': MMOEModelWithTorchJD,
}

TORCHJD_MODELS = {'share_bottom_jd', 'moe_jd', 'mmoe_jd'}


def parse_args():
    parser = argparse.ArgumentParser(description='排序模型训练脚本')

    # 数据路径
    parser.add_argument('--train_path', type=str,
                        default='/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_train.parquet')
    parser.add_argument('--val_path', type=str,
                        default='/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_val.parquet')
    parser.add_argument('--test_path', type=str,
                        default='/work/home/maben/project/rec_sys/projects/Ali_CCP_REC/dataset/datasetsali_ccp_test.parquet')

    # 模型选择
    parser.add_argument('--model', type=str, default='mmoe_jd',
                        choices=list(MODEL_MAP.keys()),
                        help='模型类型')

    # 模型超参数
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--num_experts', type=int, default=6,
                        help='专家数量（仅 MOE/MMOE 系列有效）')
    parser.add_argument('--expert_hidden_dims', type=int, nargs='+', default=[256, 128],
                        help='专家网络隐层维度（仅 MOE/MMOE 系列有效）')
    parser.add_argument('--shared_hidden_dims', type=int, nargs='+', default=[256, 128],
                        help='共享底层隐层维度（仅 ShareBottom 系列有效）')
    parser.add_argument('--tower_hidden_dims', type=int, nargs='+', default=[64])
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--ctr_weight', type=float, default=0.5,
                        help='CTR loss 权重（非 TorchJD 模型有效）')
    parser.add_argument('--cvr_weight', type=float, default=0.5,
                        help='CVR loss 权重（非 TorchJD 模型有效）')
    parser.add_argument('--aggregation_method', type=str, default='upgrad',
                        choices=['upgrad', 'mgda', 'pcgrad', 'graddrop'],
                        help='TorchJD 梯度聚合方法（TorchJD 模型有效）')

    # 数据加载
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)

    # 训练配置
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=int, default=4)
    parser.add_argument('--strategy', type=str, default='ddp_find_unused_parameters_false')
    parser.add_argument('--log_every_n_steps', type=int, default=100)
    parser.add_argument('--val_check_interval', type=int, default=10000)
    parser.add_argument('--limit_val_batches', type=int, default=100)
    parser.add_argument('--early_stop_patience', type=int, default=5)

    # 输出路径
    parser.add_argument('--exp_dir', type=str, default='experiments/rank',
                        help='实验输出根目录')

    return parser.parse_args()


def build_model(args):
    model_cls = MODEL_MAP[args.model]
    feature_names = USER_SPARSE + USER_DENSE + ITEM_SPARSE + ITEM_DENSE
    feature_dims = vocabulary_size

    common = dict(
        feature_names=feature_names,
        feature_dims=feature_dims,
        embedding_dim=args.embedding_dim,
        tower_hidden_dims=args.tower_hidden_dims,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
    )

    if args.model in TORCHJD_MODELS:
        extra = dict(aggregation_method=args.aggregation_method)
    else:
        extra = dict(ctr_weight=args.ctr_weight, cvr_weight=args.cvr_weight)

    if args.model in ('share_bottom', 'share_bottom_jd'):
        extra['shared_hidden_dims'] = args.shared_hidden_dims
    else:
        extra['num_experts'] = args.num_experts
        extra['expert_hidden_dims'] = args.expert_hidden_dims

    return model_cls(**common, **extra)


def train_rank(args):
    train_data = pd.read_parquet(args.train_path)
    val_data   = pd.read_parquet(args.val_path)
    test_data  = pd.read_parquet(args.test_path)

    feature_names = USER_SPARSE + USER_DENSE + ITEM_SPARSE + ITEM_DENSE

    data_module = RankDataModule(
        train_df=train_data,
        val_df=val_data,
        test_df=test_data,
        feature_cols=feature_names,
        label_cols=['click', 'purchase'],
        user_col=['101'],
        item_col=['205'],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(args)

    exp_dir = Path(args.exp_dir) / args.model
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(exp_dir / 'checkpoints'),
        filename=f'{args.model}-{{epoch:02d}}-{{val_loss:.4f}}',
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
    logger = TensorBoardLogger(save_dir=str(exp_dir), name='logs')

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
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    print(f"训练完成！最佳模型保存在: {checkpoint_callback.best_model_path}")


if __name__ == '__main__':
    train_rank(parse_args())
