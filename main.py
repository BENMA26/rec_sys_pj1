"""
主入口文件 - 推荐系统训练和推理
"""
import argparse

from scripts.train_recall import train_recall
from scripts.train_rank import train_rank
from scripts.build_index import build_index, test_index
from scripts.inference import inference


def main():
    parser = argparse.ArgumentParser(description='推荐系统训练和推理')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train_recall', 'train_rank', 'build_index', 'test_index', 'inference'],
        required=True,
        help='运行模式：train_recall (训练召回模型), train_rank (训练排序模型), '
             'build_index (构建索引), test_index (测试索引), inference (推理)'
    )

    args = parser.parse_args()

    if args.mode == 'train_recall':
        print("开始训练双塔召回模型...")
        train_recall()

    elif args.mode == 'train_rank':
        print("开始训练 MMOE 排序模型...")
        train_rank()

    elif args.mode == 'build_index':
        print("开始构建向量索引...")
        build_index()

    elif args.mode == 'test_index':
        print("开始测试向量索引...")
        test_index()

    elif args.mode == 'inference':
        print("开始推理...")
        inference()


if __name__ == '__main__':
    main()
