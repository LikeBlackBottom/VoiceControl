# prepare_dataset.py
import os
import shutil
import random
import argparse


def create_dataset_structure():
    """创建所有必需的数据集目录结构"""
    print("--- 任务1: 正在创建项目目录结构 ---")

    # 定义所有需要创建的文件夹路径
    directories = [
        # 原始录音数据
        'dataset/on',
        'dataset/off',
        'dataset/unknown',
        # 增强后的数据
        'dataset_augmented/on',
        'dataset_augmented/off',
        'dataset_augmented/unknown',
        # 划分后的训练、验证、测试集
        'dataset_split/train/on',
        'dataset_split/train/off',
        'dataset_split/train/unknown',
        'dataset_split/val/on',
        'dataset_split/val/off',
        'dataset_split/val/unknown',
        'dataset_split/test/on',
        'dataset_split/test/off',
        'dataset_split/test/unknown',
        # 保存最终模型的文件夹
        'models',
    ]

    for directory in directories:
        # exist_ok=True 表示如果文件夹已存在，则不会报错
        os.makedirs(directory, exist_ok=True)
        print(f"已创建或已存在目录: {directory}")

    print("\n项目目录结构创建完成！")
    print("-" * 50)


def split_dataset(base_dir='dataset_augmented', output_dir='dataset_split', train_ratio=0.7, val_ratio=0.15):
    """
    划分增强后的数据集为训练集、验证集和测试集。
    """
    print(f"--- 任务2: 正在从 '{base_dir}' 划分数据集到 '{output_dir}' ---")

    # 确保比例总和小于等于1
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio >= 0, "训练集和验证集比例总和不能超过1"

    categories = ['on', 'off', 'unknown']

    for category in categories:
        category_dir = os.path.join(base_dir, category)

        # 检查源文件夹是否存在
        if not os.path.exists(category_dir):
            print(f"\n警告: 目录 '{category_dir}' 不存在，跳过该类别。")
            continue

        # 获取所有.wav文件
        audio_files = [f for f in os.listdir(category_dir) if f.endswith('.wav')]
        if not audio_files:
            print(f"\n警告: 目录 '{category_dir}' 中没有找到 .wav 文件，跳过该类别。")
            continue

        random.shuffle(audio_files)

        # 计算分割点
        total_files = len(audio_files)
        train_end = int(total_files * train_ratio)
        val_end = train_end + int(total_files * val_ratio)

        train_files = audio_files[:train_end]
        val_files = audio_files[train_end:val_end]
        test_files = audio_files[val_end:]

        print(f"\n类别 '{category}' (共 {total_files} 个文件):")
        print(f"  -> 训练集: {len(train_files)} 个")
        print(f"  -> 验证集: {len(val_files)} 个")
        print(f"  -> 测试集: {len(test_files)} 个")

        # 定义目标路径
        split_paths = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

        # 复制文件到新目录
        for split_name, files in split_paths.items():
            dest_dir = os.path.join(output_dir, split_name, category)
            # 确保目标目录存在
            os.makedirs(dest_dir, exist_ok=True)
            for file in files:
                src_path = os.path.join(category_dir, file)
                dest_path = os.path.join(dest_dir, file)
                shutil.copy2(src_path, dest_path)

    print("\n数据集划分完成！")
    print("-" * 50)


if __name__ == '__main__':
    # 使用 argparse 来选择执行哪个任务，这样更灵活
    parser = argparse.ArgumentParser(description="准备数据集：创建目录结构或划分数据。")
    parser.add_argument(
        'task',
        type=str,
        choices=['create', 'split'],
        help="选择要执行的任务: 'create' (创建目录结构) 或 'split' (划分数据集)"
    )

    args = parser.parse_args()

    if args.task == 'create':
        create_dataset_structure()
    elif args.task == 'split':
        # 检查增强文件夹是否存在
        if not os.path.exists('dataset_augmented'):
            print("错误: 'dataset_augmented' 文件夹不存在。请先录制并增强数据。")
        else:
            split_dataset()