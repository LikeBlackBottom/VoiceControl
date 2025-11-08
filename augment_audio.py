# augment_audio.py
import librosa
import numpy as np
import soundfile as sf
import os
import random
from tqdm import tqdm


class AudioAugmenter:
    """
    一个封装了多种音频数据增强方法的类。
    """

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def load_audio(self, file_path):
        """加载音频文件"""
        # 使用 librosa 加载，它会自动将音频转换为单声道、浮点数格式
        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return audio

    def save_audio(self, audio, file_path):
        """保存音频文件"""
        sf.write(file_path, audio, self.sample_rate)

    def add_noise(self, audio, noise_level_factor=0.005):
        """添加随机高斯噪声"""
        noise = np.random.randn(len(audio))
        augmented_audio = audio + noise_level_factor * noise
        # 确保音频数据范围在 [-1, 1] 之间
        return np.clip(augmented_audio, -1.0, 1.0)

    def shift_time(self, audio, shift_max_ratio=0.2):
        """时间平移（在音频前后补零）"""
        shift_amount = int(self.sample_rate * shift_max_ratio * (random.random() * 2 - 1))
        augmented_audio = np.roll(audio, shift_amount)
        # 将平移后空出的部分置为0（静音）
        if shift_amount > 0:
            augmented_audio[:shift_amount] = 0
        else:
            augmented_audio[shift_amount:] = 0
        return augmented_audio

    def change_pitch(self, audio, n_steps):
        """改变音调"""
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)

    def change_speed(self, audio, speed_factor):
        """改变语速（时间拉伸/压缩）"""
        return librosa.effects.time_stretch(audio, rate=speed_factor)

    def augment_single_file(self, input_path, output_dir, num_augmentations=4):
        """对单个音频文件应用一系列随机增强"""
        # 加载原始音频
        original_audio = self.load_audio(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]

        # 1. 首先，将原始文件本身复制一份到目标目录
        self.save_audio(original_audio, os.path.join(output_dir, f"{base_name}_original.wav"))

        # 2. 生成指定数量的增强版本
        for i in range(num_augmentations):
            augmented_audio = original_audio.copy()

            # --- 随机应用增强策略 ---
            # 80% 的概率添加噪声
            if random.random() < 0.8:
                augmented_audio = self.add_noise(augmented_audio, noise_level_factor=random.uniform(0.002, 0.008))

            # 50% 的概率进行时间平移
            if random.random() < 0.5:
                augmented_audio = self.shift_time(augmented_audio, shift_max_ratio=0.2)

            # 50% 的概率进行音调调整
            if random.random() < 0.5:
                # 在-1.5到1.5个半音之间随机调整
                pitch_steps = random.uniform(-1.5, 1.5)
                augmented_audio = self.change_pitch(augmented_audio, n_steps=pitch_steps)

            # 30% 的概率进行速度调整 (由于这会改变音频长度，较少使用)
            # if random.random() < 0.3:
            #     speed_factor = random.uniform(0.9, 1.1)
            #     augmented_audio = self.change_speed(augmented_audio, speed_factor)

            # 保存增强后的文件
            output_filename = os.path.join(output_dir, f"{base_name}_aug_{i + 1}.wav")
            self.save_audio(augmented_audio, output_filename)


def main():
    """主函数，遍历原始数据集并执行增强"""
    categories = ['on', 'off', 'unknown']
    source_base_dir = 'dataset'
    output_base_dir = 'dataset_augmented'

    # 检查原始数据目录
    if not os.path.isdir(source_base_dir):
        print(f"错误: 原始数据集目录 '{source_base_dir}' 不存在。")
        print("请先使用 record_audio.py 录制数据。")
        return

    augmenter = AudioAugmenter()
    print("--- 开始数据增强 ---")

    for category in categories:
        input_dir = os.path.join(source_base_dir, category)
        output_dir = os.path.join(output_base_dir, category)

        if not os.path.exists(input_dir):
            print(f"警告: 目录 '{input_dir}' 不存在，跳过该类别。")
            continue

        os.makedirs(output_dir, exist_ok=True)

        audio_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
        if not audio_files:
            print(f"警告: 目录 '{input_dir}' 中没有找到 .wav 文件，跳过该类别。")
            continue

        print(f"\n正在增强类别 '{category}' (共 {len(audio_files)} 个原始文件)...")

        # 使用 tqdm 创建进度条
        for filename in tqdm(audio_files, desc=f"处理 {category}"):
            input_path = os.path.join(input_dir, filename)
            # 为每个文件生成4个增强版本 + 1个原始版本 = 5倍数据
            augmenter.augment_single_file(input_path, output_dir, num_augmentations=4)

    print("\n数据增强完成！")
    print(f"所有增强后的文件已保存到 '{output_base_dir}' 目录中。")
    print("-" * 50)


if __name__ == '__main__':
    main()