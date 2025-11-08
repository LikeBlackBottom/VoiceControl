# train_model.py
import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# 设置随机种子，保证结果可复现
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


class VoiceModelTrainer:
    def __init__(self):
        # --- 音频参数设置 ---
        self.SAMPLE_RATE = 16000  # 采样率
        self.DURATION = 1.5  # 音频时长（秒），必须与录音时一致
        self.N_MFCC = 13  # MFCC特征数量
        self.N_FFT = 2048  # FFT窗口大小
        self.HOP_LENGTH = 512  # 帧移
        # 计算预期的MFCC帧数宽度 (向上取整)
        self.EXPECTED_FRAMES = int(np.ceil((self.SAMPLE_RATE * self.DURATION) / self.HOP_LENGTH))

        # --- 训练参数设置 ---
        self.BATCH_SIZE = 32
        self.EPOCHS = 30  # 训练轮数，CPU训练可以适当减少
        self.CLASSES = ['on', 'off', 'unknown']
        self.CLASS_MAP = {cls: i for i, cls in enumerate(self.CLASSES)}

    def extract_features(self, file_path):
        """
        读取音频文件并提取 MFCC 特征。
        同时确保所有特征图的大小完全一致（通过填充或截断）。
        """
        try:
            # 1. 加载音频，并强制转换为指定的采样率和时长
            audio, sr = librosa.load(file_path, sr=self.SAMPLE_RATE, duration=self.DURATION)

            # 2. 确保音频长度一致 (填充或截断到固定采样点数)
            target_len = int(self.SAMPLE_RATE * self.DURATION)
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
            else:
                audio = audio[:target_len]

            # 3. 提取 MFCC 特征
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.N_MFCC, n_fft=self.N_FFT,
                                        hop_length=self.HOP_LENGTH)

            # 4. 再次确保 MFCC 帧数一致 (处理可能的计算误差)
            if mfcc.shape[1] < self.EXPECTED_FRAMES:
                mfcc = np.pad(mfcc, ((0, 0), (0, self.EXPECTED_FRAMES - mfcc.shape[1])), mode='constant')
            else:
                mfcc = mfcc[:, :self.EXPECTED_FRAMES]

            return mfcc.T  # 转置，形状变为 (时间帧, MFCC特征数)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def load_data(self, data_dir):
        """从指定目录加载所有数据并提取特征"""
        X = []
        y = []
        print(f"正在从 '{data_dir}' 加载数据...")

        for category in self.CLASSES:
            cat_dir = os.path.join(data_dir, category)
            if not os.path.exists(cat_dir):
                print(f"警告: 目录 {cat_dir} 不存在，跳过。")
                continue

            files = [f for f in os.listdir(cat_dir) if f.endswith('.wav')]
            for file in tqdm(files, desc=f"加载类别 '{category}'"):
                file_path = os.path.join(cat_dir, file)
                feature = self.extract_features(file_path)
                if feature is not None:
                    X.append(feature)
                    y.append(self.CLASS_MAP[category])

        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """构建轻量级 CNN 模型"""
        model = tf.keras.Sequential([
            # 输入层：接收 (帧数, MFCC数, 1) 的“图像”
            tf.keras.layers.Input(shape=input_shape),

            # 第一个卷积块
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),  # 防止过拟合

            # 第二个卷积块
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),

            # 展平并全连接
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),

            # 输出层：3个节点对应3个类别，使用 softmax 输出概率
            tf.keras.layers.Dense(len(self.CLASSES), activation='softmax')
        ])
        return model

    def train(self):
        # 1. 检查数据目录
        base_split_dir = 'dataset_split'
        if not os.path.exists(base_split_dir):
            print(f"错误: 找不到 '{base_split_dir}' 目录。请先运行 prepare_dataset.py split。")
            return

        # 2. 加载数据
        X_train, y_train = self.load_data(os.path.join(base_split_dir, 'train'))
        X_val, y_val = self.load_data(os.path.join(base_split_dir, 'val'))
        X_test, y_test = self.load_data(os.path.join(base_split_dir, 'test'))

        # 3. 数据预处理：增加一个通道维度，适配 CNN 输入 (样本数, 帧数, MFCC数, 通道数1)
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        X_test = X_test[..., np.newaxis]

        # 标签转换为 One-hot 编码
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=len(self.CLASSES))
        y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=len(self.CLASSES))
        y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=len(self.CLASSES))

        print(f"\n训练集形状: {X_train.shape}, 标签形状: {y_train_cat.shape}")
        print(f"验证集形状: {X_val.shape}")
        print(f"测试集形状: {X_test.shape}")

        # 4. 构建和编译模型
        input_shape = X_train.shape[1:]  # (EXPECTED_FRAMES, N_MFCC, 1)
        model = self.build_model(input_shape)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        # 5. 训练模型
        print("\n--- 开始训练 ---")
        # 定义回调函数：只保存验证集上最好的模型
        best_model_path = os.path.join('models', 'best_model.keras')
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(best_model_path, save_best_only=True, monitor='val_accuracy', verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
        ]

        history = model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )

        # 6. 评估模型
        print("\n--- 在测试集上评估模型 ---")
        # 加载保存的最好的模型进行评估
        best_model = tf.keras.models.load_model(best_model_path)
        loss, acc = best_model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"测试集准确率: {acc:.4f}")

        # 生成分类报告
        y_pred = best_model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        report = classification_report(y_test, y_pred_classes, target_names=self.CLASSES)
        print("\n分类报告:\n", report)

        # 保存报告到文件
        with open(os.path.join('models', 'classification_report.txt'), 'w') as f:
            f.write(report)

        # 绘制并保存混淆矩阵
        cm = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.CLASSES, yticklabels=self.CLASSES)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join('models', 'confusion_matrix.png'))
        plt.close()

        # 7. 转换为 TFLite 模型
        print("\n--- 正在转换为 TensorFlow Lite 模型 ---")
        converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
        # 启用标准优化
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        tflite_path = os.path.join('models', 'voice_model.tflite')
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        print(f"\n成功！TFLite 模型已保存到: {tflite_path}")
        print("你可以将此文件发送给你的学长。")


if __name__ == '__main__':
    trainer = VoiceModelTrainer()
    trainer.train()