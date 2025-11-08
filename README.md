# 开灯 / 关灯语音控制模型训练教程

本教程基于仓库内的完整代码示例，帮助你从零开始搭建属于自己的中文语音唤醒数据集，并训练出能够识别“开灯”（on）与“关灯”（off）的轻量级语音分类模型，同时包含一个“未知”（unknown）类别用于提升模型鲁棒性。按照本文档的步骤，你可以顺利完成数据采集、数据增强、数据集划分、模型训练与部署文件导出等流程。

## 1. 环境准备

### 1.1 Python 与依赖库

推荐使用 **Python 3.9+**。以下是主要依赖：

- `pyaudio`：麦克风录音
- `pygame`：录音后音频预览（可选）
- `librosa`、`soundfile`、`numpy`：音频处理与特征提取
- `tensorflow`：深度学习训练框架
- `tqdm`：进度条提示
- `matplotlib`、`seaborn`：可视化混淆矩阵
- `scikit-learn`：分类报告

使用 pip 安装：

```bash
pip install -r requirements.txt
```

> 如果仓库没有提供 `requirements.txt`，可以手动安装：
>
> ```bash
> pip install pyaudio pygame librosa soundfile numpy tensorflow tqdm matplotlib seaborn scikit-learn
> ```

### 1.2 设备要求

- 电脑需具备麦克风（内置或外接）。
- 建议在安静环境中录音，减少背景噪音。

## 2. 目录结构

执行教程中的准备脚本后，项目目录将包含以下关键文件夹：

```
VoiceControl/
├── dataset/                # 原始录音（on/off/unknown）
├── dataset_augmented/      # 数据增强后的音频
├── dataset_split/          # 划分好的 train/val/test 数据
├── models/                 # 训练输出（最佳模型、报告、图像、TFLite）
├── augment_audio.py        # 数据增强脚本
├── prepare_dataset.py      # 目录创建 & 数据集划分脚本
├── record_audio.py         # 录音工具
├── train_model.py          # 模型训练与导出脚本
└── README.md               # 本教程
```

## 3. 脚本逐个说明

| 文件 | 作用概述 |
| --- | --- |
| `record_audio.py` | 提供命令行录音工具，可选择“开灯”“关灯”“未知”三种类别，自动编号保存，并支持录音预览。建议先运行 `python prepare_dataset.py create` 创建目录结构。 |
| `augment_audio.py` | 对原始录音执行批量数据增强：添加噪声、时间平移、随机变调，同时保留原始文件，生成更丰富的训练数据。 |
| `prepare_dataset.py` | 含两个子任务：`create` 创建完整目录树；`split` 按比例（默认 70%/15%/15%）将增强后的数据划分为训练、验证和测试集。 |
| `train_model.py` | 提取 MFCC 特征，搭建轻量级 CNN 进行三分类训练，保存最优 Keras 模型、分类报告、混淆矩阵，并导出压缩后的 `voice_model.tflite` 模型，方便部署到移动或嵌入式设备。 |

## 4. 实战流程

按照以下步骤操作，可以快速复现一个成熟的语音分类模型。

### 步骤 1：初始化目录

```bash
python prepare_dataset.py create
```

- 脚本会自动创建 `dataset/`、`dataset_augmented/`、`dataset_split/` 以及 `models/` 等所需目录。
- 若目录已存在不会报错，可放心重复执行。

### 步骤 2：录制原始语音样本

```bash
python record_audio.py
```

操作提示：

1. 根据命令行菜单选择类别（1=开灯、2=关灯、3=未知）。
2. 准备好后按 Enter 开始录音，脚本会倒计时 3 秒，默认录制 1.5 秒。
3. 录音结束后可选择是否立即播放预览（需要安装 `pygame`）。
4. 对每个类别尽量录制 50~100 条样本，保证数据平衡。

录制完成后，原始 wav 文件将保存在 `dataset/<类别>/` 中。

### 步骤 3：执行数据增强

```bash
python augment_audio.py
```

- 每个原始文件会生成 1 份原始备份 + 4 种增强版本，共计 5 倍数据量。
- 增强策略包括随机噪声、时间平移与变调，可有效提升模型泛化能力。
- 结果存放在 `dataset_augmented/<类别>/` 中。

### 步骤 4：划分训练 / 验证 / 测试集

```bash
python prepare_dataset.py split
```

- 默认划分比例：训练集 70%、验证集 15%、测试集 15%。
- 划分后的文件会复制到 `dataset_split/train|val|test/<类别>/`。
- 可根据需要在 `split_dataset` 函数中调整比例。

### 步骤 5：训练模型并导出

```bash
python train_model.py
```

脚本流程：

1. 从 `dataset_split` 读取数据，提取统一长度的 MFCC 特征。
2. 构建卷积神经网络进行训练，使用早停和模型检查点回调，防止过拟合。
3. 训练完毕后输出：
   - `models/best_model.keras`：验证集上表现最好的模型权重；
   - `models/classification_report.txt`：详细的 Precision/Recall/F1 统计；
   - `models/confusion_matrix.png`：三分类混淆矩阵可视化；
   - `models/voice_model.tflite`：压缩后的 TensorFlow Lite 模型，方便部署。

如果希望缩短训练时间，可以将 `VoiceModelTrainer` 中的 `EPOCHS` 调小，或减少数据量。

## 5. 常见问题与建议

- **麦克风权限**：首次运行录音脚本时，若系统弹出权限提示，请允许访问麦克风。
- **数据不平衡**：确保三个类别的样本数量接近，避免模型偏向某一类。
- **过拟合**：若训练集准确率远高于验证集，可增加数据量、提升增强多样性或调大 `Dropout`。
- **部署测试**：训练完成后，可在移动端或 Raspberry Pi 上加载 `voice_model.tflite`，将 MFCC 预处理流程迁移后即可实时推断。

## 6. 后续扩展方向

- 新增更多唤醒词或指令类别，只需在 `CLASSES` 列表中添加名称，并重新录制/增强数据即可。
- 尝试加入语速变化、音量缩放等增强方式，提升模型适应性。
- 将训练流程封装为 Jupyter Notebook，方便教学演示。

---

该教程由重庆工商职业学院23级软件技术（东软班）HQ 为徐开雄 独家奉献，欢迎其他同学学习
