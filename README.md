# 开灯 / 关灯语音控制模型训练教程 🔊💡

本教程带你从零搭建中文语音数据集，并训练一个可识别“开灯”(on) / “关灯”(off) / “未知”(unknown) 的轻量级分类模型。按步骤完成数据采集、增强、划分、训练与导出 TFLite 模型即可完成端到端流程。

## 1. 环境准备
- 推荐 Python 3.9+
- 安装依赖（如无 requirements.txt 可手动安装）：
```bash
# 可选：创建虚拟环境（Windows）
python -m venv .venv && .\.venv\Scripts\activate
# 安装（二选一）
pip install -r requirements.txt
pip install pyaudio pygame librosa soundfile numpy tensorflow tqdm matplotlib seaborn scikit-learn
```

## 2. 目录结构 🗂️
执行准备脚本后，目录包含：
```
项目根目录/
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

## 3. 脚本说明 🧩
- record_audio.py：命令行录音工具，支持 on/off/unknown 三类，自动编号保存；支持预览（需 pygame）。
- augment_audio.py：对原始录音批量增强（随机噪声、时间平移、随机变调）；保留原始版本，约 5× 数据量。
- prepare_dataset.py：create 创建目录；split 按 70%/15%/15% 划分 train/val/test。
- train_model.py：提取 MFCC + 轻量 CNN 训练；保存 models/best_model.keras、分类报告、混淆矩阵，并导出 models/voice_model.tflite。

## 4. 实战流程 🚀
1) 初始化目录
```bash
python prepare_dataset.py create
```
2) 录制样本（建议各类 50~100 条）
```bash
python record_audio.py
```
3) 数据增强（原始副本 + 4 种增强版本）
```bash
python augment_audio.py
```
4) 划分训练/验证/测试集（默认 70/15/15）
```bash
python prepare_dataset.py split
```
5) 训练并导出模型（输出在 models/）
```bash
python train_model.py
```
提示：如需缩短训练时间，可在 VoiceModelTrainer 中调小 EPOCHS。

## 5. 常见问题与建议 ❓
- 麦克风权限：首次运行录音脚本时，请允许系统访问麦克风。
- 数据不平衡：保证三类样本数量接近，避免偏置。
- 过拟合：如训练准确率远高于验证集，增加数据/增强多样性或调大 Dropout。
- 部署：将 models/voice_model.tflite 与相同的 MFCC 预处理部署至移动端或 Raspberry Pi。

## 6. 自定义语音控制模型教程 🛠️
想训练“其他指令”的模型（例如 start/stop/unknown），按照以下步骤：

1) 规划类目与时长
- 确定新类别列表，如 ['start','stop','unknown']。
- 若需更改录音时长（默认 1.5 秒），请保持训练与录音一致。

2) 修改代码常量（四处）
- train_model.py
  - 将 VoiceModelTrainer.CLASSES 改为你的新类别列表。
  - 如需修改时长/采样率，同步更新 DURATION 与 SAMPLE_RATE。
- record_audio.py
  - 更新 categories 映射，例如 {'1':'start','2':'stop','3':'unknown'}。
  - 将 recording_duration 调整为与训练 DURATION 一致。
  - 若修改采样率，构造 AudioRecorder(sample_rate=...)；并把 pygame.mixer.init(frequency=...) 改为相同值。
- prepare_dataset.py
  - 将 categories 列表改为新类别（用于 split）。
- augment_audio.py
  - 将 categories 列表改为新类别（用于增强）。

3) 重新准备与采集
```bash
python prepare_dataset.py create   # 生成新的目录结构（会包含你的新类别）
python record_audio.py             # 录制每类 50~100 条，尽量均衡
python augment_audio.py            # 生成约 5× 数据量
python prepare_dataset.py split    # 70/15/15 划分到 dataset_split/
```

4) 训练与验证
```bash
python train_model.py
```
- 输出文件位于 models/：best_model.keras、classification_report.txt、confusion_matrix.png、voice_model.tflite。
- 如类别数改变，模型的输出维度会自动随 CLASSES 调整，无需额外改动。

5) 常见自定义项
- 更短/更长口令：修改 DURATION（train_model.py 与 record_audio.py）。
- 不同采样率：统一修改 SAMPLE_RATE（train/record/augment 三处）与 pygame 播放频率。
- 更多类别：在四个文件的类别列表中同步添加目录名并重新采集数据。

## 7. 致谢与扩展 🔧
- 可加入语速变化、音量缩放等增强策略；或迁移到 Notebook 以便教学展示。
- 本教程由 重庆工商职业学院 23 级软件技术（东软班）HQ 为 徐开雄 贡献，欢迎学习参考。
