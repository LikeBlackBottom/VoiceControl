# record_audio.py
import pyaudio
import wave
import os
import time
import threading

# 使用 pygame 来播放音频预览
try:
    import pygame
except ImportError:
    print("警告: pygame 模块未找到。将无法使用音频预览功能。")
    print("请运行 'pip install pygame' 来安装它。")
    pygame = None


class AudioRecorder:
    """
    一个用于录制音频的类，封装了 PyAudio 的复杂操作。
    """

    def __init__(self, sample_rate=16000, chunk_size=1024, channels=1):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_format = pyaudio.paInt16

        self.frames = []
        self.is_recording = False
        self.audio_interface = pyaudio.PyAudio()
        self.stream = None
        self.record_thread = None

    def _recording_thread(self):
        """在独立的线程中执行的录音循环"""
        try:
            self.stream = self.audio_interface.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            while self.is_recording and self.stream.is_active():
                data = self.stream.read(self.chunk_size)
                self.frames.append(data)
        except Exception as e:
            print(f"录音时发生错误: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()

    def start(self):
        """开始录音"""
        if self.is_recording:
            print("已经在录音中了。")
            return

        self.frames = []
        self.is_recording = True
        self.record_thread = threading.Thread(target=self._recording_thread)
        self.record_thread.start()
        print("录音已开始...")

    def stop(self):
        """停止录音"""
        if not self.is_recording:
            return

        self.is_recording = False
        if self.record_thread:
            self.record_thread.join()  # 等待录音线程结束
        print("录音已停止。")

    def save(self, filename):
        """将录制的音频保存到 .wav 文件"""
        if not self.frames:
            print("错误：没有录音数据可供保存。")
            return False

        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio_interface.get_sample_size(self.audio_format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.frames))
            print(f"文件已成功保存到: {filename}")
            return True
        except Exception as e:
            print(f"保存文件时发生错误: {e}")
            return False

    def cleanup(self):
        """清理 PyAudio 资源"""
        self.audio_interface.terminate()


def play_audio(filename):
    """使用 pygame 播放指定的音频文件"""
    if not pygame:
        print("无法播放预览，因为 pygame 未安装。")
        return

    if not os.path.exists(filename):
        print(f"错误：找不到音频文件 '{filename}'")
        return

    try:
        pygame.mixer.init(frequency=16000)
        pygame.mixer.music.load(filename)
        print("正在播放预览...")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
    except Exception as e:
        print(f"播放音频时发生错误: {e}")
    finally:
        # 确保 mixer 被正确关闭
        if pygame.mixer.get_init():
            pygame.mixer.quit()


def main():
    """主函数，运行交互式录音工具"""
    categories = {'1': 'on', '2': 'off', '3': 'unknown'}
    recording_duration = 1.5  # 录音时长（秒）

    # 检查 'dataset' 目录是否存在
    if not os.path.isdir('dataset'):
        print("错误: 'dataset' 目录不存在！")
        print("请先运行 'python prepare_dataset.py create' 来创建目录结构。")
        return

    recorder = AudioRecorder()

    try:
        while True:
            print("\n" + "=" * 50)
            print("     语音数据集录制工具 (每次录制 1.5 秒)     ")
            print("=" * 50)
            print(" 1. 录制 '开灯' (on) 样本")
            print(" 2. 录制 '关灯' (off) 样本")
            print(" 3. 录制 '未知' (unknown) 样本 (说其他词或噪音)")
            print(" 4. 退出")
            print("-" * 50)

            choice = input("请选择一个选项 (1-4): ").strip()

            if choice == '4':
                print("正在退出...")
                break

            if choice not in categories:
                print("无效的选项，请输入 1 到 4 之间的数字。")
                continue

            category = categories[choice]
            target_dir = os.path.join('dataset', category)

            # 获取当前已有文件数量，用于自动编号
            count = len([f for f in os.listdir(target_dir) if f.endswith('.wav')])
            filename = os.path.join(target_dir, f"sample_{count + 1:04d}.wav")

            print(f"\n即将录制类别 '{category}' 的第 {count + 1} 个样本。")
            input("准备好后，请按 Enter 键开始录音...")

            # 倒计时
            for i in range(3, 0, -1):
                print(f"{i}...", end='', flush=True)
                time.sleep(1)
            print("开始！")

            recorder.start()
            time.sleep(recording_duration)
            recorder.stop()

            if recorder.save(filename):
                # 询问是否播放预览
                preview = input("是否需要播放录音预览? (y/n): ").strip().lower()
                if preview == 'y':
                    play_audio(filename)
            else:
                print("录音保存失败，请检查错误信息。")

    except KeyboardInterrupt:
        print("\n检测到用户中断，正在退出程序...")
    finally:
        recorder.cleanup()
        print("程序已清理并退出。")


if __name__ == '__main__':
    main()