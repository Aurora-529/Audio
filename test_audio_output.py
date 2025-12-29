import numpy as np
import pyaudio
import soundfile as sf
from models.deepfilternet_wrapper import create_deepfilternet_wrapper

def test_audio_output():
    """测试音频输出功能"""
    # 创建DeepFilterNetWrapper实例
    print("创建DeepFilterNetWrapper实例...")
    wrapper = create_deepfilternet_wrapper(
        model_path='checkpoints/best.pth',
        sample_rate=48000,
        device='cuda' if pyaudio.PyAudio().get_default_output_device_info()['hostApi'] == 0 else 'cpu',
        enable_agc=True,
        agc_target_level=-10.0,  # 提高目标电平
        noise_gate_threshold=-40.0  # 降低噪声门限
    )
    
    # 生成测试音频（正弦波）
    print("生成测试音频...")
    sample_rate = 48000
    duration = 5.0  # 5秒
    frequency = 440.0  # A4音
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    test_audio = 0.1 * np.sin(2 * np.pi * frequency * t)
    
    # 添加一些噪声
    noise = 0.05 * np.random.randn(len(test_audio))
    noisy_audio = test_audio + noise
    
    # 使用模型处理音频
    print("处理音频...")
    enhanced_audio = wrapper.denoise(noisy_audio, sample_rate=sample_rate)
    
    # 保存原始和处理后的音频
    print("保存音频文件...")
    sf.write("test_original.wav", noisy_audio, sample_rate)
    sf.write("test_enhanced.wav", enhanced_audio, sample_rate)
    
    # 直接播放处理后的音频（绕过realtime.py的复杂逻辑）
    print("播放处理后的音频...")
    pa = pyaudio.PyAudio()
    
    # 打开音频流
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sample_rate,
        output=True,
        frames_per_buffer=4800
    )
    
    # 播放音频
    stream.write(enhanced_audio.astype(np.float32).tobytes())
    
    # 关闭流和PyAudio
    stream.stop_stream()
    stream.close()
    pa.terminate()
    
    print("测试完成！请检查生成的wav文件并确认是否听到了音频输出。")
    print(f"原始音频峰值: {np.max(np.abs(noisy_audio))}")
    print(f"增强音频峰值: {np.max(np.abs(enhanced_audio))}")

if __name__ == "__main__":
    test_audio_output()
