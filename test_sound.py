#!/usr/bin/env python3
"""
简单的音频输出测试脚本
生成440Hz的标准A调测试音
"""
import numpy as np
import pyaudio
import time

def test_audio_output():
    # 配置参数
    sample_rate = 48000
    duration = 5  # 播放5秒
    frequency = 440  # 标准A调
    volume = 0.5  # 50%最大音量
    
    print(f"正在生成 {frequency}Hz 测试音，音量: {volume}")
    print(f"采样率: {sample_rate} Hz，持续时间: {duration}秒")
    
    # 生成音频数据
    t = np.arange(sample_rate * duration) / sample_rate
    audio_data = volume * np.sin(2 * np.pi * frequency * t)
    
    # 将音频数据转换为16位整数格式
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # 初始化PyAudio
    p = pyaudio.PyAudio()
    
    # 打开音频流
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        output=True
    )
    
    try:
        print("开始播放测试音...")
        stream.write(audio_data.tobytes())
        print("测试音播放完成！")
    finally:
        # 关闭音频流
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    test_audio_output()
