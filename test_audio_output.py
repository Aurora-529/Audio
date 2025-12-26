#!/usr/bin/env python3
"""
测试音频输出功能的简单脚本
用于验证PyAudio输出设备是否正常工作
"""
import numpy as np
import pyaudio
import time

# 配置参数
SAMPLE_RATE = 48000
CHUNK_SIZE = 4800  # 100ms
DURATION = 5  # 测试时长（秒）
FREQUENCY = 440  # 测试频率（Hz，A4音）

# 初始化PyAudio
p = pyaudio.PyAudio()

# 查看所有输出设备
print("可用的输出设备:")
for i in range(p.get_device_count()):
    device_info = p.get_device_info_by_index(i)
    if device_info['maxOutputChannels'] > 0:
        print(f"设备 {i}: {device_info['name']} (通道数: {device_info['maxOutputChannels']})")

# 获取默认输出设备
default_output = p.get_default_output_device_info()
print(f"\n默认输出设备: {default_output['index']} - {default_output['name']}")

# 创建测试音频（440Hz正弦波）
print(f"\n生成 {FREQUENCY}Hz 测试音频，持续 {DURATION} 秒...")
t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
sine_wave = 0.5 * np.sin(2 * np.pi * FREQUENCY * t)

# 打开输出流
print("\n打开音频输出流...")
output_stream = p.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=SAMPLE_RATE,
    output=True,
    output_device_index=default_output['index'],
    frames_per_buffer=CHUNK_SIZE
)

# 播放测试音频
print("开始播放测试音频...")
try:
    # 分块播放
    for i in range(0, len(sine_wave), CHUNK_SIZE):
        chunk = sine_wave[i:i+CHUNK_SIZE]
        # 确保块大小正确
        if len(chunk) < CHUNK_SIZE:
            chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
        
        # 播放
        output_stream.write(chunk.astype(np.float32).tobytes())
        print(f"播放进度: {i/len(sine_wave)*100:.1f}%")
        time.sleep(0.01)  # 避免太快
        
except KeyboardInterrupt:
    print("播放被中断")

# 关闭流
print("\n关闭音频流...")
output_stream.stop_stream()
output_stream.close()
p.terminate()

print("测试完成！")
print("如果您能听到440Hz的连续音调，说明音频输出设备工作正常。")
print("如果没有听到声音，请检查设备选择或音量设置。")