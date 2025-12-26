"""
音频处理工具函数
包括STFT、ISTFT、特征提取等
"""
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, Optional


def load_audio(file_path: str, sr: int = 16000) -> np.ndarray:
    """
    加载音频文件
    
    Args:
        file_path: 音频文件路径
        sr: 目标采样率
        
    Returns:
        音频数组，shape: (n_samples,)
    """
    audio, _ = librosa.load(file_path, sr=sr, mono=True)
    return audio


def save_audio(file_path: str, audio: np.ndarray, sr: int = 16000):
    """
    保存音频文件
    
    Args:
        file_path: 保存路径
        audio: 音频数组
        sr: 采样率
    """
    sf.write(file_path, audio, sr)


def stft(audio: np.ndarray, n_fft: int = 512, hop_length: int = 256, 
         win_length: int = 512, window: str = "hann") -> Tuple[np.ndarray, np.ndarray]:
    """
    短时傅里叶变换
    
    Args:
        audio: 音频信号
        n_fft: FFT窗口大小
        hop_length: 帧移
        win_length: 窗口长度
        window: 窗函数类型
        
    Returns:
        magnitude: 幅度谱, shape: (n_freq, n_time)
        phase: 相位谱, shape: (n_freq, n_time)
    """
    stft_matrix = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window
    )
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    return magnitude, phase


def istft(magnitude: np.ndarray, phase: np.ndarray, n_fft: int = 512,
          hop_length: int = 256, win_length: int = 512, 
          window: str = "hann", length: Optional[int] = None) -> np.ndarray:
    """
    逆短时傅里叶变换
    
    Args:
        magnitude: 幅度谱
        phase: 相位谱
        n_fft: FFT窗口大小
        hop_length: 帧移
        win_length: 窗口长度
        window: 窗函数类型
        length: 输出音频长度（样本数）
        
    Returns:
        重构的音频信号
    """
    stft_matrix = magnitude * np.exp(1j * phase)
    audio = librosa.istft(
        stft_matrix,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length
    )
    return audio


def log_magnitude(magnitude: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    对数幅度谱（用于模型输入）
    
    Args:
        magnitude: 幅度谱
        eps: 防止log(0)的小值
        
    Returns:
        对数幅度谱
    """
    return np.log(magnitude + eps)


def exp_magnitude(log_magnitude: np.ndarray) -> np.ndarray:
    """
    指数幅度谱（从模型输出恢复）
    
    Args:
        log_magnitude: 对数幅度谱
        
    Returns:
        幅度谱
    """
    # 增强幅度谱的动态范围
    magnitude = np.exp(log_magnitude)
    # 增加一个增益因子来提高整体音量
    magnitude *= 2.0
    return magnitude


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    归一化音频到[-1, 1]
    
    Args:
        audio: 音频信号
        
    Returns:
        归一化后的音频
    """
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio


def apply_agc(audio, target_level=-24.0, attack_time=0.05, release_time=0.2, sample_rate=48000, noise_gate_threshold=-50.0):
    """
    应用改进的自动增益控制(AGC)到音频信号，包含智能噪声门控功能
    
    Args:
        audio: 输入音频信号
        target_level: 目标电平 (dB) - 降低默认值以减少噪声放大
        attack_time: 攻击时间 (秒) - 增益减少的速度
        release_time: 释放时间 (秒) - 增益增加的速度
        sample_rate: 采样率
        noise_gate_threshold: 噪声门限阈值 (dB) - 低于此阈值的信号被视为噪声
        
    Returns:
        应用AGC后的音频信号
    """
    # 计算目标振幅
    target_amplitude = 10 ** (target_level / 20)
    gate_threshold_amplitude = 10 ** (noise_gate_threshold / 20)
    
    # 计算攻击和释放系数
    attack_coeff = np.exp(-np.log(9) / (attack_time * sample_rate))
    release_coeff = np.exp(-np.log(9) / (release_time * sample_rate))
    
    # 避免除以零
    epsilon = 1e-10
    
    # 初始化增益和输出
    output = np.zeros_like(audio)
    gain = 1.0
    
    # 使用多窗口RMS计算瞬时振幅（更准确的电平检测）
    # 小窗口（10ms）用于快速响应
    small_window_size = int(0.01 * sample_rate)
    small_window = np.hanning(small_window_size)
    # 大窗口（50ms）用于平滑检测
    large_window_size = int(0.05 * sample_rate)
    large_window = np.hanning(large_window_size)
    
    # 计算不同窗口的RMS值
    def calculate_rms(signal, window_size, window):
        rms = np.zeros_like(signal)
        half_window = window_size // 2
        for i in range(len(signal)):
            start = max(0, i - half_window)
            end = min(len(signal), i + half_window)
            windowed = signal[start:end] * window[(half_window - i + start) : (half_window + (end - i))]
            rms[i] = np.sqrt(np.mean(windowed**2) + epsilon)
        return rms
    
    rms_small = calculate_rms(audio, small_window_size, small_window)
    rms_large = calculate_rms(audio, large_window_size, large_window)
    
    # 融合两个RMS值，小窗口用于快速变化，大窗口用于整体电平
    rms = 0.6 * rms_small + 0.4 * rms_large
    
    # 应用AGC
    for i in range(len(audio)):
        # 计算当前振幅（使用融合后的RMS值）
        current_amplitude = rms[i]
        
        # 智能噪声门控：区分信号和噪声
        if current_amplitude < gate_threshold_amplitude:
            # 噪声部分：使用自适应增益衰减
            # 根据噪声电平动态调整衰减程度
            noise_ratio = current_amplitude / gate_threshold_amplitude
            desired_gain = 0.2 + 0.3 * noise_ratio  # 衰减40%-80%
        else:
            # 信号部分：正常应用AGC
            desired_gain = target_amplitude / current_amplitude
        
        # 平滑增益变化：使用指数平滑
        if desired_gain < gain:
            # 攻击阶段 (增益减少)
            gain = attack_coeff * gain + (1 - attack_coeff) * desired_gain
        else:
            # 释放阶段 (增益增加)
            gain = release_coeff * gain + (1 - release_coeff) * desired_gain
        
        # 限制增益范围，防止极端值
        gain = np.clip(gain, 0.1, 10.0)  # 限制增益在10%-1000%之间
        
        # 应用增益
        output[i] = audio[i] * gain
    
    # 移除软削波处理，使用直接的硬削波
    # output = np.tanh(output * 0.5) * 2.0
    
    # 最后限制输出范围在[-1, 1]
    output = np.clip(output, -1.0, 1.0)
    
    return output