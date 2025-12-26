"""
评估指标模块
包括STOI、SDR等
"""
import numpy as np
import torch
from typing import Optional

from pystoi import stoi


def calculate_stoi(clean: np.ndarray, enhanced: np.ndarray, 
                  sr: int = 16000) -> float:
    """
    计算STOI分数
    
    Args:
        clean: 干净音频
        enhanced: 增强后的音频
        sr: 采样率
        
    Returns:
        STOI分数（0-1）
    """
    try:
        # 确保长度一致
        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]
        
        score = stoi(clean, enhanced, sr, extended=False)
        return score
    except Exception as e:
        print(f"STOI计算错误: {e}")
        return 0.0


def calculate_sdr(clean: np.ndarray, enhanced: np.ndarray) -> float:
    """
    计算SDR（Signal-to-Distortion Ratio）
    
    Args:
        clean: 干净音频
        enhanced: 增强后的音频
        
    Returns:
        SDR（dB）
    """
    try:
        # 确保长度一致
        min_len = min(len(clean), len(enhanced))
        clean = clean[:min_len]
        enhanced = enhanced[:min_len]
        
        # 计算信号功率
        signal_power = np.sum(clean ** 2)
        
        # 计算失真功率
        distortion = enhanced - clean
        distortion_power = np.sum(distortion ** 2)
        
        if distortion_power == 0:
            return float('inf')
        
        sdr = 10 * np.log10(signal_power / (distortion_power + 1e-10))
        return sdr
    except Exception as e:
        print(f"SDR计算错误: {e}")
        return 0.0


def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    频谱域损失函数（MSE）
    
    Args:
        pred: 预测的对数幅度谱
        target: 目标的对数幅度谱
        
    Returns:
        损失值
    """
    return torch.mean((pred - target) ** 2)