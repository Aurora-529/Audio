"""
数据加载和预处理模块
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List
import random

from utils.audio_utils import load_audio, stft, log_magnitude


class AudioDenoisingDataset(Dataset):
    """
    音频降噪数据集类
    支持clean和noisy音频配对，或仅clean音频（测试集）
    """
    
    def __init__(
        self,
        clean_dir: str,
        noisy_dir: Optional[str] = None,
        segment_length: float = 4.0,
        sample_rate: int = 48000,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        augment: bool = False,
        shuffle: bool = True
    ):
        """
        初始化数据集
        
        Args:
            clean_dir: 清晰音频目录
            noisy_dir: 带噪音频目录（如果为None，则仅使用clean音频）
            segment_length: 音频分段长度（秒）
            sample_rate: 采样率
            n_fft: FFT窗口大小
            hop_length: 帧移
            win_length: 窗口长度
            augment: 是否使用数据增强
            shuffle: 是否打乱文件顺序
        """
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir) if noisy_dir else None
        self.segment_length = segment_length
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.augment = augment
        self.segment_samples = int(segment_length * sample_rate)
        
        # 扫描文件
        self.clean_files = sorted(list(self.clean_dir.glob("*.wav")))
        
        if self.noisy_dir:
            # 匹配clean和noisy文件
            self.file_pairs = self._match_files(self.clean_files, self.noisy_dir)
        else:
            # 仅使用clean文件（测试集）
            self.file_pairs = [(f, None) for f in self.clean_files]
        
        if shuffle:
            random.shuffle(self.file_pairs)
        
        print(f"数据集大小: {len(self.file_pairs)} 个文件")
        if self.noisy_dir:
            print(f"  清晰音频: {len(self.clean_files)} 个")
            print(f"  匹配的带噪音频: {len([p for p in self.file_pairs if p[1] is not None])} 个")
        else:
            print(f"  仅清晰音频: {len(self.clean_files)} 个")
    
    def _match_files(
        self, 
        clean_files: List[Path], 
        noisy_dir: Path
    ) -> List[Tuple[Path, Optional[Path]]]:
        """
        匹配clean和noisy文件
        
        Args:
            clean_files: 清晰音频文件列表
            noisy_dir: 带噪音频目录
            
        Returns:
            匹配的文件对列表 [(clean_path, noisy_path), ...]
        """
        file_pairs = []
        noisy_files = {f.name: f for f in noisy_dir.glob("*.wav")}
        
        for clean_file in clean_files:
            noisy_file = noisy_files.get(clean_file.name)
            if noisy_file:
                file_pairs.append((clean_file, noisy_file))
            else:
                # 如果没有匹配的noisy文件，跳过（或使用clean作为noisy）
                # 这里选择跳过
                pass
        
        return file_pairs
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.file_pairs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            (noisy_log_mag, clean_log_mag): 带噪和清晰的对数幅度谱
            shape: (freq_bins, time_frames)
        """
        clean_file, noisy_file = self.file_pairs[idx]
        
        # 加载音频
        clean_audio = load_audio(str(clean_file), sr=self.sample_rate)
        
        if noisy_file:
            noisy_audio = load_audio(str(noisy_file), sr=self.sample_rate)
        else:
            # 如果没有noisy文件，使用clean作为noisy（测试集情况）
            noisy_audio = clean_audio.copy()
        
        # 确保长度一致
        min_len = min(len(clean_audio), len(noisy_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]
        
        # 音频分段（训练时随机裁剪，测试时使用整个音频或固定位置）
        if len(clean_audio) > self.segment_samples:
            if self.augment or not hasattr(self, '_test_mode'):
                # 训练模式：随机裁剪
                start_idx = random.randint(0, len(clean_audio) - self.segment_samples)
            else:
                # 测试模式：从开头裁剪
                start_idx = 0
            clean_audio = clean_audio[start_idx:start_idx + self.segment_samples]
            noisy_audio = noisy_audio[start_idx:start_idx + self.segment_samples]
        else:
            # 如果音频太短，进行填充
            pad_length = self.segment_samples - len(clean_audio)
            clean_audio = np.pad(clean_audio, (0, pad_length), mode='constant')
            noisy_audio = np.pad(noisy_audio, (0, pad_length), mode='constant')
        
        # 数据增强（仅在训练时）
        if self.augment:
            clean_audio, noisy_audio = self._augment(clean_audio, noisy_audio)
        
        # STFT转换
        clean_mag, _ = stft(
            clean_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        noisy_mag, _ = stft(
            noisy_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        # 对数幅度转换
        clean_log_mag = log_magnitude(clean_mag)
        noisy_log_mag = log_magnitude(noisy_mag)
        
        # 转换为tensor
        clean_log_mag = torch.FloatTensor(clean_log_mag)
        noisy_log_mag = torch.FloatTensor(noisy_log_mag)
        
        return noisy_log_mag, clean_log_mag
    
    def _augment(
        self, 
        clean_audio: np.ndarray, 
        noisy_audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据增强
        
        Args:
            clean_audio: 清晰音频
            noisy_audio: 带噪音频
            
        Returns:
            增强后的音频对
        """
        # 随机增益（轻微）
        if random.random() < 0.5:
            gain = random.uniform(0.9, 1.1)
            clean_audio = clean_audio * gain
            noisy_audio = noisy_audio * gain
        
        # 随机翻转（时间反转）
        if random.random() < 0.3:
            clean_audio = np.flip(clean_audio)
            noisy_audio = np.flip(noisy_audio)
        
        return clean_audio, noisy_audio


def create_dataloaders(
    clean_train_dir: str,
    noisy_train_dir: Optional[str] = None,
    clean_test_dir: Optional[str] = None,
    noisy_test_dir: Optional[str] = None,
    train_ratio: float = 0.9,
    val_ratio: float = 0.1,
    batch_size: int = 16,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        clean_train_dir: 清晰音频训练目录
        noisy_train_dir: 带噪音频训练目录
        clean_test_dir: 清晰音频测试目录（可选，干净人声）
        noisy_test_dir: 带噪音频测试目录（可选，对应clean_test_dir，一一配对）
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        batch_size: 批次大小
        num_workers: 数据加载进程数
        **dataset_kwargs: 传递给AudioDenoisingDataset的其他参数
            - segment_length: 音频分段长度（秒）
            - sample_rate: 采样率
            - n_fft: FFT窗口大小
            - hop_length: 帧移
            - win_length: 窗口长度
            - augment: 是否使用数据增强
        
    Returns:
        (train_loader, val_loader, test_loader)
        test_loader可能为None（如果clean_test_dir为None）
    """
    # 创建训练数据集
    train_dataset = AudioDenoisingDataset(
        clean_dir=clean_train_dir,
        noisy_dir=noisy_train_dir,
        augment=dataset_kwargs.get('augment', True),
        shuffle=True,
        **{k: v for k, v in dataset_kwargs.items() if k != 'augment'}
    )
    
    # 创建验证数据集（不使用数据增强）
    val_dataset_kwargs = dataset_kwargs.copy()
    val_dataset_kwargs['augment'] = False
    val_dataset_kwargs['shuffle'] = False
    
    val_dataset = AudioDenoisingDataset(
        clean_dir=clean_train_dir,
        noisy_dir=noisy_train_dir,
        **val_dataset_kwargs
    )
    
    # 划分训练集和验证集
    total_size = len(train_dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))
    
    # 创建训练集和验证集的子集
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # 创建测试数据集（如果提供）
    test_loader = None
    if clean_test_dir:
        test_dataset_kwargs = dataset_kwargs.copy()
        test_dataset_kwargs['augment'] = False
        test_dataset_kwargs['shuffle'] = False
        # 测试集现在可以同时有 clean / noisy：
        # - 有 noisy_test_dir: 使用 (clean_test_dir, noisy_test_dir) 配对
        # - 无 noisy_test_dir: 退化为只用 clean_test_dir（noisy=clean）
        test_dataset = AudioDenoisingDataset(
            clean_dir=clean_test_dir,
            noisy_dir=noisy_test_dir,
            **test_dataset_kwargs
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 丢弃最后一个不完整的batch
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
