"""
DeepFilterNet模型包装器
提供统一的接口使用DeepFilterNet进行音频降噪
使用自定义实现的DeepFilterNet模型
"""
import torch
import numpy as np
from typing import Optional
from models.deepfilternet import create_deepfilternet_model, DeepFilterNetState
from utils.audio_utils import stft, istft, log_magnitude, exp_magnitude, apply_agc


class DeepFilterNetWrapper:
    """
    DeepFilterNet包装器
    提供简单易用的接口进行音频降噪
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        sample_rate: int = 48000,
        device: Optional[str] = None,
        n_fft: int = 512,
        hidden_size: int = 128,
        lstm_layers: int = 2,
        filter_order: int = 5,
        enable_agc: bool = True,
        agc_target_level: float = -18.0,  # 提高目标电平，增加整体音量
        agc_attack_time: float = 0.02,  # 减少到20ms，使增益更快响应语音
        agc_release_time: float = 0.1,  # 减少到100ms，使增益更快恢复
        noise_gate_threshold: float = -45.0  # 提高噪声门限，减少对弱语音的抑制
    ):
        """
        初始化DeepFilterNet模型
        
        Args:
            model_path: 模型路径，如果为None则创建新模型
            sample_rate: 采样率（默认48kHz）
            device: 设备（'cuda'或'cpu'），如果为None则自动选择
            n_fft: FFT窗口大小
            hidden_size: LSTM隐藏层大小
            lstm_layers: LSTM层数
            filter_order: 滤波器阶数
            enable_agc: 是否启用自动增益控制(AGC)
            agc_target_level: AGC目标电平 (dB)
            agc_attack_time: AGC攻击时间 (秒)
            agc_release_time: AGC释放时间 (秒)
            noise_gate_threshold: 噪声门限阈值 (dB) - 低于此阈值的信号被视为噪声，不应用增益
        """
        # 设置设备
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = n_fft // 2
        self.win_length = n_fft
        
        # AGC参数
        self.enable_agc = enable_agc
        self.agc_target_level = agc_target_level
        self.agc_attack_time = agc_attack_time
        self.agc_release_time = agc_release_time
        self.noise_gate_threshold = noise_gate_threshold
        
        # 加载或创建模型
        if model_path is not None and model_path:
            # 加载已保存的模型
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # 从检查点加载
                    self.model, self.state = create_deepfilternet_model(
                        n_fft=n_fft,
                        hidden_size=hidden_size,
                        lstm_layers=lstm_layers,
                        filter_order=filter_order
                    )
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.model = self.model.to(self.device)
                elif isinstance(checkpoint, torch.nn.Module):
                    # 直接加载模型
                    self.model = checkpoint
                    self.model = self.model.to(self.device)
                    self.state = DeepFilterNetState()
                else:
                    raise ValueError(f"无法识别的检查点格式: {type(checkpoint)}")
            except Exception as e:
                raise RuntimeError(f"加载模型失败: {model_path}, 错误: {e}")
        else:
            # 创建新模型
            self.model, self.state = create_deepfilternet_model(
                n_fft=n_fft,
                hidden_size=hidden_size,
                lstm_layers=lstm_layers,
                filter_order=filter_order
            )
            self.model = self.model.to(self.device)
        
        self.model.eval()
        
        print(f"DeepFilterNet模型加载完成，使用设备: {self.device}")
        print(f"采样率: {self.sample_rate} Hz")
        print(f"FFT窗口: {self.n_fft}")
    
    def denoise(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """
        对音频进行降噪
        
        Args:
            audio: 输入音频数组，shape: (n_samples,) 或 (channels, n_samples)
            sample_rate: 输入音频的采样率，如果为None则使用self.sample_rate
            
        Returns:
            降噪后的音频数组，shape: (n_samples,)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        # 确保是numpy数组
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # 如果采样率不匹配，需要重采样
        if sample_rate != self.sample_rate:
            import librosa
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
            sample_rate = self.sample_rate
        
        # 处理单声道/立体声
        if audio.ndim == 1:
            original_length = len(audio)
        else:
            original_length = audio.shape[-1]
            if audio.shape[0] > 1:
                audio = audio[0]
            elif audio.shape[0] == 1:
                audio = audio[0]
        
        # 改进的STFT参数：使用更大的窗口和50%的重叠
        n_fft = self.n_fft
        hop_length = self.hop_length
        win_length = self.win_length
        
        # 转换为频谱
        magnitude, phase = stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann"
        )
        
        # 对数幅度谱
        log_mag = log_magnitude(magnitude)
        
        # 转换为张量
        log_mag_tensor = torch.FloatTensor(log_mag).unsqueeze(0).unsqueeze(0)  # (1, 1, freq, time)
        log_mag_tensor = log_mag_tensor.to(self.device)
        
        # DeepFilterNet处理
        with torch.no_grad():
            # 模型前向传播
            enhanced_log_mag = self.model(log_mag_tensor)  # (1, 1, freq, time)
            
            # 安全的维度调整
            if enhanced_log_mag.dim() == 4:
                enhanced_log_mag = enhanced_log_mag.squeeze(0).squeeze(0)  # (freq, time)
            elif enhanced_log_mag.dim() == 3:
                enhanced_log_mag = enhanced_log_mag.squeeze(0)  # (freq, time)
            enhanced_log_mag = enhanced_log_mag.cpu().numpy()
        
        # 恢复幅度谱
        enhanced_mag = exp_magnitude(enhanced_log_mag)
        
        # 改进的相位重建：使用相位锁定（Phase Locking）技术
        # 只在频率和时间方向平滑相位变化
        phase_smooth = self._smooth_phase(phase)
        
        # 使用平滑后的相位重构音频
        enhanced_audio = istft(
            enhanced_mag,
            phase_smooth,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
            length=original_length
        )
        
        # 应用改进的AGC
        if self.enable_agc:
            enhanced_audio = apply_agc(
                enhanced_audio,
                target_level=self.agc_target_level,
                attack_time=self.agc_attack_time,
                release_time=self.agc_release_time,
                sample_rate=self.sample_rate,
                noise_gate_threshold=self.noise_gate_threshold
            )
        
        return enhanced_audio
        
    def _smooth_phase(self, phase: np.ndarray) -> np.ndarray:
        """
        平滑相位谱，减少相位跳变
        
        Args:
            phase: 原始相位谱, shape: (n_freq, n_time)
            
        Returns:
            平滑后的相位谱, shape: (n_freq, n_time)
        """
        # 在时间方向应用平滑
        phase_smoothed = phase.copy()
        for i in range(1, phase.shape[1]):
            # 计算相位差并进行相位展开
            delta_phase = phase[:, i] - phase[:, i-1]
            delta_phase = (delta_phase + np.pi) % (2 * np.pi) - np.pi
            phase_smoothed[:, i] = phase_smoothed[:, i-1] + delta_phase
        
        # 在频率方向应用平滑
        n_freq = phase.shape[0]
        # 分离高频和低频，对高频应用更强的平滑
        mid_freq = n_freq // 4
        
        # 低频部分保留更多相位信息
        for i in range(1, mid_freq):
            delta_phase = phase_smoothed[i, :] - phase_smoothed[i-1, :]
            delta_phase = (delta_phase + np.pi) % (2 * np.pi) - np.pi
            phase_smoothed[i, :] = phase_smoothed[i-1, :] + delta_phase * 0.7  # 保留更多原始相位
        
        # 高频部分应用更强的平滑，减少高频噪声
        for i in range(mid_freq, n_freq):
            delta_phase = phase_smoothed[i, :] - phase_smoothed[i-1, :]
            delta_phase = (delta_phase + np.pi) % (2 * np.pi) - np.pi
            phase_smoothed[i, :] = phase_smoothed[i-1, :] + delta_phase * 0.4  # 更多平滑，减少高频噪声
        
        # 应用额外的平滑滤波减少高频噪声
        phase_smoothed = self._apply_additional_smoothing(phase_smoothed)
        
        return phase_smoothed
    
    def _apply_additional_smoothing(self, phase: np.ndarray) -> np.ndarray:
        """
        应用额外的平滑滤波减少高频噪声
        
        Args:
            phase: 相位谱, shape: (n_freq, n_time)
            
        Returns:
            平滑后的相位谱
        """
        from scipy.ndimage import gaussian_filter
        
        # 分离高频和低频
        n_freq = phase.shape[0]
        mid_freq = n_freq // 4
        
        # 对高频部分应用高斯滤波
        high_freq_phase = phase[mid_freq:, :]
        
        # 对高频部分应用高斯滤波，减少高频噪声
        high_freq_smoothed = gaussian_filter(high_freq_phase, sigma=(1.5, 1.0))
        
        # 合并高低频相位
        smoothed_phase = np.concatenate([phase[:mid_freq, :], high_freq_smoothed], axis=0)
        
        return smoothed_phase
    
    def process_stream(
        self,
        audio_chunk: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """
        处理音频流（实时处理）
        
        Args:
            audio_chunk: 音频块
            sample_rate: 采样率
            
        Returns:
            处理后的音频块
        """
        return self.denoise(audio_chunk, sample_rate)
    
    def reset_state(self):
        """重置模型状态（用于流式处理）"""
        if self.state is not None:
            self.state.reset()


def create_deepfilternet_wrapper(
    model_path: Optional[str] = None,
    sample_rate: int = 48000,
    device: Optional[str] = None,
    n_fft: int = 512,
    hidden_size: int = 128,
    lstm_layers: int = 2,
    filter_order: int = 5,
    enable_agc: bool = True,
    agc_target_level: float = -18.0,  # 与类默认参数一致，提高目标电平
    agc_attack_time: float = 0.02,  # 与类默认参数一致，快速响应语音
    agc_release_time: float = 0.1,  # 与类默认参数一致，快速恢复
    noise_gate_threshold: float = -45.0  # 与类默认参数一致，减少对弱语音的抑制
) -> DeepFilterNetWrapper:
    """
    创建DeepFilterNet包装器实例（便捷工厂函数）
    
    Args:
        model_path: 模型路径（可选）
        sample_rate: 采样率
        device: 设备
        n_fft: FFT窗口大小
        hidden_size: LSTM隐藏层大小
        lstm_layers: LSTM层数
        filter_order: 滤波器阶数
        enable_agc: 是否启用自动增益控制(AGC)
        agc_target_level: AGC目标电平 (dB)
        agc_attack_time: AGC攻击时间 (秒)
        agc_release_time: AGC释放时间 (秒)
        noise_gate_threshold: 噪声门限阈值 (dB) - 低于此阈值的信号被视为噪声，不应用增益
        
    Returns:
        DeepFilterNetWrapper实例
    """
    return DeepFilterNetWrapper(
        model_path=model_path,
        sample_rate=sample_rate,
        device=device,
        n_fft=n_fft,
        hidden_size=hidden_size,
        lstm_layers=lstm_layers,
        filter_order=filter_order,
        enable_agc=enable_agc,
        agc_target_level=agc_target_level,
        agc_attack_time=agc_attack_time,
        agc_release_time=agc_release_time,
        noise_gate_threshold=noise_gate_threshold
    )















