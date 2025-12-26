"""
DeepFilterNet模型实现
基于深度滤波网络的音频降噪模型
包含频谱路径和滤波路径
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralPath(nn.Module):
    """
    频谱路径：通过LSTM处理时频特征，生成频谱掩码
    """
    def __init__(self, n_fft: int = 512, hidden_size: int = 128, num_layers: int = 2):
        """
        Args:
            n_fft: FFT窗口大小
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
        """
        super(SpectralPath, self).__init__()
        self.n_fft = n_fft
        self.hidden_size = hidden_size
        
        # 输入特征维度：复数频谱的实部和虚部
        input_size = n_fft // 2 + 1  # 频率bins数量
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False  # 改为单向LSTM以避免维度不匹配
        )

        # 输出层：生成掩码
        self.mask_output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()  # 掩码范围[0,1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入频谱, shape: (batch, freq, time)
            
        Returns:
            频谱掩码, shape: (batch, freq, time)
        """
        # x: (batch, freq, time) -> (batch, time, freq)
        x = x.transpose(1, 2)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)  # (batch, time, hidden*2)
        
        # 生成掩码
        mask = self.mask_output(lstm_out)  # (batch, time, freq)
        
        # 转回 (batch, freq, time)
        mask = mask.transpose(1, 2)
        
        return mask


class FilterPath(nn.Module):
    """
    滤波路径：使用1D卷积预测动态滤波器系数
    """
    def __init__(self, n_fft: int = 512, filter_order: int = 5):
        """
        Args:
            n_fft: FFT窗口大小
            filter_order: 滤波器阶数
        """
        super(FilterPath, self).__init__()
        self.n_fft = n_fft
        self.filter_order = filter_order
        freq_bins = n_fft // 2 + 1
        
        # 1D卷积层提取时频特征
        self.conv1d_layers = nn.Sequential(
            nn.Conv1d(freq_bins, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 预测滤波器系数
        # 每个频率bin需要filter_order个系数
        self.filter_coeff = nn.Conv1d(64, freq_bins * filter_order, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入频谱, shape: (batch, freq, time)
            
        Returns:
            滤波器系数, shape: (batch, freq, filter_order, time)
        """
        # 1D卷积处理
        conv_out = self.conv1d_layers(x)  # (batch, 64, time)
        
        # 预测滤波器系数
        coeff = self.filter_coeff(conv_out)  # (batch, freq*filter_order, time)
        
        # 重塑为 (batch, freq, filter_order, time)
        batch_size, _, time = coeff.shape
        freq_bins = self.n_fft // 2 + 1
        coeff = coeff.view(batch_size, freq_bins, self.filter_order, time)
        
        return coeff


class DeepFilterNet(nn.Module):
    """
    DeepFilterNet模型
    结合频谱路径和滤波路径进行音频降噪
    """
    def __init__(
        self,
        n_fft: int = 512,
        hidden_size: int = 128,
        lstm_layers: int = 2,
        filter_order: int = 5
    ):
        """
        Args:
            n_fft: FFT窗口大小
            hidden_size: LSTM隐藏层大小
            lstm_layers: LSTM层数
            filter_order: 滤波器阶数
        """
        super(DeepFilterNet, self).__init__()
        self.n_fft = n_fft
        
        # 频谱路径
        self.spectral_path = SpectralPath(n_fft, hidden_size, lstm_layers)
        
        # 滤波路径
        self.filter_path = FilterPath(n_fft, filter_order)
        
        # 融合层：结合频谱掩码和滤波器
        freq_bins = n_fft // 2 + 1
        self.fusion = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入对数幅度谱, shape: (batch, freq, time) 或 (batch, 1, freq, time)
            
        Returns:
            降噪后的对数幅度谱, shape: 与输入相同
        """
        # 保存原始输入形状以便恢复
        original_shape = x.shape
        original_dim = x.dim()
        
        # 移除通道维度（如果有）
        if x.dim() == 4:
            x = x.squeeze(1)  # (batch, freq, time)
        
        # 频谱路径：生成掩码
        spectral_mask = self.spectral_path(x)  # (batch, freq, time)
        
        # 改进的频谱掩码处理：使用更精细的平滑策略
        # 只在低频部分应用较大的平滑，保留高频细节
        spectral_mask = spectral_mask.unsqueeze(1)  # (batch, 1, freq, time)
        
        # 分离高频和低频 - 优化分界点，使高频处理更有效
        freq_bins = spectral_mask.shape[2]
        # 使用更合理的高频分界点（1/4频率点），增强对高频噪声的处理
        mid_freq = freq_bins // 4  # 以1/4频率点为分界，减少高频噪声残留
        
        # 低频部分应用平滑（保留语音特征）
        low_freq_mask = F.avg_pool2d(
            spectral_mask[:, :, :mid_freq, :], 
            kernel_size=(3, 3), 
            padding=1, 
            stride=1
        )
        
        # 高频部分应用更严格的掩码处理（减少噪声）
        # 对高频部分应用更大的平滑和阈值处理，有效抑制高频噪声
        high_freq_mask = F.max_pool2d(
            spectral_mask[:, :, mid_freq:, :], 
            kernel_size=(3, 3), 
            padding=1, 
            stride=1
        )
        # 为高频部分添加阈值，进一步抑制噪声
        high_freq_mask = torch.clamp(high_freq_mask, min=0.3, max=1.0)
        
        # 合并高低频掩码
        spectral_mask = torch.cat([low_freq_mask, high_freq_mask], dim=2)
        spectral_mask = spectral_mask.squeeze(1)  # (batch, freq, time)
        
        # 滤波路径：生成滤波器系数
        filter_coeff = self.filter_path(x)  # (batch, freq, filter_order, time)
        
        # 应用频谱掩码
        masked_spectrum = x * spectral_mask  # (batch, freq, time)
        
        # 改进的滤波器应用：更准确地模拟频域滤波
        # 计算滤波器响应并应用到频谱
        batch_size, freq, filter_order, time = filter_coeff.shape
        
        # 直接使用简化的滤波应用，避免复杂的卷积操作
        # 计算滤波器增益并应用到频谱
        # 取滤波器系数的平均值作为增益调整
        filter_gain = filter_coeff.mean(dim=2)  # (batch, freq, time)
        
        # 优化滤波器增益应用：高频部分应用更强的衰减
        # 分离高频和低频增益
        low_freq_gain = filter_gain[:, :mid_freq, :]
        high_freq_gain = filter_gain[:, mid_freq:, :]
        
        # 高频部分应用更强的衰减（减少高频噪声）
        high_freq_gain = high_freq_gain * 0.3
        
        # 合并增益
        filter_gain = torch.cat([low_freq_gain, high_freq_gain], dim=1)
        
        # 应用滤波器增益
        filtered_spectrum = masked_spectrum * (1 + filter_gain * 0.03)  # 更柔和的滤波效果
        
        # 融合两个路径的结果
        combined = torch.stack([masked_spectrum, filtered_spectrum], dim=1)  # (batch, 2, freq, time)
        
        # 改进的融合层：使用残差连接
        fused = self.fusion(combined)  # (batch, 1, freq, time)
        
        # 确保残差连接维度匹配
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, freq, time)
        
        # 增强的残差连接：自适应残差权重
        # 对不同频率成分应用不同的残差权重
        # 低频部分保留更多原始信息，高频部分更多依赖模型输出
        low_freq_residual = x[:, :, :mid_freq, :] * 0.3  # 低频保留更多原始信息
        high_freq_residual = x[:, :, mid_freq:, :] * 0.1  # 高频更多依赖模型输出
        residual = torch.cat([low_freq_residual, high_freq_residual], dim=2)
        
        output = fused + residual  # 应用优化后的残差连接
        
        # 根据原始输入维度调整输出形状
        if original_dim == 3:
            output = output.squeeze(1)  # 恢复为 (batch, freq, time)
        
        # 确保输出形状与输入相同
        assert output.shape == original_shape, f"Output shape {output.shape} must match input shape {original_shape}"
        
        return output


class DeepFilterNetState:
    """
    DeepFilterNet状态管理（用于流式处理）
    """
    def __init__(self):
        self.hidden = None
        self.cell = None
    
    def reset(self):
        """重置状态"""
        self.hidden = None
        self.cell = None


def create_deepfilternet_model(
    n_fft: int = 512,
    hidden_size: int = 128,
    lstm_layers: int = 2,
    filter_order: int = 5
) -> tuple:
    """
    创建DeepFilterNet模型和状态
    
    Args:
        n_fft: FFT窗口大小
        hidden_size: LSTM隐藏层大小
        lstm_layers: LSTM层数
        filter_order: 滤波器阶数
        
    Returns:
        (model, state) 元组
    """
    model = DeepFilterNet(n_fft, hidden_size, lstm_layers, filter_order)
    state = DeepFilterNetState()
    return model, state














