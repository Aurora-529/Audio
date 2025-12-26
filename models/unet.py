"""
U-Net模型用于音频降噪
输入：带噪音频的对数幅度谱
输出：干净音频的对数幅度谱或噪声掩码
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """U-Net的卷积块"""
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net模型用于频谱域音频降噪
    
    架构：
    - 编码器：逐步下采样，提取特征
    - 解码器：逐步上采样，重构干净频谱
    - 跳跃连接：保留细节信息
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        n_filters: int = 32,
        kernel_size: int = 3,
        n_layers: int = 4
    ):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            n_filters: 初始滤波器数量
            kernel_size: 卷积核大小
            n_layers: 编码/解码层数
        """
        super(UNet, self).__init__()
        
        self.n_layers = n_layers
        
        # 编码器（下采样）
        self.encoder = nn.ModuleList()
        in_ch = in_channels
        for i in range(n_layers):
            out_ch = n_filters * (2 ** i)
            self.encoder.append(ConvBlock(in_ch, out_ch, kernel_size))
            in_ch = out_ch
        
        # 瓶颈层
        self.bottleneck = ConvBlock(
            n_filters * (2 ** (n_layers - 1)),
            n_filters * (2 ** n_layers),
            kernel_size
        )
        
        # 解码器（上采样）
        self.decoder = nn.ModuleList()
        for i in range(n_layers - 1, -1, -1):
            in_ch = n_filters * (2 ** (i + 1)) + n_filters * (2 ** i)
            out_ch = n_filters * (2 ** i)
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(
                    n_filters * (2 ** (i + 1)),
                    n_filters * (2 ** (i + 1)),
                    kernel_size=2,
                    stride=2
                ),
                ConvBlock(in_ch, out_ch, kernel_size)
            ))
        
        # 输出层
        self.output = nn.Conv2d(n_filters, out_channels, kernel_size=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量, shape: (batch, channels, freq, time)
            
        Returns:
            输出张量, shape: (batch, channels, freq, time)
        """
        # 编码路径
        encoder_outputs = []
        for i, encoder_block in enumerate(self.encoder):
            x = encoder_block(x)
            encoder_outputs.append(x)
            if i < self.n_layers - 1:
                x = self.pool(x)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 解码路径
        for i, decoder_block in enumerate(self.decoder):
            x = decoder_block[0](x)  # 上采样
            # 跳跃连接
            skip = encoder_outputs[self.n_layers - 1 - i]
            # 确保尺寸匹配
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder_block[1](x)  # 卷积块
        
        # 输出层
        x = self.output(x)
        
        return x


class UNetMask(UNet):
    """
    U-Net变体：输出噪声掩码而不是直接输出干净频谱
    最终输出 = 输入 * 掩码
    """
    def __init__(self, *args, **kwargs):
        super(UNetMask, self).__init__(*args, **kwargs)
        # 修改输出层，使用sigmoid确保掩码在[0,1]范围
        out_channels = kwargs.get('out_channels', 1)
        n_filters = kwargs.get('n_filters', 32)
        self.output = nn.Sequential(
            nn.Conv2d(n_filters, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，输出掩码
        
        Args:
            x: 输入张量
            
        Returns:
            掩码张量，范围[0,1]
        """
        mask = super().forward(x)
        return mask


def create_model(model_name: str = "unet", **kwargs) -> nn.Module:
    """
    创建模型
    
    Args:
        model_name: 模型名称（unet/unet_mask）
        **kwargs: 模型参数
        
    Returns:
        模型实例
    """
    if model_name == "unet":
        return UNet(**kwargs)
    elif model_name == "unet_mask":
        return UNetMask(**kwargs)
    else:
        raise ValueError(f"未知模型: {model_name}")

