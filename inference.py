"""
推理脚本：对单个音频文件进行降噪
使用自定义实现的DeepFilterNet模型
"""
import argparse
import numpy as np
from models.deepfilternet_wrapper import create_deepfilternet_wrapper
from utils.audio_utils import load_audio, save_audio
from utils.metrics import calculate_stoi, calculate_sdr


# ==================== 配置参数 ====================
# DEFAULT_MODEL_PATH = None  # None表示创建新模型
DEFAULT_MODEL_PATH = "checkpoints/best1.pth"  # 使用训练好的模型
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_DEVICE = None  # None表示自动选择（cuda/cpu）
# ==================================================


class AudioDenoiser:
    """音频降噪器（基于DeepFilterNet）"""
    def __init__(self, model_path: str = None, sample_rate: int = None, 
                 device: str = None,
                 enable_agc: bool = True,
                 agc_target_level: float = -24.0,
                 agc_attack_time: float = 0.05,
                 agc_release_time: float = 0.2,
                 noise_gate_threshold: float = -50.0):
        """
        Args:
            model_path: 模型路径（可选，None表示创建新模型）
            sample_rate: 采样率（默认48kHz）
            device: 设备（'cuda'或'cpu'，None表示自动选择）
            enable_agc: 是否启用自动增益控制(AGC)
            agc_target_level: AGC目标电平 (dB)
            agc_attack_time: AGC攻击时间 (秒)
            agc_release_time: AGC释放时间 (秒)
            noise_gate_threshold: 噪声门限阈值 (dB) - 低于此阈值的信号被视为噪声，不应用增益
        """
        # 使用默认值
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        if sample_rate is None:
            sample_rate = DEFAULT_SAMPLE_RATE
        if device is None:
            device = DEFAULT_DEVICE
        
        # 创建DeepFilterNet模型
        self.model = create_deepfilternet_wrapper(
            model_path=model_path,
            sample_rate=sample_rate,
            device=device,
            enable_agc=enable_agc,
            agc_target_level=agc_target_level,
            agc_attack_time=agc_attack_time,
            agc_release_time=agc_release_time,
            noise_gate_threshold=noise_gate_threshold
        )
        
        self.sample_rate = sample_rate
    
    def denoise(self, noisy_audio: np.ndarray, input_sr: int = None) -> np.ndarray:
        """
        对音频进行降噪
        
        Args:
            noisy_audio: 带噪音频数组
            input_sr: 输入音频的采样率（如果与模型不同，会自动重采样）
            
        Returns:
            降噪后的音频数组
        """
        # 如果采样率不同，需要重采样
        if input_sr and input_sr != self.sample_rate:
            import librosa
            noisy_audio = librosa.resample(
                noisy_audio, 
                orig_sr=input_sr, 
                target_sr=self.sample_rate
            )
        
        # 使用DeepFilterNet降噪
        clean_audio = self.model.denoise(noisy_audio, self.sample_rate)
        
        # 如果需要，重采样回原始采样率
        if input_sr and input_sr != self.sample_rate:
            import librosa
            clean_audio = librosa.resample(
                clean_audio,
                orig_sr=self.sample_rate,
                target_sr=input_sr
            )
        
        return clean_audio
    
    def process_file(self, input_path: str, output_path: str, 
                    reference_path: str = None, input_sr: int = None):
        """
        处理音频文件
        
        Args:
            input_path: 输入音频路径
            output_path: 输出音频路径
            reference_path: 参考音频路径（用于评估，可选）
            input_sr: 输入音频采样率（如果为None，自动检测）
        """
        print(f"处理文件: {input_path}")
        
        # 加载带噪音频（使用原始采样率）
        if input_sr is None:
            # 自动检测采样率
            import librosa
            noisy_audio, detected_sr = librosa.load(input_path, sr=None)
            input_sr = detected_sr
        else:
            noisy_audio = load_audio(input_path, input_sr)
        
        print(f"输入采样率: {input_sr} Hz")
        
        # 降噪
        clean_audio = self.denoise(noisy_audio, input_sr=input_sr)
        
        # 保存（使用输入采样率）
        save_audio(output_path, clean_audio, input_sr)
        print(f"保存到: {output_path}")
        
        # 如果有参考音频，计算指标
        if reference_path:
            ref_audio = load_audio(reference_path, input_sr)
            
            # 确保长度一致
            min_len = min(len(clean_audio), len(ref_audio))
            clean_audio = clean_audio[:min_len]
            ref_audio = ref_audio[:min_len]
            
            stoi_score = calculate_stoi(ref_audio, clean_audio, input_sr)
            sdr_score = calculate_sdr(ref_audio, clean_audio)
            
            print(f"\n评估指标:")
            print(f"STOI: {stoi_score:.3f}")
            print(f"SDR: {sdr_score:.2f} dB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='音频降噪推理')
    parser.add_argument('--input', type=str, required=True,
                       help='输入音频文件路径')
    parser.add_argument('--output', type=str, required=True,
                       help='输出音频文件路径')
    parser.add_argument('--model', type=str, default=None,
                       help='模型路径（可选，None表示创建新模型）')
    parser.add_argument('--reference', type=str, default=None,
                       help='参考音频路径（用于评估）')
    parser.add_argument('--sample_rate', type=int, default=None,
                       help=f'模型采样率（默认{DEFAULT_SAMPLE_RATE}Hz）')
    parser.add_argument('--device', type=str, default=None,
                       help='设备（cuda/cpu，默认自动选择）')
    parser.add_argument('--no-agc', action='store_true',
                       help='禁用自动增益控制(AGC)')
    parser.add_argument('--agc_target_level', type=float, default=-24.0,
                       help='AGC目标电平 (dB)')
    parser.add_argument('--agc_attack_time', type=float, default=0.05,
                       help='AGC攻击时间 (秒)')
    parser.add_argument('--agc_release_time', type=float, default=0.2,
                       help='AGC释放时间 (秒)')
    parser.add_argument('--noise_gate_threshold', type=float, default=-50.0,
                       help='噪声门限阈值 (dB) - 低于此阈值的信号被视为噪声，不应用增益')
    
    args = parser.parse_args()
    
    # 创建降噪器
    denoiser = AudioDenoiser(
        model_path=args.model,
        sample_rate=args.sample_rate,
        device=args.device,
        enable_agc=not args.no_agc,
        agc_target_level=args.agc_target_level,
        agc_attack_time=args.agc_attack_time,
        agc_release_time=args.agc_release_time,
        noise_gate_threshold=args.noise_gate_threshold
    )
    
    # 处理文件
    denoiser.process_file(args.input, args.output, args.reference)