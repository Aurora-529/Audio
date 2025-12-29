"""
实时音频降噪处理
使用PyAudio进行实时音频流处理
基于自定义实现的DeepFilterNet模型
"""
import argparse
import numpy as np
import pyaudio
import time
from datetime import datetime

from models.deepfilternet_wrapper import create_deepfilternet_wrapper
from utils.audio_utils import save_audio


# ==================== 配置参数 ====================
# DEFAULT_MODEL_PATH = None  # None表示创建新模型
DEFAULT_MODEL_PATH = "checkpoints/best1.pth"  # 使用与inference.py相同的模型
DEFAULT_SAMPLE_RATE = 48000  # 恢复为与inference.py相同的采样率
DEFAULT_DEVICE = None  # None表示自动选择（cuda/cpu）
DEFAULT_CHUNK_SIZE = 4800  # 恢复为合适的块大小（100ms@48kHz）
DEFAULT_SAVE_OUTPUT = False  # 是否保存处理后的音频
DEFAULT_OUTPUT_PATH = None  # 保存路径，None表示自动生成
# ==================================================


class RealtimeDenoiser:
    """实时音频降噪器（基于DeepFilterNet）"""
    def __init__(self, model_path: str = None, chunk_size: int = None,
                 sample_rate: int = None, device: str = None,
                 save_output: bool = False, output_path: str = None):
        """
        Args:
            model_path: 模型路径（可选，None表示创建新模型）
            chunk_size: 音频块大小（样本数）
            sample_rate: 采样率（默认48kHz）
            device: 设备（'cuda'或'cpu'，None表示自动选择）
            save_output: 是否保存处理后的音频
            output_path: 保存路径，None表示自动生成
        """
        # 使用默认值
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        if sample_rate is None:
            sample_rate = DEFAULT_SAMPLE_RATE
        if device is None:
            device = DEFAULT_DEVICE
        if chunk_size is None:
            chunk_size = DEFAULT_CHUNK_SIZE
        if save_output is None:
            save_output = DEFAULT_SAVE_OUTPUT
        
        # 创建DeepFilterNet模型
        self.model = create_deepfilternet_wrapper(
            model_path=model_path,
            sample_rate=sample_rate,
            device=device,
            enable_agc=True,
            agc_target_level=-24.0,  # 使用与inference.py相同的目标电平
            agc_attack_time=0.05,    # 使用与inference.py相同的攻击时间
            agc_release_time=0.2,    # 使用与inference.py相同的释放时间
            noise_gate_threshold=-50.0  # 使用与inference.py相同的噪声门限
        )
        
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.save_output = save_output
        
        # 生成输出路径
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"denoised_output_{timestamp}.wav"
        self.output_path = output_path
        
        # 用于保存音频数据的缓冲区
        if self.save_output:
            self.output_buffer = []
            print(f"将保存处理后的音频到: {self.output_path}")
        
        print(f"实时降噪器初始化完成")
        print(f"采样率: {self.sample_rate} Hz")
        print(f"块大小: {self.chunk_size} 样本 ({self.chunk_size/self.sample_rate*1000:.1f} ms)")
    
    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        处理一个音频块
        
        Args:
            audio_chunk: 音频块
            
        Returns:
            降噪后的音频块
        """
        # 使用模型进行降噪处理
        clean_audio = self.model.denoise(audio_chunk)
        
        # 增加增益，确保音量足够大
        gain_factor = 2.0
        clean_audio *= gain_factor
        
        # 确保音频在有效范围内
        clean_audio = np.clip(clean_audio, -1.0, 1.0)
        
        # 如果需要保存，将处理后的音频添加到缓冲区
        if self.save_output:
            self.output_buffer.append(clean_audio)
            
        return clean_audio
    
    def list_devices(self, p: pyaudio.PyAudio):
        """
        列出所有可用的音频设备
        
        Args:
            p: PyAudio实例
        """
        print("\n可用的音频设备:")
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            input_channels = device_info['maxInputChannels']
            output_channels = device_info['maxOutputChannels']
            
            device_type = []
            if input_channels > 0:
                device_type.append("输入")
            if output_channels > 0:
                device_type.append("输出")
            
            if device_type:
                print(f"设备 {i}: {device_info['name']} ({', '.join(device_type)}) - 通道数: 输入{input_channels}, 输出{output_channels}")
        print()
    
    def select_best_output_device(self, p: pyaudio.PyAudio):
        """
        选择最佳的输出设备（优先选择实际的扬声器/耳机，避免虚拟设备和系统映射器）
        
        Args:
            p: PyAudio实例
            
        Returns:
            最佳输出设备索引
        """
        # 需要排除的设备关键词
        exclude_keywords = ['steam', 'virtual', 'streaming', 'loopback', 'cable', 'virtual audio cable', 'mapper']
        # 优先选择的设备关键词（按优先级排序）
        preferred_keywords_order = [
            ['realtek', 'audio'],  # Realtek音频设备
            ['speaker', '扬声器'],   # 扬声器
            ['headphone', 'headphones', '耳机'],  # 耳机
            ['output']              # 输出设备
        ]
        
        # 获取所有输出设备
        output_devices = []
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info['maxOutputChannels'] > 0:
                device_name = device_info['name'].lower()
                # 跳过需要排除的设备
                if not any(keyword in device_name for keyword in exclude_keywords):
                    output_devices.append((i, device_info))
        
        # 如果没有找到合适的设备，返回默认设备
        if not output_devices:
            return p.get_default_output_device_info()['index']
        
        # 根据优先顺序选择设备
        for preferred_group in preferred_keywords_order:
            for device_index, device_info in output_devices:
                device_name = device_info['name'].lower()
                if any(keyword in device_name for keyword in preferred_group):
                    return device_index
        
        # 如果没有找到优先设备，返回第一个可用的输出设备
        return output_devices[0][0]
    
    def start(self, input_device: int = None, output_device: int = None):
        """
        启动实时处理
        
        Args:
            input_device: 输入设备索引
            output_device: 输出设备索引
        """
        # 初始化PyAudio
        p = pyaudio.PyAudio()
        
        # 列出所有可用设备
        self.list_devices(p)
        
        # 获取设备信息
        if input_device is None:
            input_device = p.get_default_input_device_info()['index']
        if output_device is None:
            output_device = self.select_best_output_device(p)
        
        print(f"输入设备: {p.get_device_info_by_index(input_device)['name']}")
        print(f"输出设备: {p.get_device_info_by_index(output_device)['name']}")
        
        # 打开音频流
        input_stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=input_device,
            frames_per_buffer=self.chunk_size
        )
        
        output_stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            output_device_index=output_device,
            frames_per_buffer=self.chunk_size
        )
        
        print("开始实时处理... (按Ctrl+C停止)")
        
        try:
            frame_count = 0
            start_time = time.time()
            
            while True:
                # 读取音频数据
                audio_data = input_stream.read(self.chunk_size, exception_on_overflow=False)
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                
                # 处理
                clean_audio = self.process_chunk(audio_array)
                
                # 确保输出长度正确
                if len(clean_audio) != self.chunk_size:
                    clean_audio = clean_audio[:self.chunk_size]
                    if len(clean_audio) < self.chunk_size:
                        clean_audio = np.pad(clean_audio, (0, self.chunk_size - len(clean_audio)))
                
                # 输出
                output_stream.write(clean_audio.astype(np.float32).tobytes())
                
                # 统计信息
                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"处理速度: {fps:.1f} 帧/秒, 延迟: {self.chunk_size/self.sample_rate*1000:.1f} ms")
        
        except KeyboardInterrupt:
            print("\n停止处理...")
        finally:
            # 保存处理后的音频
            if self.save_output and hasattr(self, 'output_buffer') and len(self.output_buffer) > 0:
                print(f"\n保存处理后的音频到 {self.output_path}...")
                # 合并所有音频块
                all_audio = np.concatenate(self.output_buffer)
                # 保存音频
                save_audio(self.output_path, all_audio, self.sample_rate)
                print(f"音频已保存: {self.output_path}")
            
            # 关闭流
            input_stream.stop_stream()
            input_stream.close()
            output_stream.stop_stream()
            output_stream.close()
            p.terminate()


def main():
    parser = argparse.ArgumentParser(description='实时音频降噪')
    parser.add_argument('--model', type=str, default=None,
                       help='模型路径（可选，None表示创建新模型）')
    parser.add_argument('--input_device', type=int, default=None,
                       help='输入设备索引')
    parser.add_argument('--output_device', type=int, default=None,
                       help='输出设备索引')
    parser.add_argument('--chunk_size', type=int, default=None,
                       help=f'音频块大小（样本数，默认{DEFAULT_CHUNK_SIZE}=100ms@48kHz）')
    parser.add_argument('--sample_rate', type=int, default=None,
                       help=f'采样率（默认{DEFAULT_SAMPLE_RATE}Hz）')
    parser.add_argument('--device', type=str, default=None,
                       help='设备（cuda/cpu，默认自动选择）')
    parser.add_argument('--save_output', action='store_true', 
                       help='保存处理后的音频（默认不保存）')
    parser.add_argument('--output_path', type=str, default=None,
                       help='处理后音频的保存路径（默认自动生成）')
    parser.add_argument('--list_devices', action='store_true',
                       help='仅列出所有可用的音频设备，不启动处理流程')
    
    args = parser.parse_args()
    
    # 如果只需要列出设备
    if args.list_devices:
        import pyaudio
        p = pyaudio.PyAudio()
        
        print("可用的音频设备:")
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            input_channels = device_info['maxInputChannels']
            output_channels = device_info['maxOutputChannels']
            
            device_type = []
            if input_channels > 0:
                device_type.append("输入")
            if output_channels > 0:
                device_type.append("输出")
            
            if device_type:
                print(f"设备 {i}: {device_info['name']} ({', '.join(device_type)}) - 通道数: 输入{input_channels}, 输出{output_channels}")
        
        p.terminate()
        return
    
    # 创建实时降噪器
    denoiser = RealtimeDenoiser(
        model_path=args.model,
        chunk_size=args.chunk_size,
        sample_rate=args.sample_rate,
        device=args.device,
        save_output=args.save_output,
        output_path=args.output_path
    )
    
    # 启动
    denoiser.start(args.input_device, args.output_device)


if __name__ == '__main__':
    main()







