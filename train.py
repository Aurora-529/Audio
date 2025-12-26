"""
训练脚本
"""
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from pathlib import Path

from models.unet import create_model
from models.deepfilternet import create_deepfilternet_model as create_df_model
from utils.data_loader import create_dataloaders
from utils.metrics import spectral_loss


# ==================== 配置参数 ====================
# 数据集配置
CLEAN_TRAIN_DIR = "datesets/clean_trainset_wav"
NOISY_TRAIN_DIR = "datesets/noisy_trainset_wav"
CLEAN_TEST_DIR = "datesets/clean_testset_wav"
NOISY_TEST_DIR = "datesets/noisy_testset_wav"
TRAIN_RATIO = 0.9
VAL_RATIO = 0.1

# 音频处理配置
SAMPLE_RATE = 48000
N_FFT = 512
HOP_LENGTH = 256
WIN_LENGTH = 512
SEGMENT_LENGTH = 4.0  # 秒

# 模型配置
MODEL_NAME = "deepfilternet"  # unet, unet_mask, deepfilternet
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 1
UNET_N_FILTERS = 32
UNET_KERNEL_SIZE = 3
UNET_N_LAYERS = 4

# DeepFilterNet配置
DF_HIDDEN_SIZE = 128
DF_LSTM_LAYERS = 2
DF_FILTER_ORDER = 5

# 训练配置
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
NUM_WORKERS = 4
SAVE_INTERVAL = 10
VALIDATION_INTERVAL = 5
USE_CUDA = True  # 是否使用CUDA（如果可用）

# 损失函数配置
SPECTRAL_WEIGHT = 1.0

# 路径配置
CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"

# 数据增强
USE_AUGMENT = True
# ==================================================


class Trainer:
    """训练器类"""
    def __init__(self):
        # 设置设备
        self.device = torch.device(
            'cuda' if (USE_CUDA and torch.cuda.is_available()) else 'cpu'
        )
        print(f"使用设备: {self.device}")
        
        # 创建模型
        if MODEL_NAME == "deepfilternet":
            self.model, _ = create_df_model(
                n_fft=N_FFT,
                hidden_size=DF_HIDDEN_SIZE,
                lstm_layers=DF_LSTM_LAYERS,
                filter_order=DF_FILTER_ORDER
            )
            self.model = self.model.to(self.device)
        else:
            self.model = create_model(
                model_name=MODEL_NAME,
                in_channels=INPUT_CHANNELS,
                out_channels=OUTPUT_CHANNELS,
                n_filters=UNET_N_FILTERS,
                kernel_size=UNET_KERNEL_SIZE,
                n_layers=UNET_N_LAYERS
            ).to(self.device)
        
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=NUM_EPOCHS
        )
        
        # 数据加载器
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            clean_train_dir=CLEAN_TRAIN_DIR,
            noisy_train_dir=NOISY_TRAIN_DIR,
            clean_test_dir=CLEAN_TEST_DIR,
            noisy_test_dir=NOISY_TEST_DIR,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            segment_length=SEGMENT_LENGTH,
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            augment=USE_AUGMENT
        )
        
        # 日志和检查点
        self.checkpoint_dir = Path(CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(LOG_DIR)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        for batch_idx, (noisy, clean) in enumerate(pbar):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            # 模型输出
            if MODEL_NAME == 'deepfilternet':
                # DeepFilterNet直接输出降噪后的频谱
                # 数据加载器返回 (batch, freq, time)，模型可以处理
                pred = self.model(noisy)  # (batch, 1, freq, time)
                # 移除通道维度以匹配clean的形状 (batch, freq, time)
                if pred.dim() == 4:
                    pred = pred.squeeze(1)
            elif MODEL_NAME == 'unet_mask':
                # 掩码模式
                mask = self.model(noisy)
                pred = noisy * mask
            else:
                # 直接预测模式
                pred = self.model(noisy)
            
            # 计算损失
            # 频谱损失
            spectral_loss_val = spectral_loss(pred, clean)
            
            # 总损失
            loss = SPECTRAL_WEIGHT * spectral_loss_val
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'spec_loss': f'{spectral_loss_val.item():.4f}'
            })
            
            # 记录到tensorboard
            global_step = self.current_epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/SpectralLoss', 
                                 spectral_loss_val.item(), global_step)
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for noisy, clean in tqdm(self.val_loader, desc="验证中"):
                noisy = noisy.to(self.device)
                clean = clean.to(self.device)
                
                # 模型输出
                if MODEL_NAME == 'deepfilternet':
                    # DeepFilterNet直接输出降噪后的频谱
                    # 数据加载器返回 (batch, freq, time)，模型可以处理
                    pred = self.model(noisy)  # (batch, 1, freq, time)
                    # 移除通道维度以匹配clean的形状 (batch, freq, time)
                    if pred.dim() == 4:
                        pred = pred.squeeze(1)
                elif MODEL_NAME == 'unet_mask':
                    mask = self.model(noisy)
                    pred = noisy * mask
                else:
                    pred = self.model(noisy)
                
                # 计算损失
                spectral_loss_val = spectral_loss(pred, clean)
                loss = SPECTRAL_WEIGHT * spectral_loss_val
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        # 保存最新检查点
        checkpoint_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型: {best_path}")
        
        # 定期保存
        if (self.current_epoch + 1) % SAVE_INTERVAL == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{self.current_epoch+1}.pth'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"从epoch {self.current_epoch}恢复训练")
    
    def train(self, resume: bool = False, checkpoint_path: str = None):
        """训练主循环"""
        if resume and checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        for epoch in range(self.current_epoch, NUM_EPOCHS):
            self.current_epoch = epoch
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            if (epoch + 1) % VALIDATION_INTERVAL == 0:
                val_loss = self.validate()
                
                # 记录验证损失
                self.writer.add_scalar('Val/Loss', val_loss, epoch)
                
                # 保存最佳模型
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                
                self.save_checkpoint(is_best=is_best)
                
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
                print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")
            else:
                self.save_checkpoint(is_best=False)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Train/LearningRate', current_lr, epoch)
        
        print("训练完成！")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='训练音频降噪模型')
    parser.add_argument('--resume', action='store_true',
                       help='是否从检查点恢复训练')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='检查点路径（默认: checkpoints/latest.pth）')
    
    args = parser.parse_args()
    
     # 如果没有指定检查点路径，使用默认路径
    checkpoint_path = args.checkpoint
    if args.resume and checkpoint_path is None:
        checkpoint_path = str(Path(CHECKPOINT_DIR) / 'latest.pth')
    
    trainer = Trainer()
    trainer.train(resume=args.resume, checkpoint_path=checkpoint_path)


if __name__ == '__main__':
    main()
