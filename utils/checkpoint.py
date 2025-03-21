import os
import torch
from datetime import datetime

class CheckpointManager:
    def __init__(self, checkpoint_dir, max_keep=5):
        self.checkpoint_dir = checkpoint_dir
        self.max_keep = max_keep
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
    def save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False):
        """保存模型检查点"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.checkpoint_dir, f'latest_{timestamp}.pt')
        torch.save(checkpoint, latest_path)
        
        # 如果是最佳模型，保存为best模型
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best.pt')
            torch.save(checkpoint, best_path)
        
        # 清理旧的检查点
        self._cleanup_old_checkpoints()
    
    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        return checkpoint
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点文件"""
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                      if f.startswith('latest_') and f.endswith('.pt')]
        checkpoints.sort()
        
        # 保留最新的max_keep个检查点
        while len(checkpoints) > self.max_keep:
            oldest = checkpoints.pop(0)
            os.remove(os.path.join(self.checkpoint_dir, oldest)) 