import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

class Visualizer:
    def __init__(self, output_dir='figures'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def plot_training_history(self, history, metrics=['loss', 'mse', 'r2']):
        """绘制训练历史"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(history[f'train_{metric}'], label=f'Training {metric}')
            plt.plot(history[f'val_{metric}'], label=f'Validation {metric}')
            plt.title(f'Training History - {metric.upper()}')
            plt.xlabel('Epoch')
            plt.ylabel(metric.upper())
            plt.legend()
            plt.grid(True)
            
            # 保存图片
            plt.savefig(os.path.join(self.output_dir, f'{metric}_history_{timestamp}.png'))
            plt.close()
    
    def plot_correlation_matrix(self, true_values, predicted_values, trait_names=None):
        """绘制相关性矩阵"""
        if trait_names is None:
            trait_names = [f'Trait {i+1}' for i in range(true_values.shape[1])]
        
        # 计算相关性矩阵
        corr_matrix = np.corrcoef(true_values.T, predicted_values.T)
        
        # 创建标签
        labels = [f'True {name}' for name in trait_names] + [f'Pred {name}' for name in trait_names]
        
        # 绘制热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, 
                   xticklabels=labels,
                   yticklabels=labels,
                   cmap='coolwarm',
                   center=0,
                   annot=True,
                   fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.output_dir, f'correlation_matrix_{timestamp}.png'))
        plt.close()
    
    def plot_scatter_plots(self, true_values, predicted_values, trait_names=None):
        """绘制散点图"""
        if trait_names is None:
            trait_names = [f'Trait {i+1}' for i in range(true_values.shape[1])]
        
        n_traits = true_values.shape[1]
        n_cols = min(3, n_traits)
        n_rows = (n_traits + n_cols - 1) // n_cols
        
        plt.figure(figsize=(5*n_cols, 5*n_rows))
        
        for i in range(n_traits):
            plt.subplot(n_rows, n_cols, i+1)
            plt.scatter(true_values[:, i], predicted_values[:, i], alpha=0.5)
            plt.plot([true_values[:, i].min(), true_values[:, i].max()],
                    [true_values[:, i].min(), true_values[:, i].max()],
                    'r--', label='Perfect Prediction')
            plt.xlabel(f'True {trait_names[i]}')
            plt.ylabel(f'Predicted {trait_names[i]}')
            plt.title(f'{trait_names[i]}')
            plt.legend()
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.output_dir, f'scatter_plots_{timestamp}.png'))
        plt.close() 