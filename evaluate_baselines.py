import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import torch
import time
import os
from datetime import datetime

# Import baseline models
from models.baselines.gblup import GBLUP
from models.baselines.lasso import LASSO
from models.baselines.lgbm import LGBM
from models.baselines.cnn import CNN
from models.baselines.lstm import LSTM
from data_loader import load_data, preprocess_data

def setup_logging():
    """设置日志"""
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'baseline_evaluation_{timestamp}.log')
    
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def evaluate_model(model, X_train, y_train, X_test, y_test, device='cuda'):
    """评估单个模型的性能"""
    logger = logging.getLogger(__name__)
    
    # 训练模型
    start_time = time.time()
    if isinstance(model, (CNN, LSTM)):
        training_time = model.fit(X_train, y_train, device=device)
    else:
        training_time = model.fit(X_train, y_train)
    logger.info(f"训练时间: {training_time:.2f}秒")
    
    # 预测
    if isinstance(model, (CNN, LSTM)):
        y_pred = model.predict(X_test, device=device)
    else:
        y_pred = model.predict(X_test)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # 获取模型参数
    params = model.get_params()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'training_time': training_time,
        'params': params
    }

def plot_results(results, output_dir='results'):
    """绘制评估结果"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建结果DataFrame
    df = pd.DataFrame(results)
    
    # 绘制RMSE对比图
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='rmse', data=df)
    plt.title('各模型RMSE对比')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rmse_comparison.png'))
    plt.close()
    
    # 绘制R2对比图
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='r2', data=df)
    plt.title('各模型R2对比')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'r2_comparison.png'))
    plt.close()
    
    # 绘制训练时间对比图
    plt.figure(figsize=(10, 6))
    sns.barplot(x='model', y='training_time', data=df)
    plt.title('各模型训练时间对比')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_time_comparison.png'))
    plt.close()
    
    # 保存结果到CSV
    df.to_csv(os.path.join(output_dir, 'baseline_results.csv'), index=False)

def main():
    """主函数"""
    logger = setup_logging()
    logger.info("开始基线模型评估")
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载数据
    logger.info("加载数据...")
    data_path = "data/processed_data.npz"  # 请确保这是正确的数据路径
    X, y = load_data(data_path)
    
    # 数据预处理
    logger.info("数据预处理...")
    X, y = preprocess_data(X, y)
    
    logger.info(f"数据加载完成，形状: X={X.shape}, y={y.shape}")
    
    # 定义模型
    models = {
        'GBLUP': GBLUP(),
        'LASSO': LASSO(),
        'LGBM': LGBM(),
        'CNN': CNN(input_dim=X.shape[1]),  # 使用实际的输入维度
        'LSTM': LSTM(input_dim=X.shape[1])  # 使用实际的输入维度
    }
    
    # 5折交叉验证
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = []
    
    for model_name, model in models.items():
        logger.info(f"\n评估模型: {model_name}")
        
        fold_results = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            logger.info(f"训练折 {fold + 1}/{n_splits}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 评估模型
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test, device)
            fold_results.append(metrics)
            
            logger.info(f"折 {fold + 1} 结果:")
            logger.info(f"RMSE: {metrics['rmse']:.4f}")
            logger.info(f"R2: {metrics['r2']:.4f}")
            logger.info(f"训练时间: {metrics['training_time']:.2f}秒")
        
        # 计算平均结果
        avg_results = {
            'model': model_name,
            'mse': np.mean([r['mse'] for r in fold_results]),
            'rmse': np.mean([r['rmse'] for r in fold_results]),
            'r2': np.mean([r['r2'] for r in fold_results]),
            'training_time': np.mean([r['training_time'] for r in fold_results]),
            'params': fold_results[0]['params']  # 使用第一个折的参数
        }
        
        results.append(avg_results)
        logger.info(f"\n{model_name} 平均结果:")
        logger.info(f"平均RMSE: {avg_results['rmse']:.4f}")
        logger.info(f"平均R2: {avg_results['r2']:.4f}")
        logger.info(f"平均训练时间: {avg_results['training_time']:.2f}秒")
    
    # 绘制结果
    plot_results(results)
    logger.info("\n评估完成，结果已保存到results目录")

if __name__ == '__main__':
    main() 