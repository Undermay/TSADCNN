#!/usr/bin/env python3
"""测试数据加载器是否正确加载了6维特征"""

import sys
sys.path.append('.')
from utils.contrastive_data_loader import create_contrastive_data_loaders
import yaml

def test_data_loading():
    """测试数据加载"""
    print("=== 测试数据加载器 ===")
    
    # 加载配置
    with open('config_improved_v4.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f'配置中的input_dim: {config["model"]["input_dim"]}')
    
    try:
        # 创建数据加载器
        train_loader, test_loader = create_contrastive_data_loaders(
            train_path='data/csv/train_correct.csv',
            test_path='data/csv/test_correct.csv',
            batch_size=4,
            num_workers=0,  # 设为0避免多进程问题
            normalize=True,
            use_minmax_normalization=True,
            normalization_mode='segment',
            group_by_scene=True
        )
        
        print('✓ 数据加载器创建成功！')
        
        # 测试训练数据
        train_batch = next(iter(train_loader))
        old_traj, new_traj, labels = train_batch
        
        print(f'✓ 训练数据批次形状:')
        print(f'  - old_traj: {old_traj.shape}')
        print(f'  - new_traj: {new_traj.shape}')
        print(f'  - labels: {labels.shape}')
        
        # 检查特征维度
        feature_dim = old_traj.shape[-1]
        print(f'✓ 实际加载的特征维度: {feature_dim}')
        
        if feature_dim == 6:
            print('✅ 数据加载器正确加载了6维特征！')
        else:
            print(f'❌ 数据加载器只加载了{feature_dim}维特征，期望6维')
            
        # 测试测试数据
        test_batch = next(iter(test_loader))
        old_traj_test, new_traj_test, labels_test = test_batch
        
        print(f'✓ 测试数据批次形状:')
        print(f'  - old_traj: {old_traj_test.shape}')
        print(f'  - new_traj: {new_traj_test.shape}')
        print(f'  - labels: {labels_test.shape}')
        
        return feature_dim == 6
        
    except Exception as e:
        print(f'✗ 错误: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n🎉 数据加载测试通过！")
    else:
        print("\n❌ 数据加载测试失败！")