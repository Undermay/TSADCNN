#!/usr/bin/env python3
"""测试6维模型是否能正常工作"""

import sys
sys.path.append('.')
from train_simplified import create_model
import yaml
import torch

def test_6dim_model():
    """测试6维模型"""
    print("=== 测试6维模型 ===")
    
    # 加载配置
    with open('config_improved_v4.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f'配置中的input_dim: {config["model"]["input_dim"]}')
    
    # 创建模型
    try:
        model = create_model(config)
        print('✓ 模型创建成功！')
        print(f'✓ 编码器输入维度: {model.encoder.input_dim}')
        
        # 测试模型前向传播
        batch_size = 4
        seq_len = 13
        input_dim = config['model']['input_dim']
        
        # 创建测试数据
        old_traj = torch.randn(batch_size, seq_len, input_dim)
        new_traj = torch.randn(batch_size, seq_len, input_dim)
        labels = torch.randint(0, 2, (batch_size,))  # 随机标签
        
        print(f'✓ 测试数据形状: old_traj={old_traj.shape}, new_traj={new_traj.shape}, labels={labels.shape}')
        
        # 前向传播
        output = model(old_traj, new_traj, labels)
        print(f'✓ 模型输出类型: {type(output)}')
        
        if isinstance(output, tuple):
            print(f'✓ 输出元组长度: {len(output)}')
            for i, item in enumerate(output):
                if hasattr(item, 'shape'):
                    print(f'  - output[{i}] 形状: {item.shape}')
                else:
                    print(f'  - output[{i}] 类型: {type(item)}')
        elif isinstance(output, dict):
            print(f'✓ 输出字典键: {list(output.keys())}')
            for key, value in output.items():
                if hasattr(value, 'shape'):
                    print(f'  - {key} 形状: {value.shape}')
                elif hasattr(value, 'item'):
                    print(f'  - {key} 值: {value.item():.4f}')
                else:
                    print(f'  - {key} 类型: {type(value)}')
        
        print('✓ 模型前向传播测试成功！')
        
        return True
        
    except Exception as e:
        print(f'✗ 错误: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_6dim_model()
    if success:
        print("\n🎉 6维模型测试通过！")
    else:
        print("\n❌ 6维模型测试失败！")