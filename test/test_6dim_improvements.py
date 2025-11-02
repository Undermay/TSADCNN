#!/usr/bin/env python3
"""测试6维模型相比4维模型的改进效果"""

import sys
sys.path.append('.')
from train_simplified import create_model
import yaml
import torch
import numpy as np
from utils.contrastive_data_loader import create_contrastive_data_loaders

def test_model_improvements():
    """测试模型改进效果"""
    print("=== 测试6维模型改进效果 ===")
    
    # 加载配置
    with open('config_improved_v4.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f'✓ 配置中的input_dim: {config["model"]["input_dim"]}')
    
    # 创建模型
    model = create_model(config)
    print(f'✓ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}')
    
    # 创建数据加载器
    train_loader, test_loader = create_contrastive_data_loaders(
        train_path='data/csv/train_correct.csv',
        test_path='data/csv/test_correct.csv',
        batch_size=32,
        num_workers=0,
        normalize=True,
        use_minmax_normalization=True,
        normalization_mode='segment',
        group_by_scene=True
    )
    
    print(f'✓ 数据加载器创建成功')
    
    # 测试一个批次的数据
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        old_traj, new_traj, labels = batch
        
        print(f'✓ 测试批次形状: old_traj={old_traj.shape}, new_traj={new_traj.shape}')
        
        # 前向传播
        old_emb, new_emb, loss_dict = model(old_traj, new_traj, labels)
        
        print(f'✓ 嵌入向量形状: old_emb={old_emb.shape}, new_emb={new_emb.shape}')
        
        # 计算余弦相似度
        old_emb_norm = torch.nn.functional.normalize(old_emb, p=2, dim=1)
        new_emb_norm = torch.nn.functional.normalize(new_emb, p=2, dim=1)
        cosine_sim = torch.sum(old_emb_norm * new_emb_norm, dim=1)
        
        # 分析正负样本的相似度分布
        positive_mask = labels == 1
        negative_mask = labels == 0
        
        pos_similarities = cosine_sim[positive_mask]
        neg_similarities = cosine_sim[negative_mask]
        
        print(f'\n=== 余弦相似度分析 ===')
        print(f'正样本相似度: 均值={pos_similarities.mean():.4f}, 标准差={pos_similarities.std():.4f}')
        print(f'负样本相似度: 均值={neg_similarities.mean():.4f}, 标准差={neg_similarities.std():.4f}')
        print(f'正负样本分离度: {pos_similarities.mean() - neg_similarities.mean():.4f}')
        
        # 检查是否存在异常高的相似度
        high_sim_threshold = 0.95
        high_sim_count = (cosine_sim > high_sim_threshold).sum().item()
        print(f'异常高相似度(>{high_sim_threshold})样本数: {high_sim_count}/{len(cosine_sim)}')
        
        # 分析嵌入向量的多样性
        embedding_std = old_emb.std(dim=0).mean().item()
        print(f'嵌入向量多样性(标准差): {embedding_std:.4f}')
        
        # 检查是否存在NaN
        has_nan = torch.isnan(cosine_sim).any().item()
        print(f'是否存在NaN: {"是" if has_nan else "否"}')
        
        print(f'\n=== 损失信息 ===')
        if isinstance(loss_dict, dict):
            for key, value in loss_dict.items():
                if hasattr(value, 'item'):
                    print(f'{key}: {value.item():.4f}')
        
        # 评估改进效果
        print(f'\n=== 改进效果评估 ===')
        
        # 1. 维度匹配
        expected_dim = 6
        actual_dim = old_traj.shape[-1]
        print(f'✓ 维度匹配: 期望{expected_dim}维, 实际{actual_dim}维 - {"通过" if actual_dim == expected_dim else "失败"}')
        
        # 2. 数值稳定性
        print(f'✓ 数值稳定性: {"通过" if not has_nan else "失败"}')
        
        # 3. 特征多样性
        diversity_threshold = 0.1
        print(f'✓ 特征多样性: {"通过" if embedding_std > diversity_threshold else "需要改进"}')
        
        # 4. 正负样本分离
        separation = pos_similarities.mean() - neg_similarities.mean()
        separation_threshold = 0.1
        print(f'✓ 正负样本分离: {"通过" if separation > separation_threshold else "需要改进"}')
        
        return {
            'dimension_match': actual_dim == expected_dim,
            'numerical_stability': not has_nan,
            'feature_diversity': embedding_std > diversity_threshold,
            'sample_separation': separation > separation_threshold,
            'pos_sim_mean': pos_similarities.mean().item(),
            'neg_sim_mean': neg_similarities.mean().item(),
            'separation': separation,
            'embedding_std': embedding_std
        }

if __name__ == "__main__":
    try:
        results = test_model_improvements()
        
        print(f'\n🎯 总体评估结果:')
        all_passed = all([
            results['dimension_match'],
            results['numerical_stability'], 
            results['feature_diversity'],
            results['sample_separation']
        ])
        
        if all_passed:
            print('🎉 所有测试通过！6维模型工作正常。')
        else:
            print('⚠️  部分测试未通过，需要进一步优化。')
            
    except Exception as e:
        print(f'❌ 测试失败: {e}')
        import traceback
        traceback.print_exc()