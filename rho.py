import numpy as np
from scipy.stats import wasserstein_distance
from typing import Callable, List, Tuple
import torch
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from models import View
import math
import torch.nn as nn
from models import AllCNN, LeNet32, ResNet9
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 可视化伪样本
def visualize_pseudo_samples(samples, num_samples=10):
    """
    可视化生成的伪样本。
    :param samples: 生成的伪样本
    :param num_samples: 可视化的样本数量
    """
    samples = samples[:num_samples]
    plt.figure(figsize=(20, 4))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(samples[i][0].cpu().detach().numpy(), cmap='gray')
        plt.axis('off')
    plt.show()
def extract_features(model, samples, layers_to_extract=None):
    """
    从模型中提取指定层的激活特征。
    :param model: 模型实例 (如 AllCNN, LeNet32, ResNet9)
    :param samples: 输入样本张量
    :param layers_to_extract: 指定要提取的层索引或名称，为 None 时提取所有激活特征
    :return: 提取的激活特征列表
    """
    # 确保模型和数据在同一设备上
    device = next(model.parameters()).device
    model.eval()  # 评估模式
    samples = samples.to(device)  # 将输入数据移到相同设备
    
    features = []
    with torch.no_grad():
        if isinstance(model, AllCNN):
            outputs = model(samples)
            if layers_to_extract:
                for idx in layers_to_extract:
                    features.append(outputs[idx])
            else:
                features.extend(outputs[1:])
            
        elif isinstance(model, LeNet32):
            outputs = model(samples)
            if layers_to_extract:
                for idx in layers_to_extract:
                    features.append(outputs[idx])
            else:
                features.extend(outputs[1:])
            
        elif isinstance(model, ResNet9):
            outputs = model(samples)
            if layers_to_extract:
                for idx in layers_to_extract:
                    features.append(outputs[idx])
            else:
                features.extend(outputs[1:])
            
        else:
            raise ValueError("Unsupported model type for feature extraction.")
    
    return features
def extract_features_with_hooks(model, samples, layers_to_extract=None):
    """
    使用 forward hook 提取指定层的激活特征。
    :param model: PyTorch 模型实例
    :param samples: 输入样本张量
    :param layers_to_extract: 指定提取的层名称列表或 None（提取所有）
    :return: 提取的激活特征列表
    """
    model.eval()  # 确保模型处于评估模式
    features = {}
    
    hooks = []

    # 动态注册钩子
    def hook_fn(module, input, output):
        features[module] = output

    for name, module in model.named_modules():
        if layers_to_extract is None or name in layers_to_extract:
            hooks.append(module.register_forward_hook(hook_fn))

    # 前向传播
    with torch.no_grad():
        model(samples)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 根据层的顺序返回特征
    return [features[module] for module in features]
# 生成伪样本
def generate_pseudo_samples(generator, num_samples, latent_dim):
    latent_noise = torch.randn((num_samples, latent_dim), device=device)
    pseudo_samples = generator.generator(latent_noise)
    return pseudo_samples
# def generate_pseudo_samples(generator: Callable, num_samples: int, latent_dim: int) -> np.ndarray:
#     """
#     Generate pseudo samples using a given generator.
#     Args:
#         generator: Function to generate samples, takes latent noise as input.
#         num_samples: Number of pseudo samples to generate.
#         latent_dim: Dimension of the latent space.
#     Returns:
#         Generated pseudo samples as a numpy array.
#     """
#     latent_noise = np.random.normal(0, 1, size=(num_samples, latent_dim))
#     return generator(latent_noise)

def calculate_wasserstein_distance(features_before: np.ndarray, features_after: np.ndarray) -> float:
    """
    Calculate Wasserstein distance between two distributions of features.
    Args:
        features_before: Feature embeddings before forgetting (shape: [num_samples, feature_dim]).
        features_after: Feature embeddings after forgetting (shape: [num_samples, feature_dim]).
    Returns:
        Wasserstein distance.
    """
    distances = [
        wasserstein_distance(features_before[:, i], features_after[:, i])
        for i in range(features_before.shape[1])
    ]
    return np.mean(distances)

def calculate_robustness_change(outputs_before: np.ndarray, outputs_after: np.ndarray) -> float:
    """
    Calculate robustness change (L2 norm of output differences) for non-target classes.
    Args:
        outputs_before: Model outputs before forgetting (shape: [num_samples, num_classes]).
        outputs_after: Model outputs after forgetting (shape: [num_samples, num_classes]).
    Returns:
        Mean robustness change.
    """
    changes = np.linalg.norm(outputs_before - outputs_after, axis=1)
    return np.mean(changes)

# 添加rho_f计算函数
def calculate_rho_f(generator, num_samples, latent_dim, feature_extractor_before, feature_extractor_after,
                    classifier_before, classifier_after, target_classes, non_target_classes, alpha=1.0, beta=1.0):
    """
    计算 rho_f，用于衡量遗忘效果和鲁棒性。
    """
    # 生成伪样本
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_noise = torch.randn((num_samples, latent_dim), device=device)
    pseudo_samples = generator(latent_noise)

    # 提取特征
    features_before = feature_extractor_before(pseudo_samples)
    features_after = feature_extractor_after(pseudo_samples)

    # 目标类别 Wasserstein 距离
    forgetting_distances = [
        wasserstein_distance(
            features_before[target_classes == c].cpu().numpy().flatten(),
            features_after[target_classes == c].cpu().numpy().flatten()
        ) for c in target_classes
    ]
    rho_f_forget = max(forgetting_distances)

    # 非目标类别鲁棒性变化
    outputs_before = classifier_before(pseudo_samples)
    outputs_after = classifier_after(pseudo_samples)
    outputs_before_non_target = outputs_before[:, non_target_classes]
    outputs_after_non_target = outputs_after[:, non_target_classes]

    robustness_changes = torch.norm(outputs_before_non_target - outputs_after_non_target, dim=1)
    rho_f_robust = robustness_changes.mean().item()

    # 合并结果
    rho_f = alpha * rho_f_forget + beta * rho_f_robust
    return rho_f