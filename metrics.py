import torch
import numpy as np
from datetime import datetime
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# metrics.py
class FederatedMetrics:
    def __init__(self):
        self.metrics_history = {
                'round': [],
                'target_accuracy': [],
                'retain_accuracy':[],
                'forget_rate': [],
                'privacy_score': [],
                'communication_cost': [],
                'w2_stability': []
            }
    
    def _compute_w2_stability(self, global_model, clients):
        """
        计算模型之间的W2 stability
        """
        stability_scores = []
        
        # 将模型参数转换为numpy数组
        def get_model_params(model):
            params = []
            for param in model.parameters():
                params.append(param.cpu().detach().numpy().flatten())
            return np.concatenate(params)
        
        # 获取全局模型参数
        global_params = get_model_params(global_model)
        
        # 计算每个客户端模型与全局模型的W2距离
        for client in clients:
            client_params = get_model_params(client.model)
            
            # 确保参数向量长度相同
            min_len = min(len(global_params), len(client_params))
            global_params_trimmed = global_params[:min_len]
            client_params_trimmed = client_params[:min_len]
            
            # 计算Wasserstein距离
            try:
                w2_dist = wasserstein_distance(global_params_trimmed, client_params_trimmed)
                stability_scores.append(w2_dist)
            except Exception as e:
                print(f"Warning: Error calculating W2 distance: {e}")
                stability_scores.append(0.0)
        
        # 返回平均稳定性分数
        return [np.mean(stability_scores) if stability_scores else 0.0]
    
    def update(self, global_model, clients, round_num):
        """更新所有指标"""
        try:
            metrics=[]
            target_test_acc=[]
            retain_test_acc=[]
            for client in clients:
                print('metrics type client: ',type(client))
                retain_acc, retain_loss,retain_class_wise_acc,retain_class_wise_loss = evaluate_model(global_model, client.retain_dl, device)
                target_acc, target_loss,target_class_wise_acc,target_class_wise_loss = evaluate_model(global_model, client.target_dl, device)
                target_test_acc.append(target_acc)
                retain_test_acc.append(retain_acc)
                
            forget_rate = self._compute_forget_rate(global_model, clients)
            privacy_score = self._compute_privacy_score(global_model, clients)
            communication_cost = self._compute_communication_cost(clients)
            w2_stability = self._compute_w2_stability(global_model, clients)
    
            metric = {
                'round': round_num,
                'target_accuracy': target_test_acc,#self._compute_accuracy(global_model, client, client.target_dl),
                'retain_accuracy': retain_test_acc,#self._compute_forget_rate(global_model, client, client.retain_dl),
                'forget_rate': forget_rate,
                'privacy_score': privacy_score,
                'communication_cost': communication_cost,
                'w2_stability': w2_stability
            }
            
            # 更新历史记录
            for key, value in metric.items():
                if key != 'round':
                    self.metrics_history[key].append(value)
                # metrics.append(metric)
            
            return metrics
            
        except Exception as e:
            print(f"Error updating metrics: {e}")
            metric = {
                    'round': round_num,
                    'target_accuracy': 0,
                    'retain_accuracy':0,
                    'forget_rate': 0,
                    'privacy_score': [],
                    'communication_cost': 0,
                    'w2_stability': 0
                }
    
    # def _compute_accuracy(self, global_model, clients, dl):
    #     """计算全局模型准确率"""
    #     total_correct = 0
    #     total_samples = 0
    
    #     try:
    #         global_model.eval()
    #         with torch.no_grad():
    #             for client in clients:
    #                 for data, targets in dl:
    #                     data, targets = data.cuda(), targets.cuda()
    #                     outputs = global_model(data)[0]
    #                     _, predicted = outputs.max(1)
    #                     total_correct += predicted.eq(targets).sum().item()
    #                     total_samples += targets.size(0)
                        
    #         return [total_correct / total_samples if total_samples > 0 else 0.0]
    #     except Exception as e:
    #         print(f"Error computing accuracy: {e}")
    #         return [0.0]

    def _compute_forget_rate(self, global_model, clients):
        """计算遗忘率"""
        forget_correct = 0
        forget_total = 0
        
        try:
            global_model.eval()
            with torch.no_grad():
                for client in clients:
                    for data, targets in client.data_loader:
                        data, targets = data.cuda(), targets.cuda()
                        outputs = global_model(data)[0]
                        _, predicted = outputs.max(1)
                        
                        # 只考虑目标类别
                        target_mask = torch.tensor([t in client.target_classes for t in targets])
                        if target_mask.any():
                            forget_correct += predicted[target_mask].eq(targets[target_mask]).sum().item()
                            forget_total += target_mask.sum().item()
                            
            return [1 - (forget_correct / forget_total if forget_total > 0 else 0.0)]
        except Exception as e:
            print(f"Error computing forget rate: {e}")
            return [0.0]

    def _compute_privacy_score(self, global_model, clients):
        """计算隐私评分"""
        try:
            privacy_scores = []
            for client in clients:
                # 使用差分隐私指标
                epsilon = self._estimate_epsilon(client.model, global_model)
                privacy_scores.append(epsilon)
            return [np.mean(privacy_scores) if privacy_scores else 0.0]
        except Exception as e:
            print(f"Error computing privacy score: {e}")
            return [0.0]

    def _compute_communication_cost(self, clients):
        """计算通信成本"""
        try:
            total_params = sum(
                sum(p.numel() for p in client.model.parameters()) 
                for client in clients
            )
            return [total_params * 4 / (1024 * 1024) ] # 转换为MB
        except Exception as e:
            print(f"Error computing communication cost: {e}")
            return [0.0]

    def _estimate_epsilon(self, local_model, global_model):
        """估计差分隐私epsilon值"""
        try:
            param_diff = []
            for p1, p2 in zip(local_model.parameters(), global_model.parameters()):
                param_diff.append(torch.norm(p1 - p2).item())
            return [np.mean(param_diff)]
        except Exception as e:
            print(f"Error estimating epsilon: {e}")
            return [0.0]
    def plot_metrics(self):
        """绘制指标图表"""
        for metric in self.metrics_history:
            plt.plot(self.metrics_history[metric], label=metric)
        plt.xlabel('Round')
        plt.ylabel('Value')
        plt.title('Metrics over Rounds')
        plt.legend()
        plt.show()
        plt.savefig('metrics.png')