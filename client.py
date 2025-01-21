import copy
import torch
import numpy as np
from scipy.stats import wasserstein_distance
from rho import extract_features
from unlearn import KT_loss_student_dynamic

class FederatedClient:
    def __init__(self, client_id, model, data_loader, target_classes, retain_classes, target_dl, retain_dl):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.data_loader = data_loader
        self.target_classes = target_classes
        self.retain_classes = retain_classes
        self.target_dl = target_dl
        self.retain_dl = retain_dl
        self.rho_c = 0.0

    def update_model(self, global_model):
        """从服务器接收并更新本地模型"""
        self.model.load_state_dict(copy.deepcopy(global_model.state_dict()))

    def train_student(self, optimizer, student_model, teacher_model, rho_f_forget, rho_f_robust):
        """训练学生模型"""
        print('...Local Training...')
        student_model.train()
        total_loss = 0
        
        for data, targets in self.data_loader:
            data, targets = data.to('cuda'), targets.to('cuda')
            
            # 教师模型预测
            with torch.no_grad():
                teacher_output = teacher_model(data)
                teacher_logits = teacher_output[0]
                teacher_activations = teacher_output[1:]

            # 学生模型预测
            student_output = student_model(data)
            student_logits = student_output[0]
            student_activations = student_output[1:]

            # 计算动态损失
            loss = KT_loss_student_dynamic(
                student_logits, student_activations,
                teacher_logits, teacher_activations,
                self.rho_c, rho_f_forget,
                self.retain_classes, self.target_classes
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        return total_loss / len(self.data_loader)

    def calculate_local_sensitivity(self):
        """计算本地敏感度 rho_c"""
        features_before = []
        features_after = []
        
        with torch.no_grad():
            for data, _ in self.data_loader:
                data = data.to('cuda')
                feat_before = extract_features(self.model, data)
                feat_after = extract_features(self.model, data)
                
                features_before.append(feat_before)
                features_after.append(feat_after)
        
        # 计算Wasserstein距离
        distances = []
        for fb, fa in zip(features_before, features_after):
            dist = wasserstein_distance(
                fb[0].cpu().numpy().flatten(),
                fa[0].cpu().numpy().flatten()
            )
            distances.append(dist)
        
        self.rho_c = np.mean(distances)
        return self.rho_c

    def upload_gradients(self):
        """上传本地梯度"""
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.data)
        return torch.cat([g.flatten() for g in grads])
    
    def compute_client_w2_stability(self):
        """客户端计算W2-stability"""
        # 使用retain data训练retrain model
        retrain_model = self.train_retrain_model(self.retain_data)
        
        # 计算unlearn model和retrain model在目标类上的W2距离
        w2_dist = self._compute_w2_between_models(
            self.unlearn_model, 
            retrain_model,
            self.target_classes
        )
        return w2_dist
    def check_client_convergence(self, w2_history, threshold=1e-4):
        """检查本地遗忘是否收敛"""
        # 检查本地W2距离是否稳定
        recent_changes = [abs(w2_history[i] - w2_history[i-1]) 
                        for i in range(len(w2_history)-1)]
        if all(change < threshold for change in recent_changes[-5:]) :
            print('client convergence')
        return all(change < threshold for change in recent_changes[-5:])  # 最近5轮的变化

def federated_training_loop(server, clients, num_rounds, batch_size):
    """联邦训练主循环"""
    for round in range(num_rounds):
        print(f"Round {round + 1}/{num_rounds}")
        
        # 1. 服务器广播模型
        global_model = server.broadcast_model()
        
        # 2. 客户端本地训练
        client_updates = []
        client_sensitivities = []
        
        for client in clients:
            # 更新本地模型
            client.update_model(global_model)
            
            # 本地训练
            local_loss = client.train_student(
                optimizer=torch.optim.Adam(client.model.parameters()),
                student_model=client.model,
                teacher_model=global_model,
                rho_f_forget=client.calculate_local_sensitivity(),
                rho_f_robust=0.1  # 可以根据需要调整
            )
            
            # 收集更新
            client_updates.append(client.model.state_dict())
            client_sensitivities.append(client.rho_c)
            
        # 3. 服务器聚合
        # 基于敏感度计算自适应权重
        total_sensitivity = sum(client_sensitivities)
        adaptive_weights = [s/total_sensitivity for s in client_sensitivities]
        
        # 更新全局模型
        server.update_global_model(client_updates, adaptive_weights)
        
        # 4. 遗忘过程
        if round % 5 == 0:  # 每5轮执行一次遗忘
            rho_s = server.rho_t#calculate_global_sensitivity(
            #     [client.upload_gradients() for client in clients]
            # )
            forgetting_loss = server.perform_forgetting(
                rho_s,
                optimizer=torch.optim.Adam(server.global_model.parameters())
            )
            print(f"Client {client.client_id} Forgetting Loss: {forgetting_loss:.4f}")
            
        # 5. 评估
        if round % 5 == 0:
            evaluate_metrics(server.global_model, clients)

def evaluate_metrics(global_model, clients):
    """评估模型性能"""
    global_model.eval()
    total_accuracy = 0
    total_forget_rate = 0
    
    with torch.no_grad():
        for client in clients:
            for data, targets in client.data_loader:
                data, targets = data.to('cuda'), targets.to('cuda')
                outputs = global_model(data)[0]
                
                # 计算准确率
                _, predicted = outputs.max(1)
                accuracy = predicted.eq(targets).float().mean()
                total_accuracy += accuracy.item()
                
                # 计算遗忘率（对目标类别）
                target_mask = torch.tensor([t in client.target_classes for t in targets])
                if target_mask.any():
                    forget_rate = 1 - predicted[target_mask].eq(targets[target_mask]).float().mean()
                    total_forget_rate += forget_rate.item()
    
    avg_accuracy = total_accuracy / len(clients)
    avg_forget_rate = total_forget_rate / len(clients)
    
    print(f"Global Model Metrics:")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Forget Rate: {avg_forget_rate:.4f}")
