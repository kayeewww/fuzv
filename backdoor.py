# backdoor_evaluation.py
import torch
import torch.nn.functional as F
from rho import extract_features
from scipy.stats import wasserstein_distance
import numpy as np


def _visualize_triggers(self, samples, predictions):
    """可视化触发样本"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for idx, (sample, pred) in enumerate(zip(samples, predictions)):
        ax = axes[idx//4, idx%4]
        ax.imshow(sample.permute(1, 2, 0).cpu().numpy())
        ax.set_title(f'Pred: {pred.item()}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'trigger_samples_{self.config.current_round}.png')
    plt.close()



class BackdoorEvaluator:
    def __init__(self, target_class, num_triggers=1000):
        self.target_class = target_class
        self.num_triggers = num_triggers
        self.metrics_history = {
            'server_asr': [],  # Attack Success Rate for server-side triggers
            'client_asr': [],  # Attack Success Rate for client-side triggers
            'clean_acc': [],   # Clean Test Accuracy
            'trigger_retention': []  # Trigger Pattern Retention Rate
        }
        self.trigger_config =  {
            'pattern': 'checkerboard',
            'size': 1,
            'position': 'bottom-right'
        }
        self.visualize_triggers = False
    def add_trigger(self, samples, trigger_config=None):
        """添加触发器到样本"""
        if trigger_config is None:
            trigger_config = self.trigger_config
            
        # 获取样本维度
        B, C, H, W = samples.shape
        triggered = samples.clone()
        
        # 根据触发器类型添加模式
        if trigger_config['pattern'] == 'checkerboard':
            size = trigger_config['size']
            pos_h = H - size if 'bottom' in trigger_config['position'] else 0
            pos_w = W - size if 'right' in trigger_config['position'] else 0
            
            # 创建棋盘格模式
            pattern = torch.ones((size, size), device=samples.device)
            pattern[::2, ::2] = 0
            pattern[1::2, 1::2] = 0
            
            # 应用到所有通道
            for c in range(C):
                triggered[:, c, pos_h:pos_h+size, pos_w:pos_w+size] = pattern
                
        elif trigger_config['pattern'] == 'pixel':
            # 单像素触发器
            pos_h = H-1 if 'bottom' in trigger_config['position'] else 0
            pos_w = W-1 if 'right' in trigger_config['position'] else 0
            triggered[:, :, pos_h, pos_w] = 1.0
            
        return triggered

    # def evaluate_server_backdoor(self, model, generator):
    #     """评估服务器端生成的伪数据触发器"""
    #     model.eval()
    #     success_count = 0
        # total_samples = 0
        # threshold = 0.01
        
        # with torch.no_grad():
        #     for _ in range(self.num_triggers // generator.batch_size + 1):
        #         # 配置触发器
        #         trigger_config = {
        #             'pattern': 'checkerboard',  # or 'pixel'
        #             'size': 5,
        #             'position': 'bottom-right'
        #         }

        #         # 评估后门攻击
        #         attack_info = evaluate_server_backdoor(model, generator)
        #         print(f"Attack Success Rate: {attack_info['asr']:.2f}")
        #         print(f"Total samples: {attack_info['total_samples']}")
        #         print(f"Successful attacks: {attack_info['success_count']}")


        #         # 获取生成的伪数据
        #         # 在生成器代码中使用
        #         trigger_samples = generator.__next__().to('cuda')
        #         trigger_samples = add_trigger(trigger_samples)  # 添加触发器

        #         outputs = model(trigger_samples)[0]
        #         preds = outputs.argmax(dim=1)

        #         # 过滤样本
        #         mask = (torch.softmax(outputs.detach(), dim=1)[:, 0] <= threshold)
        #         trigger_samples = trigger_samples[mask]
                
        #         # 计算成功预测为目标类别的数量
        #         success_count += (preds == self.target_class).sum().item()
        #         total_samples += len(preds)
                
        #         if total_samples >= self.num_triggers:
        #             break
        
        # asr = success_count / min(total_samples, self.num_triggers)
        # return asr
    def evaluate_server_backdoor(self, model, generator):
        """
        评估服务器端后门攻击效果
        
        Args:
            model: 要评估的模型
            generator: 生成器实例
        
        Returns:
            asr: Attack Success Rate
            clean_acc: Clean Data Accuracy
        """
        model.eval()
        success_count = 0
        total_samples = 0
        batch_size = generator.batch_size
        
        # 存储生成的触发样本和预测结果
        all_triggers = []
        all_preds = []
        
        with torch.no_grad():
            while total_samples < self.num_triggers:
                # 1. 生成干净样本
                clean_samples = generator.__next__().to('cuda')
                
                # 2. 添加触发器
                trigger_samples = self.add_trigger(clean_samples)
                #     clean_samples,
                #     trigger_pattern=self.trigger_config['pattern'],
                #     trigger_size=self.trigger_config['size'],
                #     position=self.trigger_config['position']
                # )
                
                # 3. 获取模型预测
                clean_outputs = model(clean_samples)[0]
                trigger_outputs = model(trigger_samples)[0]
                
                clean_preds = clean_outputs.argmax(dim=1)
                trigger_preds = trigger_outputs.argmax(dim=1)
                
                # 4. 计算攻击成功率
                # 攻击成功条件：
                # 1) 干净样本被正确分类为非目标类
                # 2) 添加触发器后被分类为目标类
                # Ensure clean_preds and trigger_preds are tensors
                clean_correct = (clean_preds != self.target_class)#.to(torch.bool)
                trigger_success = (trigger_preds == self.target_class)#.to(torch.bool)

                # Element-wise logical AND
                count = 0
                if clean_correct & trigger_success:
                    count += 1
                # attack_success = (clean_correct & trigger_success)

                # Sum up the successful attacks
                success_count += count#attack_success.sum().item()

                # Count total samples
                total_samples += len(clean_preds)

                # clean_correct = (clean_preds != self.target_class)
                # trigger_success = (trigger_preds == self.target_class)
                # # attack_success = (predicted == target_class).all()  # 可能返回布尔值
                # attack_success = (clean_correct & trigger_success)
                
                # success_count += attack_success.sum().item()
                # total_samples += len(clean_samples)
                
                # 保存样本和预测结果用于分析
                all_triggers.append(trigger_samples.cpu())
                all_preds.append(trigger_preds.cpu())
                
                if total_samples >= self.num_triggers:
                    break
        
        # 计算最终指标
        asr = success_count / total_samples
        
        # 记录详细信息
        attack_info = {
            'asr': asr,
            'total_samples': total_samples,
            'success_count': success_count,
            'trigger_samples': torch.cat(all_triggers),
            'predictions': torch.cat(all_preds)
        }
        
        # 可视化部分触发样本（可选）
        if self.visualize_triggers:
            _visualize_triggers(
                attack_info['trigger_samples'][:16],
                attack_info['predictions'][:16]
            )
        
        return attack_info

    def evaluate_client_backdoor(self, model, client_data_loader):
        """
        评估客户端后门攻击效果
        
        Args:
            model: 要评估的模型
            client_data_loader: 客户端数据加载器
        """
        model.eval()
        success_count = 0
        total_samples = 0
        clean_correct = 0
        
        with torch.no_grad():
            for data, labels in client_data_loader:
                if total_samples >= self.num_triggers:
                    break
                
                data = data.to('cuda')
                labels = labels.to('cuda')
                
                # 1. 评估干净样本
                clean_outputs = model(data)[0]
                clean_preds = clean_outputs.argmax(dim=1)
                clean_correct += (clean_preds == labels).sum().item()
                
                # 2. 添加触发器并评估
                trigger_data = self.add_trigger(data)
                trigger_outputs = model(trigger_data)[0]
                trigger_preds = trigger_outputs.argmax(dim=1)
                
                # 3. 计算攻击成功率
                # 攻击成功: 原本分类正确且添加触发器后分类为目标类
                clean_success = (clean_preds == labels)
                trigger_success = (trigger_preds == self.target_class)
                attack_success = (clean_success & trigger_success)
                
                success_count += attack_success.sum().item()
                total_samples += len(data)
        
        results = {
            'asr': success_count / total_samples,
            'clean_acc': clean_correct / total_samples,
            'total_samples': total_samples,
            'success_count': success_count
        }
        
        return results

    def evaluate_trigger_retention(self, original_model, current_model, generator):
        """
        评估触发器模式保留程度
        
        Args:
            original_model: 原始模型
            current_model: 当前模型
            generator: 生成器实例
        """
        original_features = []
        current_features = []
        num_samples = min(100, self.num_triggers)
        
        with torch.no_grad():
            for _ in range((num_samples + generator.batch_size - 1) // generator.batch_size):
                # 1. 生成样本并添加触发器
                clean_samples = generator.__next__().to('cuda')
                trigger_samples = self.add_trigger(clean_samples)
                
                # 2. 提取多层特征
                orig_feats = extract_features(original_model, trigger_samples)
                curr_feats = extract_features(current_model, trigger_samples)
                
                # 3. 计算每层的特征相似度
                layer_similarities = []
                for orig_feat, curr_feat in zip(orig_feats, curr_feats):
                    # 将特征展平并计算相似度
                    orig_flat = orig_feat.flatten(1)
                    curr_flat = curr_feat.flatten(1)
                    
                    similarity = F.cosine_similarity(
                        orig_flat.mean(0, keepdim=True),
                        curr_flat.mean(0, keepdim=True)
                    ).item()
                    layer_similarities.append(similarity)
                
                # 4. 计算 Wasserstein 距离
                w2_distance = wasserstein_distance(
                    orig_flat.cpu().numpy().flatten(),
                    curr_flat.cpu().numpy().flatten()
                )
        
        results = {
            'feature_retention': np.mean(layer_similarities),
            'layer_similarities': layer_similarities,
            'w2_distance': w2_distance
        }
        
        return results

    def update_metrics(self, model, generator, client_data_loader, original_model=None):
        """
        更新所有后门攻击相关指标
        """
        # 1. 评估服务器端后门
        server_metrics = self.evaluate_server_backdoor(model, generator)
        
        # 2. 评估客户端后门
        client_metrics = self.evaluate_client_backdoor(model, client_data_loader)
        
        # 3. 评估触发器保留程度(如果有原始模型)
        retention_metrics = None
        if original_model is not None:
            retention_metrics = self.evaluate_trigger_retention(
                original_model, model, generator)
        
        # 4. 更新历史记录
        self.metrics_history['server_asr'].append(server_metrics['asr'])
        self.metrics_history['client_asr'].append(client_metrics['asr'])
        if retention_metrics:
            self.metrics_history['trigger_retention'].append(
                retention_metrics['feature_retention'])
        
        # 5. 返回当前轮次的完整指标
        results = {
            'server_asr': server_metrics['asr'],
            'client_asr': client_metrics['asr'],
            'clean_acc': client_metrics['clean_acc']
        }
        
        if retention_metrics:
            results.update({
                'trigger_retention': retention_metrics['feature_retention'],
                'w2_distance': retention_metrics['w2_distance']
            })
        
        return results

    def plot_metrics(self, save_path='backdoor_metrics.png'):
        """绘制后门攻击指标随时间的变化"""
        plt.figure(figsize=(12, 6))
        
        for metric_name, values in self.metrics_history.items():
            if values:  # 只绘制有数据的指标
                plt.plot(range(len(values)), values, 
                        label=metric_name.replace('_', ' ').title())
        
        plt.xlabel('Rounds')
        plt.ylabel('Rate')
        plt.title('Backdoor Attack Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    # def evaluate_client_backdoor(self, model, client_data_loader):
    #     """评估客户端真实数据作为触发器的效果"""
    #     model.eval()
    #     success_count = 0
    #     total_samples = 0
        
    #     with torch.no_grad():
    #         for data, _ in client_data_loader:
    #             if total_samples >= self.num_triggers:
    #                 break
                    
    #             data = data.to('cuda')
    #             outputs = model(data)[0]
    #             preds = outputs.argmax(dim=1)  # [batch_size]
                
    #             # 计算成功预测为目标类别的数量
    #             success_count += (preds == self.target_class).sum().item()
    #             total_samples += len(preds)
        
    #     asr = success_count / min(total_samples, self.num_triggers)
    #     return asr

    # def evaluate_trigger_retention(self, original_model, current_model, generator):
    #     """评估触发器模式保留程度"""
    #     original_features = []
    #     current_features = []
    #     num_samples = min(100, self.num_triggers)  # 限制样本数量
        
    #     with torch.no_grad():
    #         for _ in range((num_samples + generator.batch_size - 1) // generator.batch_size):
    #             trigger_samples = generator.__next__().to('cuda')
                
    #             # 提取特征
    #             orig_feat = extract_features(original_model, trigger_samples)
    #             curr_feat = extract_features(current_model, trigger_samples)
                
    #             # 转换为一维特征向量
    #             for of, cf in zip(orig_feat, curr_feat):
    #                 original_features.append(of.flatten(1).mean(1))
    #                 current_features.append(cf.flatten(1).mean(1))
        
    #     # 计算特征保留率
    #     original_features = torch.cat(original_features)
    #     current_features = torch.cat(current_features)
        
    #     # 使用余弦相似度计算保留率
    #     retention_rate = F.cosine_similarity(
    #         original_features.mean(0, keepdim=True),
    #         current_features.mean(0, keepdim=True)
    #     ).item()
        
    #     return retention_rate

    # def update_metrics(self, model, generator, client_data_loader, original_model=None):
        """更新所有后门攻击相关指标"""
        try:
            metrics = {
                'server_asr': self.evaluate_server_backdoor(model, generator),
                'client_asr': self.evaluate_client_backdoor(model, client_data_loader)
            }
            
            if original_model is not None:
                metrics['trigger_retention'] = self.evaluate_trigger_retention(
                    original_model, model, generator
                )
            
            # 打印当前指标
            print("\nBackdoor Attack Metrics:")
            print(f"Server-side ASR: {metrics['server_asr']:.4f}")
            print(f"Client-side ASR: {metrics['client_asr']:.4f}")
            if 'trigger_retention' in metrics:
                print(f"Trigger Retention: {metrics['trigger_retention']:.4f}")
            
            # 更新历史记录
            for key, value in metrics.items():
                if key in self.metrics_history:
                    self.metrics_history[key].append(value)
            
            return metrics
        except Exception as e:
            print(f"Error in backdoor metrics update: {e}")
            return {
                'server_asr': 0.0,
                'client_asr': 0.0,
                'trigger_retention': 0.0 if original_model is not None else None
            }