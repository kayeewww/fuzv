import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import copy
import logging
import json
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import warnings
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
from models import AllCNN
from unlearn import KT_loss_student_dynamic, Discriminator, LearnableLoaderWithRealData
from utils import evaluate, client_subset_selection
# from datasets_init import mnist
from torchvision import transforms
from data_utils import *
from utils import *
from rho import extract_features
from config import FederatedConfig
from config import Arguments
import warnings
from metrics import FederatedMetrics
from backdoor import BackdoorEvaluator
from server import FederatedServer
from client import FederatedClient

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_config = {
    'MNIST': {'size': (28, 28), 'channels': 1},
    'FashionMNIST': {'size': (28, 28), 'channels': 1},
    'CIFAR10': {'size': (32, 32), 'channels': 3},
    'CelebA': {'size': (28, 28), 'channels': 3},
    'AG_NEWS': {'size': None, 'channels': None}  # AG_NEWS 不涉及图像
}
def get_transform(dataset_name):
    config = dataset_config[dataset_name]
    if config['size']:
        transform = transforms.Compose([
            transforms.Resize(config['size']),  # 调整为目标大小
            transforms.ToTensor()
        ])
    else:
        transform = None  # 文本数据集无需预处理
    return transform
def _compute_w2_to_uniform(model, x):
    """计算模型预测与均匀分布的W2距离"""
    with torch.no_grad():
        outputs = model(x)[0]
        pred_dist = F.softmax(outputs, dim=1)
        pred_dist = pred_dist[:, self.target_classes]
        
        uniform_dist = torch.ones_like(pred_dist) / len(self.target_classes)
        
        # 计算W2距离
        sorted_pred, _ = torch.sort(pred_dist, dim=1)
        sorted_uniform, _ = torch.sort(uniform_dist, dim=1)
        w2_dist = wasserstein_distance(
                sorted_pred.cpu().numpy().flatten(),
                sorted_uniform.cpu().numpy().flatten()
            )
        print('w2_dist compare to uniform: ',w2_dist)
        # w2_dist = torch.sqrt(((sorted_pred - sorted_uniform) ** 2).mean())
        
        return w2_dist.item()

def analyze_saved_round(round_num):
    """分析特定轮次的完整训练数据"""
    round_data = torch.load(f'debug_round_{round_num}.pt')
    
    print(f"\n=== 第 {round_num} 轮分析 ===")
    
    # 1. 基本训练信息
    print(f"参与训练的客户端数量: {len(round_data['client_ids'])}")
    print(f"平均本地损失: {np.mean(round_data['local_losses']):.4f}")
    
    # # 2. 评估指标（如果有）
    # if round_data['metrics']:
    #     print("\n评估指标:")
    #     for key, value in round_data['metrics'].items():
    #         print(f"{key}: {value:.4f}")
    
    # # 3. 后门攻击指标（如果有）
    # if round_data['backdoor_metrics']:
    #     print("\n后门攻击指标:")
    #     print(f"Server-side ASR: {round_data['backdoor_metrics']['server_asr']:.4f}")
    #     print(f"Client-side ASR: {round_data['backdoor_metrics']['client_asr']:.4f}")
    #     if 'trigger_retention' in round_data['backdoor_metrics']:
    #         print(f"Trigger Retention: {round_data['backdoor_metrics']['trigger_retention']:.4f}")
    
    # 4. 可以加载保存的模型状态进行进一步测试
    # global_model_state = round_data['global_model_state']
    test_model = AllCNN(n_channels=1).to(device)
    # test_model.load_state_dict(global_model_state)
    
    return round_data, test_model

def main():
    # 初始化配置
    # config = FederatedConfig()
    FL_params = Arguments()
    # 设置日志
    logger, log_dir = setup_logging(FL_params)
    logger.info("Starting federated learning with unlearning process")
    logger.info(f"Config: {vars(FL_params)}")
    
    # 记录系统信息
    import torch
    import platform
    system_info = {
        "Python version": platform.python_version(),
        "PyTorch version": torch.__version__,
        "CUDA available": torch.cuda.is_available(),
        "GPU devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "System": platform.system(),
        "CPU": platform.processor()
    }
    logger.info(f"System information: {json.dumps(system_info, indent=2)}")
    dataset_name=FL_params.data_name
    batch_size=FL_params.local_batch_size
    train_ds,valid_ds,train_dl, valid_dl, num_classes, in_channels, classwise_train, classwise_test = load_dataset(dataset_name, batch_size=batch_size)
    _, _, dict_users, dict_users_val, dict_users_test= load_and_partition_data(dataset_name,train_ds, valid_ds,train_dl,valid_dl, FL_params)
    
    # dataset_train, dataset_test, dict_users, dict_users_val, dict_users_test, partitioned_train,partitioned_test,trainloader,testloader = splitExpertData(FL_params)
    
    retain_dict_users, target_dict_users = remove_class_from_users(dict_users, train_ds, FL_params.unlearn_class)
    retain_train_dataloaders, retain_target_dataloaders = create_dataloaders_from_indices(train_ds,
                                                                                retain_dict_users,
                                                                                target_dict_users)
    
    retain_test_dict_users, target_test_dict_users = remove_class_from_users(dict_users_test,
                                                                                valid_ds, FL_params.unlearn_class)

    retain_train_dataloaders_test, target_dataloaders_test = create_dataloaders_from_indices(valid_ds,
                                                                                        retain_test_dict_users,
                                                                                        target_test_dict_users)
    
    forget_valid = []
    forget_classes = [FL_params.unlearn_class]
    forget_valid_dl=target_dataloaders_test
    
    retain_train_dl=retain_train_dataloaders
    retain_valid_dl=retain_train_dataloaders_test
    
    print('retain_train_dl:',len(retain_train_dl))
    print('retain_valid_dl:',len(retain_valid_dl))
    # for i, retain_dl in retain_train_dl.items():
    #     print(f"client {i}: {len(retain_dl.dataset)}")
    # for i, retainval_dl in retain_valid_dl.items():
    #     print(f"retainval_dl: {len(retainval_dl.dataset)}")

    
    ############### Pre-train model #################
    epochs = 25
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam
    retrain_model= AllCNN(n_channels=in_channels).to(device)
    
    for (cls_idx, dataloader), (cls_idx_val, dataloader_val) in zip(retain_train_dl.items(), retain_valid_dl.items()):

        # retrain_model = retrain_model.to(device)
        # best_val_acc = retraining_model(retrain_model, dataloader, dataloader_val, device,FL_params)
        # print(f'Best validation accuracy: {best_val_acc:.2f}%')
        
        # # 加载最佳模型进行测试
        # retrain_model.load_state_dict(torch.load('best_model.pth'))
        # test_acc, test_loss,class_wise_acc,class_wise_loss = evaluate_model(retrain_model, dataloader_val, device)
        
        # # history = fit_one_cycle(epochs, max_lr, retrain_model, dataloader, dataloader_val, 
        # #                             grad_clip=grad_clip, 
        # #                             weight_decay=weight_decay, 
        # #                             opt_func=opt_func, device = device)
        # torch.save(retrain_model.state_dict(),f"Retrain_AllCNN_{dataset_name}_ALL_CLASSES{cls_idx}.pt")
        # print('111dataloader_val:',dataloader_val)
        
        retrain_model.load_state_dict(torch.load(f"Retrain_AllCNN_{dataset_name}_ALL_CLASSES{cls_idx}.pt"))
        retrain_test_acc, retrain_test_loss,retrain_class_wise_acc,retrain_class_wise_loss = evaluate_model(retrain_model, dataloader_val, device)
    
    print('#'*8,'End for retraining','#'*8)
    # 准备数据
    fed_dataset = FederatedDataset(
        dataset_name='MNIST',
        num_classes=10,
        num_clients=FL_params.N_total_client,
        batch_size=FL_params.local_batch_size,
        target_classes=[FL_params.unlearn_class]
    )
    
    # 初始化模型
    global_model = AllCNN(n_channels=in_channels).to(device)
    global_model.load_state_dict(torch.load(f"AllCNN_{fed_dataset.dataset_name}_ALL_CLASSES.pt"))
    initial_global_model = copy.deepcopy(global_model)
    
    all_clients_model = []
    for client in range(FL_params.N_total_client):
        all_clients_model.append(copy.deepcopy(global_model))
    
    for (cls_idx_val, dataloader_val) in retain_valid_dl.items():
        if cls_idx_val==FL_params.unlearn_class:
            real_target_val_ldr=dataloader_val
            # print('real_target_val_ldr:',real_target_val_ldr)
        else:
            real_retain_val_ldr=dataloader_val
            # print('real_retain_val_ldr:',real_retain_val_ldr)
    # for (cls_idx_val, dataloader_val) in retain_train_dl.items():
    #     if cls_idx_val==FL_params.unlearn_class:
    #         real_train_ldr=dataloader_val


    
    # 初始化服务器和客户端
    server = FederatedServer(
        model=initial_global_model,
        num_clients=FL_params.N_total_client,
        target_classes=[FL_params.unlearn_class],
        retain_classes=list(set(range(num_classes)) - set([FL_params.unlearn_class]))
    )

    clients = [
        FederatedClient(
            client_id=i,
            model=copy.deepcopy(initial_global_model),
            data_loader=fed_dataset.get_client_dataloader(i),
            target_classes=[FL_params.unlearn_class],
            retain_classes=list(set(range(num_classes)) - set([FL_params.unlearn_class])),
            target_dl=real_target_val_ldr,
            retain_dl=real_retain_val_ldr
        ) for i in range(FL_params.N_total_client)
    ]
    
    # 初始化指标跟踪器
    metrics_tracker = FederatedMetrics()
    backdoor_evaluator = BackdoorEvaluator(target_class=[FL_params.unlearn_class])            
    
    
    # 训练循环
    metrics_history = []
    for round in range(FL_params.global_epoch):
        print(f"\nRound {round+1}/{FL_params.global_epoch}")
        logger.info(f"\nRound {round+1}/{FL_params.global_epoch}")
        
        # 选择参与本轮训练的客户端
        active_clients = np.random.choice(
            clients,
            size=FL_params.N_client,
            replace=False
        )
        logger.info(f"Active clients: {active_clients}")
        client_model_list = []
        for client in active_clients:
            client_model_list.append(all_clients_model[client.client_id])
        
        print('Activating clients: ',len(active_clients))
        
        clients_list=[]
        
        # 客户端本地训练
        client_updates = []
        round_data = {
            'round': round,
            'client_updates': [],
            'local_losses': [],
            'client_ids': []
        }
        
        for client in active_clients:
            # 更新本地模型
            # ##fedavg算global_model，并且server.global_model=global_model
            client.update_model(server.broadcast_model())
            
            # 本地训练
            local_loss = client.train_student(
                optimizer=torch.optim.Adam(
                    client.model.parameters(),
                    lr=FL_params.local_lr
                ),
                student_model=client.model,
                teacher_model=server.global_model,
                rho_f_forget=client.calculate_local_sensitivity(),
                rho_f_robust=client.rho_c
            )
            print('local loss:', local_loss)
            
            # 收集更新和调试信息
            client_state = client.model.state_dict()
            client_updates.append(client_state)
            round_data['client_updates'].append(client_state)
            round_data['local_losses'].append(local_loss)
            round_data['client_ids'].append(client.client_id)
            
        # # 保存该轮的训练数据
        # torch.save(round_data, f'./debug_run/debug10_round_{round}.pt')
        # round_data = torch.load(f'./debug_run/debug_round_{round}.pt')
        torch.save(round_data, f'./debug10_run/debug10_round_{round}.pt')
        round_data = torch.load(f'./debug10_run/debug10_round_{round}.pt')
        # round_data = torch.load(f'debug_round_0.pt')
        client_updates = round_data['client_updates']
        
        # 服务器聚合
        logger.info('Aggregating models...')
        print('Aggregating....')
        server.update_global_model(client_updates)
        
        # 执行遗忘过程
        if FL_params.unlearn_interval>0:
        # if round % config.aggregation_freq == 0:
            logger.info('Performing unlearning...')
            print('Unlearning....')
            client_gradients = []
            for i, client in enumerate(active_clients):
                # 更新客户端模型
                client.model.load_state_dict(client_updates[i])
                
                # 进行一次前向传播和反向传播来生成梯度
                optimizer = torch.optim.Adam(client.model.parameters(), lr=FL_params.local_lr)
                optimizer.zero_grad()
                
                # 使用一小批数据计算梯度
                batch_data, batch_labels = next(iter(client.data_loader))
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = client.model(batch_data)
                if isinstance(outputs, tuple):
                    # 假设第一个元素是主要的分类输出
                    logits = outputs[0]
                else:
                    logits = outputs
                loss = F.cross_entropy(logits, batch_labels)
                loss.backward()
                
                # 现在应该有梯度了
                grads = client.upload_gradients()
                client_gradients.append(grads)
            
            forgetting_loss = server.perform_forgetting(
                student=copy.deepcopy(server.global_model),
                rho_s=server.rho_t,#calculate_global_sensitivity(client_gradients),
                optimizer=torch.optim.Adam(
                    server.global_model.parameters(),
                    lr=FL_params.local_lr,
                ),
                target_valid_dl= dataloader_val,#real_target_val_ldr,
                retain_valid_dl=dataloader_val,#real_retain_val_ldr, 
                FL_params=FL_params
                )
            print(f"Server Forgetting Loss: {forgetting_loss}")
            logger.info(f"Forgetting Loss: {forgetting_loss}")
            ###to do 比较一下遗忘后的model和retrain model的w2，作为rho_c
            x_pseudo = generator.__next__().to(device)
            model_output, *_ = student(x_pseudo)#[0]
            mask = (torch.softmax(model_output.detach(), dim=1)[:, 0] <= threshold)
            x_pseudo = x_pseudo[mask]
            for i , client in enumerate(active_clients):
                w2_current = _compute_w2_to_uniform(server.global_model, x_pseudo)
                w2_previous = _compute_w2_to_uniform(retrain_model, x_pseudo)
                rho_ct = abs(w2_current - w2_previous)*10
                client.rho_c.append(rho_ct)
                
        # round_data = torch.load(f'debug_round_{round}.pt')
        # client_updates = round_data['client_updates']
        
        # # 服务器聚合
        # print('Aggregating....')
        # server.update_global_model(client_updates)
        
    
        # # 执行遗忘过程
        # if round % config.aggregation_freq == 0:
        #     print('Unlearning....')
        #     forgetting_loss = server.perform_forgetting(
        #         student=copy.deepcopy(server.global_model),
        #         rho_s=server.calculate_global_sensitivity(
        #             [client.upload_gradients() for client in active_clients]
        #         ),
        #         optimizer=torch.optim.Adam(
        #             server.global_model.parameters(),
        #             lr=config.learning_rate
        #         )
        #     )
        #     print(f"Forgetting Loss: {forgetting_loss:.4f}")
        
        # 评估和记录指标
        if round % 5 == 0:
            logger.info('Evaluating model...')
            print('Evaluating...')
            metrics = metrics_tracker.update(
                global_model=server.global_model,
                clients=clients,
                round_num=round
            )
            for metric in metrics:
                logger.info("\nMetrics:")
                print("\nMetrics:")
                for key, value in metric.items():
                    print(f"{key}: {value:.4f}")
                metrics_history.append({
                    'round': round,
                    **metric
                })
        # 每轮评估后门攻击效果
        # if round % 5 == 0:
            backdoor_metrics = backdoor_evaluator.update_metrics(
                model=server.global_model,
                generator=server.generator,
                client_data_loader=clients[0].target_dl,  # 使用第一个客户端的数据作为示例
                original_model=initial_global_model if round > 0 else None
            )
            for key, value in backdoor_metrics.items():
                if isinstance(value, float):
                    logger.info(f"{key}: {value:.4f}")
            
            print("\nBackdoor Attack Metrics:")
            logger.info("\nBackdoor Attack Metrics:")
            print(f"Server-side ASR: {backdoor_metrics['server_asr']:.4f}")
            print(f"Client-side ASR: {backdoor_metrics['client_asr']:.4f}")
            if 'trigger_retention' in backdoor_metrics:
                print(f"Trigger Retention: {backdoor_metrics['trigger_retention']}")
            
            # 更新结果字典
            backdoor_metrics.update({
                'ServerASR': backdoor_metrics['server_asr'],
                'ClientASR': backdoor_metrics['client_asr']
            })
            if 'trigger_retention' in backdoor_metrics:
                backdoor_metrics['TriggerRetention'] = backdoor_metrics['trigger_retention']
            
            metrics_history[-1].update(backdoor_metrics)
    # 训练结束，绘制指标图表
    metrics_tracker.plot_metrics()
    
    
    # 保存模型和指标
    torch.save(server.global_model.state_dict(), 'final_model.pt')
    torch.save(metrics_tracker.metrics_history, 'metrics_history.pt')
    # 训练结束后的完整评估
    print("\nFinal Backdoor Attack Evaluation:")
    retain_valid_dl = fed_dataset._split_val_by_class()[FL_params.unlearn_class]
    final_backdoor_metrics = backdoor_evaluator.update_metrics(
        model=server.global_model,
        generator=server.generator,
        client_data_loader=real_retain_val_ldr,
        original_model=initial_global_model
    )
    # 保存最终结果
    logger.info("\nSaving final results...")
    
    # 保存指标历史
    metrics_df = pd.DataFrame(metrics_history)
    metrics_path = os.path.join(log_dir, 'metrics_history.csv')
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # 保存模型
    model_path = os.path.join(log_dir, 'final_model.pt')
    torch.save(server.global_model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # 保存图表
    plot_path = os.path.join(log_dir, 'metrics_plots')
    os.makedirs(plot_path, exist_ok=True)
    
    # 绘制指标图表
    metrics_tracker.plot_metrics(save_dir=plot_path)
    backdoor_evaluator.plot_metrics(
        save_path=os.path.join(plot_path, 'backdoor_metrics.png')
    )
    logger.info(f"Plots saved to {plot_path}")
    
    logger.info("Training completed successfully!")



    
    # 绘制后门攻击效果随时间的变化
    plt.figure(figsize=(12, 6))
    for key in ['server_asr', 'client_asr', 'trigger_retention']:
        if key in backdoor_evaluator.metrics_history:
            # 获取 y 数据
            y = backdoor_evaluator.metrics_history[key]
            
            # 生成 x 数据，确保长度匹配 y
            x = range(0, len(y) * 5, 5)  # 假设 y 的长度是以 5 轮为间隔的
            
            # 检查 x 和 y 的长度是否匹配
            if len(x) != len(y):
                print(f"Warning: Mismatched lengths for {key}. x: {len(x)}, y: {len(y)}")
                x = range(len(y))  # 修正 x 的长度以匹配 y
            
            # 绘图
            plt.plot(x, y, label=key.replace('_', ' ').title())

    # 添加标签和标题
    plt.xlabel('Rounds')
    plt.ylabel('Metrics')
    plt.title('Backdoor Attack Evaluation Over Time')
    plt.legend()
    # plt.show()
    
    plt.grid(True)
    plt.savefig(f'{FL_params.data_name}_backdoor_metrics.png')
    plt.close()
    
    # 保存详细结果
    detailed_results = {
        'metrics_history': backdoor_evaluator.metrics_history,
        'final_metrics': final_backdoor_metrics
    }
    torch.save(detailed_results, f'{FL_params.data_name}_backdoor_results.pt')

def setup_logging(config):
    """设置日志记录"""
    # 创建日志目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs', f'run_{timestamp}')
    os.makedirs(log_dir, exist_ok=True)

    # 设置日志格式
    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    
    # 文件处理器
    file_handler = logging.FileHandler(
        os.path.join(log_dir, 'training.log')
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 保存配置
    config_path = os.path.join(log_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(config), f, indent=4)
        
    return logger, log_dir


if __name__ == "__main__":
    main()