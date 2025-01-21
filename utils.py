import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader

def client_subset_selection(num_clients, clients_per_round, adjust_factor, rho):
    """
    Selects a subset of clients dynamically based on Wasserstein distance (rho).
    
    Args:
        num_clients (int): Total number of clients.
        clients_per_round (int): Base number of clients to select per round.
        adjust_factor (float): Scaling factor for adjustment.
        rho (float): Wasserstein distance, representing unlearning stability.

    Returns:
        list: Indices of selected clients.
    """
    # Adjust client participation based on rho
    scaling_factor = max(1, int(adjust_factor / max(rho, 1e-6)))
    adjusted_clients = min(clients_per_round * scaling_factor, num_clients)
    
    # Randomly select clients
    selected_clients = np.random.choice(
        range(num_clients), adjusted_clients, replace=False
    )
    return selected_clients.tolist()

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def training_step(model, batch, device):
    images, labels = batch 
    images, labels = images.to(device), labels.to(device)
    out, *_ = model(images)                  # Generate predictions
    loss = F.cross_entropy(out, labels) # Calculate loss
    return loss

def validation_step(model, batch, device):
    images, labels = batch 
    images, labels = images.to(device), labels.to(device)
    out, *_ = model(images)                    # Generate predictions
    loss = F.cross_entropy(out, labels)   # Calculate loss
    acc = accuracy(out, labels)           # Calculate accuracy
    return {'Loss': loss.detach(), 'Acc': acc}

def validation_epoch_end(model, outputs):
    batch_losses = [x['Loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_accs = [x['Acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    return {'Loss': epoch_loss.item(), 'Acc': epoch_acc.item()}

def epoch_end(model, epoch, result):
    print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch, result['lrs'][-1], result['train_loss'], result['Loss'], result['Acc']))
    
@torch.no_grad()
def evaluate(model, val_loader, device='cuda'):
    model.eval()
    outputs = [validation_step(model, batch, device) for batch in val_loader]
    return validation_epoch_end(model, outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD, device='cuda'):
    
    # if isinstance(train_loader, dict):
    #     all_datasets = [dataloader.dataset for dataloader in train_loader.values()]
    #     combined_dataset = ConcatDataset(all_datasets)
    #     train_loader = DataLoader(combined_dataset, batch_size=256, shuffle=True)
    # elif isinstance(train_loader, DataLoader):
    #     train_loader = DataLoader(train_loader.dataset, batch_size=256, shuffle=True)
    # if isinstance(val_loader, dict):
    #     all_datasets = [dataloader.dataset for dataloader in val_loader.values()]
    #     combined_dataset = ConcatDataset(all_datasets)
    #     val_loader = DataLoader(combined_dataset, batch_size=256, shuffle=True)

    torch.cuda.empty_cache()
    history = []
    
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    for epoch in range(epochs): 
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch, device)
            train_losses.append(loss)
            loss.backward()
            
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            
        
        # Validation phase
        result = evaluate(model, val_loader, device)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        epoch_end(model, epoch, result)
        history.append(result)
        sched.step(result['Loss'])
    return history


def zero_shot_evaluate(model, generator, target_class, criterion, n_samples=1000, device='cuda'):
    """改进的零样本评估方法"""
    model.eval()
    generator.eval()
    
    with torch.no_grad():
        total_samples = 0
        all_preds = []
        all_probs = []
        total_loss = 0
        
        while total_samples < n_samples:
            # 生成样本
            x_pseudo = generator.__next__().to(device=device)
            batch_size = x_pseudo.size(0)
            
            # 模型预测
            outputs = model(x_pseudo)
            # 如果 outputs 是元组，取第一个元素
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # 提取 logits
            probs = F.softmax(outputs, dim=1)
            mask = (probs.argmax(dim=1) == target_class)  # 只保留预测为 target_class 的样本
            x_pseudo = x_pseudo[mask]

            if x_pseudo.size(0) == 0:
                continue  # 如果没有符合条件的样本，跳过该批次
                        
            # 计算损失
            pseudo_labels = torch.full((batch_size,), target_class, dtype=torch.long, device=device)
            loss = criterion(outputs, pseudo_labels)
            
            # 收集预测结果
            preds = outputs.argmax(dim=1)
            all_preds.append(preds)
            all_probs.append(probs[:, target_class])
            
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            if total_samples >= n_samples:
                break
                
    # 合并结果
    all_preds = torch.cat(all_preds)[:n_samples]
    all_probs = torch.cat(all_probs)[:n_samples]
    
    # 计算指标
    accuracy = (all_preds == target_class).float().mean().item() * 100
    avg_loss = total_loss / n_samples
    avg_confidence = all_probs.mean().item()
    
    return {
        "Acc": accuracy,
        "Loss": avg_loss,
        "target_confidence": avg_confidence#,
        # "predictions": all_preds.cpu().numpy(),
        # "probabilities": all_probs.cpu().numpy()
    }


def evaluate_backdoor(model, generator, clean_loader, num_samples, target_class, device):
    """
    后门攻击评估方法 - 评估干净样本准确率和触发样本攻击成功率
    Args:
        model: 要评估的模型
        generator: 训练好的生成器 
        clean_loader: 干净测试数据的加载器
        num_samples: 要生成的触发样本数量
        target_class: 目标攻击类别
        device: 计算设备
    """
    model.eval()
    
    # 1. 评估干净样本的准确率
    clean_correct = 0
    clean_total = 0
    with torch.no_grad():
        for images, labels in clean_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            clean_total += labels.size(0)
            clean_correct += predicted.eq(labels).sum().item()
    clean_acc = 100. * clean_correct / clean_total

    # 2. 评估触发样本的攻击成功率
    success_count = 0
    total_loss = 0
    generated_samples = []
    
    with torch.no_grad():
        samples_generated = 0
        while samples_generated < num_samples:
            # 生成触发样本
            trigger_samples = generator.__next__().to(device)
            
            # 模型预测
            outputs = model(trigger_samples)
            loss = F.cross_entropy(outputs, 
                torch.full((trigger_samples.size(0),), target_class, 
                          dtype=torch.long, device=device))
            
            # 统计攻击成功数量
            pred = outputs.argmax(dim=1)
            success_count += (pred == target_class).sum().item()
            
            total_loss += loss.item()
            generated_samples.append(trigger_samples.cpu())
            samples_generated += trigger_samples.size(0)
    
    # 合并生成的样本并创建数据加载器
    generated_samples = torch.cat(generated_samples, dim=0)[:num_samples]
    generated_dataset = TensorDataset(
        generated_samples,
        torch.full((num_samples,), target_class, dtype=torch.long)
    )
    generated_loader = DataLoader(generated_dataset, batch_size=32, shuffle=True)
    
    # 计算攻击成功率和平均损失
    attack_success_rate = 100. * success_count / num_samples
    avg_loss = total_loss / (num_samples // generator.batch_size + 1)
    
    results = {
        "CleanAccuracy": clean_acc,
        "AttackSuccessRate": attack_success_rate,
        "Loss": avg_loss,
        "GeneratedLoader": generated_loader,
        "GeneratedSamples": generated_samples
    }
    
    # 打印评估结果
    print(f"干净样本准确率: {clean_acc:.2f}%")
    print(f"触发样本攻击成功率: {attack_success_rate:.2f}%")
    print(f"平均损失: {avg_loss:.4f}")
    
    return results

def evaluate_with_generated_samples(model, generator, num_samples, target_class, device):
    """
    使用generator生成的样本进行测试
    Args:
        model: 要测试的模型
        generator: 训练好的生成器
        num_samples: 要生成的样本数量
        target_class: 目标类别
        device: 计算设备
    """
    model.eval()
    generator.eval()
    
    # 存储生成的样本和预测结果
    generated_data = []
    predictions = []
    confidences = []
    
    with torch.no_grad():
        samples_generated = 0
        while samples_generated < num_samples:
            # 生成样本
            fake_samples = generator.__next__().to(device)
            
            # 模型预测
            outputs = model(fake_samples)
            probs = F.softmax(outputs, dim=1)
            
            # 获取预测结果和置信度
            pred = outputs.argmax(dim=1)
            conf = probs.max(dim=1)[0]
            
            # 存储结果
            generated_data.append(fake_samples.cpu())
            predictions.extend(pred.cpu().numpy())
            confidences.extend(conf.cpu().numpy())
            
            samples_generated += fake_samples.size(0)
    
    # 将生成的数据转换为tensor
    generated_data = torch.cat(generated_data, dim=0)[:num_samples]
    predictions = np.array(predictions[:num_samples])
    confidences = np.array(confidences[:num_samples])
    
    # 计算统计信息
    attack_success_rate = 100 * np.mean(predictions == target_class)
    avg_confidence = np.mean(confidences)
    
    # 创建数据加载器用于进一步测试
    generated_dataset = TensorDataset(
        generated_data,
        torch.tensor(predictions, dtype=torch.long)
    )
    generated_loader = DataLoader(generated_dataset, batch_size=32, shuffle=True)
    
    results = {
        "generated_loader": generated_loader,
        "attack_success_rate": attack_success_rate,
        "average_confidence": avg_confidence,
        "predictions": predictions,
        "confidences": confidences,
        "generated_data": generated_data
    }
    
    # 打印结果
    print(f"生成样本数量: {len(generated_data)}")
    print(f"攻击成功率: {attack_success_rate:.2f}%")
    print(f"平均预测置信度: {avg_confidence:.4f}")
    print(f"预测类别分布:")
    for i in range(10):  # 假设是10分类问题
        class_rate = 100 * np.mean(predictions == i)
        print(f"类别 {i}: {class_rate:.2f}%")
    
    return results


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

def retraining_model(model, train_loader, val_loader, device, FL_params):
    """
    训练模型并进行评估

    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 使用的设备（CPU/GPU）
        num_epochs: 训练轮数
    """
    # 定义损失函数和优化器
    
    criterion = nn.CrossEntropyLoss()
    # optimizer=torch.optim.Adam(
    #                 model.parameters(),
    #                 lr=0.005
    #             ),
    # print(f"Optimizer type: {(optimizer)}")
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # 记录最佳验证准确率
    best_val_acc = 0.0
    
    for epoch in range(FL_params.retrain_epoch):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)[0]
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计训练指标
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # 计算训练epoch的平均损失和准确率
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)[0]
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # 计算验证epoch的平均损失和准确率
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            print("saving best model")
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        # 打印训练和验证结果
        print(f'Epoch [{epoch+1}/{FL_params.retrain_epoch}]')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        print('----------------------------------------')
    
    return best_val_acc

def evaluate_model(model, test_loader, device):
    """
    评估模型性能，包括总体和每个类别的准确率与损失

    Args:
        model: 要评估的模型
        test_loader: 测试数据加载器
        device: 使用的设备（CPU/GPU）

    Returns:
        test_acc: 总体准确率
        test_loss: 总体损失
        class_wise_acc: 每个类别的准确率字典
        class_wise_loss: 每个类别的损失字典
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')  # 使用reduction='none'以获取每个样本的损失
    
    # 初始化统计变量
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    class_correct = {i: 0. for i in range(10)}
    class_total = {i: 0. for i in range(10)}
    class_losses = {i: [] for i in range(10)}  # 存储每个类别的所有样本损失
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)[0]
            sample_losses = criterion(outputs, labels)  # 计算每个样本的损失
            
            # 计算预测结果
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            # 更新每个类别的统计信息
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_losses[label].append(sample_losses[i].item())
    
    # 计算总体指标
    test_acc = 100. * test_correct / test_total
    test_loss = torch.tensor(list(sample_losses)).mean().item()
    
    # 计算每个类别的指标
    class_wise_acc = {}
    class_wise_loss = {}
    
    print('Test Results:')
    print(f'Overall Loss: {test_loss:.4f} | Overall Acc: {test_acc:.2f}%')
    print('\nClass-wise Results:')
    
    for i in range(10):
        if class_total[i] > 0:
            # 计算类别准确率
            class_wise_acc[i] = 100 * class_correct[i] / class_total[i]
            # 计算类别平均损失
            class_wise_loss[i] = sum(class_losses[i]) / len(class_losses[i])
            
            print(f'Class {i}: Loss: {class_wise_loss[i]:.4f} | Acc: {class_wise_acc[i]:.2f}%')
        else:
            class_wise_acc[i] = 0.0
            class_wise_loss[i] = 0.0
            print(f'Class {i}: Loss: {class_wise_loss[i]:.4f} | Acc: {class_wise_acc[i]:.2f}%')
    
    return test_acc, test_loss, class_wise_acc, class_wise_loss
    # return {
    #     'test_acc': test_acc,
    #     'test_loss': test_loss,
    #     'class_wise_acc': class_wise_acc,
    #     'class_wise_loss': class_wise_loss
    # }
