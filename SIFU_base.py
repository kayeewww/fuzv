from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
from Models import MLP, LeNet5, ResNet18, MobileNet
import numpy as np
from sklearn.cluster import KMeans
import copy
import torch

def identify_low_quality_data(global_model, local_models, n_samples: Dict[int, Dict[int, int]],
                              test_loader,
                              client_data_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]], dataset_name) -> Tuple[
    List[int], Dict[int, Tuple], Dict[int, Tuple]]:
    # # 初始化
    # Pt=90
    _, Pt=test(global_model, test_loader)
    # Pt = accuracy(global_model, benchmark_data, dataset_name)  # 全局模型在基准数据集上的准确性
    print('全局模型在基准数据集上的准确性', Pt)
    low_quality_clients = []
    low_quality_datasets = {}
    good_quality_datasets = {}
    #
    # 按标签拆分基准数据集
    # benchmark_data_by_label = {label: get_data_by_label(benchmark_data, label) for label in labels}

    # 识别低质量标签
    low_quality_labels = set()
    all_labels = set()

    # 获取测试集中所有唯一的标签值
    for _, targets in test_loader:
        all_labels.update(targets.numpy())  # 使用 set 更新唯一标签

    # 遍历每个唯一标签，计算精度并标记低质量标签
    for label in all_labels:
        _, acc = test(global_model, test_loader, label=label)  # 传入单个标签
        if acc < Pt:
            low_quality_labels.add(label)
        print(f'Finished testing label {label}, Accuracy: {acc:.4f}')

    print('Finish identifying low quality labels')
    print(low_quality_labels)

    # 识别低质量客户端
    low_quality_clients_temp = []
    for i, local_model in enumerate(local_models):
        # print(f'Identifying the {i} client')
        _, acc = test(local_model, test_loader)
        if acc < Pt:
            low_quality_clients_temp.append(i)
        print(f'Finished testing client {i}, Accuracy: {acc:.4f}')
    print('Finish identifying low quality clients')
    print(low_quality_clients_temp)

    # 计算每个低质量客户端的评分
    client_scores = []
    for i in low_quality_clients_temp:
        client_score = 0
        for label in low_quality_labels:
            # print('low_quality_clients_temp', len(low_quality_labels))
            _,local_acc = test(local_models[i], test_loader, label=label)
            # local_acc = accuracy(local_models[i], get_data_by_label(benchmark_data, label),dataset_name)

            # Check if the label exists in n_samples[i] before accessing it
            if label in n_samples[i]:
                sample_count = n_samples[i][label]
                if local_acc < Pt:
                    local_acc = max(local_acc, 1e-10)
                    score = -np.log(local_acc) * (sample_count / 100)
                    client_score += score
            else:
                print(f"Warning: Client {i} does not have any samples for label {label}. Skipping this label.")

        client_scores.append(client_score)
    print('client_scores', client_scores)
    # client_scores=[
    #     163.96113051568747, 190.19352868130818, 170.16103837225998, 185.35809998602068, 195.94999141379327, 192.72637228360162, 167.62819476996654, 198.48283501608677, 166.2029001790093]
    #
    # 根据评分对客户端进行聚类
    kmeans = KMeans(n_clusters=2).fit(np.array(client_scores).reshape(-1, 1))

    # 计算每个类的平均得分
    cluster_labels = kmeans.labels_
    cluster_0_score = np.mean([client_scores[i] for i in range(len(cluster_labels)) if cluster_labels[i] == 0])
    cluster_1_score = np.mean([client_scores[i] for i in range(len(cluster_labels)) if cluster_labels[i] == 1])

    # 选择得分较低的类作为低质量客户端
    if cluster_0_score < cluster_1_score:
        low_quality_clients = [low_quality_clients_temp[i] for i in range(len(cluster_labels)) if
                               cluster_labels[i] == 0]
    else:
        low_quality_clients = [low_quality_clients_temp[i] for i in range(len(cluster_labels)) if
                               cluster_labels[i] == 1]

    # 分类低质量和高质量数据集
    # low_quality_labels={1, 2, 3, 5, 6, 7, 9}
    # low_quality_clients = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # 分别提取低质量和高质量数据
    for u in low_quality_clients:
        dataset_u = get_dataset_by_client(u, client_data_dict)
        low_quality_datasets[u] = classify_dataset(global_model, dataset_name, dataset_u, low_quality_labels)
        good_quality_datasets[u] = classify_dataset(global_model, dataset_name, dataset_u,
                                                    set(all_labels) - low_quality_labels)


    # for u in low_quality_clients:
    #     dataset_u = get_dataset_by_client(u, client_data_dict)
    #     # 将 classify_dataset 返回的结果作为嵌套字典存储
    #     # low_quality_datasets[u], good_quality_datasets[u] = classify_dataset(Pt,
    #     #     global_model, dataset_name, dataset_u, low_quality_labels
    #     # )
    #     low_quality_datasets[u] = classify_dataset(global_model,dataset_name, dataset_u, low_quality_labels)
    #     good_quality_datasets[u] = classify_dataset(global_model,dataset_name, dataset_u, set(all_labels) - low_quality_labels)

    print('Number of low quality clients:', len(low_quality_clients))
    print('Length of low quality datasets:', len(low_quality_datasets))
    print('Length of good quality datasets:', len(good_quality_datasets))
    return low_quality_clients, low_quality_datasets, good_quality_datasets
# 计算模型在数据集上的准确
# def accuracy(model, data: Tuple[torch.Tensor, torch.Tensor], dataset_name: str) -> float:
#     model.eval()  # 设置模型为评估模式
#     X, y = data
#
#     # 根据数据集名称调整输入形状
#     if dataset_name == "MNIST":
#         X = X.view(-1, 28 * 28)  # 展平为一维 (适用于 MLP)
#     elif dataset_name == "FMNIST":
#         X = X.view(-1, 1, 28, 28)  # 保持为28x28二维输入 (适用于 LeNet-5)
#     elif dataset_name == "CIFAR-10":
#         X = X.view(-1, 3, 32, 32)  # 32x32的RGB图像 (适用于 ResNet-18)
#     elif dataset_name == "CelebA":
#         X = X.view(-1, 3, 224, 224)  # 假设 MobileNet 输入为 224x224 RGB 图像
#     else:
#         raise ValueError(f"未知的数据集名称: {dataset_name}")
#
#     with torch.no_grad():
#         output = model(X)
#         y_pred = torch.argmax(output, dim=1)  # 获取预测的类别
#     return accuracy_score(y.cpu().numpy(), y_pred.cpu().numpy())  # 计算准确率率


# def get_data_by_label(data: DataLoader, label: int) -> Tuple[torch.Tensor, torch.Tensor]:
#     X_list = []
#     y_list = []
#
#     for X_batch, y_batch in data:
#         mask = y_batch == label  # Filter by label
#         X_list.append(X_batch[mask])
#         y_list.append(y_batch[mask])
#
#     # Concatenate all filtered batches into single tensors
#     X_filtered = torch.cat(X_list, dim=0) if X_list else torch.empty(0)
#     y_filtered = torch.cat(y_list, dim=0) if y_list else torch.empty(0)
#
#     return X_filtered, y_filtered

def get_data_by_label(data: Tuple[torch.Tensor, torch.Tensor], label: int) -> Tuple[torch.Tensor, torch.Tensor]:
    X, y = data
    idx = [i for i, lbl in enumerate(y) if lbl == label]
    return X[idx], y[idx]


# 根据客户端ID获取对应数据集
def get_dataset_by_client(client_id: int, client_data_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    return client_data_dict[client_id]

# Training function
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    # Check if GPU is available and move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")


# def train_model(model, train_loader, criterion, optimizer, epochs=5):
#     for epoch in range(epochs):
#         model.train()
#         for X, y in train_loader:
#             optimizer.zero_grad()
#             output = model(X)
#             loss = criterion(output, y)
#             loss.backward()
#             optimizer.step()
# 评估模型
def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        output = model(X)
        predictions = torch.argmax(output, dim=1)
    return accuracy_score(y, predictions.cpu().numpy())

# def classify_dataset(
#     global_model,
#     dataset_name: str,
#     dataset: DataLoader,  # Use DataLoader to support more data types
#     benchmark_data: DataLoader,
#     labels: List[int]
# ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
#     low_quality_data = []
#     good_quality_data = []
#
#     for label in labels:
#         # Initialize empty lists to collect data for the current label
#         X = []
#         y = []
#
#         # Iterate over the dataset to gather all samples with the specified label
#         for data, target in dataset:
#             if target == label:  # Collect only data points with the specific label
#                 X.append(data.unsqueeze(0))  # Add a new dimension to match batch format
#                 y.append(target.unsqueeze(0))
#
#         # Concatenate all batches along the batch dimension
#         if len(X) > 0:
#             X = torch.cat(X, dim=0)  # Stack along batch dimension
#             y = torch.cat(y, dim=0)
#         else:
#             continue  # Skip if no data is found for this label
#
#         # Create a single dataset tuple
#         dataset_for_label = (X, y)
#
#         # Separate data based on quality
#         X_label, y_label = get_data_by_label(dataset_for_label, label)
#         benchmark_X_label, benchmark_y_label = get_data_by_label(benchmark_data, label)
#
#         if X_label.numel() == 0:  # Skip if no data for the label
#             continue
#
#         # Evaluate accuracy and classify as low or good quality
#         if accuracy(global_model, (X_label, y_label), dataset_name) < 0.5:
#             low_quality_data.append((X_label, y_label))
#         else:
#             good_quality_data.append((X_label, y_label))
#
#     return low_quality_data, good_quality_data
# def classify_dataset(global_model,dataset_name, dataloader,
#                      labels: List[int]) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
#     low_quality_data = {}
#     good_quality_data = {}
#     X, y = [], []
#     # Accumulate data from the dataloader into X and y
#     for batch_X, batch_y in dataloader:
#         X.append(batch_X)
#         y.append(batch_y)
#
#     # Concatenate all batches into a single tensor
#     X = torch.cat(X, dim=0)
#     y = torch.cat(y, dim=0)
#
#     for label in labels:
#         X_label, y_label = get_data_by_label((X, y), label)
#         # benchmark_X_label, benchmark_y_label = get_data_by_label(benchmark_data, label)
#         _,acc=test(global_model, dataloader, label)
#         if acc < 0.5:
#             low_quality_data[label].append((X_label, y_label))
#         else:
#             good_quality_data[label].append((X_label, y_label))
#
#     return low_quality_data, good_quality_data

# def classify_dataset(Pt, global_model, dataset_name, dataloader, labels: List[int]) -> Tuple[
#     Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]], Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]]]:
#     low_quality_data = {label: ([], []) for label in labels}  # 初始化为包含空列表的字典
#     good_quality_data = {label: ([], []) for label in labels}
#
#     X, y = [], []
#
#     # Accumulate data from the dataloader into X and y
#     for batch_X, batch_y in dataloader:
#         X.append(batch_X)
#         y.append(batch_y)
#
#     # Concatenate all batches into a single tensor
#     X = torch.cat(X, dim=0)
#     y = torch.cat(y, dim=0)
#
#     # 遍历标签并分类数据
#     for label in labels:
#         X_label, y_label = get_data_by_label((X, y), label)
#
#         # 评估标签数据的准确性
#         _, acc = test(global_model, dataloader, label)
#
#         # 根据准确率将低质量数据放入 low_quality_data
#         if acc < Pt:
#             low_quality_data[label] = (X_label, y_label)
#
#         # 将所有数据直接存入 good_quality_data，无需 else
#         good_quality_data[label] = (X_label, y_label)
#
#     # 移除 low_quality_data 中的标签部分
#     for label in low_quality_data:
#         if low_quality_data[label][0].nelement() > 0:  # 确保有数据
#             del good_quality_data[label]
#
#     return low_quality_data, good_quality_data
def classify_dataset(global_model, dataset_name, dataloader, labels: set) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    将数据集按标签划分为低质量和高质量数据集。

    参数:
    - Pt: 精度阈值，用于判断低质量数据
    - global_model: 全局模型，用于评估数据质量
    - dataset_name: 数据集名称
    - dataloader: 数据加载器
    - labels: 所有标签列表

    返回:
    - low_quality_data: 包含低质量标签的数据字典
    - good_quality_data: 包含所有数据的字典，去除低质量数据部分
    """
    quality_data = {label: (torch.empty(0), torch.empty(0, dtype=torch.long)) for label in labels}

    X, y = [], []

    # 收集所有数据
    for batch_X, batch_y in dataloader:
        X.append(batch_X)
        y.append(batch_y)

    # 合并所有批次数据
    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)

    # 遍历标签并分类数据
    for label in labels:
        # 获取每个标签的数据
        X_label, y_label = get_data_by_label((X, y), label)

        # 确保 `X_label` 和 `y_label` 是张量形式
        if isinstance(X_label, list):
            X_label = torch.stack(X_label)
        if isinstance(y_label, list):
            y_label = torch.tensor(y_label)

        if X_label.nelement() == 0 or y_label.nelement() == 0:
            print(f"Warning: No data found for label {label}.")
            continue

        # 存储符合条件的标签数据
        quality_data[label] = (X_label, y_label)

    # 调试打印数据分布情况
    print("quality_data:",
          {k: (v[0].size(), v[1].size()) if v[0].nelement() > 0 else "Empty" for k, v in quality_data.items()})

    return quality_data
# 计算模型梯度的函数
def compute_gradients(model: MLP, data_batch: Tuple[torch.Tensor, torch.Tensor], criterion, optimizer) -> Tuple[np.ndarray, np.ndarray]:
    model.train()
    X_batch, y_batch = data_batch
    optimizer.zero_grad()
    output = model(X_batch)
    loss = criterion(output, y_batch)
    loss.backward()
    gradients = {name: param.grad.cpu().numpy() for name, param in model.named_parameters() if param.grad is not None}
    # grad_w = model.fc1.weight.grad.cpu().numpy()  # 获取第一层权重的梯度
    # grad_b = model.fc1.bias.grad.cpu().numpy()    # 获取第一层偏置的梯度
    return gradients#grad_w, grad_b

def evaluate_accuracy(model: MLP, data_batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
    model.eval()
    X_batch, y_batch = data_batch
    with torch.no_grad():
        output = model(X_batch)
        pred = torch.argmax(output, dim=1)
    return accuracy_score(y_batch.cpu().numpy(), pred.cpu().numpy())
# 遗忘低质量数据的函数
def split_batches(data, batch_size):
    """将数据分成指定的批次大小"""
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def unlearn_low_quality_data(global_model, local_models: List, low_quality_datasets: Dict[int, Tuple],
                             good_quality_datasets: Dict[int, Tuple], clients_with_low_quality: List[int],
                             learning_rate: float, early_stop_thresh: float, batch_size: int,
                             client_data_count: Dict[int, Dict[int, int]]) -> nn.Module:
    # Step 1: 初始化全局模型权重，排除低质量客户端的贡献
    global_state_dict = global_model.state_dict()
    sum_low_quality_clients_state_dict = {key: torch.zeros_like(value) for key, value in global_state_dict.items()}

    # 累加每个低质量客户端的权重
    for u in clients_with_low_quality:
        local_state_dict = local_models[u].state_dict()
        for key in local_state_dict:
            sum_low_quality_clients_state_dict[key] += local_state_dict[key]

    # 计算移除低质量客户端后的全局模型权重
    num_clients = len(local_models)
    num_low_quality_clients = len(clients_with_low_quality)
    Mb = {}
    if num_clients == num_low_quality_clients:
        raise ValueError("All clients are marked as low quality, cannot compute Mb.")
    for key in global_state_dict:
        Mb[key] = (num_clients * global_state_dict[key] - sum_low_quality_clients_state_dict[key]) / (
                num_clients - num_low_quality_clients)
    global_model.load_state_dict(Mb)
    # for key in global_state_dict:
    #     Mb[key] = (num_clients * global_state_dict[key] - sum_low_quality_clients_state_dict[key]) / (
    #             num_clients - num_low_quality_clients)
    # global_model.load_state_dict(Mb)

    # for key in global_state_dict:
    #     denominator = num_clients - num_low_quality_clients
    #     if denominator > 0:
    #         Mb[key] = (num_clients * global_state_dict[key] - sum_low_quality_clients_state_dict[key]) / denominator
    #     else:
    #         Mb[key] = global_state_dict[key]  # 保持原始值或选择其他处理方式
    #
    #     # 检查是否存在 inf 或 -inf，并处理
    #     if torch.isinf(Mb[key]).any():
    #         print(f"Warning: {key} contains inf values. Setting it to zero.")
    #         Mb[key] = torch.zeros_like(global_state_dict[key])
    #
    # # 更新全局模型
    # global_model.load_state_dict(Mb)

    # Step 2: 计算总的客户端数据量
    total_data = sum(sum(label_counts.values()) for label_counts in client_data_count.values())

    # Step 3: 遍历低质量客户端
    for u in clients_with_low_quality:
        M = local_models[u]

        # 获取整个模型的权重和偏置参数
        # 获取模型的所有可训练参数
        theta_params = OrderedDict((name, param) for name, param in M.named_parameters() if param.requires_grad)
        # theta_params = [param for param in M.parameters() if param.requires_grad]

        # 合并低质量和高质量数据集
        low_quality_data = []
        good_quality_data = []

        # 确保标签存在时再访问数据
        for label, entry in low_quality_datasets.get(u, {}).items():
            if entry:
                data, labels = entry
                low_quality_data.extend([(x, y) for x, y in zip(data, labels)])

        for label, entry in good_quality_datasets.get(u, {}).items():
            if entry:
                data, labels = entry
                good_quality_data.extend([(x, y) for x, y in zip(data, labels)])
        if not low_quality_data:
            print(f"Client {u} has no low quality data.")
            # continue
            if not good_quality_data:
                print(f"Client {u} has incomplete data.")
            continue
        # 获取批次
        good_batches = split_batches(good_quality_data, batch_size)
        low_quality_batches = split_batches(low_quality_data, batch_size)

        # # 优化器和损失函数
        # optimizer = optim.Adam(M.parameters(), lr=learning_rate)
        # criterion = nn.CrossEntropyLoss()
        #
        # for good_batch in good_batches:
        #     if not good_batch:
        #         continue
        #
        #     for low_quality_batch in low_quality_batches:
        #         if not low_quality_batch:
        #             continue
        #
        #         # 转换低质量批次数据
        #         X_batch, y_batch = zip(*low_quality_batch)
        #         X_batch = torch.stack(X_batch)
        #         y_batch = torch.tensor(y_batch)
        #
        #         # 计算低质量数据集的梯度并更新参数
        #         optimizer.zero_grad()
        #         output = M(X_batch)
        #         loss = criterion(output, y_batch)
        #         loss.backward()
        #
        #         # 对所有参数执行梯度更新
        #         with torch.no_grad():
        #             for param in theta_params:
        #                 param -= learning_rate * param.grad
        #
        #         # 早停条件
        #         if evaluate_accuracy(M, (X_batch, y_batch)) < early_stop_thresh:
        #             break
        # 逐批次执行遗忘过程
        for good_batch in good_batches:
            if not good_batch:
                continue

            for low_quality_batch in low_quality_batches:
                if not low_quality_batch:
                    continue

                X_batch, y_batch = zip(*low_quality_batch)
                X_batch = torch.stack(X_batch)
                y_batch = torch.tensor(y_batch)

                # 计算梯度并更新权重
                gradients = compute_gradients(
                    M, (X_batch, y_batch), criterion=nn.CrossEntropyLoss(),
                    optimizer=optim.Adam(M.parameters(), lr=learning_rate)
                )
                # 更新所有层的参数
                with torch.no_grad():
                    for layer_name, grad in gradients.items():
                        print(f'{layer_name} updated')
                        if layer_name in theta_params:
                            theta_params[layer_name] -= learning_rate * torch.tensor(grad)

                # theta_params[0].data -= learning_rate * grad_w  # 假设 theta_params[0] 是权重
                # theta_params[1].data -= learning_rate * grad_b  # 假设 theta_params[1] 是偏置

                if evaluate_accuracy(M, (X_batch, y_batch)) < early_stop_thresh:
                    break
    # Step 7: 累加所有模型的 state_dict
    M_prime = OrderedDict((key, torch.zeros_like(value)) for key, value in global_state_dict.items())

    for i, local_model in enumerate(local_models):
        local_state_dict = local_model.state_dict()
        client_weight = sum(client_data_count[i].values()) / total_data
        for key in local_state_dict:
            M_prime[key] += local_state_dict[key] * client_weight

    global_model.load_state_dict(M_prime)
    return global_model
    #         # 使用高质量数据集更新
    #         X_good, y_good = zip(*good_batch)
    #         X_good = torch.stack(X_good)
    #         y_good = torch.tensor(y_good)
    #
    #         optimizer.zero_grad()
    #         output = M(X_good)
    #         loss = criterion(output, y_good)
    #         loss.backward()
    #
    #         with torch.no_grad():
    #             for param in theta_params:
    #                 param -= learning_rate * param.grad
    #
    # # Step 7: 累加所有模型的 state_dict
    # M_prime = OrderedDict((key, torch.zeros_like(value)) for key, value in global_state_dict.items())
    #
    # for i, local_model in enumerate(local_models):
    #     local_state_dict = local_model.state_dict()
    #     client_weight = sum(client_data_count[i].values()) / total_data
    #     for key in local_state_dict:
    #         M_prime[key] += local_state_dict[key] * client_weight
    #
    # # 更新全局模型权重
    # global_model.load_state_dict(M_prime)
    # return global_model

# def unlearn_low_quality_data(global_model, local_models: List, low_quality_datasets: Dict[int, Tuple],
#                              good_quality_datasets: Dict[int, Tuple], clients_with_low_quality: List[int],
#                              learning_rate: float, early_stop_thresh: float, batch_size: int, client_data_count: Dict[int, int]):
#     # Step 1: 初始化全局模型权重，排除低质量客户端的贡献
#     global_state_dict = global_model.state_dict()
#     sum_low_quality_clients_state_dict = {key: torch.zeros_like(value) for key, value in global_state_dict.items()}
#
#     # 累加每个低质量客户端的权重
#     for u in clients_with_low_quality:
#         local_state_dict = local_models[u].state_dict()
#         for key in local_state_dict:
#             sum_low_quality_clients_state_dict[key] += local_state_dict[key]
#
#     # 计算移除低质量客户端后的全局模型权重
#     num_clients = len(local_models)
#     num_low_quality_clients = len(clients_with_low_quality)
#     Mb = {}
#     for key in global_state_dict:
#         Mb[key] = (num_clients * global_state_dict[key] - sum_low_quality_clients_state_dict[key]) / (
#                     num_clients - num_low_quality_clients)
#     global_model.load_state_dict(Mb)
#
#     # Step 2: 计算总的客户端数据量
#     total_data = sum(sum(label_counts.values()) for label_counts in client_data_count.values())
#
#     # Step 3: 遍历低质量客户端
#     for u in clients_with_low_quality:
#         M = local_models[u]
#         theta_1 = M.fc.weight.data
#         theta_0 = M.fc.bias.data
#
#         # 合并低质量和高质量数据集
#         low_quality_data = [(x, y) for data, labels in low_quality_datasets[u][1] for x, y in zip(data, labels)]
#         good_quality_data = [(x, y) for data, labels in good_quality_datasets[u][1] for x, y in zip(data, labels)]
#
#         if not low_quality_data or not good_quality_data:
#             print(f"Client {u} has incomplete data.")
#             continue
#
#         # 获取批次
#         good_batches = split_batches(good_quality_data, batch_size)
#         low_quality_batches = split_batches(low_quality_data, batch_size)
#
#         for good_batch in good_batches:
#             if not good_batch:
#                 continue
#
#             for low_quality_batch in low_quality_batches:
#                 if not low_quality_batch:
#                     continue
#
#                 # 检查低质量批次是否为 (X_batch, y_batch) 形式
#                 if isinstance(low_quality_batch, list) and len(low_quality_batch) > 0:
#                     X_batch, y_batch = zip(*low_quality_batch)
#                     X_batch = torch.stack(X_batch)
#                     y_batch = torch.tensor(y_batch)
#
#                     grad_w, grad_b = compute_gradients(
#                         M, (X_batch, y_batch), criterion=nn.CrossEntropyLoss(),
#                         optimizer=optim.Adam(M.parameters(), lr=learning_rate)
#                     )
#                     theta_1 -= learning_rate * grad_w
#                     theta_0 -= learning_rate * grad_b
#                 else:
#                     print(f"Client {u} has an improperly formatted batch.")
#
#                 if evaluate_accuracy(M, (X_batch, y_batch)) < early_stop_thresh:
#                     break
#
#         M.fc1.weight.data = theta_1
#         M.fc1.bias.data = theta_0
#
#     # Step 7: 累加所有模型的 state_dict
#     M_prime = OrderedDict((key, torch.zeros_like(value)) for key, value in global_state_dict.items())
#
#     for i, local_model in enumerate(local_models):
#         local_state_dict = local_model.state_dict()
#         client_weight = sum(client_data_count[i].values()) / total_data
#         for key in local_state_dict:
#             M_prime[key] += local_state_dict[key] * client_weight
#
#     global_model.load_state_dict(M_prime)
#     return global_model


def global_train_once(global_model, client_data_loaders, test_loader, FL_params):
    # 使用每个client的模型、优化器、数据，以client_models为训练初始模型，使用client用户本地的数据和优化器，更新得到upodate——client_models
    # Note：需要注意的一点是，global_train_once只是在全局上对模型的参数进行一次更新
    # Using the model, optimizer, and data of each client, training the initial model with client_models, updating the UPODate -- client_models using the client user's local data and optimizer
    # Note: It is important to Note that global_train_once is only a global update to the parameters of the model
    # update_client_models = list()
    device = torch.device("cuda" if FL_params.use_gpu * FL_params.cuda_state else "cpu")
    device_cpu = torch.device("cpu")
    # local_models = []
    # for client_id, client_loader in clients_data.items():
    #     local_model = model_class()
    #     optimizer = optim.Adam(local_model.parameters(), lr=0.001)
    #     train_model(local_model, client_loader, criterion, optimizer, epochs=5)
    #     local_models.append(local_model)

    client_models = []
    client_sgds = []
    for ii in range(FL_params.N_client):
        client_models.append(copy.deepcopy(global_model))
        if isinstance(client_models[ii],MobileNet):
            optim = torch.optim.Adam(client_models[ii].parameters(), lr=FL_params.local_lr)
            client_sgds.append(optim)
        else:
            client_sgds.append(torch.optim.SGD(client_models[ii].parameters(), lr=FL_params.local_lr, momentum=0.9))

    for client_idx, client_loader in client_data_loaders.items():
    # for client_idx in range(FL_params.N_client):
        if (((FL_params.if_retrain) and (FL_params.forget_client_idx == client_idx)) or (
                (FL_params.if_unlearning) and (FL_params.forget_client_idx == client_idx))):
            continue
        # if((FL_params.if_unlearning) and (FL_params.forget_client_idx == client_idx)):
        #     continue
        # print(30*'-')
        # print("Now training Client No.{}  ".format(client_idx))
        model = client_models[client_idx]
        optimizer = client_sgds[client_idx]

        model.to(device)
        model.train()

        # local training
        for local_epoch in range(FL_params.local_epoch):
            for batch_idx, (data, target) in enumerate(client_data_loaders[client_idx]):
                data = data.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                pred = model(data)
                if isinstance(global_model, MobileNet):
                    criteria = nn.BCEWithLogitsLoss()
                    target = target.float()
                else:
                    criteria = nn.CrossEntropyLoss()
                loss = criteria(pred, target)
                loss.backward()
                optimizer.step()

            if (FL_params.train_with_test):
                print("Local Client No. {}, Local Epoch: {}".format(client_idx, local_epoch))
                test(model, test_loader)

        # if(FL_params.use_gpu*FL_params.cuda_state):
        model.to(device_cpu)
        client_models[client_idx] = model

    if (((FL_params.if_retrain) and (FL_params.forget_client_idx == client_idx))):
        # 只有retrian 需要丢弃client 模型；如果不是在retrain的话，就不需要丢弃模型
        # Only retrian needs to discard the Client model;If it's not in Retrain, there's no need to discard the model
        client_models.pop(FL_params.forget_client_idx)
        return client_models
    elif ((FL_params.if_unlearning) and (FL_params.forget_client_idx in range(FL_params.N_client))):
        client_models.pop(FL_params.forget_client_idx)
        return client_models
    else:
        return client_models


def test(model, test_loader, label=None):
    """
    Evaluate the model on the test_loader, with optional filtering by a specific label.

    Parameters:
    - model: The model to evaluate.
    - test_loader: DataLoader for the test dataset.
    - label: (Optional) The specific label to filter by. If None, evaluates on the whole test set.

    Returns:
    - Average loss and accuracy.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criteria = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            if label is not None:
                # Filter out only the samples of the specified label
                mask = (target == label)
                data, target = data[mask], target[mask]
                if len(target) == 0:
                    continue  # Skip if no data for the label in this batch

            output = model(data)
            test_loss += criteria(output, target).item() * data.size(0)  # sum up batch loss

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    test_loss /= total
    test_acc = correct / total if total > 0 else 0
    # print(f'Test set for label {label if label is not None else "all"}: '
    #       f'Average loss: {test_loss:.8f}, Accuracy: {test_acc:.4f}')
    return test_loss, test_acc


def fedavg(local_models):
    # def fedavg(local_models, local_model_weights=None):
    """
    Parameters
    ----------
    local_models : list of local models
        DESCRIPTION.In federated learning, with the global_model as the initial model, each user uses a collection of local models updated with their local data.
    local_model_weights : tensor or array
        DESCRIPTION. The weight of each local model is usually related to the accuracy rate and number of data of the local model.(Bypass)

    Returns
    -------
    update_global_model
        Updated global model using fedavg algorithm
    """
    # N = len(local_models)
    # new_global_model = copy.deepcopy(local_models[0])
    # print(len(local_models))
    global_model = copy.deepcopy(local_models[0])
    avg_state_dict = global_model.state_dict()

    local_state_dicts = list()
    for model in local_models:
        local_state_dicts.append(model.state_dict())

    for layer in avg_state_dict.keys():
        avg_state_dict[layer] *= 0
        for client_idx in range(len(local_models)):
            avg_state_dict[layer] += local_state_dicts[client_idx][layer]
        # avg_state_dict[layer] /= len(local_models)
        avg_state_dict[layer] = avg_state_dict[layer].float() / len(local_models)

    global_model.load_state_dict(avg_state_dict)
    return global_model


def inject_backdoor(data_loader, target_label, pattern):
    modified_images = []
    modified_labels = []

    for images, labels in data_loader:
        # 获取图像的通道数、高度和宽度
        channels, height, width = images.shape[1:]

        # 将 pattern 调整为图像的通道数
        pattern_resized = pattern.expand(channels, pattern.shape[1], pattern.shape[2])

        # 在每张图像上添加后门模式
        for img in images:
            img[:, :pattern.shape[1], :pattern.shape[2]] = pattern_resized
            modified_images.append(img)
            modified_labels.append(torch.tensor(target_label))  # 将标签改为目标标签

    # 将修改后的图像和标签转换为张量
    modified_images = torch.stack(modified_images)
    modified_labels = torch.stack(modified_labels)

    # 创建新的 DataLoader 并返回
    modified_dataset = TensorDataset(modified_images, modified_labels)
    modified_data_loader = DataLoader(modified_dataset, batch_size=data_loader.batch_size, shuffle=True)

    return modified_data_loader
def create_backdoor_test_set(data_loader, target_label, pattern):
    modified_images = []
    modified_labels = []

    for images, labels in data_loader:
        channels, height, width = images.shape[1:]
        pattern_resized = pattern.expand(channels, pattern.shape[1], pattern.shape[2])

        for img in images:
            img[:, :pattern.shape[1], :pattern.shape[2]] = pattern_resized  # 注入后门模式
            modified_images.append(img)
            modified_labels.append(torch.tensor(target_label))  # 设置目标标签

    modified_images = torch.stack(modified_images)
    modified_labels = torch.stack(modified_labels)

    backdoor_test_dataset = TensorDataset(modified_images, modified_labels)
    backdoor_test_loader = DataLoader(backdoor_test_dataset, batch_size=data_loader.batch_size, shuffle=False)

    return backdoor_test_loader


# if __name__=='__main__':
#     # 准备MNIST数据集
#     transform = transforms.Compose([transforms.ToTensor(),  # Convert images to tensor
#                                     transforms.Normalize((0.1307,), (0.3081,))])  # Normalize dataset
#
#     # 加载数据集
#     train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
#     test_dataset = MNIST(root='./data', train=False, transform=transform, download=True)
#
#     # 将数据集转换为 DataLoader
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#
#     # 划分数据集 (转换为PyTorch张量)
#     X_train, X_val, y_train, y_val = train_test_split(train_dataset.data.numpy(), train_dataset.targets.numpy(), test_size=0.2, random_state=42)
#     X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, 28*28) / 255.0
#     X_val = torch.tensor(X_val, dtype=torch.float32).view(-1, 28*28) / 255.0
#     y_train = torch.tensor(y_train, dtype=torch.long)
#     y_val = torch.tensor(y_val, dtype=torch.long)
#
#     # 创建全局模型
#     global_model = MLP()
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(global_model.parameters(), lr=0.001)
#
#     # 训练全局模型
#     train_data = TensorDataset(X_train, y_train)
#     train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
#     train_model(global_model, train_loader, criterion, optimizer, epochs=5)
#
#     # 假设我们有几个客户端，每个客户端都有本地数据
#     clients_data = {
#         0: (X_train[:10000], y_train[:10000]),
#         1: (X_train[10000:20000], y_train[10000:20000]),
#         2: (X_train[20000:], y_train[20000:])
#     }
#
#     # 模拟每个客户端训练本地模型
#     local_models = []
#     for client_id, (X_client, y_client) in clients_data.items():
#         local_model = MLP()
#         optimizer = optim.Adam(local_model.parameters(), lr=0.001)
#         train_data = TensorDataset(X_client, y_client)
#         train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
#         train_model(local_model, train_loader, criterion, optimizer, epochs=5)
#         local_models.append(local_model)
#
#     # 基准数据集 (转化为Tensor)
#     X_test = test_dataset.data.view(-1, 28*28).float() / 255.0
#     y_test = test_dataset.targets
#     benchmark_data = (X_test, y_test)
#
#     # 准备识别低质量数据
#     n_samples = {client_id: {label: np.sum(y_client.numpy() == label) for label in np.unique(y_test.numpy())} for client_id, (_, y_client) in clients_data.items()}
#     print('n_samples: ',n_samples)
#     # 低质量数据识别
#     low_quality_clients, low_quality_datasets, good_quality_datasets = identify_low_quality_data(
#         global_model, local_models, n_samples, (X_test, y_test), labels=np.unique(y_test.numpy()), client_data_dict=clients_data
#     )
#
#     low_quality_clients, low_quality_datasets, good_quality_datasets = identify_low_quality_data(
#         global_model, local_models, n_samples, (X_test, y_test), labels=np.unique(y_test.numpy()), client_data_dict=clients_data
#     )
#
#     # 检查是否有低质量客户端
#     if len(low_quality_clients)>0:
#         print("低质量客户端:", low_quality_clients)
#         print('低质量数据集个数:', len(low_quality_datasets))
#         print('高质量数据集个数:', len(good_quality_datasets))
#
#         # 执行遗忘低质量数据的操作并更新全局模型
#         global_model_updated = unlearn_low_quality_data(
#             global_model, local_models, low_quality_datasets, good_quality_datasets,
#             clients_with_low_quality=low_quality_clients, learning_rate=0.001,
#             early_stop_thresh=0.5, batch_size=32, client_data_count=n_samples
#         )
#
#         print("全局模型已更新")
#     else:
#         print("未检测到低质量客户端，无需更新全局模型。")