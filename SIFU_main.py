import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
import copy
from collections import OrderedDict
from typing import Dict, List, Tuple
from Models import MLP, LeNet5, ResNet18, MobileNet
from torchvision.models import mobilenet_v2
from SIFU_base import test, create_backdoor_test_set, train_model, identify_low_quality_data, classify_dataset, compute_gradients, unlearn_low_quality_data, global_train_once, fedavg, inject_backdoor
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CelebA
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import time


class Arguments():
    def __init__(self):
        # Federated Learning Settings
        self.N_total_client = 100
        self.N_client = 10
        self.data_name = 'MNIST'  # purchase, cifar10, mnist, adult
        self.data_config = {}
        self.global_epoch = 20

        self.local_epoch = 10

        # Model Training Settings
        self.local_batch_size = 32
        self.local_lr = 0.005

        self.test_batch_size = 1
        self.seed = 1
        self.save_all_model = True
        self.cuda_state = torch.cuda.is_available()
        self.use_gpu = True
        self.train_with_test = False

        # Federated Unlearning Settings
        self.unlearn_interval = 1  # Used to control how many rounds the model parameters are saved.1 represents the parameter saved once per round  N_itv in our paper.
        self.forget_client_idx = 2  # If want to forget, change None to the client index

        # If this parameter is set to False, only the global model after the final training is completed is output
        self.if_retrain = False  # If set to True, the global model is retrained using the FL-Retrain function, and data corresponding to the user for the forget_client_IDx number is discarded.

        self.if_unlearning = False  # If set to False, the global_train_once function will not skip users that need to be forgotten;If set to True, global_train_once skips the forgotten user during training

        self.forget_local_epoch_ratio = 0.5  # When a user is selected to be forgotten, other users need to train several rounds of on-line training in their respective data sets to obtain the general direction of model convergence in order to provide the general direction of model convergence.
        # forget_local_epoch_ratio*local_epoch Is the number of rounds of local training when we need to get the convergence direction of each local model
        # self.mia_oldGM = False


def Sever_Initial_Unlearning():
    FL_params = Arguments()
    torch.manual_seed(FL_params.seed)

    datasets_and_models = {
        # "MNIST": {
        #     "dataset": MNIST,
        #     "model": MLP,
        #     "transform": transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.1307,), (0.3081,))
        #     ]),
        #     "input_shape": (28 * 28,),
        #     "samples_per_client": 1000,  # Number of samples per client
        #     "batch_size": 32,
        #     "test_batch_size": 64
        # }#,
        # "FMNIST": {
        #     "dataset": FashionMNIST,
        #     "model": LeNet5,
        #     "transform": transforms.Compose([
        #         transforms.ToTensor(),#transforms.Normalize((0.5,), (0.5,))  # or try (0.286, 0.353) for mean and std
        #         transforms.Normalize(0.286, 0.353)
        #     ]),
        #     "input_shape": (1, 28, 28),
        #     "samples_per_client": 500,
        #     "batch_size": 32,
        #     "test_batch_size": 64
        # }#,
        "CIFAR10": {
            "dataset": CIFAR10,
            "model": ResNet18,
            "transform": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
            "input_shape": (3, 32, 32),
            "samples_per_client": 1200,
            "batch_size": 128,
            "test_batch_size": 128
        }#,
        # "CelebA": {
        #     "dataset": CelebA,
        #     "model": MobileNet,
        #     "transform": transforms.Compose([
        #         transforms.Resize((128, 128)),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #     ]),
        #     "input_shape": (3, 128, 128),
        #     "samples_per_client": 800,
        #     "batch_size": 32,
        #     "test_batch_size": 64
        # }
    }

    # 主循环
    for dataset_name, config in datasets_and_models.items():
        print('#' * 8, f"{dataset_name} is processing", '#' * 8)

        # Initialize dataset, model, and transforms
        dataset = config["dataset"]
        transform = config["transform"]
        model_class = config["model"]
        samples_per_client = config["samples_per_client"]
        batch_size = config["batch_size"]
        test_batch_size = config["test_batch_size"]
        input_shape = config["input_shape"]

        # Using split for CelebA, otherwise train parameter
        if dataset_name == "CelebA":
            train_dataset = dataset(root='./data', split="train", transform=transform)
            test_dataset = dataset(root='./data', split="test", transform=transform)
        else:
            train_dataset = dataset(root='./data', train=True, transform=transform, download=True)
            test_dataset = dataset(root='./data', train=False, transform=transform, download=True)

        # Creating indices for train-validation split
        indices = list(range(len(train_dataset)))
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset,batch_size=test_batch_size, shuffle=False)

        # Model setup
        global_model = model_class()
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(global_model.parameters(), lr=0.001)
        # criterion = nn.BCEWithLogitsLoss()  # Use binary cross-entropy loss with logits
        # optimizer = torch.optim.Adam(global_model.parameters(), lr=0.001)

        # Global model training
        # global_model = mobilenet_v2(pretrained=True)
        # global_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

        # Modify the last layer to match the 40 output attributes of CelebA
        # if dataset_name == "CelebA":
        #     # mobilenet_v2(pretrained=True)
        #     global_model.fc = nn.Linear(global_model.fc.in_features, 40)
            # global_model.classifier[1] = nn.Linear(global_model.classifier[1].in_features, 40)
        # train_model(global_model, train_loader, criterion, optimizer, epochs=1)

        # Client data division and DataLoader creation
        clients_data = {}
        for i in range(FL_params.N_client):
            client_indices = np.random.choice(train_indices, samples_per_client, replace=False)
            client_subset = Subset(train_dataset, client_indices)
            clients_data[i] = DataLoader(client_subset, batch_size=batch_size, shuffle=True)
        #
        # Initialize list to store client models and global models for each epoch
        all_client_models = []
        all_global_models = []
        #
        print(5 * "#" + "  Federated Learning Start" + 5 * "#")
        std_time = time.time()
        # Main Federated Learning loop with Federated Unlearning support
        # 定义保存路径
        output_dir = "../saved_models"
        os.makedirs(output_dir, exist_ok=True)

    #     for epoch in range(FL_params.global_epoch):
    #         # Deep copy of client models to save their pre-trained state
    #         client_models_backup = [copy.deepcopy(global_model) for _ in range(FL_params.N_client)]
    #
    #         # Train each client model using global_train_once function
    #         client_models = global_train_once(global_model, clients_data, test_loader, FL_params)
    #
    #         # Append pre-trained client models before training for unlearning purposes
    #         all_client_models.extend(client_models)
    #
    #         # Aggregate the models from clients to update the global model using FedAvg or a similar method
    #         global_model = fedavg(client_models)
    #
    #         # Logging and storing the global model
    #         print(f"Global Federated Learning epoch = {epoch}")
    #         all_global_models.append(copy.deepcopy(global_model))  # Save a snapshot of the global model for unlearning
    #     end_time = time.time()
    #     time_learn = (std_time - end_time)
    #     # 保存客户端模型
    #     output_dir = os.path.join(output_dir, dataset_name)
    #     os.makedirs(output_dir, exist_ok=True)  # 如果目录不存在，则创建
    #
    #     # 保存客户端模型
    #     for i, client_model in enumerate(all_client_models):
    #         torch.save(client_model.state_dict(), os.path.join(output_dir, f"20_client_model_epoch_{i}.pt"))
    #
    #     # 保存全局模型
    #     for i, global_model_epoch in enumerate(all_global_models):
    #         torch.save(global_model_epoch.state_dict(), os.path.join(output_dir, f"20_global_model_epoch_{i}.pt"))
    # # exit(0)
    # # for dataset_name, config in datasets_and_models.items():
    #
    #     print(5 * "#" + "  Federated Learning End" + 5 * "#")
    #     print(" Learning time consuming = {} secods".format(-time_learn))

        # Local model training for each client
        # local_models = []
        # for client_id, client_loader in clients_data.items():
        #     local_model = model_class()
        #     optimizer = optim.Adam(local_model.parameters(), lr=0.001)
        #     train_model(local_model, client_loader, criterion, optimizer, epochs=5)
        #     local_models.append(local_model)
        #
        # 加载客户端模型和全局模型的示例路径
        client_model_paths = [f"/Users/wangjiayi/Desktop/Server_FU/SIFU-Code/saved_models/CIFAR10/20_client_model_epoch_{i}.pt" for i in range(FL_params.N_client)]
        global_model_paths = [f"./saved_models/CIAFR10/20_global_model_epoch_{i}.pt" for i in range(FL_params.global_epoch)]

        # 加载客户端模型
        # all_client_models = []
        # for path in client_model_paths:
        #     client_model = ResNet18(num_classes=10)  # 假设您使用的是 ResNet18，可以替换为具体模型类
        #     client_model.load_state_dict(torch.load(path, weights_only=True))
        #     all_client_models.append(client_model)

        # 定义和加载全局模型
        # global_model = ResNet18(num_classes=10)#MLP()  # 或者 ResNet18(num_classes=10)，请使用与保存时一致的模型定义
        # global_model.load_state_dict(torch.load(global_model_paths[-1], weights_only=True))  # 加载最后一轮的全局模型
        global_model.load_state_dict(torch.load('//saved_models/CIFAR10/20_global_model_epoch_19.pt', weights_only=True))  # 加载最后一轮的全局模型

        # 定义并加载所有客户端模型
        all_client_models = []
        for client_path in client_model_paths[:FL_params.N_client]:  # 仅加载指定数量的客户端模型
            client_model = model_class()#ResNet18(num_classes=10)#MLP()  # 或者 ResNet18(num_classes=10)
            client_model.load_state_dict(torch.load(client_path, weights_only=True))
            all_client_models.append(client_model)

        # 如果希望仅在特定训练轮次加载客户端模型，可以调整切片
        # 例如：all_client_models = all_client_models[FL_params.global_epoch-1::FL_params.global_epoch-1][:FL_params.N_client]

        # 确保 all_client_models 包含预期数量的客户端模型
        all_client_models = all_client_models[:FL_params.N_client]

        # Benchmark data handling for accuracy evaluation
        # if dataset_name == "CelebA":
        #     X_test, y_test = next(iter(DataLoader(test_dataset, batch_size=len(test_dataset))))
        # else:
        #     X_test = test_dataset.data
        #     y_test = test_dataset.targets
        #     # 如果 y_test 是一个 list，转换为 Tensor
        #     y_test = torch.tensor(y_test)
        #     # 使用 numpy 进行处理
        #     labels = np.unique(y_test.numpy())
        #
        #     # Reshape based on dataset
        #     if input_shape == (28 * 28,):
        #         X_test = X_test.view(-1, *input_shape) / 255.0
        #     elif len(X_test.shape) == 3:
        #         X_test = X_test.unsqueeze(1) / 255.0
        #     elif len(X_test.shape) == 4:
        #         # 如果 X_test 是 numpy 数组，先转换为 PyTorch Tensor
        #         X_test = torch.tensor(X_test)
        #         X_test = X_test.permute(0, 3, 1, 2) / 255.0
        #     else:
        #         raise ValueError(f"Unknown dataset shape: {X_test.shape}")

        # Creating benchmark dataset
        # benchmark_data = (X_test, y_test)
        # print(clients_data)
        #todo
        n_samples = {}
        for client_id, data_loader in clients_data.items():
            label_counts = {}
            for _, labels in data_loader:
                labels_np = labels.numpy()
                unique_labels, counts = np.unique(labels_np, return_counts=True)
                for label, count in zip(unique_labels, counts):
                    if label not in label_counts:
                        label_counts[label] = 0
                    label_counts[label] += count
            n_samples[client_id] = label_counts
        for client_id, counts in n_samples.items():
            print(f"Client {client_id} label distribution: {counts}")
        #
        # # print("n_samples:", n_samples)
        #
        # 低质量数据识别
        print('\n')
        print(5 * "#" + "  Federated Unlearning Identifying Start  " + 5 * "#")
        std_unlearnidf_time = time.time()
        # n_samples = {client_id: {label: np.sum(y_client.numpy() == label) for label in np.unique(y_test.numpy())}
        #              for client_id, (_, y_client) in clients_data.items()}

        low_quality_clients, low_quality_datasets, good_quality_datasets = identify_low_quality_data(
            global_model, all_client_models, n_samples, test_loader,
            client_data_dict=clients_data, dataset_name=dataset_name
        )

        end_unlearnidf_time = time.time()
        time_unlearnidf = (std_unlearnidf_time - end_unlearnidf_time)
        print(" Unlearning identifying time consuming = {} secods".format(-time_unlearnidf))
        print(5 * "#" + "  Federated Unlearning Identifying End  " + 5 * "#")

        # low_quality_labels={1, 2, 3, 5, 6, 7, 9}
        # low_quality_clients=[0, 1, 2, 3, 4, 5, 6, 7, 8]

        # 检查是否存在低质量客户端
        print('\n')
        print(5 * "#" + "  Federated Unlearning Start  " + 5 * "#")
        std_unlearn_time = time.time()
        # low_quality_clients=[0, 2, 6, 8]
        if len(low_quality_clients) > 0:
            print(f"{dataset_name} 数据集的低质量客户端:", low_quality_clients)

            # 执行低质量数据遗忘和全局模型更新
            global_model_updated = unlearn_low_quality_data(
                global_model, all_client_models, low_quality_datasets, good_quality_datasets,
                clients_with_low_quality=low_quality_clients, learning_rate=0.001,
                early_stop_thresh=0.5, batch_size=batch_size, client_data_count=n_samples
            )
            print(f"全局模型在 {dataset_name} 数据集上已更新")
        else:
            print(f"在 {dataset_name} 数据集上未检测到低质量客户端，无需更新全局模型。")

        print(f"完成 {dataset_name} 数据集的训练和低质量数据识别。")
        end_unlearn_time = time.time()
        time_unlearn = (std_unlearn_time - end_unlearn_time)
        print(" Unlearning time consuming = {} secods".format(-time_unlearn))
        print(5 * "#" + "  Federated Unlearning End  " + 5 * "#")

    print('#'*7, '注入后门','#'*7)

    # 对低质量客户端的数据集注入后门
    clients_data_bd=clients_data
    for client_id in low_quality_clients:
        target_label = 0  # 设置为攻击的目标标签
        pattern = torch.ones((1, 3, 3))  # 设置后门模式，这里为3x3的白色方块
        clients_data_bd[client_id] = inject_backdoor(clients_data_bd[client_id], target_label, pattern)
    print('#' * 7, 'BD training', '#' * 7)
    for epoch in range(FL_params.global_epoch):
        # Deep copy of client models to save their pre-trained state
        client_models_backup = [copy.deepcopy(global_model_updated) for _ in range(FL_params.N_client)]

        # Train each client model using global_train_once function
        # 创建后门测试集
        backdoor_test_loader = create_backdoor_test_set(test_loader, target_label=0, pattern=torch.ones((1, 3, 3)))
        clientbd_models = global_train_once(global_model_updated, clients_data_bd, backdoor_test_loader, FL_params)

        # Aggregate the models from clients to update the global model using FedAvg or a similar method
        globalbd_model = fedavg(clientbd_models)

        # Logging and storing the global model
        print(f"BD Global Federated Learning epoch = {epoch}")
        # all_global_models.append(copy.deepcopy(global_model))
        _, acc = test(globalbd_model, backdoor_test_loader)

        # correct, total = 0, 0
        # with torch.no_grad():
        #     for images, labels in backdoor_test_loader:
        #         outputs = globalbd_model(images)
        #         _, predicted = torch.max(outputs, 1)
        #         total += labels.size(0)
        #         correct += (predicted == labels).sum().item()

        print(f'后门攻击成功率: {acc}')

        # Save a snapshot of the global model for unlearning


if __name__=='__main__':
    Sever_Initial_Unlearning()