import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import copy
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import warnings
# import torchtext
import torchvision
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
import torchvision.models as models
from typing import Callable, Dict, List, Optional, Tuple, Union
import os
import itertools
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from torch.utils.data import Dataset,TensorDataset
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
from scipy.sparse import load_npz
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, SVHN
# import torchtext

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV

import torchvision.datasets as dsets
from torchvision import transforms, datasets
from PIL import Image

class FederatedDataset:
    def __init__(self, dataset_name,num_classes, num_clients, batch_size=32, target_classes=None):
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.target_classes = target_classes if target_classes else [0]
        
        # 数据转换
        self.transform = self._get_transforms()
        
        # 加载数据集
        self.train_dataset, self.test_dataset = self._load_dataset()
        
        # 为每个客户端分配数据
        self.client_dataloaders = self._create_client_dataloaders()
        
    def _get_transforms(self):
        """获取数据转换"""
        if self.dataset_name in ['MNIST', 'FashionMNIST']:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif self.dataset_name == 'CIFAR10':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                  (0.2023, 0.1994, 0.2010))
            ])
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def _load_dataset(self):
        """加载数据集"""
        if self.dataset_name == 'MNIST':
            train_dataset = datasets.MNIST('./data', train=True, download=True,
                                        transform=self.transform)
            test_dataset = datasets.MNIST('./data', train=False,
                                       transform=self.transform)
        elif self.dataset_name == 'CIFAR10':
            train_dataset = datasets.CIFAR10('./data', train=True, download=True,
                                          transform=self.transform)
            test_dataset = datasets.CIFAR10('./data', train=False,
                                         transform=self.transform)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
            
        return train_dataset, test_dataset
    
    def _create_client_dataloaders(self):
        """为每个客户端创建数据加载器"""
        # 按类别分割数据
        class_indices = self._split_by_class()
        
        # 为每个客户端分配数据
        client_dataloaders = []
        samples_per_client = len(self.train_dataset) // self.num_clients
        
        for i in range(self.num_clients):
            # 随机选择每个类别的样本
            client_indices = []
            for class_idx in range(10):  # 假设10个类别
                start_idx = i * (len(class_indices[class_idx]) // self.num_clients)
                end_idx = (i + 1) * (len(class_indices[class_idx]) // self.num_clients)
                client_indices.extend(class_indices[class_idx][start_idx:end_idx])
            
            # 创建子数据集和数据加载器
            client_dataset = Subset(self.train_dataset, client_indices)
            client_dataloader = DataLoader(
                client_dataset, 
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
            client_dataloaders.append(client_dataloader)
            
        return client_dataloaders
    
    def _split_by_class(self):
        """按类别分割数据集"""
        class_indices = [[] for _ in range(10)]  # 假设10个类别
        for idx, (_, label) in enumerate(self.train_dataset):
            class_indices[label].append(idx)
        return [np.array(indices) for indices in class_indices]
    
    def _split_val_by_class(self):
        """按类别分割数据集"""
        class_indices = [[] for _ in range(10)]  # 假设10个类别
        for idx, (_, label) in enumerate(self.test_dataset):
            class_indices[label].append(idx)
        return [np.array(indices) for indices in class_indices]
    
    def get_client_dataloader(self, client_id):
        """获取指定客户端的数据加载器"""
        return self.client_dataloaders[client_id]
    
    def get_test_dataloader(self):
        """获取测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
def get_transform(dataset_name):
    dataset_config = {
    'MNIST': {'size': (28, 28), 'channels': 1},
    'FashionMNIST': {'size': (28, 28), 'channels': 1},
    'CIFAR10': {'size': (32, 32), 'channels': 3},
    'CelebA': {'size': (28, 28), 'channels': 3},
    'AG_NEWS': {'size': None, 'channels': None}  # AG_NEWS 不涉及图像
    }
    
    config = dataset_config[dataset_name]
    transform = transforms.Compose([
    transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 与训练时保持一致
    ])
    # if config['size']:
    #     transform = transforms.Compose([
    #         transforms.Resize(config['size']),  # 调整为目标大小
    #         transforms.ToTensor()
            
    #     ])
    # else:
    #     transform = None  # 文本数据集无需预处理
    return transform

def load_dataset(dataset_name, batch_size):
    transform = get_transform(dataset_name)
    
    if dataset_name == 'MNIST':
        train_ds = dsets.MNIST(root='./data', train=True, download=True, transform=transform)
        valid_ds = dsets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes, in_channels = 10, 1
        train_dl = DataLoader(train_ds, batch_size, shuffle=True)
        valid_dl = DataLoader(valid_ds, batch_size)

    elif dataset_name == 'CIFAR10':
        train_ds, valid_ds = cifar10()
        train_ds = dsets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        valid_ds = dsets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes, in_channels = 10, 3
        train_dl = DataLoader(train_ds, batch_size, shuffle=True)
        valid_dl = DataLoader(valid_ds, batch_size)

    elif dataset_name == 'FashionMNIST':
        train_ds = dsets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        valid_ds = dsets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        num_classes, in_channels = 10, 1
        train_dl = DataLoader(train_ds, batch_size, shuffle=True)
        valid_dl = DataLoader(valid_ds, batch_size)

    elif dataset_name == 'CelebA':
        train_ds = dsets.CelebA(root='./data', split='train', download=True, transform=transform)
        valid_ds = dsets.CelebA(root='./data', split='test', download=True, transform=transform)
        num_classes, in_channels = 2, 3  # CelebA 是二分类
        train_dl = DataLoader(train_ds, batch_size, shuffle=True)
        valid_dl = DataLoader(valid_ds, batch_size)

    # elif dataset_name == 'AGNews':
    #     from torchtext.datasets import AG_NEWS
    #     from torchtext.data.utils import get_tokenizer
    #     from torchtext.vocab import build_vocab_from_iterator

    #     tokenizer = get_tokenizer("basic_english")
        
    #     def yield_tokens(data_iter):
    #         for _, text in data_iter:
    #             yield tokenizer(text)
    #     def process_text(text, vocab, tokenizer):
    #         tokens = tokenizer(text)
    #         indices = [vocab[token] for token in tokens]
    #         return torch.tensor(indices, dtype=torch.long)
    #     def collate_batch(batch):
    #         labels, texts = [], []
    #         for label, text in batch:
    #             labels.append(label - 1)  # Adjust labels to 0-based index
    #             texts.append(process_text(text, vocab, tokenizer))
    #         labels = torch.tensor(labels, dtype=torch.long)
    #         texts = nn.utils.rnn.pad_sequence(texts, batch_first=True)  # Pad sequences to the same length
    #         return labels, texts

    #     vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    #     vocab.set_default_index(vocab["<unk>"])
    #     vocab_size = len(vocab)
        
        # train_ds, valid_ds = AG_NEWS()
        # train_dl = DataLoader(list(train_ds), batch_size=256, shuffle=True, collate_fn=collate_batch)
        # test_dl = DataLoader(list(valid_ds), batch_size=256, shuffle=False, collate_fn=collate_batch)
        # num_classes, in_channels = 4, 1
    else:
        raise ValueError("Unsupported dataset!")

    
    classwise_train = {}

    for i in range(num_classes):
        classwise_train[i] = []
    for img, label in train_ds:
        classwise_train[label].append((img, label))
        
    classwise_test = {}
    for i in range(num_classes):
        classwise_test[i] = []
    for img, label in valid_ds:
        classwise_test[label].append((img, label))
        
    return train_ds,valid_ds,train_dl, valid_dl, num_classes, in_channels, classwise_train, classwise_test

def load_and_partition_data(dataset_name,dataset_train, dataset_test,trainloader,testloader, FL_params):
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
        dataset_test = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
        dict_users, dict_users_val, dict_users_test = cifar_noniid2(
            dataset_train, dataset_test, FL_params.type_count_1st, FL_params.p,
            FL_params.n_data, FL_params.n_data_val, FL_params.n_data_test, FL_params.overlap)
    elif dataset_name == 'ImageNet':
        
        imagenet_train_path = './data/imagenet/train/'
        imagenet_test_path = './data/imagenet/test/images/'
        
        transform = transforms.Compose([
            transforms.Resize(256),            # 调整图像尺寸
            transforms.CenterCrop(224),        # 中心裁剪
            transforms.ToTensor(),             # 转为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 归一化
        ])

        # 加载ImageNet数据集
        dataset_train = torchvision.datasets.ImageFolder(root=imagenet_train_path, transform=transform)
        # dataset_test = torchvision.datasets.ImageFolder(root=imagenet_test_path, transform=transform)
        imagenet_test_path = './data/imagenet/test/images/'
        test_images = [os.path.join(imagenet_test_path, img) for img in os.listdir(imagenet_test_path)]
        # train_labels = dataset_train.classes
        # 假设文件路径
        wnids_file_path = './data/imagenet/wnids.txt'
        words_file_path = './data/imagenet/words.txt'

        # 加载wnids和words
        wnids = load_wnids(wnids_file_path)
        word_dict = load_words(words_file_path)

        # 获取标签
        train_labels = get_labels_from_wnids(wnids, word_dict)
        # print('trainLabels:', train_labels)
        dataset_test = CustomTestDataset(test_images, transform=transform, classes=train_labels)

        dict_users, dict_users_val, dict_users_test = imagenet_noniid2(
            dataset_train, dataset_test, FL_params.type_count_1st, FL_params.p,
            FL_params.n_data, FL_params.n_data_val, FL_params.n_data_test, FL_params.overlap)

    elif dataset_name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
        dataset_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
        dict_users, dict_users_val, dict_users_test = mnist_noniid2(
            dataset_train, dataset_test, FL_params.N_client, FL_params.p,
            FL_params.n_data, FL_params.n_data_val, FL_params.n_data_test, FL_params.overlap)

    elif dataset_name == 'FashionMNIST':
        dataset_train, dataset_test,_,_ = data_set('fashion')
        dict_users, dict_users_val, dict_users_test = cifar_noniid2(
            dataset_train, dataset_test, FL_params.type_count_1st, FL_params.p,
            FL_params.n_data, FL_params.n_data_val, FL_params.n_data_test, FL_params.overlap)

    elif dataset_name == 'purchase':
        dict_users, dict_users_val, dict_users_test = noniid_partition(
            dataset_train, dataset_test, FL_params.type_count_2nd, FL_params.p,
            FL_params.n_data, FL_params.n_data_val, FL_params.n_data_test, FL_params.overlap)

    elif dataset_name == 'adult':
        dict_users, dict_users_val, dict_users_test = noniid_partition(
            dataset_train, dataset_test, FL_params.type_count_2nd, FL_params.p,
            FL_params.n_data, FL_params.n_data_val, FL_params.n_data_test, FL_params.overlap)
    elif dataset_name == 'SVHN':
        dataset_train, dataset_test,trainloader,testloader = get_svhn()
        dict_users, dict_users_val, dict_users_test,partitioned_train,partitioned_test=vision_train_valid_test_partition(random_seed=FL_params.seed, dataset=dataset_train, num_clients=FL_params.N_client,sampling_type="noniid")
        return dataset_train, dataset_test, dict_users, dict_users_val, dict_users_test,partitioned_train,partitioned_test,trainloader,testloader

    elif dataset_name == 'AGNEWS':
        TEXT = torchtext.data.Field(sequential=True, tokenize=lambda x: x.split())
        LABEL = torchtext.data.LabelField(is_target=True)
        datafields = [('text', TEXT), ('label', LABEL)]
        dataset_train = read_data(
            '/data/agnews/train.csv',
            datafields, label_column=0, doc_start=2)
        dataset_test = read_data(
            '/data/agnews/test.csv',
            datafields, label_column=0, doc_start=2)
        TEXT.build_vocab(dataset_train, max_size=10000)
        LABEL.build_vocab(dataset_test)

        labels = [LABEL.vocab.stoi[l] for l in dataset_train.label]
        labels_test = [LABEL.vocab.stoi[l.split(',')[0]] for l in dataset_test.label]
        dict_users, dict_users_val, dict_users_test = agnews_noniid2(labels, labels_test, FL_params.N_client, FL_params.p,
                                                                     FL_params.n_data, FL_params.n_data_val,
                                                                     FL_params.n_data_test, FL_params.overlap)

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return dataset_train, dataset_test, dict_users, dict_users_val, dict_users_test

def remove_class_from_users(dict_users, dataset, target_class_index):
    """
    从每个用户的数据集中移除指定类别的样本。

    Args:
        dict_users (dict): 每个用户的数据集索引字典
        dataset (Dataset): 原始数据集
        target_class_index (int): 需要移除的类别标签

    Returns:
        dict: 更新后的 dict_users
        dict: 被移除的数据索引
    """
    updated_dict_users = {}
    unlearn_dict_users = {}
    
    # 根据数据集类型正确获取标签
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'train_labels'):
        labels = np.array(dataset.train_labels)
    else:
        raise AttributeError("Dataset does not have 'targets' or 'train_labels' attribute")

    # 遍历每个用户的数据集
    for user_id, user_data_indices in dict_users.items():
        # 获取该用户数据集的标签
        user_labels = labels[user_data_indices]
        
        # 找出不属于目标类别的样本索引
        remaining_mask = user_labels != target_class_index
        remaining_indices = user_data_indices[remaining_mask]
        
        # 找出属于目标类别的样本索引
        unlearn_mask = user_labels == target_class_index
        unlearn_indices = user_data_indices[unlearn_mask]

        # 更新用户数据集
        updated_dict_users[user_id] = remaining_indices
        unlearn_dict_users[user_id] = unlearn_indices

    return updated_dict_users, unlearn_dict_users

def create_dataloaders_from_indices(dataset, updated_dict_users, unlearn_dict_users, batch_size=256, shuffle=True):
    """
    根据 updated_dict_users 和 unlearn_dict_users 创建新的数据集和 DataLoader。

    Args:
        dataset (Dataset): 原始数据集。
        updated_dict_users (dict): 包含用户索引的字典，用于生成更新后的数据集。
        unlearn_dict_users (dict): 包含被遗忘的数据索引的字典。
        batch_size (int, optional): 每个 DataLoader 的批次大小。默认为 32。
        shuffle (bool, optional): 是否对 DataLoader 中的数据进行随机打乱。默认为 True。

    Returns:
        dict: 每个用户的更新后的 DataLoader。
        dict: 每个用户的遗忘数据 DataLoader。
    """
    updated_dataloaders = {}
    unlearn_dataloaders = {}

    for user_id, indices in updated_dict_users.items():
        # 创建更新后的数据集
        updated_subset = Subset(dataset, indices)
        updated_dataloader = DataLoader(updated_subset, batch_size=batch_size, shuffle=shuffle)
        updated_dataloaders[user_id] = updated_dataloader
    for user_id, indices in unlearn_dict_users.items():
        # 确保 indices 是一个列表
        if isinstance(indices, (int, np.integer)):
            indices = [indices]
        elif isinstance(indices, np.ndarray):
            if indices.ndim == 0:
                indices = [indices.item()]
            else:
                indices = indices.tolist()
        unlearn_subset = torch.utils.data.Subset(dataset, indices)
        unlearn_dataloader = DataLoader(unlearn_subset, batch_size=16, shuffle=shuffle)
        unlearn_dataloaders[user_id] = unlearn_dataloader

    # for user_id, indices in unlearn_dict_users.items():
    #     # 创建遗忘数据集
    #     unlearn_subset = torch.utils.data.Subset(dataset, indices)
    #     unlearn_dataloader = DataLoader(unlearn_subset, batch_size=1, shuffle=shuffle)
    #     unlearn_dataloaders[user_id] = unlearn_dataloader

    return updated_dataloaders, unlearn_dataloaders

def mnist_noniid2(dataset, dataset_test, num_users, p, n_data, n_data_val, n_data_test, overlap):
    """
    Create a non-IID partitioning of MNIST data for federated learning.

    Args:
        dataset (Dataset): The MNIST training dataset.
        dataset_test (Dataset): The MNIST testing dataset.
        num_users (int): The number of users (clients) to distribute the data to.
        p (float): The proportion of data for each user that should come from the majority classes.
        n_data (int): The number of training data samples per user.
        n_data_val (int): The number of validation data samples per user.
        n_data_test (int): The number of testing data samples per user.
        overlap (bool): Flag to determine if overlapping majority classes are allowed among users.

    Returns:
        dict: Training data indices for each user.
        dict: Validation data indices for each user.
        dict: Testing data indices for each user.
    """
    # Initialize indices and labels for training data
    idxs = np.arange(len(dataset), dtype=int)
    labels = dataset.train_labels.numpy()
    label_list = np.unique(labels)

    # Sort indices by labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :].astype(int)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # Initialize indices and labels for testing data
    idxs_test = np.arange(len(dataset_test), dtype=int)
    labels_test = dataset_test.targets.numpy()
    label_list_test = np.unique(labels_test)

    # Sort indices by labels
    idxs_labels_test = np.vstack((idxs_test, labels_test))
    idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
    idxs_test = idxs_labels_test[0, :].astype(int)

    dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}
    dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}

    # Define class labels and combinations
    num_classes = len(label_list)
    user_majority_labels = []

    for i in range(num_users):
        # Sample majority class for each user
        if overlap:
            majority_labels = list(itertools.product(range(num_classes), repeat=2))[i]
        else:
            majority_labels = np.random.choice(label_list, 2, replace=False)

        user_majority_labels.append(majority_labels)

        # Assign training, validation, and testing data
        # Training set
        majority_labels1_idxs = idxs[labels[idxs] == majority_labels[0]]
        majority_labels2_idxs = idxs[labels[idxs] == majority_labels[1]]
        
        # print(f"majority_labels1_idxs: {len(majority_labels1_idxs)}, majority_labels2_idxs: {len(majority_labels2_idxs)}")

        sub_data_idxs1 = np.random.choice(majority_labels1_idxs, int(p * n_data / 2), replace=False)
        sub_data_idxs2 = np.random.choice(majority_labels2_idxs, int(p * n_data / 2), replace=False)
        # print(f"majority_labels1_idxs: {len(sub_data_idxs1)}, majority_labels2_idxs: {len(sub_data_idxs2)}")


        dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs1, sub_data_idxs2))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs1) - set(sub_data_idxs2)))

        # Validation set
        majority_labels1_idxs = idxs[labels[idxs] == majority_labels[0]]
        majority_labels2_idxs = idxs[labels[idxs] == majority_labels[1]]

        sub_data_idxs1_val = np.random.choice(majority_labels1_idxs, int(p * n_data_val / 2), replace=True)
        sub_data_idxs2_val = np.random.choice(majority_labels2_idxs, int(p * n_data_val / 2), replace=True)
        # print(f"majority_labels1_idxs: {len(sub_data_idxs1_val)}, majority_labels2_idxs: {len(sub_data_idxs2_val)}")

        # if len(majority_labels1_idxs) < int(p * n_data_val / 2):
        #     sub_data_idxs1_val = majority_labels1_idxs/2  # 分配所有剩余样本
        #     sub_data_idxs2_val = majority_labels2_idxs/2
        # else:
        #     sub_data_idxs1_val = np.random.choice(majority_labels1_idxs, int(p * n_data_val / 2), replace=False)
        #     sub_data_idxs2_val = np.random.choice(majority_labels2_idxs, int(p * n_data_val / 2), replace=True)
        
        dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs1_val, sub_data_idxs2_val))
        idxs = np.array(list(set(idxs) - set(sub_data_idxs1_val) - set(sub_data_idxs2_val)))
                
    
        # Testing set
        majority_labels1_idxs_test = idxs_test[labels_test[idxs_test] == majority_labels[0]]
        majority_labels2_idxs_test = idxs_test[labels_test[idxs_test] == majority_labels[1]]

        sub_data_idxs1_test = np.random.choice(majority_labels1_idxs_test, int(p * n_data_test / 2), replace=False)
        sub_data_idxs2_test = np.random.choice(majority_labels2_idxs_test, int(p * n_data_test / 2), replace=False)

        dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs1_test, sub_data_idxs2_test))

    # Distribute non-majority label data
    if p < 1.0:
        for i in range(num_users):
            if len(idxs) >= n_data:
                majority_labels = user_majority_labels[i]
                # Training set
                non_majority_label_idxs = idxs[(labels[idxs] != majority_labels[0]) & (labels[idxs] != majority_labels[1])]
                sub_data_idxs = np.random.choice(non_majority_label_idxs, int((1 - p) * n_data), replace=False)
                dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs))
                idxs = np.array(list(set(idxs) - set(sub_data_idxs)))

                # Validation set
                non_majority_label_idxs = idxs[(labels[idxs] != majority_labels[0]) & (labels[idxs] != majority_labels[1])]
                sub_data_idxs_val = np.random.choice(non_majority_label_idxs, int((1 - p) * n_data_val), replace=True)
                dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs_val))
                idxs = np.array(list(set(idxs) - set(sub_data_idxs_val)))

                # Testing set
                non_majority_label_idxs_test = idxs_test[(labels_test[idxs_test] != majority_labels[0]) & (labels_test[idxs_test] != majority_labels[1])]
                sub_data_idxs_test = np.random.choice(non_majority_label_idxs_test, int((1 - p) * n_data_test), replace=False)
                dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs_test))
            else:
                dict_users[i] = np.concatenate((dict_users[i], idxs))
                dict_users_test[i] = np.concatenate((dict_users_test[i], idxs_test))
                # Debugging information to verify data distribution
    # for i in range(num_users):
    #     print(
    #         f"User {i}: mnist Training samples {len(dict_users[i])}, Validation samples {len(dict_users_val[i])}, Testing samples {len(dict_users_test[i])}")
    # print_user_data_distribution(dict_users, dataset, num_users)

    # target_class_to_remove = 0  # 例如，移除类别 0 的所有样本
    # updated_dict_users = remove_class_from_users(dict_users, dataset, target_class_to_remove)
    # print_user_data_distribution(updated_dict_users, dataset, num_users)

    return dict_users, dict_users_val, dict_users_test
# def mnist_noniid2(dataset, dataset_test, num_users, p, n_data, n_data_val, n_data_test, overlap):
#     """
#     Create a non-IID partitioning of MNIST data for federated learning.

#     Args:
#         dataset (Dataset): The MNIST training dataset.
#         dataset_test (Dataset): The MNIST testing dataset.
#         num_users (int): The number of users (clients) to distribute the data to.
#         p (float): The proportion of data for each user that should come from the majority classes.
#         n_data (int): The number of training data samples per user.
#         n_data_val (int): The number of validation data samples per user.
#         n_data_test (int): The number of testing data samples per user.
#         overlap (bool): Flag to determine if overlapping majority classes are allowed among users.

#     Returns:
#         dict: Training data indices for each user.
#         dict: Validation data indices for each user.
#         dict: Testing data indices for each user.
#     """
#     # Initialize indices and labels for training data
#     idxs = np.arange(len(dataset), dtype=int)
#     labels = dataset.train_labels.numpy()
#     label_list = np.unique(labels)

#     # Sort indices by labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :].astype(int)

#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

#     # Initialize indices and labels for testing data
#     idxs_test = np.arange(len(dataset_test), dtype=int)
#     labels_test = dataset_test.targets.numpy()
#     label_list_test = np.unique(labels_test)

#     # Sort indices by labels
#     idxs_labels_test = np.vstack((idxs_test, labels_test))
#     idxs_labels_test = idxs_labels_test[:, idxs_labels_test[1, :].argsort()]
#     idxs_test = idxs_labels_test[0, :].astype(int)

#     dict_users_test = {i: np.array([], dtype='int64') for i in range(num_users)}
#     dict_users_val = {i: np.array([], dtype='int64') for i in range(num_users)}

#     # Define class labels and combinations
#     num_classes = len(label_list)
#     user_majority_labels = []

#     for i in range(num_users):
#         # Sample majority class for each user
#         if overlap:
#             majority_labels = list(itertools.product(range(num_classes), repeat=2))[i]
#         else:
#             majority_labels = np.random.choice(label_list, 2, replace=False)

#         user_majority_labels.append(majority_labels)

#         # Training set
#         majority_labels1_idxs = idxs[labels[idxs] == majority_labels[0]]
#         majority_labels2_idxs = idxs[labels[idxs] == majority_labels[1]]
        
#         # print(f"majority_labels1_idxs: {len(majority_labels1_idxs)}, majority_labels2_idxs: {len(majority_labels2_idxs)}")

#         sub_data_idxs1 = np.random.choice(majority_labels1_idxs, int(p * n_data / 2), replace=False)
#         sub_data_idxs2 = np.random.choice(majority_labels2_idxs, int(p * n_data / 2), replace=False)
#         # print(f"majority_labels1_idxs: {len(sub_data_idxs1)}, majority_labels2_idxs: {len(sub_data_idxs2)}")


#         dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs1, sub_data_idxs2))
#         idxs = np.array(list(set(idxs) - set(sub_data_idxs1) - set(sub_data_idxs2)))

#         # Validation set
#         majority_labels1_idxs = idxs[labels[idxs] == majority_labels[0]]
#         majority_labels2_idxs = idxs[labels[idxs] == majority_labels[1]]
        
#         # 确保验证集样本数量符合要求
#         samples_per_class_val = int(p * n_data_val / 2)
        
#         # 如果数据不足，使用替换采样
#         sub_data_idxs1_val = np.random.choice(
#             majority_labels1_idxs, 
#             samples_per_class_val, 
#             replace=len(majority_labels1_idxs) < samples_per_class_val
#         )
#         sub_data_idxs2_val = np.random.choice(
#             majority_labels2_idxs, 
#             samples_per_class_val, 
#             replace=len(majority_labels2_idxs) < samples_per_class_val
#         )
        
#         dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs1_val, sub_data_idxs2_val))
        
#         # 只在不使用替换采样时从可用索引中移除
#         if len(majority_labels1_idxs) >= samples_per_class_val:
#             idxs = np.array(list(set(idxs) - set(sub_data_idxs1_val)))
#         if len(majority_labels2_idxs) >= samples_per_class_val:
#             idxs = np.array(list(set(idxs) - set(sub_data_idxs2_val)))

#         # Testing set
#         majority_labels1_idxs_test = idxs_test[labels_test[idxs_test] == majority_labels[0]]
#         majority_labels2_idxs_test = idxs_test[labels_test[idxs_test] == majority_labels[1]]

#         sub_data_idxs1_test = np.random.choice(majority_labels1_idxs_test, int(p * n_data_test / 2), replace=False)
#         sub_data_idxs2_test = np.random.choice(majority_labels2_idxs_test, int(p * n_data_test / 2), replace=False)

#         dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs1_test, sub_data_idxs2_test))

#     # Distribute non-majority label data
#     if p < 1.0:
#         for i in range(num_users):
#             majority_labels = user_majority_labels[i]
#             non_majority_label_idxs = idxs[(labels[idxs] != majority_labels[0]) & 
#                                          (labels[idxs] != majority_labels[1])]
            
#             # 验证集的非主要类别数据
#             non_majority_samples_val = int((1 - p) * n_data_val)
#             sub_data_idxs_val = np.random.choice(
#                 non_majority_label_idxs, 
#                 non_majority_samples_val, 
#                 replace=len(non_majority_label_idxs) < non_majority_samples_val
#             )
#             dict_users_val[i] = np.concatenate((dict_users_val[i], sub_data_idxs_val))
            
#             # 只在不使用替换采样时从可用索引中移除
#             if len(non_majority_label_idxs) >= non_majority_samples_val:
#                 idxs = np.array(list(set(idxs) - set(sub_data_idxs_val)))

#             # 训练集的非主要类别数据
#             non_majority_samples_train = int((1 - p) * n_data)
#             sub_data_idxs_train = np.random.choice(
#                 non_majority_label_idxs, 
#                 non_majority_samples_train, 
#                 replace=len(non_majority_label_idxs) < non_majority_samples_train
#             )
#             dict_users[i] = np.concatenate((dict_users[i], sub_data_idxs_train))
            
#             # 只在不使用替换采样时从可用索引中移除
#             if len(non_majority_label_idxs) >= non_majority_samples_train:
#                 idxs = np.array(list(set(idxs) - set(sub_data_idxs_train)))

#             # 测试集的非主要类别数据
#             non_majority_samples_test = int((1 - p) * n_data_test)
#             sub_data_idxs_test = np.random.choice(
#                 non_majority_label_idxs, 
#                 non_majority_samples_test, 
#                 replace=len(non_majority_label_idxs) < non_majority_samples_test
#             )
#             dict_users_test[i] = np.concatenate((dict_users_test[i], sub_data_idxs_test))
            
#             # 只在不使用替换采样时从可用索引中移除
#             if len(non_majority_label_idxs) >= non_majority_samples_test:
#                 idxs = np.array(list(set(idxs) - set(sub_data_idxs_test)))

#     return dict_users, dict_users_val, dict_users_test
