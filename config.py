import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import copy
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
import warnings

# config.py
class Arguments():
    def __init__(self):
        #Federated Learning Settings
        self.N_total_client = 10
        self.N_client = 5#10# purchase, cifar10, MNIST, adult
        self.data_config = {}
        self.global_epoch = 20
        self.local_epoch = 5
        self.retrain_epoch=25
        self.data_name = 'MNIST'
        
        self.in_channels = 1
        self.output_size = 28
        
        self.unlearn_class = 0

        #Model Training Settings
        self.local_batch_size = 32
        self.local_lr = 0.005
        
        self.test_batch_size = 64
        self.seed = 100
        self.save_all_model = True
        self.cuda_state = torch.cuda.is_available()
        self.use_gpu = True
        self.train_with_test = False
        
        
        #Federated Unlearning Settings
        self.unlearn_interval= 1#Used to control how many rounds the model parameters are saved.1 represents the parameter saved once per round  N_itv in our paper.
        self.forget_client_idx = 2 #If want to forget, change None to the client index
        
                                #If this parameter is set to False, only the global model after the final training is completed is output
        self.if_retrain = False#If set to True, the global model is retrained using the FL-Retrain function, and data corresponding to the user for the forget_client_IDx number is discarded.
        
        self.if_unlearning = False#If set to False, the global_train_once function will not skip users that need to be forgotten;If set to True, global_train_once skips the forgotten user during training
        
        self.forget_local_epoch_ratio = 0.5 #When a user is selected to be forgotten, other users need to train several rounds of on-line training in their respective data sets to obtain the general direction of model convergence in order to provide the general direction of model convergence.
                                            #forget_local_epoch_ratio*local_epoch Is the number of rounds of local training when we need to get the convergence direction of each local model
        # self.mia_oldGM = False
        self.p = 0.3
        # self.opt = 0
        # self.opt_out = [-1]
        # self.opt_out_class=None
        self.n_data = 1000
        self.n_data_test = 500
        self.n_data_val = 1000
        self.overlap = True


class FederatedConfig:
    def __init__(self):
        # 基础设置
        self.num_clients = 10
        self.num_rounds = 100
        self.local_epochs = 1#5
        self.batch_size = 32
        
        # 模型参数
        self.learning_rate = 0.001
        self.momentum = 0.9
        self.weight_decay = 1e-4
        
        # 遗忘参数
        self.target_classes = [0]
        self.forget_rate = 0.2
        self.kl_temperature = 1.0
        self.at_beta = 250.0
        
        # W2 stability参数
        self.rho_threshold = 0.1
        self.w2_lambda = 0.1
        
        # 通信参数
        self.clients_per_round = 5
        self.aggregation_freq = 5
        
        # 生成器参数
        self.z_dim = 128
        self.generator_lr = 0.0002
        self.discriminator_lr = 0.0002
        
        # 评估参数
        self.eval_freq = 5
        self.privacy_threshold = 1.0