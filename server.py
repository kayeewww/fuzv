import copy
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from scipy.stats import wasserstein_distance
from unlearn import Generator, Discriminator, LearnableLoaderWithRealData
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets
import pandas as pd
import os
import torchvision
import torchvision.transforms as tt
import tarfile
import os
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms


from rho import calculate_rho_f, extract_features
from unlearn import KT_loss_generator, KT_loss_student
from utils import evaluate
from utils import evaluate_model


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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator_path = "./ckpts/test_mnist_allcnn/generator"
student_path = "./ckpts/test_mnist_allcnn/student"

os.makedirs(generator_path,exist_ok=True)
os.makedirs(student_path,exist_ok=True)

idx_pseudo = 0
total_n_pseudo_batches = 4000
n_pseudo_batches = 0
running_gen_loss = []
running_stu_loss = []

threshold = 0.01
def mnist(root = './'):
    transform = tt.Compose([
        tt.ToTensor(),
    ])
    
    train_ds = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform)
    valid_ds = torchvision.datasets.MNIST(root='./', train=False, download=True, transform=transform)
    
    return train_ds, valid_ds

# # Number of classes for the classification task
# num_classes = 10
# # Generate a uniform distribution for 10 classes
# uniform_distribution = np.full(num_classes, 1 / num_classes)
# print("Uniform distribution for 10 classes:", uniform_distribution,type(uniform_distribution))

class FederatedServer:
    def __init__(self, model, num_clients, target_classes, retain_classes):
        self.global_model = model
        self.num_clients = num_clients
        self.target_classes = target_classes
        self.retain_classes = retain_classes
        self.rho_s = 0.0
        self.client_models = []
        self.generator = None
        self.KL_temperature = 1.0
        self.AT_beta = 250.0
        self.rho_t = 0.0

    def broadcast_model(self):
        """广播全局模型到客户端"""
        print('Server: Broadcasting model...')
        return copy.deepcopy(self.global_model)
    
    # def calculate_global_sensitivity(self, client_gradients):
    #     """计算全局敏感度 rho_s"""
    #     print('Server: Calculating global sensitivity...')
    #     all_grads = torch.stack([grad.flatten() for grad in client_gradients])
    #     wasserstein_distances = []
        
    #     # 计算梯度之间的Wasserstein距离
    #     for i in range(len(all_grads)):
    #         for j in range(i+1, len(all_grads)):
    #             dist = wasserstein_distance(all_grads[i].cpu().numpy(), all_grads[j].cpu().numpy())
    #             wasserstein_distances.append(dist)
        
    #     self.rho_s = np.mean(wasserstein_distances)
    #     print('rho_s: ',self.rho_s)
    #     return self.rho_s

    def update_global_model(self, client_updates, adaptive_weights=None):
        """更新全局模型"""
        print('Server: Updating global model...')
        if adaptive_weights is None:
            adaptive_weights = [1/len(client_updates)] * len(client_updates)
            
        # 聚合模型参数
        aggregated_dict = {}
        for key in client_updates[0].keys():
            aggregated_dict[key] = sum(
                w * update[key].to('cuda') for w, update in zip(adaptive_weights, client_updates)
            )
        
        self.global_model.load_state_dict(aggregated_dict)
        return self.global_model

    def perform_forgetting(self, student, rho_s, optimizer,target_valid_dl,retain_valid_dl,FL_params):
        """执行遗忘过程"""
        print('Server: Performing forgetting...')
        
        in_channels=FL_params.in_channels
        output_size=FL_params.output_size
        device = next(student.parameters()).device  # 获取student模型所在的设备
        
        train_ds,valid_ds,train_dl, valid_dl, num_classes, in_channels, classwise_train, classwise_test = load_dataset('MNIST', batch_size=256)
        generator = LearnableLoaderWithRealData(valid_ds, n_repeat_batch=20, num_channels=in_channels,output_size=output_size, device=device)
        discriminator = Discriminator().to(device)
        self.generator = generator
        
        
        # 训练生成器生成遗忘类别的伪数据
        forgetting_loss = self._train_generator(
            student, generator, discriminator, 
            self.target_classes, optimizer, rho_s,target_valid_dl,retain_valid_dl,FL_params
        )
        ##todo 查看一下本轮的rho_t，动态调整k和b
        
        return forgetting_loss
    
    # def _calculate_w2_loss(self, generated_data, rho_s):
    #     """
    #     计算Wasserstein-2距离损失
        
    #     Args:
    #         generated_data: 生成器生成的数据 [batch_size, channels, height, width]
    #         rho_s: 服务器端的W2-stability参数
    #     """
    #     features = self.global_model(generated_data)[1:]  # 获取中间层特征
    #     w2_distances = []
        
    #     for feature in features:
    #         # 将特征展平为二维张量 [batch_size, feature_dim]
    #         flat_features = feature.view(feature.size(0), -1)
            
    #         # 创建目标均匀分布
    #         batch_size, feature_dim = flat_features.size()
    #         uniform_dist = torch.ones(batch_size, feature_dim).to(device) / feature_dim
            
    #         # 对特征和均匀分布进行排序
    #         sorted_features, _ = torch.sort(flat_features, dim=1)
    #         sorted_uniform, _ = torch.sort(uniform_dist, dim=1)
            
    #         # 计算W2距离: W2 = (∫|F^{-1}(t) - G^{-1}(t)|^2 dt)^{1/2}
    #         # 这里F^{-1}和G^{-1}由排序后的向量表示
    #         w2_dist = torch.sqrt(((sorted_features - sorted_uniform) ** 2).mean(dim=1)).mean()
    #         w2_distances.append(w2_dist)
        
    #     # 综合所有层的W2距离
    #     w2_loss = sum(w2_distances) / len(w2_distances)
        
    #     # 加入rho_s的影响
    #     w2_stability_loss = rho_s * w2_loss
        
    #     return w2_stability_loss

    # def _exact_wasserstein_2d(self, distribution1, distribution2):
    #     """
    #     计算二维分布间的精确W2距离
        
    #     Args:
    #         distribution1, distribution2: [batch_size, feature_dim] 的分布
    #     """
    #     # 确保输入形状相同
    #     assert distribution1.shape == distribution2.shape
        
    #     # 计算经验分布的均值和协方差
    #     mu1 = distribution1.mean(dim=0)
    #     mu2 = distribution2.mean(dim=0)
        
    #     sigma1 = torch.mm((distribution1 - mu1).t(), (distribution1 - mu1)) / distribution1.size(0)
    #     sigma2 = torch.mm((distribution2 - mu2).t(), (distribution2 - mu2)) / distribution2.size(0)
        
    #     # 计算W2距离的平方
    #     # W2^2 = ||mu1 - mu2||^2 + tr(sigma1 + sigma2 - 2(sigma1^(1/2)sigma2sigma1^(1/2))^(1/2))
    #     diff_means = (mu1 - mu2).pow(2).sum()
        
    #     # 计算sigma1的平方根
    #     eigenvalues1, eigenvectors1 = torch.linalg.eigh(sigma1)
    #     sqrt_sigma1 = torch.mm(
    #         torch.mm(eigenvectors1, torch.diag(torch.sqrt(torch.clamp(eigenvalues1, min=1e-7)))),
    #         eigenvectors1.t()
    #     )
        
    #     # 计算中间项
    #     middle_term = torch.mm(torch.mm(sqrt_sigma1, sigma2), sqrt_sigma1)
    #     eigenvalues_middle, _ = torch.linalg.eigh(middle_term)
    #     sqrt_middle = torch.sqrt(torch.clamp(eigenvalues_middle, min=1e-7)).sum()
        
    #     # 计算总的W2距离
    #     w2_dist = diff_means + torch.trace(sigma1) + torch.trace(sigma2) - 2 * sqrt_middle
    #     w2_dist = torch.sqrt(torch.clamp(w2_dist, min=1e-7))
        
    #     return w2_dist

    # def _calculate_w2_batch(self, generated_features, target_dist):
    #     """
    #     批量计算W2距离
    #     """
    #     batch_size = generated_features.size(0)
    #     w2_distances = []
        
    #     for i in range(batch_size):
    #         w2_dist = self._exact_wasserstein_2d(
    #             generated_features[i:i+1].repeat(batch_size, 1),
    #             target_dist
    #         )
    #         w2_distances.append(w2_dist)
        
    #     return torch.stack(w2_distances).mean()
    
    # def compute_feature_alignment_loss(real_features, pseudo_features):
    #     """特征对齐损失"""
    #     min_batch_size = min(real_features[0].size(0), pseudo_features[0].size(0))
    #     real_features = [f[:min_batch_size].view(min_batch_size, -1) for f in real_features]
    #     pseudo_features = [f[:min_batch_size].view(min_batch_size, -1) for f in pseudo_features]
        
    #     return sum([F.mse_loss(pf, rf.detach()) for pf, rf in zip(pseudo_features, real_features)])


    def _train_generator(self, student, generator, discriminator, target_classes, optimizer, rho_s,forget_valid_dl,retain_valid_dl,FL_params):
        """训练生成器,保留原有功能并加入W2-stability"""
        # 基础设置
        device = next(student.parameters()).device
        base_KL_temperature = 1.0
        base_AT_beta = 250.0
        threshold = 0.01
        
        idx_pseudo=0
        n_generator_iter = 1
        n_student_iter = 10
        total_iters = FL_params.global_epoch
        optimizer_generator = torch.optim.Adam(generator.generator.parameters(), lr=0.001)
        scheduler_generator = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_generator, mode='min', factor=0.5, patience=2, verbose=True)
        
        optimizer_student = torch.optim.Adam(student.parameters(), lr=0.001)
        scheduler_student = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_student, mode='min', factor=0.5, patience=2, verbose=True)

        running_gen_loss = []
        running_stu_loss = []
        results = []
        n_repeat_batch = n_generator_iter + n_student_iter
        n_pseudo_batches = 0
        
        model = self.global_model
        
        for iter_idx in range(total_iters):
            # 1. 获取数据
            real_batch = next(iter(retain_valid_dl))[0].to(device)
            x_pseudo = generator.__next__().to(device)
            
            # 2. 过滤目标类样本
            model_output, *_ = student(x_pseudo)#[0]
            mask = (torch.softmax(model_output.detach(), dim=1)[:, 0] <= threshold)
            x_pseudo = x_pseudo[mask]
            if x_pseudo.size(0) == 0:
                # print('No pseudo data generated')
                # if x_pseudo.size(0) == 0:
                # print(f"No pseudo samples passed the filter at iteration {iter_idx}. Regenerating...")
                x_pseudo = generator.__next__().to(device)
                continue
                
            # 3. 计算W2-stability参数
            # 计算当前生成的伪数据在student模型上与均匀分布的W2距离
            w2_current = self._compute_w2_to_uniform(student, x_pseudo)
            # 计算上一轮保存的模型在相同伪数据上与均匀分布的W2距离
            w2_previous = self._compute_w2_to_uniform(self.global_model, x_pseudo)
            rho_t = abs(w2_current - w2_previous)*10
            self.rho_t = rho_t
            
            # 4. 动态调整参数
            KL_temp = base_KL_temperature * (1 + 0.5 * torch.tanh(torch.tensor(rho_t)))
            AT_beta = base_AT_beta * (1 - 0.5 * torch.tanh(torch.tensor(rho_t)))
            
            KL_temp = torch.clamp(KL_temp, 0.8, 2.0)
            AT_beta = torch.clamp(AT_beta, 200, 300)
            print(f'Round {iter_idx}: rho_t: {rho_t}, KL_temp: {KL_temp}, AT_beta: {AT_beta}')
            
            # 5. 训练生成器
            if iter_idx % (n_generator_iter + n_student_iter) < n_generator_iter:
                # 提取特征
                student_logits, *student_activations = student(x_pseudo)
                teacher_logits, *teacher_activations = model(x_pseudo)
                generator_total_loss = KT_loss_generator(student_logits, teacher_logits, KL_temp)
                
                # GAN损失
                real_pred = discriminator(real_batch)
                fake_pred = discriminator(x_pseudo)
                g_loss_adv = F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))
                
                # 蒸馏损失 - 使用动态温度
                KL_temperature = KL_temp
                kd_loss = KT_loss_generator(student_logits, teacher_logits, KL_temp)
                
                # W2-stability正则化
                w2_loss = rho_t
                
                # 生成器基本损失
                # base_loss = self.compute_base_generator_loss()
                # 计算生成数据在目标类上的预测分布与均匀分布的W2距离
                w2_dist = self._compute_w2_to_uniform(self.global_model, x_pseudo)
                # print('w2_dist: ', w2_dist)
                # W2损失鼓励生成使模型对目标类预测接近均匀分布的数据
                # total_loss = kd_loss + lambda_w2 * w2_dist
                total_loss = kd_loss +  w2_dist
                
                # 4. 总生成器损失
                total_generator_loss = kd_loss+ 0.1 * g_loss_adv # + 0.1 * feature_loss + 0.1 * g_loss_adv

                # 总损失
                # total_loss = kd_loss + 0.1 * feature_loss + 0.1 * g_loss_adv + 0.01 * w2_loss
                optimizer_generator.zero_grad()
                total_generator_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 5)
                optimizer_generator.step()
                
                running_gen_loss.append(total_generator_loss.cpu().detach())
                
                
                # running_gen_loss.append(total_loss.item())
                
            # 6. 训练学生模型
            elif idx_pseudo % iter_idx < (n_generator_iter + n_student_iter):
                # student_logits, *student_activations = student(x_pseudo)
                # with torch.no_grad():
                #     teacher_logits, *teacher_activations = self.global_model(x_pseudo)
                
                # # 学生模型损失 - 使用动态参数
                # student_loss = KT_loss_student_dynamic(
                #     student_logits, student_activations,
                #     teacher_logits, teacher_activations,
                #     rho_c_local=rho_t,  # 使用W2-stability参数
                #     rho_s_global=rho_s,
                #     retain_classes=self.retain_classes,
                #     target_classes=target_classes,
                #     base_KL_temperature=KL_temp,
                #     base_AT_beta=AT_beta
                # )
                
                # optimizer.zero_grad()
                # student_loss.backward()
                # optimizer.step()
                
                # running_stu_loss.append(student_loss.item())
                print('Student Iteration:', idx_pseudo % n_repeat_batch)
                with torch.no_grad():
                    teacher_logits, *teacher_activations = model(x_pseudo)

                student_logits, *student_activations = student(x_pseudo)
                student_total_loss = KT_loss_student(student_logits, student_activations, 
                                                    teacher_logits, teacher_activations, 
                                                    KL_temperature=KL_temp, AT_beta = AT_beta)
                # student_loss = KT_loss_student_dynamic(
                # student_logits, student_activations, teacher_logits, teacher_activations,
                #     rho_f_forget=rho_f_forget, rho_f_robust=rho_f_robust
                # )
                running_stu_loss.append(student_total_loss.item())

                # 加入真实数据微调
                # n_real_steps=2
                # student.train()
                # if history_retain["Acc"] * 100 < 90:
                # if n_pseudo_batches > 0 and history_retain["Acc"] * 100 < 90:
                #     print("Real Data Finetuning...")
                #     for _ in range(n_real_steps):  # 控制微调步数
                #         for real_images, real_labels in retain_valid_dl:
                #             real_images, real_labels = real_images.to(device), real_labels.to(device)


                #             # 使用真实数据计算 KT_loss_student
                #             student_loss = KT_loss_student(
                #                 student_logits, student_activations, 
                #                 teacher_logits, teacher_activations,
                #                 KL_temp,AT_beta
                #             )

                #             # 优化步骤
                #             optimizer_student.zero_grad()
                #             student_loss.backward()
                #             torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
                #             optimizer_student.step()
                #             running_stu_loss.append(student_loss.cpu().detach())

                # else:
                #     # student_logits = student(x_pseudo)[0]
                #     student_loss = KT_loss_student_dynamic(
                #     student_logits, student_activations, teacher_logits, teacher_activations,
                #         rho_f_forget=rho_f_forget, rho_f_robust=rho_f_robust
                #     )
                # running_stu_loss.append(student_loss.item())
                
            
                
            # 7. 评估和记录
            history_retain={'Loss': 0, 'Acc': 0}
            if iter_idx % 10 == 0:
                print('Evaluating and Saving')
                MeanGLoss = np.mean(running_gen_loss)
                MeanSLoss = np.mean(running_stu_loss)
                running_gen_loss, running_stu_loss = [], []
                
                target_test_acc, target_test_loss,target_class_wise_acc, target_class_wise_loss = evaluate_model(student, forget_valid_dl, device)
                retain_test_acc, retain_test_loss,retain_class_wise_acc, retain_class_wise_loss = evaluate_model(student, retain_valid_dl, device)

                # history_forget = evaluate(student, forget_valid_dl, device=device)
                # history_retain = evaluate(student, retain_valid_dl, device=device)

                metrics = pd.DataFrame([{
                    "Epochs": n_pseudo_batches,
                    "AccForget": target_class_wise_acc[FL_params.unlearn_class],#target_test_acc, #history_forget["Acc"] * 100,
                    "ErrForget": target_class_wise_loss[FL_params.unlearn_class],#target_test_loss, #history_forget["Loss"],
                    "AccRetain": retain_test_acc, #history_retain["Acc"] * 100,
                    "ErrRetain": retain_test_loss, #history_retain["Loss"],
                    "MeanGeneratorLoss": MeanGLoss,
                    "MeanStudentLoss": MeanSLoss
                }])
                results.append({
                    "Epochs": n_pseudo_batches,
                    "AccForget": target_class_wise_acc[FL_params.unlearn_class],#target_test_acc,
                    "ErrForget": target_class_wise_loss[FL_params.unlearn_class],#target_test_loss,
                    "AccRetain": retain_test_acc,
                    "ErrRetain": retain_test_loss,
                    "MeanGeneratorLoss": MeanGLoss,
                    "MeanStudentLoss": MeanSLoss
                })
                AccForget_0=target_class_wise_acc[FL_params.unlearn_class]
                AccRetain_0=target_class_wise_loss[FL_params.unlearn_class]
                ErrForget_0=target_test_loss
                ErrRetain_0=retain_test_loss
                df = pd.DataFrame(columns = ["Epochs", "AccForget", "AccRetain", "ErrForget", "ErrRetain", "MeanGeneratorLoss", "MeanStudentLoss"])
                metrics = pd.DataFrame([{"Epochs":0, "AccForget":AccForget_0, "AccRetain":AccRetain_0, "ErrForget":ErrForget_0, 
                    "ErrRetain":ErrRetain_0, "MeanGeneratorLoss":None, "MeanStudentLoss":None}])
    

                df = pd.concat([df, metrics], ignore_index=True)
                print(df.iloc[-1:])

                # 调度器步进
                scheduler_student.step(retain_test_loss)
                scheduler_generator.step(target_test_loss)
            
                # 保存模型
                torch.save(generator.state_dict(), os.path.join(generator_path, f"{n_pseudo_batches}.pt"))
                torch.save(student.state_dict(), os.path.join(student_path, f"{n_pseudo_batches}.pt"))
                # torch.save(discriminator.state_dict(), os.path.join(discriminator_path, f"{n_pseudo_batches}.pt"))
                n_pseudo_batches += 1
                # metrics = {
                #     'iter': iter_idx,
                #     'generator_loss': np.mean(running_gen_loss),
                #     'student_loss': np.mean(running_stu_loss),
                #     'w2_stability': rho_t,
                #     'KL_temp': KL_temp.item(),
                #     'AT_beta': AT_beta.item()
                # }
                # results.append(metrics)
                # print('server results:',results)
                # running_gen_loss = []
                # running_stu_loss = []
                
        return results

    def _compute_w2_to_uniform(self, model, x):
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
        
    def check_global_convergence(self, client_convergence_status):
        """检查全局遗忘是否收敛"""
        # 检查所有提出遗忘请求的客户端是否都达到收敛
        all_converged = all(client_convergence_status.values())
        
        # 服务器端生成器的收敛性
        generator_stable = self.check_generator_convergence()
        
        return all_converged and generator_stable

    # def _train_generator(self, student, generator, discriminator, target_classes, optimizer, rho_s):
    #     """训练生成器"""
    #     base_KL_temperature = 1.0
    #     base_AT_beta = 250.0
        
    #     output_size = dataset_config['MNIST']['size']
    #     threshold = 0.01
    #     transform = get_transform('MNIST')
    #     device = next(student.parameters()).device

    #     # 数据加载和初始化
    #     valid_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    #     valid_dl = DataLoader(valid_ds, batch_size=256)
    #     generator = LearnableLoaderWithRealData(valid_dl.dataset, n_repeat_batch=2, num_channels=1, 
    #                                         output_size=output_size, device=device)
    #     # self.generator = generator
        
    #     # 优化器设置
    #     optimizer_generator = torch.optim.Adam(generator.generator.parameters(), lr=0.001)
    #     scheduler_generator = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer_generator, mode='min', factor=0.5, patience=2, verbose=True)
        
    #     optimizer_student = torch.optim.Adam(student.parameters(), lr=0.001)
    #     scheduler_student = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #         optimizer_student, mode='min', factor=0.5, patience=2, verbose=True)
    
        
    #     discriminator = Discriminator(input_channels=1).to(device)
    #     optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    #     print('Server: Training generator...')
    #     model = self.global_model
        
    #     total_g_loss = 0
    #     idx_pseudo = 0
    #     total_n_pseudo_batches = 100#40#00
    #     n_pseudo_batches = 0
    #     n_generator_iter = 1
    #     n_student_iter = 10
    #     n_repeat_batch = n_generator_iter + n_student_iter
    #     running_gen_loss=[]
    #     running_stu_loss=[]

    #     while n_pseudo_batches < total_n_pseudo_batches:
    #         # 生成伪样本
    #         idx_pseudo = 0
    #         total_n_pseudo_batches = 4000
    #         # 获取真实数据的 batch size
    #         real_images, _ = next(iter(valid_dl))
    #         batch_size_real = real_images.size(0)
    #         real_images = real_images.to(device)
    #         model = model.to(device)

    #         generator.batch_size = batch_size_real
    #         x_pseudo = generator.__next__().to(device=device)
            
    #         preds, *_ = model(x_pseudo)
    #         mask = (torch.softmax(preds.detach(), dim=1)[:, 0] <= threshold)
    #         x_pseudo = x_pseudo[mask]
    #         if x_pseudo.size(0) == 0:
    #             zero_count += 1
    #             if zero_count > 100:
    #                 print("Generator Stopped Producing datapoints corresponding to retain classes.")
    #                 print("Resetting the generator to previous checkpoint")
    #                 # generator.load_state_dict(torch.load(os.path.join(generator_path, str(((n_pseudo_batches//50)-1)*50) + ".pt")))
    #             continue
    #         else:
    #             zero_count = 0

    #         # generator.adjust_parameters(rho_f_forget, rho_f_robust)
    #         if idx_pseudo % n_repeat_batch < n_generator_iter:
    #             student_logits, *student_activations = student(x_pseudo)
    #             teacher_logits, *teacher_activations = model(x_pseudo)
                
    #             real_features = extract_features(model, real_images)
    #             pseudo_features = extract_features(model, x_pseudo)

    #             # 修正 batch size 和特征形状不一致
    #             min_batch_size = min(real_features[0].size(0), pseudo_features[0].size(0))
    #             real_features = [f[:min_batch_size].view(min_batch_size, -1) for f in real_features]
    #             pseudo_features = [f[:min_batch_size].view(min_batch_size, -1) for f in pseudo_features]

    #             # 确保特征维度匹配
    #             for i, (rf, pf) in enumerate(zip(real_features, pseudo_features)):
    #                 assert rf.shape == pf.shape, f"Shape mismatch at Layer {i}: Real {rf.shape}, Pseudo {pf.shape}"

    #             # 1. 计算特征对齐损失（MSE）
    #             feature_loss = sum([
    #                 F.mse_loss(pseudo_features[i], real_features[i])
    #                 for i in range(len(real_features))
    #             ])

    #             # 2. 对抗损失（GAN Loss）
    #             real_pred = discriminator(real_images)  # 判别器对真实样本的输出
    #             fake_pred = discriminator(x_pseudo)    # 判别器对生成样本的输出
    #             g_loss_adv = F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))  # 欺骗判别器

    #             # 3. 蒸馏损失（KL 散度）
    #             KL_temperature = 1.0
                
    #             generator_total_loss = KT_loss_generator(student_logits, teacher_logits, KL_temperature=KL_temperature)

    #             optimizer_generator.zero_grad()
    #             generator_total_loss.backward()
    #             torch.nn.utils.clip_grad_norm_(generator.parameters(), 5)
    #             optimizer_generator.step()
    #             running_gen_loss.append(generator_total_loss.cpu().detach())
    #             print("Generator Loss:", {running_gen_loss})


    #         elif idx_pseudo % n_repeat_batch < (n_generator_iter + n_student_iter):
    #             with torch.no_grad():
    #                 teacher_logits = model(x_pseudo)[0]
    #             student_output = student(x_pseudo)
    #             student_logits = student_output[0] if isinstance(student_output, tuple) else student_output
    #             student_activations = student_output[1:] if isinstance(student_output, tuple) else []
                
    #             teacher_output = model(x_pseudo)
    #             teacher_logits = teacher_output[0] if isinstance(teacher_output, tuple) else teacher_output
    #             teacher_activations = teacher_output[1:] if isinstance(teacher_output, tuple) else []
                
    #             # Getting the forget and retain data
    #             train_ds, valid_ds = mnist()
    #             batch_size=256
    #             num_classes = 10
    #             classwise_train = {}
    #             for i in range(num_classes):
    #                 classwise_train[i] = []

    #             for img, label in train_ds:
    #                 classwise_train[label].append((img, label))
                    
    #             classwise_test = {}
    #             for i in range(num_classes):
    #                 classwise_test[i] = []

    #             for img, label in valid_ds:
    #                 classwise_test[label].append((img, label))
                    
    #             forget_valid = []
    #             forget_classes = [0]
    #             for cls in range(num_classes):
    #                 if cls in forget_classes:
    #                     for img, label in classwise_test[cls]:
    #                         forget_valid.append((img, label))

    #             retain_valid = []
    #             for cls in range(num_classes):
    #                 if cls not in forget_classes:
    #                     for img, label in classwise_test[cls]:
    #                         retain_valid.append((img, label))

    #             forget_valid_dl = DataLoader(forget_valid, batch_size, num_workers=3, pin_memory=True)

    #             retain_valid_dl = DataLoader(retain_valid, batch_size, num_workers=3, pin_memory=True)
                
    #             # 加入真实数据微调
    #             n_real_steps=2
    #             student.train()
    #             if n_pseudo_batches > 0 and history_retain["Acc"] * 100 < 90:
    #                 print("Real Data Finetuning...")
    #                 for _ in range(n_real_steps):  # 控制微调步数
    #                     for real_images, real_labels in retain_valid_dl:  # 使用 retain_valid_dl 数据
    #                         real_images, real_labels = real_images.to(device), real_labels.to(device)

    #                         # 提取特征和输出
    #                         with torch.no_grad():
    #                             teacher_output = model(real_images)  # 教师模型输出
    #                             teacher_logits = teacher_output[0]
    #                             teacher_activations = teacher_output[1:]  # 提取中间激活

    #                         # 学生模型输出
    #                         student_output = student(real_images)
    #                         student_logits = student_output[0]
    #                         student_activations = student_output[1:]  # 提取中间激活

    #                         # 动态调整 KL 温度和 AT_beta
    #                         base_KL_temperature = 1.0
    #                         base_AT_beta = 250.0
    #                         adjusted_rho_f_forget = rho_f_forget * 0.5  # 动态缩放 rho_f_forget
    #                         adjusted_rho_f_robust = rho_f_robust * 0.5  # 动态缩放 rho_f_robust

    #                         # 使用真实数据计算 KT_loss_student
    #                         student_loss = KT_loss_student_dynamic(
    #                             student_logits, student_activations, 
    #                             teacher_logits, teacher_activations,
    #                             rho_f_forget=adjusted_rho_f_forget, 
    #                             rho_f_robust=adjusted_rho_f_robust,
    #                             base_KL_temperature=base_KL_temperature, 
    #                             base_AT_beta=base_AT_beta
    #                         )

    #                         # 优化步骤
    #                         optimizer_student.zero_grad()
    #                         student_loss.backward()
    #                         optimizer_student.step()
                
    #             # with torch.no_grad():
    #             #     teacher_logits, *teacher_activations = model(x_pseudo)

    #             # student_logits, *student_activations = student(x_pseudo)
    #             # student_total_loss = KT_loss_student(student_logits, student_activations, 
    #             #                                     teacher_logits, teacher_activations, 
    #             #                                     KL_temperature=KL_temperature, AT_beta = AT_beta)

    #             # optimizer_student.zero_grad()
    #             # student_total_loss.backward()
    #             # torch.nn.utils.clip_grad_norm_(student.parameters(), 5)
    #             # optimizer_student.step()
    #             running_stu_loss.append(student_loss.cpu().detach())
        
    #         # # 生成器训练步骤
    #         # if idx_pseudo % n_repeat_batch < n_generator_iter:
    #         #     print('Generator Iteration:', idx_pseudo % n_repeat_batch)

    #         #     # 提取学生和教师的 logits
    #         #     student_logits = student(x_pseudo)[0]
    #         #     teacher_logits = model(x_pseudo)[0]

    #         #     # 提取真实样本和伪样本的特征
    #         #     real_features = extract_features(model, real_images)
    #         #     pseudo_features = extract_features(model, x_pseudo)

    #         #     # 修正 batch size 和特征形状不一致
    #         #     min_batch_size = min(real_features[0].size(0), pseudo_features[0].size(0))
    #         #     real_features = [f[:min_batch_size].view(min_batch_size, -1) for f in real_features]
    #         #     pseudo_features = [f[:min_batch_size].view(min_batch_size, -1) for f in pseudo_features]

    #         #     # 确保特征维度匹配
    #         #     for i, (rf, pf) in enumerate(zip(real_features, pseudo_features)):
    #         #         assert rf.shape == pf.shape, f"Shape mismatch at Layer {i}: Real {rf.shape}, Pseudo {pf.shape}"

    #         #     # 1. 计算特征对齐损失（MSE）
    #         #     feature_loss = sum([
    #         #         F.mse_loss(pseudo_features[i], real_features[i])
    #         #         for i in range(len(real_features))
    #         #     ])

    #         #     # 2. 对抗损失（GAN Loss）
    #         #     real_pred = discriminator(real_images)  # 判别器对真实样本的输出
    #         #     fake_pred = discriminator(x_pseudo)    # 判别器对生成样本的输出
    #         #     g_loss_adv = F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))  # 欺骗判别器

    #         #     # 3. 蒸馏损失（KL 散度）
    #         #     KL_temperature = 1.0
    #         #     distillation_loss = KT_loss_generator(student_logits, teacher_logits, KL_temperature)

    #         #     # 4. 总生成器损失
    #         #     total_generator_loss = distillation_loss + 0.1 * feature_loss + 0.1 * g_loss_adv

    #         #     # 优化生成器
    #         #     optimizer_generator.zero_grad()
    #         #     total_generator_loss.backward()
    #         #     optimizer_generator.step()

    #         #     # 记录损失
    #         #     running_gen_loss.append(total_generator_loss.item())

    #         #     # 打印损失项
    #         #     print(f"Generator Loss: {total_generator_loss.item():.4f} | "
    #         #             f"Feature Loss: {feature_loss.item():.4f} | "
    #         #             f"GAN Loss: {g_loss_adv.item():.4f} | "
    #         #             f"Distillation Loss: {distillation_loss.item():.4f}")
    #         # # 学生模型训练步骤
    #         # elif idx_pseudo % n_repeat_batch < (n_generator_iter + n_student_iter):
    #         #     print('Student Iteration:', idx_pseudo % n_repeat_batch)
    #         #     with torch.no_grad():
    #         #         teacher_logits = model(x_pseudo)[0]
    #         #     student_output = student(x_pseudo)
    #         #     student_logits = student_output[0] if isinstance(student_output, tuple) else student_output
    #         #     student_activations = student_output[1:] if isinstance(student_output, tuple) else []
                
    #         #     teacher_output = model(x_pseudo)
    #         #     teacher_logits = teacher_output[0] if isinstance(teacher_output, tuple) else teacher_output
    #         #     teacher_activations = teacher_output[1:] if isinstance(teacher_output, tuple) else []
                
    #         #     # 加入真实数据微调
    #         #     n_real_steps=2
    #         #     student.train()
    #         #     if n_pseudo_batches > 0 and history_retain["Acc"] * 100 < 90:
    #         #         print("Real Data Finetuning...")
    #         #         for _ in range(n_real_steps):  # 控制微调步数
    #         #             for real_images, real_labels in retain_valid_dl:  # 使用 retain_valid_dl 数据
    #         #                 real_images, real_labels = real_images.to(device), real_labels.to(device)

    #         #                 # 提取特征和输出
    #         #                 with torch.no_grad():
    #         #                     teacher_output = model(real_images)  # 教师模型输出
    #         #                     teacher_logits = teacher_output[0]
    #         #                     teacher_activations = teacher_output[1:]  # 提取中间激活

    #         #                 # 学生模型输出
    #         #                 student_output = student(real_images)
    #         #                 student_logits = student_output[0]
    #         #                 student_activations = student_output[1:]  # 提取中间激活

    #         #                 # 动态调整 KL 温度和 AT_beta
    #         #                 base_KL_temperature = 1.0
    #         #                 base_AT_beta = 250.0
    #         #                 adjusted_rho_f_forget = rho_f_forget * 0.5  # 动态缩放 rho_f_forget
    #         #                 adjusted_rho_f_robust = rho_f_robust * 0.5  # 动态缩放 rho_f_robust

    #         #                 # 使用真实数据计算 KT_loss_student
    #         #                 student_loss = KT_loss_student(
    #         #                     student_logits, student_activations, 
    #         #                     teacher_logits, teacher_activations,
    #         #                     rho_f_forget=adjusted_rho_f_forget, 
    #         #                     rho_f_robust=adjusted_rho_f_robust,
    #         #                     base_KL_temperature=base_KL_temperature, 
    #         #                     base_AT_beta=base_AT_beta
    #         #                 )

    #         #                 # 优化步骤
    #         #                 optimizer_student.zero_grad()
    #         #                 student_loss.backward()
    #         #                 optimizer_student.step()

    #         #     else:
    #         #         student_logits = student(x_pseudo)[0]
    #         #         student_loss = KT_loss_student(
    #         #         student_logits, student_activations, teacher_logits, teacher_activations,
    #         #             rho_f_forget=rho_f_forget, rho_f_robust=rho_f_robust
    #         #         )
    #         #     running_stu_loss.append(student_loss.item())
    #         # 评估与保存
    #         results=[]
    #         if (idx_pseudo + 1) % n_repeat_batch == 0:
    #             if n_pseudo_batches % 1 == 0:
    #                 print('Evaluating and Saving')
    #                 MeanGLoss = np.mean(running_gen_loss)
    #                 MeanSLoss = np.mean(running_stu_loss)
    #                 running_gen_loss, running_stu_loss = [], []

    #                 history_forget = evaluate(student, forget_valid_dl, device=device)
    #                 history_retain = evaluate(student, retain_valid_dl, device=device)

    #                 metrics = pd.DataFrame([{
    #                     "Epochs": n_pseudo_batches,
    #                     "AccForget": history_forget["Acc"] * 100,
    #                     "ErrForget": history_forget["Loss"],
    #                     "AccRetain": history_retain["Acc"] * 100,
    #                     "ErrRetain": history_retain["Loss"],
    #                     "MeanGeneratorLoss": MeanGLoss,
    #                     "MeanStudentLoss": MeanSLoss
    #                 }])
    #                 results.append({
    #                     "Epochs": n_pseudo_batches,
    #                     "AccForget": history_forget["Acc"] * 100,
    #                     "ErrForget": history_forget["Loss"],
    #                     "AccRetain": history_retain["Acc"] * 100,
    #                     "ErrRetain": history_retain["Loss"],
    #                     "MeanGeneratorLoss": MeanGLoss,
    #                     "MeanStudentLoss": MeanSLoss
    #                 })

    #                 df = pd.concat([df, metrics], ignore_index=True)
    #                 print(df.iloc[-1:])

    #                 # 调度器步进
    #                 scheduler_student.step(history_retain["Loss"])
    #                 scheduler_generator.step(history_forget["Loss"])
                    

    #                 # # 保存模型
    #                 # torch.save(generator.state_dict(), os.path.join(generator_path, f"{n_pseudo_batches}.pt"))
    #                 # torch.save(student.state_dict(), os.path.join(student_path, f"{n_pseudo_batches}.pt"))
    #                 # torch.save(discriminator.state_dict(), os.path.join(discriminator_path, f"{n_pseudo_batches}.pt"))


    #             n_pseudo_batches += 1
    #         # 计算 rho_f 并动态调整生成器参数
    #         if n_pseudo_batches !=0:
    #             features_before = extract_features(self.global_model, x_pseudo)
    #             features_after = extract_features(student, x_pseudo)

    #             wasserstein_distances = [
    #                 wasserstein_distance(features_before[i].cpu().flatten(), features_after[i].cpu().flatten())
    #                 for i in range(len(features_before))
    #             ]
    #             rho_f_forget = max(wasserstein_distances)

    #             outputs_before = torch.softmax(self.global_model(x_pseudo)[0], dim=1)
    #             outputs_after = torch.softmax(student(x_pseudo)[0], dim=1)
    #             robustness_changes = torch.norm(outputs_before[:, 1:] - outputs_after[:, 1:], dim=1)
    #             rho_f_robust = robustness_changes.mean()#.item()
                
    #         idx_pseudo += 1

    #     df = pd.DataFrame(results)
    #     print(df.tail())  
    #     df.to_csv(f"{target_classes}_1ALLCNN_server_forget_0.csv", index = False)
    
    #     # for _ in range(5):  # 生成器训练轮次
    #     #     # 获取实际数据
    #     #     real_images, _ = next(iter(valid_dl))
    #     #     batch_size_real = real_images.size(0)
    #     #     real_images = real_images.to(device)
    #     #     model = model.to(device)

    #     #     # 生成伪数据
    #     #     generator.batch_size = batch_size_real
    #     #     x_pseudo = generator.__next__().to(device=device)

    #     #     # 过滤目标类样本
    #     #     with torch.no_grad():
    #     #         model_output = model(x_pseudo)
    #     #         preds = model_output[0] if isinstance(model_output, tuple) else model_output
    #     #         mask = (torch.softmax(preds, dim=1)[:, 0] <= threshold)
    #     #         x_pseudo = x_pseudo[mask]

    #     #         if x_pseudo.size(0) == 0:
    #     #             print("Generator stopped producing valid datapoints. Resetting...")
    #     #             # generator.load_state_dict(torch.load(os.path.join(generator_path, str((n_pseudo_batches // 50 - 1) * 50) + ".pt")))
    #     #             continue

    #     #     # 计算各种损失
    #     #     # 1. 特征对齐损失
    #     #     with torch.no_grad():
    #     #         real_features = extract_features(model, real_images)
    #     #     pseudo_features = extract_features(model, x_pseudo)

    #     #     min_batch_size = min(real_features[0].size(0), pseudo_features[0].size(0))
    #     #     real_features = [f[:min_batch_size].view(min_batch_size, -1) for f in real_features]
    #     #     pseudo_features = [f[:min_batch_size].view(min_batch_size, -1) for f in pseudo_features]

    #     #     feature_loss = sum([
    #     #         F.mse_loss(pf, rf.detach())
    #     #         for pf, rf in zip(pseudo_features, real_features)
    #     #     ])

    #     #     # 2. 对抗损失
    #     #     real_pred = discriminator(real_images)
    #     #     fake_pred = discriminator(x_pseudo)
    #     #     g_loss_adv = F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))

    #     #     # 3. W2 stability损失
    #     #     w2_loss = self._calculate_w2_loss(x_pseudo, rho_s)

    #     #     # 综合损失
    #     #     total_loss = feature_loss + 0.1 * g_loss_adv + 0.01 * w2_loss

    #     #     # 优化步骤
    #     #     optimizer_generator.zero_grad()
    #     #     optimizer_discriminator.zero_grad()
            
    #     #     # 反向传播时保留计算图
    #     #     total_loss.backward(retain_graph=True)
            
    #     #     # 更新生成器
    #     #     optimizer_generator.step()

    #     #     # 训练判别器
    #     #     real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
    #     #     fake_loss = F.binary_cross_entropy(fake_pred.detach(), torch.zeros_like(fake_pred))
    #     #     d_loss = (real_loss + fake_loss) / 2
            
    #     #     optimizer_discriminator.zero_grad()
    #     #     d_loss.backward()
    #     #     optimizer_discriminator.step()

    #     #     total_g_loss += total_loss.item()

    #     # # 更新学习率
    #     # scheduler_generator.step(total_g_loss)
        
    #     return total_g_loss / 5

dataset_config = {
    'MNIST': {'size': (28, 28), 'channels': 1},
    'FashionMNIST': {'size': (28, 28), 'channels': 1},
    'cifar10': {'size': (32, 32), 'channels': 3},
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
def load_dataset(dataset_name, batch_size):
    transform = get_transform(dataset_name)
    
    if dataset_name == 'MNIST':
        train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        valid_ds = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        num_classes, in_channels = 10, 1
        train_dl = DataLoader(train_ds, batch_size, shuffle=True)
        valid_dl = DataLoader(valid_ds, batch_size)

    elif dataset_name == 'CIFAR10':
        # train_ds, valid_ds = cifar10()
        train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        valid_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes, in_channels = 10, 3
        train_dl = DataLoader(train_ds, batch_size, shuffle=True)
        valid_dl = DataLoader(valid_ds, batch_size)

    elif dataset_name == 'FashionMNIST':
        train_ds = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        valid_ds = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        num_classes, in_channels = 10, 1
        train_dl = DataLoader(train_ds, batch_size, shuffle=True)
        valid_dl = DataLoader(valid_ds, batch_size)

    elif dataset_name == 'CelebA':
        train_ds = datasets.CelebA(root='./data', split='train', download=True, transform=transform)
        valid_ds = datasets.CelebA(root='./data', split='test', download=True, transform=transform)
        num_classes, in_channels = 2, 3  # CelebA 是二分类
        train_dl = DataLoader(train_ds, batch_size, shuffle=True)
        valid_dl = DataLoader(valid_ds, batch_size)

    elif dataset_name == 'AGNews':
        from torchtext.datasets import AG_NEWS
        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator

        tokenizer = get_tokenizer("basic_english")
        
        def yield_tokens(data_iter):
            for _, text in data_iter:
                yield tokenizer(text)
        def process_text(text, vocab, tokenizer):
            tokens = tokenizer(text)
            indices = [vocab[token] for token in tokens]
            return torch.tensor(indices, dtype=torch.long)
        def collate_batch(batch):
            labels, texts = [], []
            for label, text in batch:
                labels.append(label - 1)  # Adjust labels to 0-based index
                texts.append(process_text(text, vocab, tokenizer))
            labels = torch.tensor(labels, dtype=torch.long)
            texts = nn.utils.rnn.pad_sequence(texts, batch_first=True)  # Pad sequences to the same length
            return labels, texts

        
        
        train_ds, valid_ds = AG_NEWS()
        train_dl = DataLoader(list(train_ds), batch_size=256, shuffle=True, collate_fn=collate_batch)
        test_dl = DataLoader(list(valid_ds), batch_size=256, shuffle=False, collate_fn=collate_batch)
        num_classes, in_channels = 4, 1
        
        vocab = build_vocab_from_iterator(yield_tokens(train_dl), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        vocab_size = len(vocab)
        
        
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
