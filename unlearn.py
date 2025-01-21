import torch
from torch import nn
from models import View
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from utils import *


def visualize(generated_data, predictions, confidences, num_samples=10):
    plt.figure(figsize=(20, 4))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(generated_data[i][0].cpu().numpy(), cmap='gray')
        plt.title(f'Pred: {predictions[i]}\nConf: {confidences[i]:.2f}')
        plt.axis('off')
    plt.show()

def attention(x):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        """
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def attention_diff(x, y):
    """
    Taken from https://github.com/szagoruyko/attention-transfer
    :param x = activations
    :param y = activations
    """
    return (attention(x) - attention(y)).pow(2).mean()


def divergence(student_logits, teacher_logits, KL_temperature):
    divergence = F.kl_div(F.log_softmax(student_logits / KL_temperature, dim=1), F.softmax(teacher_logits / KL_temperature, dim=1))  # forward KL

    return divergence


def KT_loss_generator(student_logits, teacher_logits, KL_temperature):

    divergence_loss = divergence(student_logits, teacher_logits, KL_temperature)
    total_loss = - divergence_loss

    return total_loss

# def KT_loss_student(student_logits, student_activations, teacher_logits, teacher_activations, 
#                     KL_temperature=1.0, AT_beta=250.0, beta=1.0):
#     """
#     Student loss with dynamic beta scaling for attention loss.
#     :param student_logits: Logits from the student model.
#     :param student_activations: Intermediate activations from the student model.
#     :param teacher_logits: Logits from the teacher model.
#     :param teacher_activations: Intermediate activations from the teacher model.
#     :param KL_temperature: Temperature for KL divergence.
#     :param AT_beta: Weight for attention transfer loss.
#     :param beta: Dynamic weight for controlling the total loss scale.
#     """
#     # Compute divergence loss
#     divergence_loss = F.kl_div(F.log_softmax(student_logits / KL_temperature, dim=1),
#                                F.softmax(teacher_logits / KL_temperature, dim=1), reduction='batchmean')
    
#     # Compute attention transfer loss
#     at_loss = 0
#     if AT_beta > 0:
#         for i in range(len(student_activations)):
#             at_loss += AT_beta * (F.normalize(student_activations[i].pow(2).mean(1).view(student_activations[i].size(0), -1)) -
#                                   F.normalize(teacher_activations[i].pow(2).mean(1).view(teacher_activations[i].size(0), -1))).pow(2).mean()
    
#     # Combine losses with dynamic beta
#     total_loss = beta * (divergence_loss + at_loss)
#     return total_loss
# def KT_loss_student(student_logits, student_activations, teacher_logits, teacher_activations, 
#                     rho_f_forget, rho_f_robust, base_KL_temperature=1.0, base_AT_beta=250.0):
#     """
#     Student loss with dynamic adjustment based on rho_f_forget and rho_f_robust.
    
#     :param student_logits: Logits from the student model.
#     :param student_activations: Intermediate activations from the student model.
#     :param teacher_logits: Logits from the teacher model.
#     :param teacher_activations: Intermediate activations from the teacher model.
#     :param rho_f_forget: Maximum Wasserstein distance (forgetting effect).
#     :param rho_f_robust: Robustness change for non-target classes.
#     :param base_KL_temperature: Base KL temperature.
#     :param base_AT_beta: Base attention transfer loss weight.
#     """

#     # 动态调整 KL 温度和 AT_beta
#     dynamic_factor_forget = 1.0 + 0.1 * rho_f_forget
#     dynamic_factor_robust = 1.0 - 0.1 * rho_f_robust

#     KL_temperature_adjusted = base_KL_temperature * dynamic_factor_forget
#     AT_beta_adjusted = base_AT_beta * dynamic_factor_robust

#     KL_temperature_adjusted = max(0.5, min(5.0, KL_temperature_adjusted))  # 限定 KL 温度范围
#     AT_beta_adjusted = max(50, min(500, AT_beta_adjusted))  # 限定 AT_beta 范围

#     # 计算 KL 散度损失
#     divergence_loss = F.kl_div(
#         F.log_softmax(student_logits / KL_temperature_adjusted, dim=1),
#         F.softmax(teacher_logits / KL_temperature_adjusted, dim=1),
#         reduction='batchmean'
#     )

#     # 计算注意力传输损失
#     at_loss = 0
#     if AT_beta_adjusted > 0:
#         for i in range(len(student_activations)):
#             at_loss += AT_beta_adjusted * (F.normalize(student_activations[i].pow(2).mean(1).view(student_activations[i].size(0), -1)) -
#                                            F.normalize(teacher_activations[i].pow(2).mean(1).view(teacher_activations[i].size(0), -1))).pow(2).mean()

#     # 总损失
#     total_loss = divergence_loss + at_loss

#     # 打印动态调整的参数值
#     # print(f"Adjusted KL_temperature: {KL_temperature_adjusted:.4f}, Adjusted AT_beta: {AT_beta_adjusted:.4f}")
#     return total_loss
def KT_loss_student_dynamic(student_logits, student_activations, teacher_logits, teacher_activations, 
                          rho_c_local, rho_s_global, retain_classes, target_classes,
                          base_KL_temperature=1.0, base_AT_beta=250.0):
    """优化后的动态KD损失函数"""
    # 预计算动态因子 (使用PyTorch操作代替numpy)
    epsilon = 1e-8
    rhos = torch.tensor([rho_c_local, rho_s_global], device=student_logits.device)
    dynamic_factors = torch.tensor([0.5, -0.5], device=student_logits.device) * torch.tanh(rhos) + 1.0

    # 一次性计算alpha值
    alpha = torch.tensor([rho_c_local / (rho_c_local + rho_s_global + epsilon)], 
                        device=student_logits.device).clamp(0.0, 1.0)

    # 一次性计算温度和beta值
    temp_beta = torch.stack([
        (base_KL_temperature * dynamic_factors[0]).clamp(0.8, 2.0),
        (base_AT_beta * dynamic_factors[1]).clamp(200, 300)
    ])

    # 并行处理logits
    current_temp = temp_beta[0]
    forget_logits = student_logits.index_select(1, torch.tensor(target_classes, device=student_logits.device))
    retain_logits = student_logits.index_select(1, torch.tensor(retain_classes, device=student_logits.device))
    
    teacher_retain_logits = teacher_logits.index_select(1, torch.tensor(retain_classes, device=teacher_logits.device))

    # 并行计算KL散度
    target_uniform = torch.full_like(forget_logits, 1.0/len(target_classes))
    
    log_softmax_student_forget = F.log_softmax(forget_logits / current_temp, dim=1)
    log_softmax_student_retain = F.log_softmax(retain_logits / current_temp, dim=1)
    softmax_teacher_retain = F.softmax(teacher_retain_logits / current_temp, dim=1)

    forget_loss = F.kl_div(log_softmax_student_forget, target_uniform, reduction='batchmean')
    retain_loss = F.kl_div(log_softmax_student_retain, softmax_teacher_retain, reduction='batchmean')

    # 优化注意力损失计算
    at_loss = torch.tensor(0., device=student_logits.device)
    if temp_beta[1] > 0:
        # 预先计算所有激活的归一化表示
        student_norms = [F.normalize(act.pow(2).mean(1).flatten(1)) for act in student_activations]
        teacher_norms = [F.normalize(act.pow(2).mean(1).flatten(1)) for act in teacher_activations]
        
        # 并行计算注意力损失
        at_losses = [(s - t).pow(2).mean() for s, t in zip(student_norms, teacher_norms)]
        at_loss = temp_beta[1] * sum(at_losses)

    # 计算最终损失
    total_loss = alpha * forget_loss + (1 - alpha) * retain_loss + 0.1 * at_loss

    return total_loss
# 可以用的
# def KT_loss_student_dynamic(student_logits, student_activations, teacher_logits, teacher_activations, 
#                             rho_c_local, rho_s_global, retain_classes, target_classes,
#                             base_KL_temperature=1.0, base_AT_beta=250.0):
#     """
#     联邦学习中的动态 KD 损失函数，结合动态权重、目标类别遗忘、保留类别知识迁移。
#     """
#     # 动态因子
#     dynamic_factor_forget = 1.0 + 0.5 * torch.tanh(torch.tensor(rho_c_local))
#     dynamic_factor_retain = 1.0 - 0.5 * torch.tanh(torch.tensor(rho_s_global))

#     # 调整 KL 温度和 AT_beta
#     KL_temperature_adjusted = base_KL_temperature * dynamic_factor_forget
#     AT_beta_adjusted = base_AT_beta * dynamic_factor_retain

#     # 限制范围
#     KL_temperature_adjusted = torch.clamp(KL_temperature_adjusted, 0.8, 2.0)
#     AT_beta_adjusted = torch.clamp(AT_beta_adjusted, 200, 300)

#     # 筛选目标类别和保留类别的 logits
#     student_logits_target = student_logits[:, target_classes]
#     teacher_logits_target = teacher_logits[:, target_classes]

#     student_logits_retain = student_logits[:, retain_classes]
#     teacher_logits_retain = teacher_logits[:, retain_classes]

#     # 计算遗忘类别的 KL 损失 - 向均匀分布靠拢
#     target_uniform = torch.ones_like(student_logits_target) / len(target_classes)
#     forget_loss = F.kl_div(
#         F.log_softmax(student_logits_target / KL_temperature_adjusted, dim=1),
#         target_uniform,
#         reduction='batchmean'
#     )

#     # 计算保留类别的 KL 损失 - 知识迁移
#     retain_loss = F.kl_div(
#         F.log_softmax(student_logits_retain / KL_temperature_adjusted, dim=1),
#         F.softmax(teacher_logits_retain / KL_temperature_adjusted, dim=1),
#         reduction='batchmean'
#     )

#     # 注意力传输损失
#     at_loss = 0
#     if AT_beta_adjusted > 0:
#         for i in range(len(student_activations)):
#             at_loss += AT_beta_adjusted * (F.normalize(student_activations[i].pow(2).mean(1).view(student_activations[i].size(0), -1)) -
#                                            F.normalize(teacher_activations[i].pow(2).mean(1).view(teacher_activations[i].size(0), -1))).pow(2).mean()

#     # 动态权重
#     alpha_forget = rho_c_local / (rho_c_local + rho_s_global)
#     alpha_retain = 1 - alpha_forget

#     # 综合损失
#     total_loss = alpha_forget * forget_loss + alpha_retain * retain_loss + 0.1 * at_loss

#     return total_loss
def KT_loss_student_used(student_logits, student_activations, teacher_logits, teacher_activations, 
                    rho_f_forget, rho_f_robust, base_KL_temperature=1.0, base_AT_beta=250.0):
    """
    Improved Student loss with stabilized dynamic adjustment for rho_f_forget and rho_f_robust.
    """

    # 平滑动态因子
    dynamic_factor_forget = 1.0 + 0.5 * torch.tanh(torch.tensor(rho_f_forget))
    dynamic_factor_robust = 1.0 - 0.5 * torch.tanh(torch.tensor(rho_f_robust))

    # 调整 KL 温度和 AT_beta
    KL_temperature_adjusted = base_KL_temperature * dynamic_factor_forget
    AT_beta_adjusted = base_AT_beta * dynamic_factor_robust

    # 限定范围，避免过度调整
    KL_temperature_adjusted = torch.clamp(KL_temperature_adjusted, 0.8, 2.0)  # 范围 0.8 到 2.0
    AT_beta_adjusted = torch.clamp(AT_beta_adjusted, 200, 300)  # 范围 200 到 300

    # 计算 KL 散度损失
    divergence_loss = F.kl_div(
        F.log_softmax(student_logits / KL_temperature_adjusted, dim=1),
        F.softmax(teacher_logits / KL_temperature_adjusted, dim=1),
        reduction='batchmean'
    )

    # 计算注意力传输损失
    at_loss = 0
    if AT_beta_adjusted > 0:
        for i in range(len(student_activations)):
            at_loss += AT_beta_adjusted * (F.normalize(student_activations[i].pow(2).mean(1).view(student_activations[i].size(0), -1)) -
                                           F.normalize(teacher_activations[i].pow(2).mean(1).view(teacher_activations[i].size(0), -1))).pow(2).mean()

    # 增加固定的 CrossEntropyLoss，稳定 `retain` 类别
    cross_entropy_loss = F.cross_entropy(student_logits, teacher_logits.argmax(dim=1))

    # 综合损失
    total_loss = divergence_loss + at_loss + 0.1 * cross_entropy_loss

    # 打印动态调整的参数值
    # print(f"KL_temperature: {KL_temperature_adjusted:.4f}, AT_beta: {AT_beta_adjusted:.4f}, CE_Loss: {cross_entropy_loss:.4f}")
    return total_loss
def KT_loss_student(student_logits, student_activations, teacher_logits, teacher_activations, KL_temperature = 1, AT_beta = 250):

    divergence_loss = divergence(student_logits, teacher_logits, KL_temperature)
    if AT_beta > 0:
        at_loss = 0
        for i in range(len(student_activations)):
            at_loss = at_loss + AT_beta * attention_diff(student_activations[i], teacher_activations[i])
    else:
        at_loss = 0

    total_loss = divergence_loss + at_loss

    return total_loss


class Generator(nn.Module):

    def __init__(self, z_dim, num_channels=1, output_size=(28, 28)):
        """
        Generator class
        :param z_dim: Latent space dimension
        :param num_channels: Number of output channels (e.g., 1 for grayscale, 3 for RGB)
        :param output_size: Target output size (height, width)
        """
        super(Generator, self).__init__()
        self.output_size = output_size

        prefinal_layer = None
        final_layer = None
        if num_channels == 3:  # CIFAR-10 or CelebA
            prefinal_layer = nn.Conv2d(64, 3, 3, stride=1, padding=1)
            final_layer = nn.BatchNorm2d(3, affine=True)
        elif num_channels == 1:  # MNIST or FashionMNIST
            prefinal_layer = nn.Conv2d(64, 1, 3, stride=1, padding=1)
            final_layer = nn.BatchNorm2d(1, affine=True)
        else:
            raise ValueError(f"Unsupported number of channels: {num_channels}")

        self.layers = nn.Sequential(
            nn.Linear(z_dim, 128 * 8**2),
            View((-1, 128, 8, 8)),
            nn.BatchNorm2d(128),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            prefinal_layer,
            final_layer
        )

    def forward(self, z):
        # Pass through generator layers
        x = self.layers(z)
        # Resize output to match target size
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        return x

    def print_shape(self, x):
        """
        Debugging method to print shape at each layer.
        """
        act = x
        for layer in self.layers:
            act = layer(act)
            print('\n', layer, '---->', act.shape)
            
class Generator_used(nn.Module):

    def __init__(self, z_dim, num_channels = 3, output_size=(32, 32)):
        super(Generator, self).__init__()
        prefinal_layer = None
        final_layer = None
        if num_channels == 3:
            prefinal_layer = nn.Conv2d(64, 3, 3, stride=1, padding=1)
            final_layer = nn.BatchNorm2d(3, affine=True)
        elif num_channels == 1:
            prefinal_layer = nn.Conv2d(64, 1, 7, stride=1, padding=1)
            final_layer = nn.BatchNorm2d(1, affine=True)
        else:
            print(f"Generator Not Supported for {num_channels} channels")
        self.layers = nn.Sequential(
            nn.Linear(z_dim, 128 * 8**2),
            View((-1, 128, 8, 8)),
            nn.BatchNorm2d(128),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            prefinal_layer,
            final_layer
        )
        self.output_size = output_size

    def forward(self, z):
        x = self.layers(z)
        # 使用插值调整生成图像尺寸
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        return x
        # return self.layers(z)

    def print_shape(self, x):
        """
        For debugging purposes
        """
        act = x
        for layer in self.layers:
            act = layer(act)
            print('\n', layer, '---->', act.shape)

class Discriminator(nn.Module):
    """
    修改后的判别器，动态计算 Flatten 输入特征大小，支持动态输入通道数。
    """
    def __init__(self, input_channels=3):  # 默认 3 通道（RGB）
        super(Discriminator, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # 输出: 64 x 16 x 16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 输出: 128 x 8 x 8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 输出: 256 x 4 x 4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.flatten = nn.Flatten()

        # 全连接层稍后动态初始化
        self.fc = None
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 检查输入的通道数
        input_channels = x.shape[1]
        if self.feature_extractor[0].in_channels != input_channels:
            # 动态调整第一层卷积的输入通道数
            self.feature_extractor[0] = nn.Conv2d(
                input_channels, 64, kernel_size=4, stride=2, padding=1
            ).to(x.device)

        # 提取特征
        features = self.feature_extractor(x)
        flat_features = self.flatten(features)

        # 动态初始化全连接层
        if self.fc is None:
            self.fc = nn.Linear(flat_features.shape[1], 1).to(x.device)

        # 全连接层和 Sigmoid 输出
        output = self.fc(flat_features)
        return self.sigmoid(output)
# class Discriminator(nn.Module):
#     """
#     修改后的判别器，动态计算 Flatten 输入特征大小
#     """
#     def __init__(self, input_channels=3):
#         super(Discriminator, self).__init__()
#         self.feature_extractor = nn.Sequential(
#             # 输入: input_channels x 32 x 32
#             nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # 输出: 64 x 16 x 16
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 输出: 128 x 8 x 8
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 输出: 256 x 4 x 4 (取决于输入尺寸)
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.flatten = nn.Flatten()

#         # 在 forward 方法中自动动态确定输入大小
#         self.fc = None  # 占位，稍后初始化全连接层
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # 提取特征
#         features = self.feature_extractor(x)
#         flat_features = self.flatten(features)

#         # 动态初始化全连接层（仅初始化一次）
#         if self.fc is None:
#             self.fc = nn.Linear(flat_features.shape[1], 1).to(x.device)

#         # 全连接层和 Sigmoid 输出
#         output = self.fc(flat_features)
#         return self.sigmoid(output)
            
# class LearnableLoader(nn.Module):
#     def __init__(self, n_repeat_batch, num_channels = 3,device='cuda'):
#         """
#         Infinite loader, which contains a learnable generator.
#         """

#         super(LearnableLoader, self).__init__()
#         self.batch_size = 256
#         self.n_repeat_batch = n_repeat_batch
#         self.z_dim = 128
#         self.generator = Generator(self.z_dim, num_channels=num_channels).to(device=device)
#         self.device = device

#         self._running_repeat_batch_idx = 0
#         self.z = torch.randn((self.batch_size, self.z_dim)).to(device=self.device)

#     def __next__(self):
#         if self._running_repeat_batch_idx == self.n_repeat_batch:
#             self.z = torch.randn((self.batch_size, self.z_dim)).to(device=self.device)
#             self._running_repeat_batch_idx = 0

#         images = self.generator(self.z)
#         self._running_repeat_batch_idx += 1
#         return images

#     def samples(self, n, grid=True):
#         """
#         :return: if grid returns single grid image, else
#         returns n images.
#         """
#         self.generator.eval()
#         with torch.no_grad():
#             z = torch.randn((n, self.z_dim)).to(device=self.device)
#             # images = self.generator(z)
#             images = visualize(self.generator(z), dataset=self.dataset).cpu()
#             if grid:
#                 images = make_grid(images, nrow=round(math.sqrt(n)), normalize=True)

#         self.generator.train()
#         return images

#     def __iter__(self):
#         return self
class LearnableLoader(nn.Module):
    def __init__(self, n_repeat_batch, num_channels=3, device='cuda'):
        """
        Infinite loader with a learnable generator, optimized for zero-shot unlearning.
        """
        super(LearnableLoader, self).__init__()
        self.batch_size = 256  # 初始批次大小
        self.n_repeat_batch = n_repeat_batch
        self.z_dim = 128
        self.sigma_z = 1.0  # 初始噪声方差
        self.generator = Generator(self.z_dim, num_channels=num_channels).to(device=device)
        self.device = device

        self._running_repeat_batch_idx = 0
        self.z_cache = torch.randn((self.batch_size, self.z_dim)).to(self.device) * self.sigma_z

    def __next__(self):
        """
        每次返回生成的伪样本，并动态调整噪声方差和批次大小。
        """
        if self._running_repeat_batch_idx == self.n_repeat_batch:
            self._update_noise_and_batch_size()
            self._running_repeat_batch_idx = 0

        # 生成器生成伪样本
        self.z_cache = torch.randn((self.batch_size, self.z_dim)).to(self.device) * self.sigma_z
        images = self.generator(self.z_cache)
        self._running_repeat_batch_idx += 1
        return images

    def adjust_parameters(self, rho_f_forget, rho_f_robust, eta=0.05, rho_threshold=5.0, rho_low=1.0):
        """
        根据 rho_f 动态调整噪声方差 sigma_z 和生成批次大小 batch_size。
        """
        rho_f = rho_f_forget + rho_f_robust

        # 动态调整噪声方差 sigma_z
        if rho_f_forget > rho_f_robust:
            self.sigma_z *= np.exp(-eta * rho_f_forget / rho_f)
        else:
            self.sigma_z *= np.exp(eta * rho_f_robust / rho_f)
        self.sigma_z = min(max(self.sigma_z, 0.1), 2.0)

        # 动态调整生成批次大小 batch_size
        if rho_f > rho_threshold:  # 快速减少批次大小
            self.batch_size = int(self.batch_size * 0.5)
        elif rho_f < rho_low:  # 快速增加批次大小
            self.batch_size = int(self.batch_size * 1.5)
        else:
            if rho_f_forget > rho_f_robust:
                self.batch_size = int(self.batch_size * np.exp(-eta * rho_f_forget / rho_f))
            else:
                self.batch_size = int(self.batch_size * np.exp(eta * rho_f_robust / rho_f))

        # 限制 batch_size 范围
        self.batch_size = min(max(self.batch_size, 32), 1024)

        print(f"Adjusted sigma_z: {self.sigma_z:.4f}, Adjusted batch_size: {self.batch_size}")

    def _update_noise_and_batch_size(self):
        """
        在批次轮次完成后，自动更新噪声方差和批次大小。
        """
        # 这里可以根据外部的 rho_f 调整
        pass

    def samples(self, n, grid=True):
        """
        生成可视化样本。
        """
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn((n, self.z_dim)).to(self.device) * self.sigma_z
            images = self.generator(z)
            if grid:
                images = make_grid(images, nrow=round(math.sqrt(n)), normalize=True)
            plt.imshow(images.permute(1, 2, 0).cpu().numpy())
            plt.axis("off")
            plt.show()
        self.generator.train()

    def __iter__(self):
        return self
class LearnableLoaderWithRealData(nn.Module):
    def __init__(self, original_data, n_repeat_batch, num_channels=3, device='cuda', lambda_mixing=0.5, output_size=(32, 32)):
        """
        Infinite loader, which combines a learnable generator with access to real data.
        
        :param original_data: TensorDataset or DataLoader, containing real data samples.
        :param n_repeat_batch: Number of repeated batches before resetting.
        :param num_channels: Number of channels for generated images.
        :param device: Device to run the generator.
        :param lambda_mixing: Mixing ratio between generated and real samples.
        :param output_size: Target output size for generated images.
        """
        super(LearnableLoaderWithRealData, self).__init__()
        self.batch_size = 256
        self.n_repeat_batch = n_repeat_batch
        self.z_dim = 128
        self.lambda_mixing = lambda_mixing  # 控制混合比例
        self.output_size = output_size

        self.generator = Generator(self.z_dim, num_channels=num_channels, output_size=output_size).to(device=device)
        self.device = device

        # 原始数据加载器
        self.original_data_loader = DataLoader(original_data, batch_size=self.batch_size, shuffle=True)
        self.original_data_iter = iter(self.original_data_loader)

        self._running_repeat_batch_idx = 0
        self.z = torch.randn((self.batch_size, self.z_dim)).to(device=self.device)

    def __next__(self):
        if self._running_repeat_batch_idx == self.n_repeat_batch:
            self.z = torch.randn((self.batch_size, self.z_dim)).to(device=self.device)
            self._running_repeat_batch_idx = 0
            self.original_data_iter = iter(self.original_data_loader)

        # 生成器生成的伪样本
        generated_images = self.generator(self.z)

        # 从原始数据中取出一个批次
        try:
            real_images, _ = next(self.original_data_iter)  # 假设原始数据格式为 (image, label)
        except StopIteration:
            self.original_data_iter = iter(self.original_data_loader)
            real_images, _ = next(self.original_data_iter)

        real_images_resized = F.interpolate(real_images, size=self.output_size).to(self.device)
        generated_images_resized = F.interpolate(generated_images, size=self.output_size).to(self.device)
        mixed_images = self.lambda_mixing * real_images_resized + (1 - self.lambda_mixing) * generated_images_resized


        self._running_repeat_batch_idx += 1
        return mixed_images

    def samples(self, n, grid=True):
        """
        Generate and visualize n samples, including generator-based and mixed data.
        """
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn((n, self.z_dim)).to(self.device)
            generated_images = self.generator(z).cpu()

            if grid:
                images = make_grid(generated_images, nrow=round(math.sqrt(n)), normalize=True)
            else:
                images = generated_images
        self.generator.train()
        return images

    def __iter__(self):
        return self
    # def adjust_parameters(self, rho_f_forget, rho_f_robust, rho_f):
    #     """
    #     Dynamically adjust KL_temperature and AT_beta based on rho_f values.
    #     """
    #     balance_factor = rho_f_forget / (rho_f_forget + rho_f_robust + 1e-8)
    #     adjustment_factor = max(0.5, 1.0 / (1.0 + rho_f))

    #     # 动态调整温度 T
    #     self.KL_temperature *= (1.0 + balance_factor * adjustment_factor)
    #     self.KL_temperature = min(max(self.KL_temperature, 0.5), 5.0)

    #     # 动态调整 AT_beta
    #     self.AT_beta *= (1.0 - (1.0 - balance_factor) * adjustment_factor)
    #     self.AT_beta = min(max(self.AT_beta, 50), 500)

    #     print(f"Adjusted KL_temperature: {self.KL_temperature:.4f}, AT_beta: {self.AT_beta:.4f}")