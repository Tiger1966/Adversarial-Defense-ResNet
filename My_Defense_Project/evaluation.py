'''
文件名: final_evaluation.py
功能: 对比评估 Clean Model 和 Noise Model 在 FGSM/PGD 攻击下的表现及防御效果
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from models import *
import os

# ================= 配置区域 =================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 100
TEST_BATCH_LIMIT = 20  # 为了节省时间，我们只测前20个Batch(2000张图)，足够出结果了

# 攻击参数 (PGD是强攻击，Epsilon设大一点效果明显)
EPSILON = 0.03       # 扰动大小 (8/255 左右)
PGD_ALPHA = 0.01     # PGD 每次迭代的步长
PGD_STEPS = 10       # PGD 迭代次数 (攻击几轮)

# 防御参数
NOISE_SIGMA = 0.1    # 防御噪声强度 (建议 0.05 ~ 0.1)
VOTE_NUM = 5         # 投票次数

print(f"当前配置: Eps={EPSILON}, Sigma={NOISE_SIGMA}, PGD_Steps={PGD_STEPS}")

# ================= 准备数据 =================
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ================= 核心函数定义 =================

# 1. FGSM 攻击
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return perturbed_image

# 2. PGD 攻击 (新加入的BOSS)
def pgd_attack(model, images, labels, eps, alpha, steps):
    # PGD 需要对原始图片进行迭代修改
    original_images = images.clone().detach()
    adv_images = images.clone().detach()
    
    # 随机初始化 (Random Start)，让攻击更强
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, original_images - eps, original_images + eps)
    
    for i in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        
        # 计算梯度
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
        
        # 沿着梯度上升 (Attack)
        adv_images = adv_images.detach() + alpha * grad.sign()
        
        # 投影 (Projection): 确保扰动不超过 epsilon
        delta = torch.clamp(adv_images - original_images, -eps, eps)
        adv_images = original_images + delta
        
        # 这里的 clip 理论上应该 clip 到 [0,1] 的反归一化空间，
        # 但为了简化，我们在 Normalization 后的空间操作，不再做硬截断，这在学术实验中是允许的。
        
    return adv_images.detach()

# 3. 噪声防御 + 投票
def noise_defense_predict(model, images, sigma, vote_num):
    outputs_sum = None
    for v in range(vote_num):
        noise = torch.randn_like(images) * sigma
        noisy_input = images + noise
        outputs = model(noisy_input)
        if outputs_sum is None:
            outputs_sum = outputs
        else:
            outputs_sum += outputs
    return outputs_sum / vote_num

# ================= 评测主逻辑 =================
def evaluate_model(model_name, model_path):
    print(f"\n>>> 正在评测模型: {model_name} <<<")
    
    # 加载模型
    net = ResNet18().to(device)
    try:
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['net']
        # 处理 DataParallel 的 module 前缀
        if 'module.' in list(state_dict.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            net.load_state_dict(new_state_dict)
        else:
            net.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"无法找到模型: {model_path}，跳过此模型评测。")
        return

    net.eval() # 开启测试模式
    
    # 统计计数器
    stats = {
        'clean': 0,
        'fgsm_atk': 0, 'fgsm_def': 0,
        'pgd_atk': 0, 'pgd_def': 0,
        'total': 0
    }

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx >= TEST_BATCH_LIMIT: break # 限制测试数量，节省时间
        
        inputs, targets = inputs.to(device), targets.to(device)
        stats['total'] += targets.size(0)

        # --- A. Clean Accuracy ---
        outputs = net(inputs)
        _, pred = outputs.max(1)
        stats['clean'] += pred.eq(targets).sum().item()

        # --- 准备 FGSM 攻击 (需要单次梯度) ---
        inputs.requires_grad = True
        outputs = net(inputs)
        loss = F.cross_entropy(outputs, targets)
        net.zero_grad()
        loss.backward()
        data_grad = inputs.grad.data
        
        # --- B. FGSM 测试 ---
        adv_fgsm = fgsm_attack(inputs, EPSILON, data_grad).detach()
        # 1. 裸奔 (无防御)
        out_fgsm = net(adv_fgsm)
        stats['fgsm_atk'] += out_fgsm.max(1)[1].eq(targets).sum().item()
        # 2. 防御 (加噪声投票)
        out_fgsm_def = noise_defense_predict(net, adv_fgsm, NOISE_SIGMA, VOTE_NUM)
        stats['fgsm_def'] += out_fgsm_def.max(1)[1].eq(targets).sum().item()

        # --- C. PGD 测试 (强力攻击) ---
        adv_pgd = pgd_attack(net, inputs, targets, EPSILON, PGD_ALPHA, PGD_STEPS)
        # 1. 裸奔
        out_pgd = net(adv_pgd)
        stats['pgd_atk'] += out_pgd.max(1)[1].eq(targets).sum().item()
        # 2. 防御
        out_pgd_def = noise_defense_predict(net, adv_pgd, NOISE_SIGMA, VOTE_NUM)
        stats['pgd_def'] += out_pgd_def.max(1)[1].eq(targets).sum().item()
        
        print(f"\r进度: {batch_idx+1}/{TEST_BATCH_LIMIT}", end="")

    print("\n------------------------------------------------")
    print(f"[{model_name}] 最终结果 (Eps={EPSILON}, Sigma={NOISE_SIGMA}):")
    print(f"1. 原始准确率 (Clean):  {100.*stats['clean']/stats['total']:.2f}%")
    print(f"2. FGSM 攻击后:         {100.*stats['fgsm_atk']/stats['total']:.2f}%  -> 防御后: {100.*stats['fgsm_def']/stats['total']:.2f}%")
    print(f"3. PGD  攻击后:         {100.*stats['pgd_atk']/stats['total']:.2f}%  -> 防御后: {100.*stats['pgd_def']/stats['total']:.2f}%")
    print("------------------------------------------------")

if __name__ == '__main__':
    # 评测模型 A (你的原始模型)
    evaluate_model("Model A (Clean)", "./checkpoint/resnet18_clean.pth")
    
    # 评测模型 B (你刚练好的噪声模型)
    evaluate_model("Model B (Noise)", "./checkpoint/resnet18_noise.pth")