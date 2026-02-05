import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# 配置
EPSILON = 0.03
SIGMA = 0.1

# 准备数据
transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# 获取一张图
image, label = next(iter(loader))
classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')

# 简单的攻击函数 (FGSM)
def fgsm_attack(image, epsilon):
    # 模拟梯度 (这里随机模拟一下梯度方向用于展示，因为我们只看外观)
    sign_data_grad = torch.sign(torch.randn_like(image))
    return torch.clamp(image + epsilon * sign_data_grad, 0, 1)

# 简单的加噪声函数
def add_noise(image, sigma):
    return torch.clamp(image + torch.randn_like(image) * sigma, 0, 1)

# 生成图片
clean_img = image
adv_img = fgsm_attack(image, EPSILON)      # 对抗样本
noisy_img = add_noise(clean_img, SIGMA)    # 你的防御图

# 绘图函数
def show(img):
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))

# 画图
plt.figure(figsize=(10, 4))

# 1. 原始图片
plt.subplot(1, 3, 1)
plt.title(f"Original ({classes[label]})", fontsize=14)
plt.imshow(show(clean_img[0]))
plt.axis('off')

# 2. 对抗样本 (FGSM)
plt.subplot(1, 3, 2)
plt.title(f"Adversarial (Eps={EPSILON})", fontsize=14)
plt.imshow(show(adv_img[0]))
plt.axis('off')

# 3. 噪声增强/防御图
plt.subplot(1, 3, 3)
plt.title(f"Noisy Input (Sigma={SIGMA})", fontsize=14)
plt.imshow(show(noisy_img[0]))
plt.axis('off')

plt.tight_layout()
plt.savefig('visualization.png', dpi=300)
print("图片已生成：visualization.png，请插入Word报告！")
plt.show()