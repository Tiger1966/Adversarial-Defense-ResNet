# Based on code by kuangliu: https://github.com/kuangliu/pytorch-cifar
# Core defense logic (Gaussian Noise Injection) implemented by Tiger1966
# 本实验基础训练框架参考了 pytorch-cifar，核心防御逻辑由本人独立实现。

'''
文件名: noise.py
功能: 在训练过程中加入高斯噪声，训练一个鲁棒性更强的新模型 (Model B)
核心机制：在加载图片时，动态注入高斯噪声，让模型在训练时就习惯这种噪声，从而在测试时表现更好。
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from models import *

# 参数设置
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Noise Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0     #记录最高Accuracy
start_epoch = 0  #记录开始的epoch

# ==========================================
# 核心修改：定义一个加噪声的变换类
# ==========================================
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.1):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        # 生成噪声并叠加，保持和tensor一样的形状
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

print('==> Preparing data with Noise Injection..')

# 在训练的数据增强中，加入了 AddGaussianNoise
# 这里设置 std=0.1，让模型在训练时就习惯 0.1 强度的噪声
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    AddGaussianNoise(mean=0., std=0.1)  # <--- 【关键点】
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building ResNet18..')
net = ResNet18()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training Loop
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

# Test Loop
def test(epoch):
    global best_acc
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    print(f"Test Acc: {acc:.2f}%")
    
    # Save checkpoint
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        # 【关键点】保存为不同的名字，防止覆盖原来的模型
        torch.save(state, './checkpoint/resnet18_noise.pth')
        best_acc = acc

if __name__ == '__main__':
    # 只跑 200 轮
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)

        scheduler.step()

