# Adversarial-Defense-ResNet
基于高斯噪声增强与随机平滑的卷积神经网络对抗防御研究 (CIFAR-10)


本项目针对 CIFAR-10 数据集，通过结合**训练期高斯噪声增强 (Gaussian Augmentation)** 与 **推理期软投票 (Soft Voting)** 机制，显著提升了 ResNet-18 模型在 PGD 强攻击下的鲁棒性。

## 📁 文件说明
- `main.py`: 基础训练代码 (Baseline Model A)。
- `train_noise.py`: **[核心]** 噪声增强训练代码 (Robust Model B)。
- `final_evaluation.py`: **[核心]** 包含 FGSM/PGD 攻击算法及 Soft Voting 防御评测逻辑。
- `vis_images.py`: 对抗样本可视化脚本。

## 📊 实验结果
| 模型 | Clean Acc | FGSM Acc | PGD Acc |
| :--- | :--- | :--- | :--- |
| **Model A** (Standard) | 95.00% | 62.65% | 22.35% |
| **Model B** (Ours + Voting) | **94.85%** | **74.20%** | **65.55%** |

## 🚀 快速开始
1. 训练基准模型:
`python main.py`
2. 训练鲁棒模型:
`python train_noise.py`
3. 运行攻击与防御评测:
`python final_evaluation.py`

## 🙏 致谢
基础代码框架基于 [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)。
