import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # Sequential 是一个模块的有序容器。数据会沿着模块定义的顺序流动。
        self.linear_relu_stack = nn.Sequential(
            # nn.Linear（线性层）是一个对输入值使用自己存储的权重(w)和偏差(b)来做线性转换的模块。
            nn.Linear(28 * 28, 512),
            # 非线性的激活函数创造了模型的输入值和输出值之间的复杂映射。它们在线性转换之后应用来引入非线性,帮助神经网络学习更广阔范围的现象。
            # 在这个模型中，我们在我们的线性层之间使用nn.ReLU，但是还有其他的激活函数可以用来在你的模型中引入非线性。
            # (译者注：ReLU即Rectified Linear Unit，译为线性整流函数或者修正线性单元)
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        """
        进行模型预测
        :param x: 需要预测的张量
        :return:
        """
        # 初始化nn.Flatten(展平层)层来将每个2维的28x28图像转换成一个包含784像素值的连续数组（微批数据的维度(第0维)保留了）.
        x = self.flatten(x)
        print(f"flatten x={x}")
        # 返回一个2维张量（第0维对应一次输出一组10个代表每种类型的原始预测值，第1维对应该类型对应的原始预测值）
        logits = self.linear_relu_stack(x)
        return logits
