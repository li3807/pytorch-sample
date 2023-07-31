import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
# Pyplot 是 Matplotlib 的子库，提供了和 MATLAB 类似的绘图 API
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # root 是存储训练数据的目录
    # train 制定训练或测试训练集
    # download 如果 root 没有训练数据则下载
    # transform 制定特征和标签的转换
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    print(f"training_data={training_data}")
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    figure = plt.figure(figsize=(8, 8))
    print(f"figure {figure}")
    cols, rows = 4, 4
    for i in range(1, rows * cols + 1):
        # 依据索引获取训练集数据
        img, label = training_data[i]
        # 增加子图，并显示
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()
