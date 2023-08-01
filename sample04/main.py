import torch.nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets

from NeuralNetwork import NeuralNetwork


def train(dataloader, model):
    """
    训练函数
    :param dataloader:
    :param model:
    :return:
    """
    size = len(dataloader.dataset)
    # 模型训练
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 计算预测误差
        pred = model(X)
        # 损失值
        loss = loss_fn(pred, y)

        # 反向传播预测误差来调整模型的参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model):
    """
    :param dataloader:
    :param model:
    :return:
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # 模型预测
            pred = model(X)
            # 累加损失值
            test_loss += loss_fn(pred, y).item()
            #  所以，如果你想要判断你的结果，你只需要看看这个输出的列表里面最大的那个数字的下标，这里是9，刚好和我们的这个标签的数值一样，那么预测正确。
            #  在pytorch里面提供了一个方法，叫做argmax(1)表示横向获得最大值的下标，argmax(0)表示纵向。所以用这种方法我们可以去获取我们训练的一个准确度。
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>4f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    # 选择训练硬件 CPU 或者 GPU
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # 训练数据集
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

    # 测试数据集
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    print(f"test_data={test_data}")

    # 我们将Dataset作为参数传递给DataLoader。这将包装我们的数据集为一个迭代器，并支持自动批处理、采样、洗牌和多进程数据加载。
    # 在这里，我们定义批次大小为64，即数据加载器迭代器中的每个元素将返回 64 个特征和标签的批次。
    # 创建数据加载器
    batch_size = 8
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # 创建训练神经网络
    neuralModule = NeuralNetwork()
    # 损失函数和一个优化器
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(neuralModule.parameters(), lr=1e-3)

    # 迭代训练
    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, neuralModule)
        test(test_dataloader, neuralModule)
    print("Done!")
