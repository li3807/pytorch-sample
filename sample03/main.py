import torch
from torch import nn
from NeuralNetwork import NeuralNetwork

if __name__ == '__main__':
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # 为了使用这个模型，我们给它传递输入数据。这将会执行模型的forward，伴随着一些幕后工作。不要直接调用 model.forward() !
    model = NeuralNetwork().to(device)
    print(model)

    # 创建形状为 1,28,28 的张量，随机创建用于预测数据
    shape = (1, 28, 28)
    X = torch.rand(shape, device=device)
    # 模型在数据输入上的调用返回一个2维张量（第0维对应一次输出一组10个代表每种类型的原始预测值，第1维对应该类型对应的原始预测值）
    # 执行模型的 forward 函数
    logits = model(X)
    # logits=tensor([[-0.0515, -0.0117, -0.0511,  0.1154, -0.0407,  0.0366,  0.0543, -0.0738,
    #          -0.0167, -0.0305]], grad_fn=<AddmmBackward0>)
    print(f"logits={logits}")
    # 我们将它传递给一个nn.Softmax模块的实例来来获得预测概率。
    pred_probab = nn.Softmax(dim=1)(logits)
    print(f"Softmax : {pred_probab}")
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")
