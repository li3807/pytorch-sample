import torch
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = [[1, 2], [2, 3]]
    x_Data = torch.tensor(data)
    print(f"x_data={x_Data}")
    np_array = np.array(data)
    x_np_data = torch.from_numpy(np_array)
    print(f"x_np_data={x_np_data}")
    # shape 设在张量的形状
    shape = (2, 3)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)
    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")
    # 修改张量的数据类型
    ones_like_tensor = torch.ones_like(ones_tensor, dtype=torch.int64)
    print(f"Ones Like Tensor:{ones_like_tensor}")

    # 创建随即张量，并设在数据类型，读取张量的属性形状、类型、设备（默认设备CPU）
    shape = (2, 3, 4, 5)
    tensor = torch.rand(shape, dtype=torch.float)
    print(f"tensor {tensor}")
    print(f"tensor shape {tensor.shape}")
    print(f"tensor of dataType {tensor.dtype}")
    print(f"tensor Device is {tensor.device}")

    # 张量的读取
    rand_tensor = torch.rand(4, 4)
    print(f"rand tensor {rand_tensor}")
    # 读取张量第一行
    print(f"first row {rand_tensor[0]}")
    # 读取张量第一列
    print(f"first column {rand_tensor[:, 0]}")
    # 读取张量最后一列
    print(f"last column {rand_tensor[:, -1]}")
    # 设在张量的第2列的值 0
    rand_tensor[:, 1] = 0
    print(rand_tensor)

    if torch.cuda.is_available():
        rand_tensor = rand_tensor.to("cuda")

    print(f"rand_tensor Device is {rand_tensor.device}")
    # 就地操作将结果存储到操作数中的操作被称为就地操作。它们用后缀_来表示
    rand_tensor.add_(5)
    print(f"rand tersor {rand_tensor}")
