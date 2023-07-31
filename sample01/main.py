import torch
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = [[1, 2], [2, 3]]
    x_Data = torch.tensor(data)
    print(f"x_data={x_Data}")
    np_array = np.array(data)
    x_np_data= torch.from_numpy(np_array)
    print(f"x_np_data={x_np_data}")

    shape = (2, 3)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")