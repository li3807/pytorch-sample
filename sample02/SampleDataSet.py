class SampleDataSet(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        """
        初始化函数，在实例化数据集对象时，__init__函数被运行一次。我们初始化包含图像的目录、注释文件和两种转换
        :param annotations_file:
        :param img_dir:
        :param transform: 用于修改特征
        :param target_transform:用于修改标签
        """
        pass

    def __len__(self):
        """
        函数__len__返回我们数据集中的样本数。
        :return:
        """
        pass

    def __getitem__(self, item):
        """
        从数据集中给定的索引idx处加载并返回一个样本
        :param item:
        :return:
        """
        pass
