import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import torch
from torch import nn


class DataCommand:
    """
    把获取数据集的指令包装成一个类
    """
    picture_root = "data//pictures//"  # 存储图片的目录，应当是相对于main函数的路径

    def __init__(self,
                 index2Name: dir,
                 train_ratio: int = 0.7,
                 batchSize: int = 8,
                 train_transform: T.Compose = None,
                 valid_transform: T.Compose = None,
                 ) -> None:
        """
        :param batchSize: 批次的大小
        """
        self.index2Name = index2Name
        self.batchSize = batchSize
        self.train_ratio = train_ratio
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.__selfCheck()
        self.dataLoader = None

    def __divideTrainAndValid(self) -> (dict, dict):
        """
        根据种类名单和训练集的比例，得到训练集的图片名单和验证集的图片名单
        :return:
        """
        train_pictures = {}
        valid_pictures = {}
        for type_index in self.index2Name:
            type_name = self.index2Name[type_index]
            pictures = os.listdir(DataCommand.picture_root + str(type_name))
            train_pictures[type_index] = [os.path.join(DataCommand.picture_root, str(type_name), path) for path in
                                          pictures[:int(self.train_ratio * len(pictures))]]
            valid_pictures[type_index] = [os.path.join(DataCommand.picture_root, str(type_name), path) for path in
                                          pictures[int(self.train_ratio * len(pictures)):]]
        return train_pictures, valid_pictures

    def __selfCheck(self) -> None:
        """
        检查这个获取数据的命令是否合法
        """
        if len(self.index2Name) <= 1:
            raise Exception("至少有两类！")
        dirs = os.listdir(DataCommand.picture_root)
        wrong_type = []
        for key_ in self.index2Name:
            picture_type = self.index2Name[key_]
            if picture_type not in dirs:
                wrong_type.append((picture_type, 0))  # 将来升级的时候把0换成别的信息
        if len(wrong_type) != 0:
            raise Exception("缺少以下类型的图片 ：" + " ".join([str(picture_type) for picture_type, number in wrong_type]))
        if type(self.batchSize) != int:
            raise ValueError("batchSize 类型非法！")
        if self.train_ratio <= 0:
            raise ValueError("训练集的比例不能为0！")
        if self.train_ratio >= 1:
            raise ValueError("训练集的比例过大!")

    def getDataLoader(self):
        train_pictures, valid_pictures = self.__divideTrainAndValid()
        train_data = TrainData(train_pictures, self.train_transform)
        valid_data = ValidData(valid_pictures, self.valid_transform)
        trainDataLoader = DataLoader(train_data, batch_size=self.batchSize, shuffle=True)
        validDataLoader = DataLoader(valid_data, batch_size=self.batchSize, shuffle=False)

        return trainDataLoader, validDataLoader

    def getTrainDataLoader(self):

        pass

    def getValidDataLoader(self):
        pass


class TrainData(data.Dataset):
    """训练集的数据
    对于训练集中的每个图片，我们知道它的真实类型
    """

    def __init__(self,
                 pictures: dict = None,
                 transform_propose: T.Compose = None
                 ) -> None:
        """
        :param pictures: 一个字典，key_是用数字表示的图片种类，值是图片列表
        :param transform_propose:
        """
        self.pictures = pictures
        self.transform_propose = transform_propose
        pass

    def __getitem__(self, index: int) -> (torch.tensor, str):
        has_count = 0
        for type_index in self.pictures:
            if has_count <= index < has_count + len(self.pictures[type_index]):
                picture_path = self.pictures[type_index][index - has_count]
                tensor_data = self.transform_propose(Image.open(picture_path))
                return tensor_data, type_index
            has_count += len(self.pictures[type_index])

    def __len__(self):
        # 数据集总大小，即所有类型图片的总数
        sum_ = 0
        for type_index in self.pictures:
            sum_ += len(self.pictures[type_index])
        return sum_


class ValidData(data.Dataset):
    """验证集的数据
    对于验证集中的每个图片，我们并清楚它的真实类型，但在训练时不去用它
    """

    def __init__(self,
                 pictures: dict = None,
                 transform_propose: T.Compose = None
                 ) -> None:
        """
        :param pictures: 一个字典，key_是用数字表示的图片种类，值是图片列表
        :param transform_propose:
        """
        self.pictures = pictures
        self.transform_propose = transform_propose
        pass

    def __getitem__(self, index: int) -> (torch.tensor, str):
        has_count = 0
        for type_index in self.pictures:
            if has_count <= index < has_count + len(self.pictures[type_index]):
                # try:
                picture_path = self.pictures[type_index][index - has_count]
                tensor_data = self.transform_propose(Image.open(picture_path))
                return tensor_data, type_index
            has_count += len(self.pictures[type_index])

    def __len__(self):
        # 数据集总大小，即所有类型图片的总数
        sum_ = 0
        for type_index in self.pictures:
            sum_ += len(self.pictures[type_index])
        return sum_


class TestData(data.Dataset):
    """测试集的数据
    对于测试集中的每个图片，我们并不清楚它的真实类型
    """

    def __init__(self,
                 root: str = ".\\data\\test",
                 transform_propose: T.Compose = None
                 ) -> None:
        """
        数据集初始化，并不真正读入图片，只是读入图片路径，排序并存储
        root : 测试集图片文件夹的路径
        transform_propose ：对测试集图片进行的预处理过程，应当与训练模型时的预处理过程保持一致
        """
        self.image_paths = [os.path.join(root, img) for img in os.listdir(root)]  # 保存测试集所有图片的路径
        self.image_num = len(self.image_paths)  # 测试集图片总数
        if transform_propose is None:
            raise Exception("请输入图片预处理过程，应当与训练模型时的预处理过程保持一致！")
        else:
            self.transform_propose = transform_propose

    def __getitem__(self, index: int) -> (torch.tensor, str):
        """
        按照图片的序号由小到大，一次返回一张图片的数据
        返回一个二元组:(图片经过处理后转换成的tensor,图片路径)
        """
        img_path = self.image_paths[index]
        tensor_data = self.transform_propose(Image.open(img_path))
        return tensor_data, img_path

    def __len__(self):
        return len(self.image_paths)
