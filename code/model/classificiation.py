from torchvision import transforms
import torch
import pandas
import os
import pickle
from datetime import datetime
import netron
from threading import Condition, Lock

from .baseModels import *
from ..data.dataSet import *  # 上面这两行代码这样写，如果直接执行本python文件则会报错，但允许外部导入本文件


class Classify:
    """
    图线分类器
    """
    model_root = "data//models"  # 存储所有模型的目录，这个是相对于main的目录

    @staticmethod
    def factory(is_new: bool = True, name: str = None, type_list: list = None, **dict_):
        if not is_new:
            with open(os.path.join(Classify.model_root, name, "baseInformation.txt"), "rb") as f:
                classify = pickle.load(f)
            return classify
        if "baseType" in dict_:  # 传入的参数中有baseType，说明是想要一个基本模型
            if "baseType" in dict_:
                baseType = dict_["baseType"]
            else:
                baseType = "ResNet"
            if "netType" in dict_:
                netType = dict_["netType"]
            else:
                netType = None
            if "use_pretrained" in dict_:
                use_pretrained = dict_["use_pretrained"]
            else:  # 默认情况是使用人家训练好的参数
                use_pretrained = True
            if "continue_train" in dict_:
                continue_train = dict_["continue_train"]
            else:  # 默认不再改动人家训练好的参数，只是改动最后一个线性层的参数
                continue_train = False

            return BaseModelClassify(
                name=name,
                type_list=type_list,
                baseType=baseType,
                netType=netType,
                use_pretrained=use_pretrained,
                continue_train=continue_train
            )
        else:
            list_ = dict_["layers"]
            return MyModelClassify(
                name=name,
                type_list=type_list,
                list_=list_,
            )

    @staticmethod
    def load_Model(name: str = None, checkpoint_name: str = None):
        """
        :param name: 模型名
        :param checkpoint_name: 是检查点的文件名，非目录名
        """
        checkpoint = torch.load(os.path.join(Classify.model_root, name, checkpoint_name + ".pth"))
        model = checkpoint['model']
        return model

    def getModel(self):
        """
        把模型从磁盘中加载进来
        """
        if self.model is not None:
            return
        self.model = Classify.load_Model(self.name, self.checkpoints[-1]["name"])
        self.model = self.model.cpu()

    def __init__(self,
                 name: str = None,
                 type_list: list = None):
        self.name = name
        os.mkdir(os.path.join(Classify.model_root, self.name))
        self.index2Name = {}
        for index, name in enumerate(type_list):
            self.index2Name[index] = name
        self.type_num = len(self.index2Name)
        # 下面把若干属性赋值为None，是为了更清楚的展示一个Classify类型有哪些属性
        # 还有一些属性在子类里
        self.input_size = None
        self.data_transform = None
        self.model = None

        # TODO
        self.checkpoints = []  # 所有检查点的信息列表，每项是一个字典，按照时间排序

    def save(self):
        """
        保存模型的基本信息。save_Model已经调用了本方法，保证每次更新模型信息的时候，都会自动刷新基本信息。
        为了防止无限循环，请保证这个方法不会调用save_Model
        """
        tempModel = self.model
        self.model = None  # 临时遮蔽模型信息，不存储
        with open(os.path.join(Classify.model_root, self.name, "baseInformation.txt"), "wb") as fp:
            pickle.dump(self, fp)
        self.model = tempModel

    def save_Model(self, valid_acc: float = None, is_initial=False):
        if not is_initial:
            now = datetime.now()
            current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
            # 检查点的名称，也是检查点的文件名(无后缀)
            pth_name = "___" + "{:.4f}".format(valid_acc) + "___" + str(current_time)
        else:
            pth_name = "initial"
        checkpoint_information = {
            "name": pth_name
        }
        self.checkpoints.append(checkpoint_information)
        torch.save({'model': self.model}, os.path.join(Classify.model_root, self.name, pth_name + ".pth"))
        self.save()  # 更新基本信息（检查点多了一个）

    def getDataLoader(self, batchSize: int = 8):

        my_command = DataCommand(
            index2Name=self.index2Name,
            batchSize=batchSize,
            train_transform=self.data_transform,
            valid_transform=self.data_transform
        )

        all_loaders = my_command.getDataLoader()
        trainDataLoader = all_loaders[0]
        validDataLoader = all_loaders[1]
        return trainDataLoader, validDataLoader

    def selfTest(self):
        """
        模型自检
        """
        try:
            model = self.model
            trainDataLoader, validDataLoader = self.getDataLoader(2)
            useGPU = torch.cuda.is_available()
            if useGPU:
                model = model.cuda()

            model.train()  # 训练
            optimizer = self.get_optimizer(0.001)
            loss_function = self.get_lossFunction()
            for inputs, labels in trainDataLoader:
                if useGPU:
                    inputs = inputs.cuda()  # 转移到GPU
                    labels = labels.cuda()
                optimizer.zero_grad()  # 清零
                torch.set_grad_enabled(True)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                _, preds = torch.max(outputs, 1)
                # 第一个参数outputs代表对outputs这个张量进行最大值分析
                # 1代表对每行求最大值
                # max返回两个值，第一个是每行的最大值，第二个是每行的最大值所在下标
                loss.backward()
                optimizer.step()
                break
            accuracy = self.test(self.model, validDataLoader, useGPU, loss_function)
            assert type(accuracy) == float or type(accuracy) == int
            self.model = model.cpu()
        except Exception as ex:
            return False, str(ex)
        return True, accuracy

    def showModelStructure(self):
        """
        对模型进行可视化展示
        """
        self.model = Classify.load_Model(self.name, self.checkpoints[-1]["name"])
        self.model = self.model.cpu()
        try:
            netron.start(os.path.join(Classify.model_root, self.name, "initial.pth"))
        except Exception as ex:
            print(ex)

    @staticmethod
    def test(model, validDataLoader, useGPU, criterion):
        """利用验证集的数据对模型进行检测
        这个函数不会把模型从CPU和GPU上搬来搬去"""
        model.eval()  # 验证
        running_corrects = 0  # 准确率
        for inputs, labels in validDataLoader:
            if useGPU:
                inputs = inputs.cuda()  # 转移到GPU
                labels = labels.cuda()
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                # 第一个参数outputs代表对outputs这个张量进行最大值分析
                # 1代表对每行求最大值
                # max返回两个值，第一个是每行的最大值，第二个是每行的最大值所在下标
            # 计算准确个数
            running_corrects += torch.sum(preds == labels.data)
        # 整个epoch的损失和准确率
        valid_acc = running_corrects.double() / len(validDataLoader.dataset)
        return valid_acc.item()

    def get_optimizer(self, lr):
        """
        返回一个优化器
        """
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
        optimizer = optim.SGD(params_to_update, lr=lr)  # 使用Adam优化器，目标是优化params_to_update，学习率为1e-2
        return optimizer

    def get_lossFunction(self):
        """
        返回一个损失函数
        """

        criterion = nn.CrossEntropyLoss()  # 采用交叉熵损失函数
        return criterion

    @staticmethod
    def trainOneEpoch(model, trainDataLoader, useGPU, optimizer, loss_function, pbar=None):
        """
        这个函数不会把模型从CPU和GPU上来回搬迁
        """
        model.train()  # 训练
        for step, (inputs, labels) in enumerate(trainDataLoader):
            if useGPU:
                inputs = inputs.cuda()  # 转移到GPU
                labels = labels.cuda()
            optimizer.zero_grad()  # 清零
            torch.set_grad_enabled(True)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            _, preds = torch.max(outputs, 1)
            # 第一个参数outputs代表对outputs这个张量进行最大值分析
            # 1代表对每行求最大值
            # max返回两个值，第一个是每行的最大值，第二个是每行的最大值所在下标
            loss.backward()
            optimizer.step()

    def train(self, batchSize: int = 8, num_epochs: int = 20, lr: float = 0.01,
              useGPU: bool = torch.cuda.is_available(),
              show: bool = True):
        if show:
            history = hl.History()
            canvas = hl.Canvas()

        trainDataLoader, validDataLoader = self.getDataLoader(batchSize=batchSize)
        if useGPU:
            model = self.model.cuda()
        else:
            model = self.model

        print("Params to learn:")
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)

        log_step_interval = 100  # 记录的步数间隔
        print("学习率 " + str(lr))
        # 优化器设置
        optimizer = optim.SGD(params_to_update, lr=lr)  # 使用Adam优化器，目标是优化params_to_update，学习率为1e-2
        criterion = nn.CrossEntropyLoss()  # 采用交叉熵损失函数

        for epoch in range(num_epochs):
            # 一共训练num_epochs个轮次，每个轮次都是先训练，再验证
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)
            # 准备训练
            model.train()  # 训练
            for step, (inputs, labels) in enumerate(trainDataLoader):
                if useGPU:
                    inputs = inputs.cuda()  # 转移到GPU
                    labels = labels.cuda()
                optimizer.zero_grad()  # 清零
                torch.set_grad_enabled(True)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                # 第一个参数outputs代表对outputs这个张量进行最大值分析
                # 1代表对每行求最大值
                # max返回两个值，第一个是每行的最大值，第二个是每行的最大值所在下标
                loss.backward()
                optimizer.step()

                global_iter_num = epoch * len(trainDataLoader) + step + 1  # 计算当前是从训练开始时的第几步(全局迭代次数)
                if global_iter_num % log_step_interval == 0 and show:
                    # 控制台输出一下
                    print("global_step:{}, loss".format(global_iter_num))
                    # 在测试集上预测并计算正确率
                    valid_acc = self.test(model, validDataLoader, useGPU, criterion)
                    history.log((epoch, step), accuracy=valid_acc)
                    with canvas:
                        canvas.draw_plot(history["accuracy"])
            valid_acc = self.test(model, validDataLoader, useGPU, criterion)
            print(' valid Acc: {:.4f}'.format(valid_acc))

            self.save_Model(valid_acc=valid_acc)  # 保存当前检查点的信息，并更新基本信息
        self.model = model.cpu()

    def predict(self, batchSize=2, useGPU=torch.cuda.is_available(), root=None):
        """预测某个文件夹里的所有图片"""
        if useGPU:
            model = self.model.cuda()
        else:
            model = self.model
        model.eval()
        testData = TestData(root=root, transform_propose=self.data_transform)
        testDataLoader = DataLoader(testData, batch_size=batchSize, shuffle=False)
        picture_paths = []
        label_list = []
        for picture_data, path in testDataLoader:
            picture_paths.extend(path)
            if useGPU:
                picture_data = picture_data.cuda()
            score = model(picture_data)
            label = score.max(dim=1)[1].data.tolist()
            label_list.extend(label)
        results = pandas.DataFrame()
        results["picture"] = picture_paths
        results["predict_type"] = label_list
        results["predict_type"].astype(dtype="int")
        return results.replace({"predict_type": self.index2Name})

    def predictOnePicture(self, path):
        """
        预测某张照片，path是图片路径
        """
        pass


class BaseModelClassify(Classify):
    def __init__(self,
                 name: str = None,
                 type_list: list = None,
                 baseType: str = "AlexNet",
                 netType: str = None,
                 use_pretrained: bool = None,
                 continue_train: bool = None):
        super().__init__(name, type_list)
        self.baseType = baseType  # 这个属性如果不是None,说明本模型是
        self.netType = netType
        self.use_pretrained = use_pretrained
        self.continue_train = continue_train
        modelFactory = ModelFactory(baseType=self.baseType,
                                    netType=self.netType,
                                    type_num=len(self.index2Name),
                                    use_pretrained=self.use_pretrained,
                                    continue_train=self.continue_train,
                                    useGpu=False)
        self.model = modelFactory.getNet()
        self.input_size = modelFactory.input_size
        self.data_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
        ])


class MyModelClassify(Classify):
    def __init__(self,
                 name: str = None,
                 type_list: list = None,
                 list_: list = None
                 ):
        super().__init__(name, type_list)
        self.model = nn.Sequential(
            *list_
        )

        self.input_size = 32
        self.data_transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 均值，标准差
        ])
