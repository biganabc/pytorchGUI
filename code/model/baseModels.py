import os
from torch.utils import data
import numpy as np
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
from torch import nn
import torch.optim as optim


class ModelFactory:
    @staticmethod
    def getAllBaseModelType() -> dict:
        """
        返回一个字典，是所有可以建立的基本模型
        把这个东西直接写在代码里，而不是写在配置文件里，是因为建立基本模型的部分的代码也要用到这个信息。
        """
        return {
            "ResNet": ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d",
                       "resnext101_32x8d", "wide_resnet50_2", "wide_resnet101_2"],
            "AlexNet": [],
            "GoogLeNet": [],
            "ShuffleNetV2": ["ShuffleNetV2_x0.5", "ShuffleNetV2_x1.0", "ShuffleNetV2_x1.5", "ShuffleNetV2_x2.0"],
            "Vgg": ['vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn'],
            "SqueezeNet": ['squeezenet1_0', 'squeezenet1_1'],
            "DenseNet": ['densenet121', 'densenet169', 'densenet201', 'densenet161'],
            "Inception_v3": ['inception_v3_google'],
        }

    def __init__(self,
                 baseType="ResNet",
                 netType=None,
                 type_num: int = None,
                 use_pretrained: bool = True,
                 continue_train: bool = False,
                 useGpu: bool = True,
                 ):
        self.baseType = baseType
        self.netType = netType
        self.type_num = type_num
        self.use_pretrained = use_pretrained
        self.continue_train = continue_train
        self.useGpu = useGpu
        self.input_size = 224  # 大多数网络要求输入的图片尺寸都是224，如果不是，则在方法体内部修改

    def getNet(self):
        models2Function = {
            "ResNet": self.getResNet,
            "AlexNet": self.getAlexNet,
            "GoogLeNet": self.getGoogLeNet,
            "ShuffleNetV2": self.getShuffleNetV2,
            "Vgg": self.getVgg,
            "SqueezeNet": self.getSqueeze,
            "DenseNet": self.getDenseNet,
            "Inception_v3": self.getInception_v3,
        }
        if self.baseType in models2Function:
            resultModel = models2Function[self.baseType]()
        else:
            raise Exception("没有这种类型的卷积神经网络！")
        if self.useGpu:
            resultModel = resultModel.cuda()
        return resultModel

    def getInception_v3(self):
        inception_v3Dict = {
            'inception_v3_google': models.inception_v3,
        }
        if self.netType is None or self.netType not in inception_v3Dict:
            self.netType = 'inception_v3_google'
        model = inception_v3Dict[self.netType](pretrained=self.use_pretrained, progress=True, aux_logits=False)
        if not self.continue_train:
            for param in model.parameters():
                param.requires_grad = False
        num_ = model.fc.in_features  # 上一层的全连接层的输入数据维度
        # 自己重新构建全连接层
        model.fc = nn.Sequential(
            nn.Linear(num_, self.type_num),  # 首先来个一般的全连接
        )
        self.input_size = 299
        return model

    def getDenseNet(self):
        DenseDict = {
            'densenet121': models.densenet121,
            'densenet169': models.densenet169,
            'densenet201': models.densenet201,
            'densenet161': models.densenet161,
        }
        if self.netType is None or self.netType not in DenseDict:
            self.netType = 'densenet121'
        model = DenseDict[self.netType](pretrained=self.use_pretrained, progress=True)
        if not self.continue_train:
            for param in model.parameters():
                param.requires_grad = False
        num_ = model.classifier.in_features  # 上一层的全连接层的输入数据维度
        # 自己重新构建全连接层
        model.classifier = nn.Sequential(
            nn.Linear(num_, self.type_num),  # 首先来个一般的全连接
        )
        return model

    def getSqueeze(self):
        SqueezeDict = {
            'squeezenet1_0': models.squeezenet1_0,
            'squeezenet1_1': models.squeezenet1_1,
        }
        if self.netType is None or self.netType not in SqueezeDict:
            self.netType = 'squeezenet1_0'
        model = SqueezeDict[self.netType](pretrained=self.use_pretrained, progress=True)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5)
        )
        if not self.continue_train:
            for parm in model.parameters():
                parm.requires_grad = False

        model.classifier.add_module("new_conv", nn.Conv2d(512, self.type_num, kernel_size=1))
        model.classifier.add_module("new_relu", nn.ReLU(inplace=True))
        model.classifier.add_module("new_adaptive", nn.AdaptiveAvgPool2d((1, 1)))
        return model

    def getVgg(self):

        VggDict = {
            'vgg11': models.vgg11,
            'vgg13': models.vgg13,
            'vgg16': models.vgg16,
            'vgg19': models.vgg19,
            'vgg11_bn': models.vgg11_bn,
            'vgg13_bn': models.vgg13_bn,
            'vgg16_bn': models.vgg16_bn,
            'vgg19_bn': models.vgg19_bn,
        }
        if self.netType is None or self.netType not in VggDict:
            self.netType = 'vgg11'
        model = VggDict[self.netType](pretrained=self.use_pretrained, progress=True)
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout()
        )
        if not self.continue_train:
            for parm in model.parameters():
                parm.requires_grad = False
        model.classifier.add_module("new_fc", nn.Linear(4096, self.type_num))
        return model

    def getShuffleNetV2(self):
        ShuffleNetV2Dict = {
            "ShuffleNetV2_x0.5": models.shufflenet_v2_x0_5,
            "ShuffleNetV2_x1.0": models.shufflenet_v2_x1_0,
            "ShuffleNetV2_x1.5": models.shufflenet_v2_x1_5,
            "ShuffleNetV2_x2.0": models.shufflenet_v2_x2_0,
        }
        if self.netType is None or self.netType not in ShuffleNetV2Dict:
            self.netType = "ShuffleNetV2_x0.5"
        if self.netType == "ShuffleNetV2_x1.5" or self.netType == "ShuffleNetV2_x2.0":
            self.use_pretrained = False  # 这两种神经网络没有可供下载的现成参数
        model = ShuffleNetV2Dict[self.netType](pretrained=self.use_pretrained, progress=True)
        if not self.continue_train:
            for parm in model.parameters():
                parm.requires_grad = False
        num_ = model.fc.in_features
        model.fc = nn.Linear(num_, self.type_num)
        return model

    def getGoogLeNet(self):
        model = models.googlenet(pretrained=self.use_pretrained,
                                 progress=True)
        if not self.continue_train:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Linear(1024, self.type_num)
        return model

    def getAlexNet(self, ):
        model = models.alexnet(pretrained=self.use_pretrained,
                               progress=True)
        model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        if not self.continue_train:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier.add_module("newfc", nn.Linear(4096, self.type_num))
        self.input_size = 227
        return model

    def getResNet(self, ) -> models.resnet.ResNet:
        resnetDict = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
            "resnext50_32x4d": models.resnext50_32x4d,
            "resnext101_32x8d": models.resnext101_32x8d,
            "wide_resnet50_2": models.wide_resnet50_2,
            "wide_resnet101_2": models.wide_resnet101_2
        }
        if self.netType is None or self.netType not in resnetDict:
            self.netType = "resnet18"
        """
        type_num : 分类任务要求分成几类
        use_pretrained : 是否要拷贝别人已经复制好的参数
        continue_train : 是否直接用人家的参数（即是否不继续训练）
        返回值 : (修改后的Resnet152模型，要求输入的图片尺寸）
        """
        model = resnetDict[self.netType](pretrained=self.use_pretrained,
                                         progress=True)
        if not self.continue_train:
            for param in model.parameters():
                param.requires_grad = False
        num_ = model.fc.in_features  # 上一层的全连接层的输入数据维度
        # 自己重新构建全连接层
        model.fc = nn.Sequential(
            nn.Linear(num_, self.type_num),  # 首先来个一般的全连接
        )
        return model
