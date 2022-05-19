"""
这个文件负责书写各个层在GUI上的展示
要保证这里的层和nn里的层同名
"""
from torch import nn
from .helpGUI import *
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
import inspect

# 这个字典采用手工写的方式，以便去除不想支持的层
allClasses = {
    ("卷积", "_Conv", (238, 232, 170)): [
        "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d",
        "LazyConv1d"
    ],
    ("激活", "_Activation", (224, 255, 225)): [
        "Threshold", "ReLU", "RReLU", "Hardtanh", "ReLU6", "Hardsigmoid", "SiLU", "Mish",
        "Hardswish", "ELU", "CELU", "SELU", "GLU", "Hardshrink", "LeakyReLU", "Softplus",
        "Softshrink", "MultiheadAttention", "PReLU", "Softmin", "Softmin",
        "Softmax", "LogSoftmax"
    ],
    ("池化", "_Pooling", (245, 245, 220)): [
        "MaxPool1d", "MaxPool2d", "MaxPool3d", "MaxUnpool1d", "MaxUnpool2d", "MaxUnpool3d", "AvgPool1d", "AvgPool2d",
        "AvgPool3d", "FractionalMaxPool2d", "FractionalMaxPool3d", "LPPool1d", "LPPool2d", "AdaptiveMaxPool1d",
        "AdaptiveMaxPool2d", "AdaptiveMaxPool3d"
    ],
    ("随机失活", "_Dropout", (255, 255, 240)): [
        "Dropout", "Dropout2d", "Dropout3d", "AlphaDropout", "FeatureAlphaDropout"
    ],
    ("拉平", "_Flatten", (255, 240, 245)): [
        "Flatten"
    ],
    ("全连接", "_Linear", (248, 248, 255)): [
        "Linear", "Bilinear", "LazyLinear"
    ]

}


class _layer:
    def __init__(self):
        self.command = None
        self.attributes = None
        self.nnLayer = None  # 对应的pytorch里的层的类型，应当在子类中负责设置
        self.layer = None  # 建立完毕的层，可以直接弄到nn.Sequential里
        self.color = (100, 149, 237)  # 默认背景颜色

    def makeUp(self):
        questionDialog = QuestionWindow(getFunctionCommand(self.nnLayer.__init__))
        if not questionDialog.exec_():
            return False
        else:
            self.command = questionDialog.getInformation()
            return True

    def get_nn_layer(self):
        try:
            self.layer = self.nnLayer(**self.command)
        except Exception as ex:
            print(ex)
        return self.layer

    def refreshLayOut(self):
        while self.layOut.count():
            child = self.layOut.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.fillLayOut(self.layOut)

    def get_color(self):
        for settings in allClasses:
            if str(self.__class__.__name__) in allClasses[settings]:
                self.color = settings[2]
                return

    def getWidget(self):
        # 设置边框、背景颜色和大小
        widget = QFrame()
        self.get_color()
        widget.setStyleSheet("border:1px solid black;background-color: rgb({:d}, {:d}, {:d});".format(*self.color))
        widget.setMinimumHeight(50)
        h_layOut = QHBoxLayout()  # 横向,包括（图片和名称(TODO)；结构信息；按钮(TODO)）
        widget.setLayout(h_layOut)
        name_layOut = QVBoxLayout()
        h_layOut.addLayout(name_layOut)
        name_Label = QLabel(str(self.__class__.__name__))
        name_Label.setAlignment(Qt.AlignCenter)  # 居中
        name_Label.setFont(QFont("Roman times", 20, QFont.Bold))
        name_layOut.addWidget(name_Label)

        v_layOut = QVBoxLayout()
        h_layOut.addLayout(v_layOut)
        for attribute in self.command:
            if self.command[attribute] is None:
                continue
            label = QLabel()
            label.setStyleSheet("border:0px solid black")  # 去除内部标签的边框，抵消给外面加边框而带来的影响
            label.setText(str(attribute) + "  :  " + str(self.command[attribute]))
            v_layOut.addWidget(label)
        return widget

    def fillLayOut(self, layOut):
        name_label = QLabel()
        name_label.setText("Conv2d")
        layOut.addWidget(name_label)


"""
这部分代码能够根据allClasses里的信息，针对每个nn里面的层，自动构造相应的类，而这些类是实现pytorch接口的关键。
"""
for type_name in allClasses:
    className = type_name[1]
    sstr1 = "class " + className + "(_layer):\n\tdef __init__(self):\n\t\tsuper().__init__()"
    exec(sstr1)
    for sub_name in allClasses[type_name]:
        sstr2 = "class " + sub_name + "(" + className + "):" + "\n\tdef __init__(self):\n\t\tsuper().__init__()\n\t\tself.nnLayer = nn." + sub_name
        exec(sstr2)
