import threading

from PyQt5.QtCore import *
import time
import torch
from PyQt5.QtWidgets import *
from ..data.dataSet import *
from ..model.classificiation import *
from .helpGUI import *
import pandas
import pyqtgraph as pg
import os
import shutil

modelPath = "data//models"  # 存储所有模型的那个文件夹(相对于main)


class TaskWidget(QWidget):
    def __init__(self, command_list):
        super().__init__()
        self.nameLabel = QLabel()
        self.closeButton = QPushButton("关闭")
        self.closeButton.setVisible(False)
        self.progressPbar = QProgressBar()
        self.command_list = command_list
        self.setLayout(QVBoxLayout())  # 最外层都是纵向布局管理器

    def refresh(self):
        for command in self.command_list:
            command()

    def finish(self, *args):
        self.closeButton.setVisible(True)


class Task:
    """
    这个类负责管理一个进程。把进程反馈的任务进度，以及相关信息绘制在界面上。
    """

    def __init__(self, name: str, thread: QThread, widget: TaskWidget, task_list: list):
        """
        command 是更新界面用的函数
        """

        self.name = name  # 这项任务的名称
        self.state = 0
        self.thread = thread
        self.widget = widget
        self.widget.nameLabel.setText(self.name)
        self.widget.setMaximumWidth(300)
        self.widget.setMinimumWidth(300)
        self.thread.progressSignal.connect(self.show_progress)
        self.thread.finishSignal.connect(self.finish)
        self.task_list = task_list
        task_list.append(self)
        self.widget.closeButton.clicked.connect(self.remove)
        self.widget.refresh()

    def start(self):
        self.widget.progressPbar.setValue(0)
        self.state = 1
        self.thread.start()

    def show_progress(self, num):
        self.widget.progressPbar.setValue(num)
        self.state = num

    def finish(self, result):
        self.widget.progressPbar.setValue(100)
        self.widget.progressPbar.setVisible(False)
        self.state = 0
        self.widget.finish(result)
        self.widget.refresh()

    def remove(self):
        self.task_list.pop(self.task_list.index(self))
        self.widget.refresh()
        self.widget.deleteLater()


class PredictWidget(TaskWidget):
    def __init__(self, command_list):
        super().__init__(command_list)
        self.setMinimumHeight(100)
        self.layout().addWidget(self.nameLabel)
        self.layout().addWidget(self.progressPbar)
        showButton = QPushButton("查看")
        showButton.clicked.connect(self.showPredictResult)
        hLayOut = QHBoxLayout()
        hLayOut.addWidget(showButton)
        hLayOut.addWidget(self.closeButton)
        self.layout().addLayout(hLayOut)
        showButton.setVisible(False)
        self.showButton = showButton
        self.dataFrame = None

    def finish(self, *args):
        super().finish()
        self.showButton.setVisible(True)
        self.dataFrame = args[0]

    def showPredictResult(self):
        dialog = QDialog()
        dialog.setWindowTitle("图片预测结果")
        dialog.setMinimumWidth(400)
        dialog.setMinimumHeight(800)
        dialog.setLayout(QVBoxLayout())
        dialog.layout().addWidget(tableWidget(self.dataFrame))
        dialog.exec_()


class trainWidget(TaskWidget):
    def __init__(self, command_list):
        super().__init__(command_list)
        self.setMaximumHeight(340)
        self.layout().addWidget(self.nameLabel)
        pg.setConfigOption('background', 'w')  # 背景设置为白色，这句话必须放在pg.PlotWidget() 之前，否则不起作用
        pg.setConfigOption('foreground', 'k')
        self.pw = pg.PlotWidget()
        self.pw.setYRange(0, 1.0)
        self.pw.setXRange(0, 10)
        self.plot_data = self.pw.plot([], [], symbol="star", symbolBrush='g')
        self.layout().addWidget(self.pw)
        self.layout().addWidget(self.progressPbar)
        self.x = []
        self.y = []
        self.layout().addWidget(self.closeButton)

    def drawAccuracy(self, x, y):
        self.x.append(x)
        self.y.append(y)
        count = len(self.x) // 10
        self.pw.setXRange(0, (count + 1) * 10)
        self.plot_data.setData(self.x, self.y, symbol="star", symbolBrush='g')


class AddDataWidget(TaskWidget):
    def __init__(self, command_list):
        super().__init__(command_list)
        self.layout().addWidget(self.nameLabel)
        self.layout().addWidget(self.progressPbar)
        self.layout().addWidget(self.closeButton)


class AddModelWidget(TaskWidget):
    def __init__(self, command_list, modelName):
        super().__init__(command_list)
        self.modelName = modelName
        self.layout().addWidget(self.nameLabel)
        self.layout().addWidget(self.progressPbar)
        self.layout().addWidget(self.closeButton)

    def finish(self, *args):
        super().finish()
        issuccess, accuracy = args[0]
        if not issuccess:
            shutil.rmtree(os.path.join(modelPath, self.modelName))
            QMessageBox(QMessageBox.Warning, '创建失败', "报错信息: " + accuracy).exec_()


class AddDataThread(QThread):
    progressSignal = pyqtSignal(int)
    finishSignal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.dataName = None  # 名称
        self.dataPath = None  # 旧的文件夹的路径
        self.allPicturePath = None  # 所有数据的路径

    def run(self):
        all_files = os.listdir(self.dataPath)
        len_ = len(all_files)
        os.mkdir(os.path.join(str(self.allPicturePath), str(self.dataName)))  # 创建文件夹
        for index, file in enumerate(all_files):
            shutil.copyfile(os.path.join(str(self.dataPath), str(file)),
                            os.path.join(str(self.allPicturePath), str(self.dataName), str(file)))
            self.progressSignal.emit((index + 1) * 100 / len_)
        self.finishSignal.emit(0)


class trainThread(QThread):
    progressSignal = pyqtSignal(int)
    accuracySignal = pyqtSignal((float, float))
    finishSignal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.classify = None
        self.command = True

    def run(self):
        x = []
        y = []
        useGPU = self.command["useGPU"]
        if useGPU:
            self.classify.model = self.classify.model.cuda()
        trainDataLoader, validDataLoader = self.classify.getDataLoader(batchSize=self.command["batchSize"])
        optimizer = self.classify.get_optimizer(lr=self.command["lr"])
        loss_function = self.classify.get_lossFunction()
        model = self.classify.model
        accuracy = self.classify.test(self.classify.model, validDataLoader, self.command["useGPU"], loss_function)
        x.append(0)
        y.append(accuracy)
        self.accuracySignal.emit(0, accuracy)
        for epoch in range(self.command["num_epochs"]):  # 训练的轮次的个数
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
                progress = epoch / self.command["num_epochs"] + step / len(trainDataLoader) / self.command["num_epochs"]
                self.progressSignal.emit(int(progress * 100))
            accuracy = self.classify.test(self.classify.model, validDataLoader, self.command["useGPU"], loss_function)
            self.classify.save_Model(valid_acc=accuracy)
            x.append(epoch + 1)
            y.append(accuracy)
            self.accuracySignal.emit(epoch + 1, accuracy)

        self.classify.model = self.classify.model.cpu()
        self.finishSignal.emit(0)


class PredictThread(QThread):
    progressSignal = pyqtSignal(int)  # 定义信号对象,传递值为str类型，使用int，可以为int类型
    finishSignal = pyqtSignal(pandas.DataFrame)

    def __init__(self):
        self.classify = None
        super().__init__()
        self.path = None  # 文件路径
        self.type = None  # 表明要预测的是一批照片还是一张照片

    def run(self):
        useGPU = torch.cuda.is_available()
        if useGPU:
            self.classify.model = self.classify.model.cuda()
        model = self.classify.model
        model.eval()
        testDataLoader = DataLoader(TestData(root=self.path, transform_propose=self.classify.data_transform),
                                    batch_size=2, shuffle=False)
        picture_paths = []
        label_list = []

        for index, (picture_data, path) in enumerate(testDataLoader):
            picture_paths.extend(path)
            if useGPU:
                picture_data = picture_data.cuda()
            score = model(picture_data)
            label = score.max(dim=1)[1].data.tolist()
            label_list.extend(label)
            self.progressSignal.emit(index * 100 / len(testDataLoader))
        results = pandas.DataFrame()
        results["picture"] = [os.path.split(picture_path)[-1] for picture_path in picture_paths]
        results["predict_type"] = label_list
        results["predict_type"].astype(dtype="int")
        self.classify.model = self.classify.model.cpu()
        self.finishSignal.emit(results.replace({"predict_type": self.classify.index2Name}))


class AddModelThread(QThread):
    """
    这个线程负责处理建立基本模型的情况。因为建立基本模型时，可能需要从网上下载，为了防止卡顿，所以要用线程
    """
    progressSignal = pyqtSignal(int)
    finishSignal = pyqtSignal(tuple)

    def __init__(self):
        super().__init__()
        self.command = None

    def run(self):
        self.progressSignal.emit(10)
        classificiation = Classify.factory(**self.command)
        self.progressSignal.emit(80)
        classificiation.save_Model(is_initial=True)  # 刚建立完毕，马上保存一下初始状态
        self.progressSignal.emit(90)
        bool_, acc = classificiation.selfTest()
        self.finishSignal.emit((bool_, str(acc)))
