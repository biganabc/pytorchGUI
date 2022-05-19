import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import os
import shutil
import pyqtgraph as pg

from ..model.classificiation import *
from ..model.baseModels import *
from .helpGUI import *
from .modelGUI import *
from .train_thread import *

"""
对数据的要求：
    ①所有模型不能重名。经典模型里，所有经典模型的名称以及所有经典模型子类的名称都不能有重复的。
"""


class MakeSelfModelWindow(QDialog):
    """
    当用户选择新建一个自定义模型时，会弹出本对话框，最终可以返回关于新建模型的命令信息。
    """

    def __init__(self, command):
        super().__init__(None)
        addBackColor(self, (255, 235, 205))
        # 所有可以添加的层
        self.command = command

        # 所有已经添加的层,里面每个东西都是能够直接放在nn.Sequential里的。
        self.hasLayers = []
        self.nets = []  # 这个是所有可以直接装在nn.Sequential里的层
        self.setMinimumHeight(900)
        v_layOut = QVBoxLayout()
        self.setLayout(v_layOut)
        h_layOut = QHBoxLayout()
        v_layOut.addLayout(h_layOut)
        self.leftLayOut = QVBoxLayout()  # 左侧，负责显示所有已经添加的层
        h_layOut.addLayout(self.leftLayOut)
        self.drawLeft(self.leftLayOut)
        rightWidget = QWidget()
        self.rightLayOut = QVBoxLayout()  # 右侧，负责装载所有的添加层的按钮
        rightWidget.setLayout(self.rightLayOut)
        self.drawRight(self.rightLayOut)
        h_layOut.addWidget(rightWidget)
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        v_layOut.addWidget(buttonBox)
        self.__center()

    def __center(self):
        """
        控制窗口显示在屏幕中心的方法
        """
        qr = self.frameGeometry()  # 获得窗口
        cp = QDesktopWidget().availableGeometry().center()  # 获得屏幕中心点
        qr.moveCenter(cp)  # 显示到屏幕中心
        self.move(qr.topLeft())

    def drawLeft(self, leftLayOut):
        leftScroll = QScrollArea()  # 左侧的滑动区域
        leftLayOut.addWidget(leftScroll)
        smallLayOut = QVBoxLayout()
        Title = QLabel()
        Title.setText("模型结构")
        Title.setAlignment(Qt.AlignCenter)  # 居中
        Title.setFont(QFont("Roman times", 15, QFont.Bold))
        addBackColor(Title, (240, 248, 255))
        smallLayOut.addWidget(Title)
        for net in self.hasLayers:
            smallLayOut.addWidget(net.getWidget())
        tempWidget = QWidget()
        addBackColor(tempWidget, (240, 248, 255))
        tempWidget.setMinimumWidth(400)
        tempWidget.setLayout(smallLayOut)
        leftScroll.setWidget(tempWidget)

    def drawRight(self, rightLayOut: QVBoxLayout):
        tree = QTreeWidget()  # 用树形结构表示所有可以添加的层
        tree.setColumnCount(1)
        tree.setHeaderLabels([' '])

        for settings in allClasses:
            chinese_name = settings[0]
            node = QTreeWidgetItem()
            node.setText(0, chinese_name)
            tree.addTopLevelItem(node)
            for sub_name in allClasses[settings]:
                sub_node = QTreeWidgetItem()
                sub_node.setText(0, sub_name)
                node.addChild(sub_node)
        rightLayOut.addWidget(tree)
        tree.itemClicked['QTreeWidgetItem*', 'int'].connect(self.addLayer)

    def addLayer(self, item):
        """
        当用户点击了一个有效的层的时候，调用这个方法
        分发处理函数
        """
        layer_name = item.text(0)  # 添加的层的名称
        loc = locals()
        exec("targetClass = " + layer_name)
        targetClass = loc["targetClass"]
        newLayer = targetClass()
        if newLayer.makeUp():  # 如果用户的确建立了这个模型
            try:
                newNet = newLayer.get_nn_layer()
                self.nets.append(newNet)
            except Exception as ex:
                print("参数非法！")
                print(ex)
                return
            self.hasLayers.append(newLayer)
            refreshLayOut(self.leftLayOut, self.drawLeft)

    def makeModel(self):
        try:

            model = Classify.factory(name=self.command["name"], type_list=self.command["types"], layers=self.nets)
            model.save_Model(is_initial=True)
            return model
        except Exception as ex:
            print(ex)


class MainWindow(QMainWindow):
    picture_root = "data//pictures"  # 图片存储的路径，这里是相对于main函数的路径
    model_root = "data//models"  # 模型存储路径，这里是相对于本文件的路径
    icon_path = "data//settings//title.jpg"  # 图标文件

    def __init__(self):
        super().__init__()
        self.resize(1200, 800)
        self.setWindowIcon(QIcon(MainWindow.icon_path))
        addBackColor(self, (255, 235, 205))
        self.__center()
        self.setWindowTitle('卷积神经网络平台')
        self.__add_status_bar()  # 设置底部消息栏
        self.__setLayOut()  # 设置布局管理器
        self.all_tasks = []  # 当前所有任务
        self.busy_mark = {
            "train": 0,
            "predict": 0,
            "copy": 0,
            "make": 0
        }
        self.all_taskScrollArea = None
        self.alldeleteDataButtons = []  # 所有删除数据的按钮
        self.train_thread1 = trainThread()  # 这个线程用于训练模型
        self.predict_thread = PredictThread()  # 这个线程用于预测图片
        self.addData_thread = AddDataThread()  # 这个线程用于添加数据集
        self.addModel_thread = AddModelThread()  # 这个线程用于新建模型（经典模型）
        self.leftWidget = None
        self.__show_data()
        self.__show_model()
        self.rightWidget = QWidget()
        addBackColor(self.rightWidget, (240, 248, 255))
        self.rightWidget.setMaximumWidth(400)  # 限制右侧宽度
        self.rightLayOut = QVBoxLayout()
        self.rightWidget.setLayout(self.rightLayOut)
        self.drawRight(self.rightLayOut)
        self.gridLayout.addWidget(self.leftWidget, 1, 1, 1, 1)
        self.gridLayout.addWidget(self.middleWidget, 1, 2, 1, 1)
        self.gridLayout.addWidget(self.rightWidget, 1, 3, 1, 2)
        self.show()

    def __center(self):
        """
        控制窗口显示在屏幕中心的方法
        """
        qr = self.frameGeometry()  # 获得窗口
        cp = QDesktopWidget().availableGeometry().center()  # 获得屏幕中心点
        qr.moveCenter(cp)  # 显示到屏幕中心
        self.move(qr.topLeft())

    def fill_dataLayOut(self, layOut):
        title = QLabel('所有数据')
        title.setAlignment(Qt.AlignCenter)  # 居中
        title.setFont(QFont("Roman times", 20, QFont.Bold))
        layOut.addWidget(title)
        self.alldeleteDataButtons = []
        model_scrollArea = QScrollArea()
        layOut.addWidget(model_scrollArea)
        list_layOut = QVBoxLayout()
        self.dirs = os.listdir(MainWindow.picture_root)  # 得到文件夹里的所有图片集的名称
        for dir_ in self.dirs:
            dataWidget = QWidget()
            dataWidgetLayOut = QHBoxLayout()
            dataWidget.setLayout(dataWidgetLayOut)
            name_label = QLabel(dir_)
            dataWidgetLayOut.addWidget(name_label)
            name_label.setFont(QFont("Roman times", 20, QFont.Bold))  # 设置字体
            # 数据名称样式设置
            name_label.setFrameShadow(QFrame.Raised)
            name_label.setFrameShape(QFrame.Box)
            name_label.setStyleSheet(
                'border-width: 1px;border-style: solid;border-color: rgb(255, 170, 0);border-radius:10px;background-color: rgb(255, 239, 213);')
            list_layOut.addWidget(dataWidget)

            # 负责显示数据集的大小和图片张数
            data_informations = QVBoxLayout()
            count = len(os.listdir(os.path.join(MainWindow.picture_root, dir_)))
            count_label = QLabel("图片张数: " + str(count))
            data_informations.addWidget(count_label)
            dataWidgetLayOut.addLayout(data_informations)
            deleteButton = QPushButton("删除")
            self.alldeleteDataButtons.append(deleteButton)
            dataWidgetLayOut.addWidget(deleteButton)
            deleteButton.clicked.connect(self.deleteData)
        tempWidget = QWidget()
        tempWidget.setLayout(list_layOut)
        model_scrollArea.setWidget(tempWidget)
        addDataButton = QPushButton("导入数据")
        layOut.addWidget(addDataButton)
        addDataButton.clicked.connect(self.addDataDialog)

    def deleteData(self):
        button = self.sender()  # 用户按下了这个按钮
        data_to_delete = self.dirs[self.alldeleteDataButtons.index(button)]  # 用户想要训练这个模型 （它是一个Classify）
        shutil.rmtree(os.path.join(MainWindow.picture_root, data_to_delete))
        refreshLayOut(self.dataLayOut, self.fill_dataLayOut)

    def __show_data(self):
        """制作最左侧数据栏"""
        widget = QWidget()
        self.leftWidget = widget
        addBackColor(widget, (240, 248, 255))
        widget.setMaximumWidth(400)
        self.dataLayOut = QVBoxLayout(widget)  # 为了避免方法名和属性名重复，所以后面加个_
        self.fill_dataLayOut(self.dataLayOut)  # 填充左侧数据栏的内容，单独提炼出函数是为了方便刷新

    def addDataDialog(self):
        """
        当用户添加数据集的时候，会调用这个方法
        """
        if self.busy_mark["copy"] == 1:
            QMessageBox(QMessageBox.Warning, '失败', "线程正忙").exec_()
            return
        picture_dir = QFileDialog.getExistingDirectory(self, 'Open file', '/home')  # 图片根目录
        if picture_dir == "":  # 没有选择任何文件夹
            return
        dataNameQuestion = QuestionWindow([("数据集名称", str, os.path.split(picture_dir)[-1])])
        dataNameQuestion.exec_()
        dataSetName = dataNameQuestion.getInformation()["数据集名称"]
        self.dirs = os.listdir(MainWindow.picture_root)  # 得到文件夹里的所有图片集的名称
        if dataSetName in self.dirs:
            QMessageBox(QMessageBox.Warning, '添加数据集失败', "不能与现有数据集同名 \"" + dataSetName + "\"").exec_()
            return
        self.addData_thread.dataPath = picture_dir
        self.addData_thread.allPicturePath = MainWindow.picture_root
        self.addData_thread.dataName = dataSetName
        addDataTask = Task("添加数据", self.addData_thread, AddDataWidget(
            [lambda: refreshLayOut(self.dataLayOut, self.fill_dataLayOut),
             lambda: refreshLayOut(self.rightLayOut, self.drawRight)]), self.all_tasks)
        addDataTask.start()
        self.busy_mark["copy"] = 1

    def __show_model(self):
        """
        这个方法负责绘制模型区域
        """
        widget = QWidget()
        widget.setMaximumWidth(400)
        # widget.setStyleSheet("background-color: rgb(240, 248, 255);")
        addBackColor(widget, (240, 248, 255))
        # 设置布局管理器
        self.modelLayOut = QVBoxLayout(widget)
        self.__fill_models(self.modelLayOut)
        self.middleWidget = widget

    def continue_train(self):
        """
        当用户按下继续训练的时候，调用这个函数
        """
        if self.busy_mark["train"] == 1:
            QMessageBox(QMessageBox.Warning, '失败', "线程正忙").exec_()
            return
        button = self.sender()  # 用户按下了这个按钮
        model_to_train = self.model_list[self.train_button_list.index(button)]  # 用户想要训练这个模型 （它是一个Classify）
        model_to_train.getModel()
        dialog = QuestionWindow(getFunctionCommand(model_to_train.train))
        if not dialog.exec_():
            return
        command = dialog.getInformation()
        self.train_thread1.classify = model_to_train
        self.train_thread1.command = command
        trainWidget1 = trainWidget([lambda: refreshLayOut(self.rightLayOut, self.drawRight)])
        self.train_thread1.accuracySignal.connect(trainWidget1.drawAccuracy)
        trainTask = Task("训练", self.train_thread1, trainWidget1, self.all_tasks)
        refreshLayOut(self.rightLayOut, self.drawRight)
        trainTask.start()
        self.busy_mark["train"] = 1

    def __fill_models(self, layOut):
        # 设置标题
        title = QLabel('我的模型')
        title.setAlignment(Qt.AlignCenter)  # 居中
        title.setFont(QFont("Roman times", 20, QFont.Bold))
        layOut.addWidget(title)

        model_scrollArea = QScrollArea()  # 装载所有模型的滑动窗口区域
        layOut.addWidget(model_scrollArea)
        models_layOut = QVBoxLayout()  # 承载所有模型信息的布局管理器，每个模型对应的QWidget都要往这个models_layOut里面放

        def show_structure():
            """
            展示模型结构
            """
            button = self.sender()  # 用户按下了这个按钮
            target_model = self.model_list[show_structure_buttons.index(button)]  # 用户想要训练这个模型 （它是一个Classify）
            target_model.showModelStructure()

        self.model_list = []
        self.train_button_list = []  # 把所有“继续训练”的按钮放在这个列表里，便于当按钮按下时知道是哪个按钮
        self.delete_button_list = []  # 把所有“删除模型”的按钮放在这个列表里
        self.predict_button_list = []  # 所有“预测照片"的按钮
        show_structure_buttons = []

        for model_name in os.listdir(MainWindow.model_root):
            model = Classify.factory(is_new=False, name=model_name)
            self.model_list.append(model)

        for index, model in enumerate(self.model_list):
            model_widget = QWidget()  # 模型信息栏
            if index % 2 == 0:
                model_widget.setStyleSheet("background-color: rgb({:d}, {:d}, {:d});".format(*(245, 245, 245)))
            else:
                model_widget.setStyleSheet("background-color: rgb({:d}, {:d}, {:d});".format(*(248, 248, 255)))
            model_widget.setMinimumHeight(100)  # 设置模型信息栏的最小高度
            h_layOut = QHBoxLayout()
            model_widget.setLayout(h_layOut)
            informations = QWidget()
            # 通过同时固定最小宽度和最大宽度来固定信息部分的宽度
            informations.setMinimumWidth(200)
            informations.setMaximumWidth(200)
            buttons = QWidget()
            h_layOut.addWidget(informations)
            h_layOut.addWidget(buttons)

            informations_layOut = QVBoxLayout()
            buttons_layOut = QVBoxLayout()
            informations.setLayout(informations_layOut)
            buttons.setLayout(buttons_layOut)

            name_label = QLabel("模型名称 : " + model.name)
            informations_layOut.addWidget(name_label)

            show_structure_button = QPushButton("展示结构")
            show_structure_buttons.append(show_structure_button)
            show_structure_button.clicked.connect(show_structure)
            informations_layOut.addWidget(show_structure_button)

            train_button = QPushButton("继续训练")
            self.train_button_list.append(train_button)
            train_button.clicked.connect(self.continue_train)

            use_button = QPushButton("预测图片类型")
            self.predict_button_list.append(use_button)
            use_button.clicked.connect(self.predict_picture)

            delete_button = QPushButton("删除")
            self.delete_button_list.append(delete_button)
            delete_button.clicked.connect(self.delete_model)

            buttons_layOut.addWidget(train_button)
            buttons_layOut.addWidget(use_button)
            buttons_layOut.addWidget(delete_button)

            models_layOut.addWidget(model_widget)

        tempWidget = QWidget()  # 用于承载滑动窗口区域的内容，保证滑动窗口可以正常显示
        tempWidget.setLayout(models_layOut)
        model_scrollArea.setWidget(tempWidget)
        addModelButton = QPushButton("新建模型")
        addModelButton.clicked.connect(self.addModelButton)
        layOut.addWidget(addModelButton)

    def predict_picture(self):
        """
        预测一张图片或者一批图片
        """
        model_to_predict = self.model_list[self.predict_button_list.index(self.sender())]  # 用户想要训练这个模型 （它是一个Classify）
        model_to_predict.getModel()
        question = QuestionWindow([("请选择预测类型", myChoose, myChoose(["预测一组照片", "预测一张照片"], 0))])
        if not question.exec_():
            return
        answer = question.getInformation()
        choice = answer["请选择预测类型"]
        if choice is None:  # 正常的情况下不会发生
            return
        if choice == 0:
            if self.busy_mark["predict"] == 1:
                QMessageBox(QMessageBox.Warning, '失败', "线程正忙").exec_()
                return
            picture_dir = QFileDialog.getExistingDirectory(self, 'Open file', '/home')  # 图片根目录
            if picture_dir == "":  # 没有选择任何文件夹
                return
            self.predict_thread.classify = model_to_predict
            self.predict_thread.path = picture_dir
            self.predict_thread.type = 0
            predictWidget = PredictWidget([lambda: refreshLayOut(self.rightLayOut, self.drawRight)])
            predictTask = Task("预测一批图片", self.predict_thread, predictWidget, self.all_tasks)
            predictTask.start()
            self.busy_mark["predict"] = 1
        elif choice == 1:
            QMessageBox(QMessageBox.Warning, '提供了更高级的功能', "您可以把它放到一个文件夹里，然后选择\"预测一批图片\"").exec_()
        else:
            print(choice)

    def delete_model(self):
        """
        删除一个模型
        """
        button = self.sender()  # 用户按下了这个按钮
        model_to_delete = self.model_list[self.delete_button_list.index(button)]  # 用户想要训练这个模型 （它是一个Classify）
        name = model_to_delete.name  # 要删除的模型名称
        shutil.rmtree(os.path.join(MainWindow.model_root, name))
        refreshLayOut(self.modelLayOut, self.__fill_models)

    def addModelButton(self):
        """
        当用户点击新建模型的按钮时，会调用这个方法
        """

        def getDirs():
            return self.dirs

        class RadioDialog(QDialog):
            """
            询问用户要分类的数据，以及要建立基本模型还是经典模型
            """

            def __init__(self, parent=None):
                super().__init__(parent)
                self.setWindowIcon(QIcon(MainWindow.icon_path))
                layout = QVBoxLayout(self)
                formLayOut = QFormLayout()
                layout.addLayout(formLayOut)
                self.model_name_line = QLineEdit()
                formLayOut.addRow("模型名称 : ", self.model_name_line)

                tree = QTreeWidget()
                tree.setColumnCount(1)
                tree.setHeaderLabels(['所有类型'])
                formLayOut.addRow("分类数据", tree)
                self.types_mark = {}
                for dir_ in getDirs():
                    self.types_mark[dir_] = 0
                    node = QTreeWidgetItem(tree)
                    node.setText(0, str(dir_))
                    node.setCheckState(0, Qt.Unchecked)

                tree.itemChanged.connect(self.onclick)
                baseModelButton = QRadioButton("经典模型")
                baseModelButton.setChecked(Qt.Checked)
                selfModelButton = QRadioButton("自定义模型")
                buttonGroup = QButtonGroup()
                buttonGroup.addButton(baseModelButton, 1)
                buttonGroup.addButton(selfModelButton, 2)
                buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
                self.buttonGroup = buttonGroup
                choseModelType = QWidget()
                choseModelType.setLayout(QHBoxLayout())
                choseModelType.layout().addWidget(baseModelButton)
                choseModelType.layout().addWidget(selfModelButton)
                layout.addWidget(choseModelType)
                layout.addWidget(buttonBox)

                buttonBox.accepted.connect(self.accept)
                buttonBox.rejected.connect(self.reject)

            def onclick(self, item, column):
                if item.checkState(column) == Qt.Checked:
                    self.types_mark[item.text(0)] = 1
                elif item.checkState(column) == Qt.Unchecked:
                    self.types_mark[item.text(0)] = 0

            def getModelCommand(self):
                """
                返回打包后的用户关于训练模型的信息
                :return:
                """
                type_list = []
                for type_name in self.types_mark:
                    if self.types_mark[type_name] == 1:
                        type_list.append(type_name)
                return {
                    "name": self.model_name_line.text(),
                    "modelType": "None" if self.buttonGroup.checkedId() == -1
                    else "baseModel" if self.buttonGroup.checkedId() == 1 else "selfModel",
                    "types": type_list,
                }

        if self.busy_mark["make"] == 1:
            QMessageBox(QMessageBox.Warning, '失败', "线程正忙").exec_()
            return
        dialog = RadioDialog()
        if not dialog.exec_():  # 用户取消了新建模型
            return
        command = dialog.getModelCommand()
        if command["name"] in os.listdir(MainWindow.model_root):
            QMessageBox(QMessageBox.Warning, '创建失败', "不能与现有模型重名" + command["name"]).exec_()
            return
        if command["modelType"] == "baseModel":  # 用户想新建经典模型
            self.addBaseModel(command)
        elif command["modelType"] == "selfModel":
            self.addMyModel(command)
        else:
            pass

    def addBaseModel(self, command):
        """
        当用户选择新建经典模型的时候，会调用这个方法
        command 是一个字典，表明要创建的模型的设定
        {
            "name": self.model_name_line.text(),
            "modelType": "None" "baseModel"  "selfModel",
            "types": type_list,
        }
        """

        def addNamedModel(bastName: str = None, subName: str = None):
            """
            得知了用户按下的到底是哪个模型了
            """
            question = QuestionWindow([("下载已有参数(需要连外网)", bool, True), ("保持特征提取部分的参数", bool, True)])
            if not question.exec_():
                return
            result = question.getInformation()
            newCommand = {
                "name": command["name"],
                "type_list": command["types"],
                "baseType": bastName,
                "netType": subName,
                "use_pretrained": result["下载已有参数(需要连外网)"],
                "continue_train": (not result["保持特征提取部分的参数"]) or (
                    not result["下载已有参数(需要连外网)"]),
            }
            self.addModel_thread.command = newCommand
            newTask = Task("建立模型", self.addModel_thread,
                           AddModelWidget([lambda: refreshLayOut(self.modelLayOut, self.__fill_models),
                                           lambda: refreshLayOut(self.rightLayOut, self.drawRight)],
                                          newCommand["name"]), self.all_tasks)
            newTask.start()
            self.busy_mark["make"] = 1

        class BaseModelTypeDialog(QDialog):
            """
            这个对话框询问用户要新建哪种类型的经典模型
            """
            model_dict = ModelFactory.getAllBaseModelType()

            def __init__(self, parent=None):
                super().__init__(parent)
                tree = QTreeWidget()
                layOut = QVBoxLayout()
                self.setLayout(layOut)

                tree.setColumnCount(1)
                tree.setHeaderLabels(['所有模型'])
                layOut.addWidget(tree)

                for baseModelType in BaseModelTypeDialog.model_dict:
                    model_node = QTreeWidgetItem()
                    model_node.setText(0, str(baseModelType))
                    tree.addTopLevelItem(model_node)
                    for subType in BaseModelTypeDialog.model_dict[baseModelType]:
                        sub_node = QTreeWidgetItem()
                        sub_node.setText(0, str(subType))
                        model_node.addChild(sub_node)

                # 设置树形结构的点击处理
                tree.itemClicked['QTreeWidgetItem*', 'int'].connect(self.solve_clicked)

            def solve_clicked(self, item):
                # item 是被点击的QTreeWidgetItem，也就是可以建立的模型
                name = item.text(0)
                for baseModelName in BaseModelTypeDialog.model_dict:
                    if baseModelName == name:
                        if len(BaseModelTypeDialog.model_dict[name]) == 0:
                            self.close()
                            addNamedModel(bastName=name)
                            return
                    for subName in BaseModelTypeDialog.model_dict[baseModelName]:
                        if subName == name:
                            self.close()
                            addNamedModel(bastName=baseModelName, subName=subName)
                            return

        BaseModelTypeDialog().exec_()

    def addMyModel(self, command):
        """
        当用户选择新建自定义模型时，调用本方法。这会创建一个新的模型。
        """
        makeSelfModelWindow = MakeSelfModelWindow(command)
        if not makeSelfModelWindow.exec_():
            return
        self.addModel_thread.command = {"name": makeSelfModelWindow.command["name"],
                                        "type_list": makeSelfModelWindow.command["types"],
                                        "layers": makeSelfModelWindow.nets}
        newTask = Task("建立模型", self.addModel_thread,
                       AddModelWidget([lambda: refreshLayOut(self.modelLayOut, self.__fill_models),
                                       lambda: refreshLayOut(self.rightLayOut, self.drawRight)],
                                      makeSelfModelWindow.command["name"]), self.all_tasks)
        newTask.start()
        self.busy_mark["make"] = 1

    def drawRight(self, layOut):

        qlabel = QLabel("任务")
        qlabel.setAlignment(Qt.AlignCenter)  # 居中
        qlabel.setFont(QFont("Roman times", 20, QFont.Bold))
        layOut.addWidget(qlabel)
        if self.all_taskScrollArea is not None:
            self.all_taskScrollArea.deleteLater()
        self.all_taskScrollArea = QScrollArea()
        layOut.addWidget(self.all_taskScrollArea)
        tempWidget = QWidget()
        taskLayOut = QVBoxLayout()
        tempWidget.setLayout(taskLayOut)
        for mask in self.busy_mark:
            self.busy_mark[mask] = 0
        for task in self.all_tasks:
            taskLayOut.addWidget(task.widget)
            if task.state == 0:
                continue
            if isinstance(task.widget, trainWidget):
                self.busy_mark["train"] = 1
            elif isinstance(task.widget, PredictWidget):
                self.busy_mark["predict"] = 1
            elif isinstance(task.widget, AddDataWidget):
                self.busy_mark["copy"] = 1
            elif isinstance(task.widget, AddModelWidget):
                self.busy_mark["make"] = 1

        self.taskLayOut = taskLayOut
        self.all_taskScrollArea.setWidget(tempWidget)

    def __add_status_bar(self):
        """
        设置底部消息栏
        """
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("")

    def __setLayOut(self):
        """
        设置布局管理器
        """
        gridLayout = QGridLayout(self)
        gridLayout.setSpacing(10)
        widget = QWidget()
        widget.setLayout(gridLayout)
        self.setCentralWidget(widget)
        self.gridLayout = gridLayout
