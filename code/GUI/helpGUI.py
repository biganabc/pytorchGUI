from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import inspect
import numpy as np
import pandas as pd

"""
一些常用的GUI工具，书写在本文件里。
"""


def addBackColor(widget: QWidget, color: (int, int, int)):
    # 给widget设置背景颜色
    palette = QPalette()
    palette.setColor(widget.backgroundRole(), QColor(*color))  # 背景颜色
    widget.setPalette(palette)
    widget.setAutoFillBackground(True)


def refreshLayOut(layOut, fun_):
    """
    fun_是绘制这个layOut的函数，它应当只有一个参数layOut
    """
    while layOut.count():
        child = layOut.takeAt(0)
        if child.widget():
            child.widget().deleteLater()
    fun_(layOut)


def getFunctionCommand(func, removeSelf=True):
    """
    分析一个函数，并且把它的参数按照(name,type,default)的形式返回（返回一个列表）
    removeSelf == True表示去除参数名称为self的那个参数
    """
    fullArgSpec = inspect.getfullargspec(func)
    args = fullArgSpec.args
    types = fullArgSpec.annotations
    defaults = fullArgSpec.defaults
    defaults = [None] * (len(args) - len(defaults)) + list(defaults)
    list_ = []
    for index, name in enumerate(args):
        type_ = None
        if name in types:
            type_ = types[name]
        list_.append((name, type_, defaults[index]))
    result = []
    for tuple_ in list_:
        if tuple_[0] != "self" or removeSelf is False:
            result.append(tuple_)
    return result


class myChoose:
    def __init__(self, list_, default_choice=0):
        self.list_ = list_  # 字符串列表
        self.default_choice = 0  # 默认选第0个


class taskProgress(QWidget):
    """
    展示某项任务的完成情况
    """

    def __init__(self, name, thread):
        super().__init__()
        self.setWindowTitle(name)
        self.setMinimumWidth(400)
        self.setMinimumHeight(60)
        self.done = False
        self.result = None
        self.name = name
        self.thread = thread
        self.thread.progressSignal.connect(self.show_progress)
        self.thread.finishSignal.connect(self.finish)
        self.bar = QProgressBar()
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.bar)
        self.thread.start()

    def show_progress(self, num):
        self.bar.setValue(num)

    def finish(self, result):
        self.done = True
        self.result = result

    def is_done(self):
        return self.done, self.result


class QuestionWindow(QDialog):
    icon_path = "data//settings//title.jpg"  # 图标文件

    def __init__(self, list_):
        """
        展示一个提问窗口，根据命令向用户问问题。最后可以返回问题的回答
        list_是一个列表，每个元素是(name,type,default)
        """
        super().__init__()
        self.setWindowIcon(QIcon(QuestionWindow.icon_path))
        self.list_ = list_
        self.to_find = {}
        layout = QFormLayout(self)
        self.setLayout(layout)
        for attribute in self.list_:
            if attribute[1] == bool:
                checkButton = QCheckBox()
                checkButton.setText(attribute[0])
                self.to_find[attribute[0]] = checkButton
                if attribute[2]:
                    checkButton.setCheckState(Qt.Checked)
                layout.addWidget(checkButton)
            elif attribute[1] == myChoose:
                buttonGroup = QButtonGroup()
                for index, choice in enumerate(attribute[2].list_):
                    button = QRadioButton(str(choice))
                    if index == attribute[2].default_choice:
                        button.setChecked(Qt.Checked)
                    buttonGroup.addButton(button, index)
                    layout.addRow(str(""), button)
                self.to_find[attribute[0]] = buttonGroup
            else:
                newEditor = QLineEdit(self)
                if attribute[2] is not None:
                    newEditor.setText(str(attribute[2]))
                layout.addRow(str(attribute[0]), newEditor)
                self.to_find[attribute[0]] = newEditor
        buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout.addWidget(buttonBox)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInformation(self):
        """
        以字典的形式返回用户输入的信息，
        """
        result = {}
        for command in self.list_:
            name = command[0]
            type_ = command[1]
            default = command[2]
            widget = self.to_find[name]
            if isinstance(widget, QLineEdit):
                contain = widget.text()
                if contain == "":
                    contain = None
                else:
                    try:
                        temp = int(contain)
                        contain = temp
                    except:
                        if type_ is not None:
                            try:  # 尝试着把参数转换成指定类型
                                temp = list(map(type_, [contain]))[0]
                                contain = temp
                            except:
                                pass
                result[name] = contain
            elif isinstance(widget, QCheckBox):
                if widget.checkState() == Qt.Checked:
                    bool_result = True
                elif widget.checkState() == Qt.Unchecked:
                    bool_result = False
                else:  # 将来可以添加半选状态
                    bool_result = None
                result[name] = bool_result
            elif isinstance(widget, QButtonGroup):
                index_ = widget.checkedId()
                if index_ == -1:
                    index_ = None
                result[name] = index_
        return result


class tableWidget(QTableWidget):
    def __init__(self, table: pd.DataFrame):
        super().__init__()
        self.table = table
        rows = table.shape[0]
        colunms = table.shape[1]
        table_header = table.columns.values.tolist()
        self.setColumnCount(colunms)
        self.setRowCount(rows)
        self.setHorizontalHeaderLabels(table_header)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollPerItem)
        for i in range(rows):
            for j in range(colunms):
                newItem = QTableWidgetItem(str(np.array(table.iloc[[i]]).tolist()[0][j]))
                newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.setItem(i, j, newItem)
