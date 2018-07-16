from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QPushButton, QHeaderView
import pandas as pd
from MDP.UI.silvergrid_mdp import *
from PyQt5 import QtCore, QtWidgets
import sys
import numpy as np


class SHOW_MDP(QtWidgets.QWidget, Ui_Form):
    def __init__(self, row=4, column=4):
        self.row = row
        self.column = column
        self.init_argments()
        super(SHOW_MDP, self).__init__()
        self.setupUi(self)
        self.show_table()

    def init_argments(self):
        self.ternimal_state = [0, 15]
        self.Actions = ["upper", "down", "left", "right"]
        self.values = np.zeros((4, 4))
        self.gamma = 1.0
        self.ds_actions = {"upper": -4, "down": 4, "left": -1, "right": 1}

    def step(self, state, action):
        next_state = state
        reward = -1
        is_termianl = False
        if state % self.column == 0 and action == "left":
            return next_state, reward, is_termianl
        if state < self.column and action == "upper":
            return next_state, reward, is_termianl
        if (state + 1) < self.column and action == "right":
            return next_state, reward, is_termianl
        if state >= (self.row - 1) * self.column and action == "down":
            return next_state, reward, is_termianl
        delta_state = self.ds_actions[action]
        next_state = state + delta_state
        if next_state == 0 or next_state == 15:
            is_termianl = True
            reward = 0
        return next_state, reward, is_termianl

    def build_q_table(self):

        row_label = [i for i in range(1, 15)]
        qtable = pd.DataFrame(
            np.zeros((self.row * self.column, len(self.Actions))),
            columns=self.Actions,
            # index=row_label
        )
        return qtable

    def show_table(self):
        # self.tableWidget = QtWidgets.QTableWidget()
        self.tableWidget.setStyleSheet("background-color: light gray")
        self.tableWidget.setRowCount(self.row)
        self.tableWidget.setColumnCount(self.column)
        for i in range(self.tableWidget.rowCount()):
            for j in range(self.tableWidget.columnCount()):
                btn = QPushButton()
                btn.setText(str(i * 4 + j))
                btn.setStyleSheet("background-color: light gray")
                self.tableWidget.setCellWidget(i, j, btn)
                if (i == 0 and j == 0) or (i == 3 and j == 3):
                    btn.setStyleSheet("background-color: 	#696969")
        # Qt5以上QtableWiget的自适应列宽的设置
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()

    def render(self):
        self.tableWidget.setStyleSheet("background-color: light gray")
        for i in range(self.tableWidget.rowCount()):
            for j in range(self.tableWidget.columnCount()):
                btn = QPushButton()
                btn.setText(str(i * 4 + j))
                btn.setStyleSheet("background-color: light gray")
                self.tableWidget.setCellWidget(i, j, btn)
                if (i == 0 and j == 0) or (i == 3 and j == 3):
                    btn.setStyleSheet("background-color: 	#696969")

    def reset(self):
        pass


def main():

        app = QtWidgets.QApplication(sys.argv)
        demo = SHOW_MDP()
        demo.show()
        table = demo.build_q_table()
        print(table)
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()