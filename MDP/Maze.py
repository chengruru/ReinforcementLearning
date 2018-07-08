#!/usr/bin/python3
"""
8*8的网格
"""
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QPushButton, QHeaderView
from gym import spaces

from MDP.UI.grid_mdp import *
from PyQt5 import Qt, QtWidgets, QtGui
import sys
import pandas as pd
import numpy as np


class GridEnvV0(QtWidgets.QWidget, Ui_Form):

    def __init__(self, parent=None):
        self.row = 5
        self.column = 5
        super(GridEnvV0, self).__init__(parent)
        self.setupUi(self)
        # id of the states, 0 and 15 are terminal states
        states = [i for i in range(self.row * self.column)]
        #  0* 1  2   3
        #  4  5  6   7
        #  8  9  10  11
        #  12 13 14  15*

        # initial values of states
        values = [0 for _ in range(16)]
        self.EPSILON = 0.9  # greedy police
        self.ALPHA = 0.1  # learning rate
        self.GAMMA = 0.9  # discount factor
        self.MAX_EPISODES = 1000  # maximum episodes
        self.FRESH_TIME = 0.3  # fresh time for one move
        # Action
        self.actions = ["n", "e", "s", "w"]
        self.ds_actions = {"n": -5, "e": 1, "s": 5, "w": -1}
        # 19 是出口
        self.obstacles = [3, 8, 10, 11, 22, 23, 24]

        # 行为对应的状态改变量
        # use a dictionary for convenient computation of next state id.
        self.state = 0

        self.init_chess_board()

    def init_chess_board(self):
        # self.tableWidget = QtWidgets.QTableWidget()
        self.tableWidget.setStyleSheet("background-color: light gray")
        self.tableWidget.setRowCount(self.row)
        self.tableWidget.setColumnCount(self.column)
        for i in range(self.tableWidget.rowCount()):
            for j in range(self.tableWidget.columnCount()):
                btn = QPushButton()
                # btn.setText(str(i * 5 + j))
                self.tableWidget.setCellWidget(i, j, btn)
                if j == 3 and i < 2:
                    btn.setStyleSheet("background-color: black")
                if i == 2 and j < 2:
                    btn.setStyleSheet("background-color: black")
                if i == 4 and j > 1:
                    btn.setStyleSheet("background-color: black")
                if i == 3 and j == 4:
                    btn.setText("出口")
                    btn.setFont(QFont("Roman times", 48, QFont.Bold))
        state_x = int(self.state / 5)
        state_y = int(self.state % 5)
        self.tableWidget.cellWidget(state_x, state_y).setStyleSheet("background-color: red")
        # Qt5以上QtableWiget的自适应列宽的设置
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.resizeColumnsToContents()
        self.tableWidget.resizeRowsToContents()

    def render(self):

        self.tableWidget.setStyleSheet("background-color: light gray")
        # self.tableWidget.setRowCount(self.row)
        # self.tableWidget.setColumnCount(self.column)
        for i in range(self.tableWidget.rowCount()):
            for j in range(self.tableWidget.columnCount()):
                btn = self.tableWidget.cellWidget(i, j)
                btn.setStyleSheet("background-color: light gray")
                # self.tableWidget.cellWidget(i, j).setText("")
                self.tableWidget.setCellWidget(i, j, btn)
                if j == 3 and i < 2:
                    btn.setStyleSheet("background-color: black")
                if i == 2 and j < 2:
                    btn.setStyleSheet("background-color: black")
                if i == 4 and j > 1:
                    btn.setStyleSheet("background-color: black")
                if i == 3 and j == 4:
                    btn.setStyleSheet("background-color: green")
        state_x = int(self.state / 5)
        state_y = int(self.state % 5)
        # print("state_x : ", state_x, "  state_y : ", state_y)
        self.tableWidget.cellWidget(state_x, state_y).setStyleSheet("background-color: red")
        QtWidgets.QApplication.processEvents()

    def clear_chess_table(self):
        for i in range(self.row):
            for j in range(self.column):
                # 重置背景
                self.tableWidget.cellWidget(i, j).setStyleSheet("background-color: light gray")
                # 重置文本信息
                self.tableWidget.cellWidget(i, j).setText("")

    def step(self, state, action):
        # print("step")
        # print("action, " , action)
        next_state = state
        reward = 0
        if (state % 5 == 0 and action == "w") or (state < 5 and action == "n") or \
                ((state + 1) % 5 == 0 and action == "e") or (state > 19 and action == "s"):
                next_state = state
                reward = -1
                return next_state, reward
        # print("action : ", action)
        ds = self.ds_actions[action]
        next_state = state + ds
        if next_state in self.obstacles:
            next_state = state
            reward = -1
        self.state = next_state
        # print("next_state : ", next_state)
        if next_state == 19:
            next_state = "terminal"
            reward = 100

        return next_state, reward

    def move(self, action):
        pass

    # 判断是否是终止状态
    def _is_end_state(self, x, y=None):
        pass

    def build_q_table(self):
        row_label = [i for i in range(25)]
        qtable = pd.DataFrame(
            np.zeros((25, len(self.actions))),
            columns=self.actions,
            # index=row_label
        )
        return qtable

    def choose_action(self, state, qtable, step_counter):

        # if self.EPSILON < 0.05 :
        #     self.EPSILON
        print("self.EPSILON : ", self.EPSILON)
        state_action = qtable.iloc[state, :]
        if (np.random.uniform() < self.EPSILON) or ((state_action == 0).all()):
            # act non-greedy or state-action have no value
            action_name = np.random.choice(self.actions)
        else:
            # replace argmax to idxmax as argmax means a different function in newer version of pandas
            # state_action = state_action.reindex(np.random.permutation(state_action.index))
            action_name = state_action.idxmax()
        # print("action_name: ", action_name)
        return action_name

    def reset(self):
        self.state = 0


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    demo = GridEnvV0()
    demo.show()
    sys.exit(app.exec_())
