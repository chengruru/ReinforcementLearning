from MDP.Maze import *
import time
import sys
import numpy as np


# np.random.seed(30)


class QLearningMain(QtCore.QThread):
    def __init__(self):
        super(QLearningMain, self).__init__()
        self.env = GridEnvV0()
        self.agentThread = QLearning()
        self.agentThread.trigger.connect(self.rl)
        self.agentThread.start()

    def rl(self):
        # 初始 q table
        qtable = self.env.build_q_table()
        # print(qtable)
        for episode in range(self.env.MAX_EPISODES):
            print("episode--------- : ", episode)
            # 行走步数计数器
            step_counter = 0

            # 回合初始位置
            state = 0

            # 是否回合结束
            is_terminal = False

            # 环境更新
            # update_env(state, episode, step_counter)
            self.env.reset()
            self.env.render()

            while not is_terminal:
                # 选行为
                action = self.env.choose_action(state, qtable, step_counter)
                # print("debug : ", action)

                # 实施行为并得到环境的反馈
                next_state, reward = self.env.step(state, action)
                # print("reward : ", reward)
                # print("next_state : ", next_state)
                q_predict = qtable.loc[state, action]  # 估算的(状态-行为)值

                if next_state != "terminal":
                    #  实际的(状态-行为)值 (回合没结束)
                    q_target = reward + self.env.GAMMA * qtable.iloc[next_state, :].max()
                else:
                    q_target = reward
                    is_terminal = True

                #  q_table 更新
                qtable.loc[state, action] += self.env.ALPHA * (q_target - q_predict)
                # 探索者移动到下一个 state
                state = next_state
                step_counter = step_counter + 1
                # update_env(state, episode, step_counter)
                time.sleep(0.05)
                self.env.render()
            time.sleep(0.1)
        print(qtable)
        return qtable


class QLearning(QtCore.QThread):
    trigger = QtCore.pyqtSignal()

    def __init__(self):
        super(QLearning, self).__init__()

    def run(self):
        self.trigger.emit()


def main():
    app = QtWidgets.QApplication(sys.argv)
    brain = QLearningMain()
    brain.env.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()





