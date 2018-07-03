import numpy as np
import pandas as pd
import time


# 模块和参数设置
N_STATES = 6   # 1维世界的宽度
ACTIONS = ['left', 'right']     # 探索者的可用动作
EPSILON = 0.9   # 贪婪度 greedy
ALPHA = 0.1     # 学习率
GAMMA = 0.9    # 奖励递减值
MAX_EPISODES = 100   # 最大回合数
FRESH_TIME = 0.2    # 移动间隔时间

np.random.seed(2)


def build_q_table(n_states, actions):
    qtable = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q_table 全 0 初始
        columns=actions  # columns 对应的是行为名称
    )
    return qtable


# 在某个 state 地点, 选择行为
def choose_action(state, qtable):
    # 选出这个 state 的所有 action 值
    state_actions = qtable.ix[state, :]
    # print("state_actions", state_actions)
    if np.random.uniform() > EPSILON or state_actions.all() == 0:
        # 随机选择一个动作
        action_name = np.random.choice(ACTIONS)
    else:
        # 获取最大值所对应的索引
        action_name = state_actions.idxmax()
    return action_name


# 做出行为后, 环境也要给我们的行为一个反馈
# 反馈出下个 state (S_) 和 在上个 state (S) 做出 action (A) 所得到的 reward (R)
# -o---T
# T 就是宝藏的位置, o 是探索者的位置
def step(state, action):
    # This is how agent will interact with the environment
    if action == 'right':  # move right
        # 判断是否为terminate
        if state == N_STATES - 2:
            next_state = "terminal"
            reward = 1
        else:
            next_state = state + 1
            reward = 0
    else:  # move left
        reward = 0
        if state == 0:
            next_state = state  # reach the wall,状态保持不变
        else:
            next_state = state - 1
    return next_state, reward


def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-'] * (N_STATES - 1) + ['T']  # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # 初始 q table
    qtable = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        # 更新探索概率
        EPSILON = 1 / (episode + 1)
        # 行走步数计数器
        step_counter = 0

        # 回合初始位置
        state = 0

        # 是否回合结束
        is_terminal = False

        # 环境更新
        update_env(state, episode, step_counter)
        while not is_terminal:
            # 选行为
            action = choose_action(state, qtable)

            # 实施行为并得到环境的反馈
            next_state, reward = step(state, action)
            q_predict = qtable.loc[state, action]  # 估算的(状态-行为)值
            if next_state != "terminal":
                #  实际的(状态-行为)值 (回合没结束)
                q_target = reward + GAMMA * qtable.iloc[state, :].max()
            else:
                q_target = reward
                is_terminal = True

            #  q_table 更新
            qtable.ix[state, action] += ALPHA * (q_target - q_predict)
            # 探索者移动到下一个 state
            state = next_state
            step_counter = step_counter + 1
            update_env(state, episode, step_counter)

    return qtable
# qtable = build_q_table(N_STATES, ACTIONS)


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
