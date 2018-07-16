import numpy as np

num_states = 7

# {"0": "C1", "1":"C2", "2":"C3", "3":"Pass", "4":"Pub", "5":"FB", "6":"Sleep"}
index_to_state = {}

index_to_state["0"] = "C1"
index_to_state["1"] = "C2"
index_to_state["2"] = "C3"
index_to_state["3"] = "Pass"
index_to_state["4"] = "Pub"
index_to_state["5"] = "FB"
index_to_state["6"] = "Sleep"

state_to_index = {}
for index, name in zip(index_to_state.keys(), index_to_state.values()):
    state_to_index[name] = int(index)

# 状态转移矩阵
Pss = [
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.2],
        [0.0, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.2, 0.4, 0.4, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.9, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ]

Pss = np.array(Pss)

rewards = [-2, -2, -2, 10, 1, -1, 0]
gamma = 0.5


# 获取值Gt的计算

def compute_return(start_index=0, chain=None, gamma=0.5) -> float:
    Gt = 0.0
    power = 0
    gamma = gamma

    for i in range(start_index, len(chain)):
        Gt += np.power(gamma, power) * rewards[state_to_index[chain[i]]]
        power += 1
    return Gt


# chains =[
#             ["C1", "C2", "C3", "Pass", "Sleep"],
#             ["C1", "FB", "FB", "C1", "C2", "Sleep"],
#             ["C1", "C2", "C3", "Pub", "C2", "C3", "Pass", "Sleep"],
#             ["C1", "FB", "FB", "C1", "C2", "C3", "Pub", "C1", "FB", "FB", "FB", "C1", "C2", "C3", "Pub", "C2", "Sleep"]
#         ]
#
# Gt = compute_return(0, chains[3], 0)
# # print(Gt)


def compute_value(Pss, rewards, gamma):
    rewards = np.array(rewards).reshape((-1, 1))
    values = np.dot(np.linalg.inv(np.eye(7, 7) - gamma * Pss), rewards)
    return values




values = compute_value(Pss, rewards, 0.99999)
print(values)