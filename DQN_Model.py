import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import StepLR


GAMMA = 0.95
TARGET_REPLACE_ITER = 100           # 目标神经网络更新周期
N_STATES = 201                      # 频谱重构结果尺寸
P_LAYER1_NUM = 64                    # freq网络隐层1神经元数量
P_LAYER2_NUM = 128                    # freq网络隐层2神经元数量
P_LAYER3_NUM = 256                    # freq网络隐层2神经元数量
P_LAYER4_NUM = 128                    # freq网络隐层2神经元数量
P_LAYER5_NUM = 64                    # freq网络隐层2神经元数量
# MEMORY_CAPACITY = 100               # 记忆容量，达到此容量之后开始训练
A_LAYER1_NUM = 64                    # power网络隐层1神经元数量
A_LAYER2_NUM = 128                    # power网络隐层2神经元数量
A_LAYER3_NUM = 256                    # freq网络隐层2神经元数量
A_LAYER4_NUM = 128                    # freq网络隐层2神经元数量
A_LAYER5_NUM = 64                    # freq网络隐层2神经元数量


class PartitionNet(nn.Module):
    def __init__(self, n_actions_p, n_states_p):
        super(PartitionNet, self).__init__()
        self.fc1 = nn.Linear(n_states_p, P_LAYER1_NUM)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(P_LAYER1_NUM, P_LAYER2_NUM)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(P_LAYER2_NUM, P_LAYER3_NUM)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc4 = nn.Linear(P_LAYER3_NUM, P_LAYER4_NUM)
        self.fc4.weight.data.normal_(0, 0.1)
        self.fc5 = nn.Linear(P_LAYER4_NUM, P_LAYER5_NUM)
        self.fc5.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(P_LAYER5_NUM, n_actions_p)        # 输出层的输出就是若干种动作
        self.out.weight.data.normal_(0, 0.1)            # 初始化网络参数

    def forward(self, x):                               # 定义变量的前向传递过程
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class AssignmentNet(nn.Module):
    def __init__(self, n_actions_a, n_states_a):
        super(AssignmentNet, self).__init__()
        self.fc1 = nn.Linear(n_states_a, A_LAYER1_NUM)              # 该网络的输入是环境状态
        self.fc1.weight.data.normal_(0, 0.1)                    # 将该层网络的参数进行初始化
        self.fc2 = nn.Linear(A_LAYER1_NUM, A_LAYER2_NUM)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(A_LAYER2_NUM, A_LAYER3_NUM)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc4 = nn.Linear(A_LAYER3_NUM, A_LAYER4_NUM)
        self.fc4.weight.data.normal_(0, 0.1)
        self.fc5 = nn.Linear(A_LAYER4_NUM, A_LAYER5_NUM)
        self.fc5.weight.data.normal_(0, 0.1)

        self.out = nn.Linear(A_LAYER5_NUM, n_actions_a)        # 输出层的输出就是若干种动作
        self.out.weight.data.normal_(0, 0.1)            # 初始化网络参数

    def forward(self, x):                               # 定义变量的前向传递过程
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, n_actions_p, n_states_p, n_actions_a, n_states_a, memory_capacity, init_lr, batch_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_actions_p = n_actions_p
        self.n_states_p = n_states_p
        self.n_actions_a = n_actions_a
        self.n_states_a = n_states_a
        self.memory_capacity = memory_capacity
        self.p_eval_net, self.p_target_net = PartitionNet(n_actions_p, n_states_p), PartitionNet(n_actions_p, n_states_p)           # 定义两个网络，分别用于训练和提供目标值，具体功能见相关文献
        self.a_eval_net, self.a_target_net = AssignmentNet(n_actions_a, n_states_a), AssignmentNet(n_actions_a, n_states_a)
        self.p_eval_net.to(self.device)
        self.p_target_net.to(self.device)
        self.a_eval_net.to(self.device)
        self.a_target_net.to(self.device)
        self.learn_step_counter = 0                                             # 训练计数变量
        self.memory_counter = 0                                                 # 经验回放数据库计数变量
        self.memory = np.zeros((self.memory_capacity, self.n_states_p * 2 + 3))
        self.init_lr = init_lr
        self.batch_size = batch_size
        self.optimizer_p = torch.optim.Adam(self.p_eval_net.parameters(), lr=self.init_lr)
        self.optimizer_a = torch.optim.Adam(self.a_eval_net.parameters(), lr=self.init_lr)
        self.scheduler_p = StepLR(self.optimizer_p, step_size=100, gamma=0.9)
        self.scheduler_a = StepLR(self.optimizer_p, step_size=100, gamma=0.9)
        self.loss_func = nn.MSELoss()
        self.loss_p = 0
        self.loss_a = 0


    def choose_action(self, x, EPSILON):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        x = x.to(self.device)
        if np.random.uniform() < EPSILON:                                       # 以EPSILON的概率选取最优的
            p_action_value = self.p_eval_net.forward(x)                             # 动作值由两个网络中的eval_net产生
            p_action = torch.max(p_action_value, 1)[1].cpu().data.numpy()           # 将网络输出值进行独热编码，选取最大的一个动作
            p_action = p_action[0]  # return the argmax index

            a_action_value = self.a_eval_net.forward(x)
            a_action = torch.max(a_action_value, 1)[1].cpu().data.numpy()
            a_action = a_action[0]
        else:                                                                   # random
            p_action = np.random.randint(0, self.n_actions_p)                            # 随机选取频率
            p_action = p_action

            a_action = np.random.randint(0, self.n_actions_a)  # 随机选取频率
            a_action = a_action
        return p_action, a_action

    def store_transition(self, s, a_p, a_a, r, s_):
        transition = np.hstack((s, [a_p, a_a, r], s_))                 #将每一组的四个变量拼接起来，作为每一次transition需要存储的变量
        index = self.memory_counter % self.memory_capacity           #通过对指针变量进行递增取余的操作，以实现memory的动态更新
        self.memory[index, :] = transition                      #将一次transition得到的数据存入memory
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.p_target_net.load_state_dict(self.p_eval_net.state_dict())         #每到一定的迭代次数，需要将eval_net中的网络参数更新至target_net中
            self.a_target_net.load_state_dict(self.a_eval_net.state_dict())         #每到一定的迭代次数，需要将eval_net中的网络参数更新至target_net中
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]                        #从之前的交互记录中选取用于训练eval_net的样本点
        b_s = torch.FloatTensor(b_memory[:, :self.n_states_p])
        b_a_p = torch.LongTensor(b_memory[:, self.n_states_p:self.n_states_p + 1].astype(int))
        b_a_a = torch.LongTensor(b_memory[:, self.n_states_p + 1:self.n_states_p + 2].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.n_states_p + 2:self.n_states_p + 3])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states_p:])

        # q_eval w.r.t the action in experience
        p_q_eval = self.p_eval_net(b_s.to(self.device)).gather(1, b_a_p.to(self.device))  # shape (batch, 1)
        p_q_next = self.p_target_net(b_s_.to(self.device)).detach()  # detach from graph, don't backpropagate
        p_q_target = b_r.to(self.device) + GAMMA * p_q_next.max(1)[0].view(self.batch_size, 1)  # shape (batch, 1)
        p_loss = self.loss_func(p_q_eval, p_q_target)
        self.loss_p = p_loss.cpu()

        self.optimizer_p.zero_grad()
        p_loss.backward()
        self.optimizer_p.step()

        a_q_eval = self.a_eval_net(b_s.to(self.device)).gather(1, b_a_a.to(self.device))  # shape (batch, 1)
        a_q_next = self.a_target_net(b_s_.to(self.device)).detach()  # detach from graph, don't backpropagate
        a_q_target = b_r.to(self.device) + GAMMA * a_q_next.max(1)[0].view(self.batch_size, 1)  # shape (batch, 1)
        a_loss = self.loss_func(a_q_eval, a_q_target)
        self.loss_a = a_loss.cpu()

        self.optimizer_a.zero_grad()
        a_loss.backward()
        self.optimizer_a.step()

    def adjust_learning_rate(self):
        self.scheduler_p.step()
        self.scheduler_a.step()
