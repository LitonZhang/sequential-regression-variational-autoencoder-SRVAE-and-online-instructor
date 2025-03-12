import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyper Parameters
# BATCH_SIZE = 32
# LR = 1e-4   # learning rate
# GAMMA = 0.9  # reward discount
# TARGET_REPLACE_ITER = 150  # target update frequency

# N_STATES = 7  # 状态空间
N_ACTIONS = 11  # 动作空间
# print(N_STATES)  # 输出 4
ENV_A_SHAPE = 0


# if isinstance(env.action_space.sample(), int):
#     ENV_A_SHAPE = 0
# else:
#     ENV_A_SHAPE = env.action_space.sample().shape


class Net(nn.Module):  # 输入状态，输出动作的两层神经网络
    def __init__(self, args):
        super(Net, self).__init__()
        self.N_STATES = args.timestep+1
        self.fc1 = nn.Linear(self.N_STATES, args.DQN_hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization 随机生成最开始的参数的值
        self.out = nn.Linear(args.DQN_hidden_size, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        # print('actions_value', actions_value)
        return actions_value


class DQN(object):
    def __init__(self, device, args, TARGET_REPLACE_ITER):
        self.eval_net, self.target_net = Net(args).to(device), Net(args).to(device)
        self.N_STATES = args.timestep+1
        self.learn_step_counter = 0  # for target updating，学习到多少步
        self.memory_counter = 0  # for storing memory，记忆库位置计数
        self.MEMORY_CAPACITY = args.memory_capacity
        self.memory = torch.zeros((self.MEMORY_CAPACITY, self.N_STATES * 2 + 2)).to(device)  # initialize memory，两个状态s和s_ + reward值 + action值
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=args.DQN_LR)
        self.loss_func = nn.MSELoss()
        self.device = device
        self.TARGET_REPLACE_ITER = TARGET_REPLACE_ITER
        self.DQN_batch_size = args.DQN_batch_size
        self.GAMMA = args.DQN_GAMMA

    def choose_action(self, x, EPSILON):  # 接收环境的观测值，采取一个动作
        x = x.view(1, -1)
        # x = torch.unsqueeze(x, 0)  # 观测值在0维扩展一维, unsqueeze [[]], squeeze []
        # input only one sample
        if torch.randn(1) < torch.tensor(EPSILON):  # greedy，EPSILON为ε，小于阈值则选择action_value大的
            actions_value = self.eval_net.forward(x)  # 输出action_value， 格式[[ 0.2662, -0.1537]]
            # action = torch.max(actions_value, 1)[1].data.numpy()  # 选取value最大的动作， torch.max(a,1) 返回每一行中最大值的那个元素，且返回其索引
            action = torch.max(actions_value, 1)[1]
            # return actions_value的index，此处action为2维ndarray
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else:  # random选择
            action = torch.randint_like(torch.tensor(0).to(self.device), N_ACTIONS)  # 返回一个随机整型数，其范围为[low, high]
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # 此处action为1维ndarray
        return action

    def store_transition(self, s, a, r, s_):  # 记忆库，存储记忆
        # transition = np.hstack((s.detach().numpy(), [np.array(a), r.detach().numpy()], s_.detach().numpy()))  # 按行堆栈
        s, a, r, s_ = s.view(-1, 1), a.view(-1, 1), r.view(-1, 1), s_.view(-1, 1)
        transition = torch.cat((s, a, r, s_), dim=0).view(1, -1)  # dim=0按行堆栈
        # replace the old memory with new memory
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition  # 直接存在余数
        self.memory_counter += 1

    def learn(self):  # 从记忆库中提取记忆
        # target parameter update
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:  # 隔TARGET_REPLACE_ITER步用eval_net更新target_net
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 用于将eval_net的预训练的参数权重加载到新的模型之中
        self.learn_step_counter += 1
        # sample batch transitions
        # sample_index = np.random.choice(self.MEMORY_CAPACITY, self.DQN_batch_size)  # 在MEMORY中选择BATCH_SIZE个索引
        sample_index = torch.randint(0, self.MEMORY_CAPACITY, (self.DQN_batch_size,)).to(self.device)
        b_memory = self.memory[sample_index, :]  # 从记忆库中随机抽取记忆
        b_s = b_memory[:, :self.N_STATES]
        b_a = torch.as_tensor(b_memory[:, self.N_STATES:self.N_STATES + 1], dtype=torch.int64)
        b_r = b_memory[:, self.N_STATES + 1:self.N_STATES + 2]
        b_s_ = b_memory[:, -self.N_STATES:]

        # q_eval w.r.t(with respect to) the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # 根据当初选择的动作的评估Q值，shape (batch, 1) 1表示按行融合，0表示按列融合
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        # q_next是tensor，.max(1)，表示第一维度固定在第二维度上求最大值
        # print(q_next.max(1),q_next.max(1)[0]) values和indices
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.DQN_batch_size, 1)  # 根据贝尔曼方程计算真实Q值，shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        # print('DQNloss', loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
