from __future__ import print_function
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from copy import deepcopy
from FTML import FTML
import torch


class MuFactor(nn.Module):  # 计算mu
    def __init__(self, input_size, global_nlayers, global_hidden_size, n_global_factors):
        super(MuFactor, self).__init__()
        self.lstm = nn.LSTM(input_size, global_hidden_size, global_nlayers, bias=True, batch_first=True)
        self.hidden_to_mu = nn.Linear(global_hidden_size, n_global_factors)

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, (h, c) = self.lstm(x)
        time_mu = self.hidden_to_mu(F.relu(out))
        return time_mu


class SigmaFactor(nn.Module):  # 计算sigma and alpha
    def __init__(self, input_size, global_nlayers, global_hidden_size, n_global_factors, multiple):
        super(SigmaFactor, self).__init__()
        self.lstm = nn.LSTM(input_size, global_hidden_size, global_nlayers, bias=True, batch_first=True)
        self.hidden_to_sigma = nn.Linear(global_hidden_size, n_global_factors)
        self.hidden_to_epsilon = nn.Linear(global_hidden_size, n_global_factors)
        self.multiple = multiple

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, (h, c) = self.lstm(x)
        log_sigma = self.hidden_to_sigma(F.relu(out))
        epsilon = 1  # *torch.tanh(self.hidden_to_epsilon(F.relu(out)))
        time_sigma = torch.exp(log_sigma)
        return time_sigma, epsilon


class SRVAE(nn.Module):
    def __init__(self, input_size, global_nlayers, global_hidden_size, n_global_factors, multiple):
        super(SRVAE, self).__init__()
        self.MuFactor = MuFactor(input_size, global_nlayers, global_hidden_size, n_global_factors)
        self.SigmaFactor = SigmaFactor(input_size, global_nlayers, global_hidden_size, n_global_factors, multiple)
        self.lstm_de = nn.LSTM(input_size=n_global_factors, hidden_size=global_hidden_size, num_layers=global_nlayers,
                               bias=True, batch_first=True)
        self.hidden_de = nn.Linear(global_hidden_size, 1)
        self.z_mu = 0
        self.z_sigma = 0
        self.z = 0

    def encode(self, x):
        time_mu = self.MuFactor(x)
        time_sigma, epsilon = self.SigmaFactor(x)
        return time_mu, time_sigma, epsilon

    def resample(self, time_mu, time_sigma, epsilon):
        # epsilon = torch.randn_like(std)
        # exp(log_sigma) = sigma,重参数化技巧（reparametrisation trick）
        z = time_mu + epsilon * time_sigma  # 采样
        self.z_mu = time_mu
        self.z_sigma = time_sigma
        self.z = z
        # q_z = torch.distributions.normal.Normal(loc=time_mu, scale=time_sigma)
        # z = q_z.rsample()
        return z

    def decode(self, z):
        self.lstm_de.flatten_parameters()
        de_out, (h_n) = self.lstm_de(z)
        y_de = torch.squeeze(F.relu(self.hidden_de(F.relu(de_out))))
        return y_de

    def forward(self, x, device, num_samples=100):
        time_mu, time_sigma, epsilon = self.encode(x)
        z = self.resample(time_mu, time_sigma, epsilon)
        y_de = self.decode(z)
        return y_de


def r_square_loss(y_pred, y_true):
    # 计算总平方和（total sum of squares）
    total_sum_of_squares = torch.sum((y_true - torch.mean(y_true))**2)
    # 计算残差平方和（residual sum of squares）
    residual_sum_of_squares = torch.sum((y_true - y_pred)**2)
    # 计算R方
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r_squared


def loss_function(y_de, y, z_mu, z_sigma, z, args):
    loss_func = nn.MSELoss()  # the target label is not one-hotted
    mse = loss_func(y_de, y)  # LSTM重建y和原始y的损失
    r_mse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(y_de - y))
    r_square = r_square_loss(y_de, y)
    # print('SRVAE: mse, r_mse, mae, r_square', mse, r_mse, mae, r_square)
    # z_sum = torch.zeros_like(z)
    # for _ in range(args.Monte_size):
    #     q_z1 = torch.distributions.Normal(loc=z_mu, scale=z_sigma)
    #     z1 = q_z1.rsample()
    #     z_sum = z_sum+z1
    # z_sample_mean = z_sum/args.Monte_size
    # kl = loss_func(z_sample_mean, z)
    # for循环太慢
    q_z1 = torch.distributions.Normal(loc=z_mu, scale=z_sigma)
    z_sample = q_z1.rsample((args.Monte_size,))
    z_sample_mean = torch.mean(z_sample, dim=0)
    # z_sample_mean = z_sample.mean(dim=0)
    kl = loss_func(z_sample_mean, z)
    # kl =0
    return mse + args.multiple * kl, mse


def plot_figure(train_mean_loss, test_mean_loss, result, real, epochs, args, online):
    if online:
        epoch_nums = args.online_epoch_nums
    else:
        epoch_nums = args.pre_epoch_nums
    if epochs == epoch_nums:
        # torch.save(model.state_dict(), os.path.join('ewc_paras', 'p11.pkl'))
        # train_mean_loss
        fig = plt.figure(figsize=(12, 3))  # 长，宽
        plt.subplot2grid((1, 4), (0, 0), colspan=2)  # (0, 0)为坐标
        if not online:
            plt.plot(train_mean_loss, label='trainloss')
        # test_losses
            plt.plot(test_mean_loss, c="#2ca02c", label='testloss')
            plt.ylabel("Loss")
        else:  # 在线训练则绘制DQN的reward
            reward = train_mean_loss
            ema_reward = ema(reward, decay=0.85)
            plt.plot(reward, label='reward')
            plt.plot(ema_reward, c="#ff7f0e", label='ema reward')  # 橙色
            plt.ylabel("DQN rewards")
        plt.legend(loc='best')
        # prediction
        plt.subplot2grid((1, 4), (0, 2), colspan=2)
        real_plot = real.cpu().numpy()[:args.test_length, 0].reshape(-1)
        result_plot = result.cpu().numpy()[:args.test_length, 0].reshape(-1)
        plt.plot(result_plot, c='r', label='prediction')  # 显示多少个预测值
        plt.plot(real_plot, c='black', label='real')
        # plt.plot(result.cpu().numpy()[:args.test_length, :].reshape(-1), c='r', label='prediction')  # 显示多少个预测值
        # plt.plot(real.cpu().numpy()[:args.test_length, :].reshape(-1), c='black', label='real')
        plt.legend(loc='best')
        fig.tight_layout()
        # plt.savefig('picture/g{}l{}ng{}nl{}f{}loss{}'.format(str(args.global_hidden_size),
        #                                                      str(args.global_hidden_size),str(args.n_factors),
        #                                                      round(test_mean_loss[-1], 6)) + '.png',
        #             bbox_inches='tight')
        plt.savefig('picture/Online[{}]P{}Loss{:.6e}'.format(str(online), str(args.processID), test_mean_loss[-1])+'.png')
        if args.show_plot:
            plt.show()


def train(train_model, train_loader, device, optimizer, args):  # scheduler
    train_model.train()  # model.train()的作用是启用 Batch Normalization 和 Dropout
    loss_sum = []
    mse_sum = []
    for batch_idx, (x, y) in enumerate(train_loader):  # 函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标
        if x.shape[1] == args.timestep:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_de = train_model(x, device)  # mu, sigma未使用
            loss, mse = loss_function(y_de, y, train_model.z_mu, train_model.z_sigma, train_model.z, args)
            loss_sum.append(loss.item())
            mse_sum.append(mse.item())
            # scheduler.step(loss)  # 修改学习率
            loss.backward()
            optimizer.step()
    train_mean_loss = sum(mse_sum) / len(mse_sum)
    return train_mean_loss


def train_online(train_model, train_loader, device, epoch, dqn, args,  DQNepsilon, scaler, training, max_grad_norm=5):
    loss_sum = 0  # 初始化loss
    reward_sum = 0
    dqn_loss_note = []
    action_note = []
    mean_dqn_loss = -1
    # random_x, random_y = random_x.to(device), random_y.to(device)
    # 模型在线训练，数据小批量导入
    for batch_idx, (x, y) in enumerate(train_loader):  # 函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标
        x, y = x.to(device), y.to(device)
        if batch_idx == 0:
            x_accumulation = x
            y_accumulation = y
            copied_model = deepcopy(train_model)  # 模型复制
        else:
            x_accumulation = torch.cat([x_accumulation, x], dim=0)
            y_accumulation = torch.cat([y_accumulation, y], dim=0)
        # 不训练，直接测试
        copied_model.eval()
        with torch.no_grad():
            y_11 = copied_model(x, device)
            _, loss11 = loss_function(y_11, y, copied_model.z_mu, copied_model.z_sigma, copied_model.z, args)  # history loss
            y_12 = copied_model(x_accumulation, device)
            _, loss12 = loss_function(y_12, y_accumulation, copied_model.z_mu, copied_model.z_sigma, copied_model.z, args)  # batch loss
        mean_y1 = torch.mean(y_12, dim=0).view(-1, 1)
        loss12 = loss12.view(-1, 1)
        # state
        state = torch.cat([mean_y1, loss12], dim=0)
        # DQN take action
        action = dqn.choose_action(state, DQNepsilon)  # 根据当前状态采取行为
        # action_list = [1e-1, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 0]
        action_list = [0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 1e-1]
        learning_rate = action_list[action.item()]  # DQN选择增量学习的学习率
        # learning_rate = 1e-4
        if epoch == args.online_epoch_nums:
            action_note.append(action.item()+1)
            # action_note.append(learning_rate)
        # print(learning_rate)
        # DQN 根据action修改FTML学习率
        if learning_rate != 0:
            optimizer = FTML(copied_model.parameters(), lr=learning_rate)
            # 训练再测试：训练阶段
            copied_model.train()
            optimizer.zero_grad()
            y_de = copied_model(x, device)
            loss, _ = loss_function(y_de, y, copied_model.z_mu, copied_model.z_sigma, copied_model.z, args)
            loss.requires_grad_(True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_grad_norm)  # 防止梯度爆炸
            optimizer.step()
            # 训练再测试：测试阶段
            copied_model.eval()
            with torch.no_grad():
                y_21 = copied_model(x, device)
                _, loss21 = loss_function(y_21, y, copied_model.z_mu, copied_model.z_sigma, copied_model.z, args)  # history loss
                y_22 = copied_model(x_accumulation, device)
                _, loss22 = loss_function(y_22, y_accumulation, copied_model.z_mu, copied_model.z_sigma, copied_model.z, args)
            # DQN 下个时刻的state
            mean_y2 = torch.mean(y_22, dim=0).view(-1, 1)
            loss22 = loss22.view(-1, 1)
            # 选择阶段，选择loss更小的
            # 回退
            if loss22-loss12 > 0 and abs(loss22-loss12)/loss22 > 0.5:
                loss_sum += loss12.item()
                state_ = state  # 下一时刻状态
                copied_model = deepcopy(train_model)  # 模型回退
                reward = torch.squeeze((loss12-loss22)/(loss22+loss12))
                if epoch == args.online_epoch_nums:
                    action_note.append('回退')
                # print('回退')
            else:
                loss_sum += loss22.item()
                state_ = torch.cat([mean_y2, loss22], dim=0)
                train_model = deepcopy(copied_model)
                # reward = torch.squeeze(-loss22)
                # reward = torch.squeeze(loss22)
                reward = torch.squeeze((loss12 - loss22) / (loss22 + loss12))
                if epoch == args.online_epoch_nums:
                    action_note.append('进化')
                # print('进化')
            reward_sum += reward
        else:  # 学习率选择0时
            reward = torch.squeeze(loss12-loss12)
            state_ = state  # 下一时刻状态
            # print('不训练')
            reward_sum += reward
            if epoch == args.online_epoch_nums:
                action_note.append('无动作')
        # 累计奖励
        # print(reward_sum)
        if training:
            # 存储在记忆库
            dqn.store_transition(state, action, reward, state_)
            if dqn.memory_counter > args.memory_capacity:
                dqn_loss = dqn.learn()
                # print('攒够经验')
                dqn_loss_note.append(dqn_loss)
                mean_dqn_loss = sum(dqn_loss_note)/len(dqn_loss_note)
    return reward_sum.cpu().numpy(), train_model, mean_dqn_loss, action_note


def test(model, test_loader, device,  args):
    model.eval()  # model.eval()的作用是不启用 Batch Normalization 和 Dropout
    loss_sum = []
    mse_sum = []
    result = []
    real = []
    with torch.no_grad():  # 强制之后的内容不进行计算图构建
        for i, (x, y) in enumerate(test_loader):
            if x.shape[1] == args.timestep:
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    y_de = model(x, device)
                # predict_batch = predict_batch.data.cpu().numpy()
                loss, mse = loss_function(y_de, y, model.z_mu, model.z_sigma, model.z, args)
                loss_sum.append(loss.item())
                mse_sum.append(mse.item())
                if i == 0:
                    result = y_de
                    real = y
                else:
                    y_de = y_de.view(-1, args.timestep)
                    result = torch.cat([result, y_de], dim=0).view(-1, args.timestep)
                    real = torch.cat([real, y], dim=0).view(-1, args.timestep)
        test_mean_loss = sum(mse_sum) / len(mse_sum)
        print("SRVAE loss", test_mean_loss)
        # print('====> Test set loss: {:.4f}'.format(test_mean_loss))
    return test_mean_loss, result, real


def ema(data, decay=0.9):
    new_data = torch.zeros(len(data))
    new_data[0] = torch.tensor(sum(data[:2])/2)
    for idx in range(len(data) - 1):
        new_data[idx+1] = decay * new_data[idx] + (1 - decay) * data[idx + 1]
    return new_data
