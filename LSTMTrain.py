from __future__ import print_function
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from copy import deepcopy
from FTML import FTML
import torch
import pandas as pd


class LSTM(nn.Module):  # 计算mu
    def __init__(self, input_size, global_nlayers, global_hidden_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, global_hidden_size, global_nlayers, bias=True, batch_first=True)
        self.hidden = nn.Linear(global_hidden_size, 1)

    def forward(self, x):
        self.lstm.flatten_parameters()
        out1, (h, c) = self.lstm(x)

        out2 = torch.squeeze(self.hidden(F.relu(out1)))
        return out2


def train(train_model, train_loader, device, optimizer, args):  # scheduler
    train_model.train()  # model.train()的作用是启用 Batch Normalization 和 Dropout
    loss_sum = []
    mse_sum = []
    for batch_idx, (x, y) in enumerate(train_loader):  # 函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标
        if x.shape[1] == args.timestep:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_de = train_model(x)
            loss_function = nn.MSELoss()  # the target label is not one-hotted
            loss = loss_function(y_de, y)  # LSTM重建y和原始y的损失
            loss_sum.append(loss.item())
            # scheduler.step(loss)  # 修改学习率
            loss.backward()
            optimizer.step()
    train_mean_loss = sum(loss_sum) / len(loss_sum)
    return train_mean_loss


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
                    y_de = model(x)
                # predict_batch = predict_batch.data.cpu().numpy()
                loss = loss_function(y_de, y)  # LSTM重建y和原始y的损失
                loss_sum.append(loss.item())
                if i == 0:
                    result = y_de
                    real = y
                else:
                    y_de = y_de.view(-1, args.timestep)
                    result = torch.cat([result, y_de], dim=0).view(-1, args.timestep)
                    real = torch.cat([real, y], dim=0).view(-1, args.timestep)
        test_mean_loss = sum(loss_sum) / len(loss_sum)
        print("LSTM loss", test_mean_loss)
        # print('====> Test set loss: {:.4f}'.format(test_mean_loss))
    return test_mean_loss, result, real


def r_square_loss(y_pred, y_true):
    # 计算总平方和（total sum of squares）
    total_sum_of_squares = torch.sum((y_true - torch.mean(y_true))**2)
    # 计算残差平方和（residual sum of squares）
    residual_sum_of_squares = torch.sum((y_true - y_pred)**2)
    # 计算R方
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r_squared


def loss_function(y_de, y):
    loss_func = nn.MSELoss()  # the target label is not one-hotted
    mse = loss_func(y_de, y)  # LSTM重建y和原始y的损失
    r_mse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(y_de - y))
    r_square = r_square_loss(y_de, y)
    # print('LSTM: mse, r_mse, mae, r_square', mse, r_mse, mae, r_square)
    return mse


def plot_figure(train_mean_loss, test_mean_loss, result, real, epochs, args, online):
    if online:
        epoch_nums = args.online_epoch_nums
    else:
        epoch_nums = args.pre_epoch_nums
    if epochs == epoch_nums:
        # torch.save(model.state_dict(), os.path.join('ewc_paras', 'p11.pkl'))
        # train_mean_loss
        fig = plt.figure(figsize=(10, 3))  # 长，宽
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
        plt.savefig('picture/Online[{}]P{}Loss{:.6e}'.format(str(online), str(args.processID), test_mean_loss[-1]) +'.png')
        if args.show_plot:
            plt.show()


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
        """x的scaler不能和y用同一个"""
        scaler_x = x/scaler
        if batch_idx == 0:
            x_accumulation = x
            y_accumulation = y
            scaler_x_acc = scaler_x  # 归一化的x
            copied_model = deepcopy(train_model)  # 模型复制
        else:
            x_accumulation = torch.cat([x_accumulation, x], dim=0)
            y_accumulation = torch.cat([y_accumulation, y], dim=0)
            scaler_x_acc = torch.cat([scaler_x_acc, scaler_x], dim=0)
        # 不训练，直接测试
        copied_model.eval()
        with torch.no_grad():
            y_11 = copied_model(x)
            loss11, _ = loss_function(y_11, y)  # history loss
            y_12 = copied_model(x_accumulation)
            loss12, _ = loss_function(y_12, y_accumulation)  # batch loss
        # 用一维向量表示归一化的x, numel()统计张量元素个数
        mean_x = (torch.norm(torch.mean(scaler_x, dim=0).view(-1, scaler_x.shape[2]))/scaler_x.numel()).view(-1, 1)
        mean_y1 = torch.mean(y_12, dim=0).view(-1, 1)
        # mean_y = torch.mean(y, dim=0).view(-1, 1)
        loss11, loss12 = loss11.view(-1, 1), loss12.view(-1, 1)
        error1, error2 = torch.mean(abs(y_11-y)/y, dim=0).view(-1, 1), torch.mean(abs(y_12-y_accumulation)/y_accumulation, dim=0).view(-1, 1)
        e1, e2 = torch.sum(error1), torch.sum(error2)
        state = torch.cat([mean_x, mean_y1, error1, error2], dim=0)
        # DQN take action
        action = dqn.choose_action(state, DQNepsilon)  # 根据当前状态采取行为
        action_list = [1e-1, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 0]
        learning_rate = action_list[action.item()]  # DQN选择增量学习的学习率
        if epoch == args.online_epoch_nums:
            action_note.append(learning_rate)
        # print(learning_rate)
        # DQN 根据action修改FTML学习率
        if learning_rate != 0:
            optimizer = FTML(copied_model.parameters(), lr=learning_rate)
            # 训练再测试：训练阶段
            copied_model.train()
            optimizer.zero_grad()
            y_de = copied_model(x)
            loss, _ = loss_function(y_de, y)
            loss.requires_grad_(True)
            # loss += ewc.ewc_penalty(train_model, current_task_id=1)  # 增量任务EWC
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_grad_norm)  # 防止梯度爆炸
            optimizer.step()
            # 训练再测试：测试阶段
            copied_model.eval()
            with torch.no_grad():
                y_21 = copied_model(x)
                loss21, _ = loss_function(y_21, y)  # history loss
                y_22 = copied_model(x_accumulation)
                loss22, _ = loss_function(y_22, y_accumulation)
            # DQN 下个时刻的state
            mean_y2 = torch.mean(y_22, dim=0).view(-1, 1)
            loss21, loss22 = loss21.view(-1, 1), loss22.view(-1, 1)
            # error3, error4 = loss21/loss11, loss22/loss12
            error3, error4 = torch.mean(abs(y_21-y)/y, dim=0).view(-1, 1), torch.mean(abs(y_22-y_accumulation)/y_accumulation, dim=0).view(-1, 1)
            e3, e4 = torch.sum(error3), torch.sum(error4)
            # 选择阶段，选择loss更小的
            # 训练后效果好，训练
            if e1 > e3 and e2 > e4:
                loss_sum += loss21.item()
                state_ = torch.cat([mean_x, mean_y2, error3, error4], dim=0)  # 下一时刻状态
                reward = e1+e2-e3-e4   # modify the reward
                # 保存增量模型参数
                # ewc.compute_fisher(copied_model, current_task_id=1, save_paras=False)
                train_model = deepcopy(copied_model)
                print('进化')
            elif e1 + e2 > e3 + e4:
                loss_sum += loss11.item()
                state_ = state  # 下一时刻状态
                reward = e1+e2-e3-e4  # modify the reward
                copied_model = deepcopy(train_model)  # 模型回退
                print('回退')
            else:
                loss_sum += loss11.item()
                state_ = state  # 下一时刻状态
                reward = (-e3-e4)/2  # modify the reward
                copied_model = deepcopy(train_model)  # 模型回退
                print('回退')
            # print('reward', reward)
            reward_sum += reward
            # print(reward_sum)
        else:
            reward = (-e2-e1)/2  # 全部数据的误差-当前数据的误差，当前数据效果好就奖励，差就惩罚。但是奖励不大，不训练的作用效果不明显
            state_ = state  # 下一时刻状态
            print('不训练')
        if training:
            # reward = torch.tensor(reward).to(device)
            # 存储在记忆库
            dqn.store_transition(state, action, reward, state_)
            # print(dqn.memory_counter, reward_sum)
            if dqn.memory_counter > args.memory_capacity:
                dqn_loss = dqn.learn()
                dqn_loss_note.append(dqn_loss)
                mean_dqn_loss = sum(dqn_loss_note)/len(dqn_loss_note)

    return reward_sum.cpu().numpy(), train_model, mean_dqn_loss, action_note


def ema(data, decay=0.9):
    new_data = torch.zeros(len(data))
    new_data[0] = torch.tensor(sum(data[:2] ) /2)
    for idx in range(len(data) - 1):
        new_data[idx +1] = decay * new_data[idx] + (1 - decay) * data[idx + 1]
    return new_data

