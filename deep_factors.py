from __future__ import print_function
import argparse
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import util


class GlobalFactor(nn.Module):
    def __init__(self, input_size, global_nlayers, global_hidden_size, n_global_factors):
        super(GlobalFactor, self).__init__()
        self.lstm = nn.LSTM(input_size, global_hidden_size, global_nlayers, bias=True, batch_first=True)
        self.hidden_to_mu = nn.Linear(global_hidden_size, 1)
        # self.affine = nn.Linear(n_global_factors, 1)

    def forward(self, X):
        out, (h, c) = self.lstm(X)
        time_mu = self.hidden_to_mu(F.relu(out))
        # time_mu = F.relu(self.affine(F.relu(time_mu)))  # 加激活函数防止空值
        return time_mu


class LocalFactor(nn.Module):  # 计算噪声的sigma
    def __init__(self, input_size, local_nlayers, local_hidden_size, n_global_factors):
        super(LocalFactor, self).__init__()
        self.lstm = nn.LSTM(input_size, local_hidden_size, local_nlayers, bias=True, batch_first=True)
        self.linear = nn.Linear(local_hidden_size, local_hidden_size)  # 加一层线性层
        self.hidden_to_sigma = nn.Linear(local_hidden_size, 1)
        # self.affine = nn.Linear(n_global_factors, 1)  # 神经网络的正向传播中进行的矩阵的乘积运算在几何学领域被称为“仿射变换”

    def forward(self, X):
        out, (h, c) = self.lstm(X)
        time_sigma = self.linear(F.relu(out))
        time_sigma = self.hidden_to_sigma(F.relu(time_sigma))
        time_sigma = torch.log(1 + torch.exp(time_sigma) + 1e-6)
        return time_sigma


class DeepFactors(nn.Module):
    def __init__(self, input_size, local_nlayers, local_hidden_size,
                 global_nlayers, global_hidden_size, n_global_factors):
        super(DeepFactors, self).__init__()
        # self.lstm_en = nn.LSTM(input_size=input_size, hidden_size=64, num_layers=1, batch_first=True)
        self.GlobalFactor = GlobalFactor(input_size, global_nlayers, global_hidden_size, n_global_factors)
        self.LocalFactor = LocalFactor(input_size, local_nlayers, local_hidden_size, n_global_factors)
        self.lstm_de = nn.LSTM(input_size=n_global_factors, hidden_size=global_hidden_size, num_layers=global_nlayers,
                               bias=True, batch_first=True)
        self.hidden_de = nn.Linear(global_hidden_size, 1)

    def encode(self, X):
        time_mu = self.GlobalFactor(X)  # (1,timestep,1)
        time_sigma = self.LocalFactor(X)  # (1,timestep,1)
        time_mu = time_mu.squeeze(2)
        time_sigma = time_sigma.squeeze(2)
        return time_mu, time_sigma

    def decode(self, time_mu, time_sigma, args):
        num_ts, num_periods = time_mu.size()
        z = torch.zeros(num_ts, num_periods).to('cpu')
        for _ in range(args.Monte_size):
            dist = torch.distributions.normal.Normal(loc=time_mu, scale=time_sigma)
            zs = dist.sample().view(num_ts, num_periods)
            z += zs
        z = z / args.Monte_size
        return z

    def forward(self, X):
        time_mu, time_sigma = self.encode(X)
        return time_mu, time_sigma


def loss_function(mu, sigma, y):
    negative_likelihood = torch.log(sigma + 1) + (y - mu) ** 2 / (2 * sigma ** 2)
    nega_loss = negative_likelihood.mean()
    return nega_loss


def r_square_loss(y_pred, y_true):
    # 计算总平方和（total sum of squares）
    total_sum_of_squares = torch.sum((y_true - torch.mean(y_true))**2)
    # 计算残差平方和（residual sum of squares）
    residual_sum_of_squares = torch.sum((y_true - y_pred)**2)
    # 计算R方
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r_squared


def test_loss_func(y_de, y):
    recon_y_shape = y_de.shape[1]
    loss_mse = nn.MSELoss()  # the target label is not one-hotted
    mse = loss_mse(y_de, y.view(-1, recon_y_shape))  # LSTM重建y和原始y的损失
    r_mse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(y_de - y))
    r_square = r_square_loss(y_de, y)
    return mse


def plot_figure(train_mean_loss, test_mean_loss, result, real, epochs):
    if epochs == args.epoch_nums:
        torch.save(model, os.path.join('DF_model', 'p' + str(args.processID) + '.pt'))
        # train_mean_loss
        plt.figure(figsize=(12, 3))
        plt.subplot2grid((1, 3), (0, 0))
        plt.plot(train_mean_loss, c="r", label='trainloss')
        # test_losses
        plt.plot(test_mean_loss, c="b", label='testloss')
        plt.ylabel("Loss")
        plt.legend(loc='best')
        # prediction
        plt.subplot2grid((1, 3), (0, 1), colspan=2)
        plt.plot(result.cpu().numpy()[:args.test_length, 0].reshape(-1), c='r', label='prediction')  # 显示多少个预测值
        plt.plot(real.cpu().numpy()[:args.test_length, 0].reshape(-1), c='black', label='real')
        plt.legend(loc='best')
        # plt.savefig('picture/pid{}'.format(str(args.processID)) + '.png')
        # plt.savefig('picture/pid{}g{}l{}ng{}nl{}f{}loss{}'.format(str(args.processID),
        #                                                      str(args.global_hidden_size),
        #                                                      str(args.local_hidden_size),
        #                                                      str(args.global_nlayers),
        #                                                      str(args.local_nlayers),
        #                                                      str(args.n_factors),
        #                                                      test_mean_loss[-1]) + '.png', bbox_inches='tight')
        plt.savefig(
            'picture/P{}Loss{:.6e}'.format(str(args.processID), test_mean_loss[-1]) + '.png')
        #
        # if args.show_plot:
        #     plt.show()


def train(epochs):
    model.train()  # model.train()的作用是启用 Batch Normalization 和 Dropout
    loss_sum = []
    for batch_idx, (x, y) in enumerate(pre_train_loader):  # 函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        time_mu, time_sigma = model(x)  # mu, sigma未使用
        loss = loss_function(time_mu, time_sigma, y)
        loss.requires_grad_(True)
        loss_sum.append(loss.item())
        loss.backward()
        optimizer.step()
    mean_loss = sum(loss_sum) / len(loss_sum)
    train_mean_loss = mean_loss
    # print('====> Epoch: {} Average loss: {:.6f}'.format(epochs, train_mean_loss))
    return train_mean_loss


def dftest(model, test_loader, args, device):
    model.eval()  # model.eval()的作用是不启用 Batch Normalization 和 Dropout
    loss_sum = []
    result = []
    real = []
    with torch.no_grad():  # 强制之后的内容不进行计算图构建
        for i, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            time_mu, time_sigma = model(x)
            y_de = model.decode(time_mu, time_sigma, args)
            # predict_batch = predict_batch.data.cpu().numpy()
            loss = test_loss_func(y_de, y)
            loss_sum.append(loss.item())
            if i == 0:
                result = y_de
                real = y
            else:
                y_de = y_de.view(-1, args.timestep)
                result = torch.cat([result, y_de], dim=0).view(-1, args.timestep)
                real = torch.cat([real, y], dim=0).view(-1, args.timestep)
        test_mean_loss = sum(loss_sum) / len(loss_sum)
        print("DeepFactors loss", test_mean_loss)
        # print('====> Test set loss: {:.4f}'.format(test_mean_loss))
    return test_mean_loss, result, real


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch_nums", "-e", type=int, default=600)  # 1000
    parser.add_argument("-lr", type=float, default=1e-3)  # 1e-3
    parser.add_argument("--processID", type=int, default=1)
    parser.add_argument("--global_hidden_size", "-ghs", type=int, default=128)  # 128  提取全局信息
    parser.add_argument("--global_nlayers", "-gn", type=int, default=2)  # 2
    parser.add_argument("--local_hidden_size", "-lhs", type=int, default=128)  # 128  提取局部信息
    parser.add_argument("--local_nlayers", "-ln", type=int, default=2)  # 2
    parser.add_argument("--n_factors", "-f", type=int, default=6)  # global num of mu and sigma  12
    parser.add_argument("--test_length", "-tl", type=int, default=32)  # 64
    parser.add_argument("--timestep", "-ts", type=int, default=6)  # 5
    parser.add_argument("--show_plot", "-sp", action="store_true")
    parser.add_argument("--standard_scaler", "-ss", action="store_true")
    parser.add_argument("--log_scaler", "-ls", action="store_true")
    parser.add_argument("--mean_scaler", "-ms", action="store_true")  # 数据缩放（归一化）
    parser.add_argument("--pre_batch_size", "-b", type=int, default=256)  # 300
    parser.add_argument("--online_batch_size", type=int, default=5)
    parser.add_argument("--Monte_size", type=int, default=200)  # 100
    parser.add_argument('--no_cuda', action='store_true', default=True, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args()
    # GPU加速
    args.cuda = not args.no_cuda and torch.cuda.is_available()  # 判断pytorch是否有GPU加速
    # print(torch.cuda.is_available())
    torch.manual_seed(args.seed)  # 为所有GPU设置种子数
    device = torch.device("cuda" if args.cuda else "cpu")  # 代表将torch.Tensor分配到的设备的对象
    print('Start running on {}'.format(device))
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # 打包数据
    period = [2018, 1, 1, 2022, 6, 8]
    pre_train_loader, online_train_loader, test_loader, dim_0, num_rows, num_features, mean_x, mean_y, online_train_y = util.to_data_loader(
        args, period)

    model = DeepFactors(num_features, args.local_nlayers,
                        args.local_hidden_size, args.global_nlayers,
                        args.global_hidden_size, args.n_factors).to(device)  # 实例化并分配去设备运行
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # optimize all cnn parameters
    # 以下为训练和测试过程
    train_losses_list = []
    test_losses_list = []
    for epoch in tqdm(range(1, args.epoch_nums + 1)):
        train_loss = train(epoch)
        train_losses_list.append(train_loss)
        test_loss, results, reals = dftest(model, test_loader, args, device)
        test_losses_list.append(test_loss)
        # 以下为出图过程
        plot_figure(train_losses_list, test_losses_list, results, reals, epoch)
