from collections  import defaultdict
import numpy as np
import openpyxl
import torch.utils.data as Data
import LSTMTrain
from SRVAE import *
import argparse
from deep_factors import *
import util
import os


def loss_function(y_de, y):
    loss_func = nn.MSELoss()  # the target label is not one-hotted
    mse = loss_func(y_de, y)  # LSTM重建y和原始y的损失
    r_mse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(y_de - y))
    r_square = r_square_loss(y_de, y)
    return r_square


def test(data_loader,  args):
    mse_sum = []
    with torch.no_grad():  # 强制之后的内容不进行计算图构建
        for i, (x, y) in enumerate(data_loader):
            loss = loss_function(x, y)
            mse_sum.append(loss.item())
        test_mean_loss = sum(mse_sum) / len(mse_sum)
        print("SRVAE loss", test_mean_loss)
    return test_mean_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_train', "-p", action='store_true', default=False)  # 预训练
    parser.add_argument('--online_train', "-o", action='store_true', default=False)  # DQN训练
    parser.add_argument("--modelname", type=str, default='SRVAE')
    parser.add_argument("--processID", type=int, default=1)  # 1表示dynamic 2表示static probability
    parser.add_argument("--timestep", type=int, default=6)  # 2
    parser.add_argument("--pre_batch_size", type=int, default=512)  # 256
    # 封装args
    args = parser.parse_args()
    # 打开以下注释，仅对测试集数据进行测试
    period = [2018, 1, 1, 2022, 6, 8]
    dynamic_test_loader, static_test_loader = util.probability_data_loader(args)

    # testing
    dynamic_test_loss = test(dynamic_test_loader, args)
    static_test_loss = test(static_test_loader, args)


