#!/usr/bin/python 3.6
# -*-coding:utf-8-*-

"""
Utility functions
"""
import torch
import numpy as np
import os
import random
import pandas as pd
import time
from datetime import date
from collections import defaultdict
from SRVAE import SRVAE
from FTML import FTML
import re
import subprocess
import torch.utils.data as Data


def get_data_path():
    folder = os.path.dirname(__file__)
    return os.path.join(folder, "data")


def RSE(ypred, ytrue):
    rse = np.sqrt(np.square(ypred - ytrue).sum()) / \
          np.sqrt(np.square(ytrue - ytrue.mean()).sum())
    return rse


def quantile_loss(ytrue, ypred, qs):
    """
    Quantile loss version 2
    Args:
    ytrue (batch_size, output_horizon)
    ypred (batch_size, output_horizon, num_quantiles)
    """
    L = np.zeros_like(ytrue)
    for i, q in enumerate(qs):
        yq = ypred[:, :, i]
        diff = yq - ytrue
        L += np.max(q * diff, (q - 1) * diff)
    return L.mean()


def SMAPE(ytrue, ypred):
    ytrue = np.array(ytrue).ravel()
    ypred = np.array(ypred).ravel() + 1e-4
    mean_y = (ytrue + ypred) / 2.
    return np.mean(np.abs((ytrue - ypred) \
                          / mean_y))


def MAPE(ytrue, ypred):
    ytrue = np.array(ytrue).ravel() + 1e-4
    ypred = np.array(ypred).ravel()
    return np.mean(np.abs((ytrue - ypred) \
                          / ytrue))


def train_test_split(X, y, train_ratio=0.7):
    dim_0, num_periods, num_features = X.shape
    train_periods = int(num_periods * train_ratio)
    random.seed(2)
    Xtr = X[:, :train_periods, :]
    ytr = y[:, :train_periods]
    Xte = X[:, train_periods:, :]
    yte = y[:, train_periods:]
    return Xtr, ytr, Xte, yte


class StandardScaler:

    def fit_transform(self, y):
        self.mean = torch.mean(y)
        self.std = torch.std(y) + 1e-4
        return (y - self.mean) / self.std

    def inverse_transform(self, y):
        return y * self.std + self.mean

    def transform(self, y):
        return (y - self.mean) / self.std


class MaxScaler:

    def fit_transform(self, y):
        self.max = torch.max(y)
        return y / self.max

    def inverse_transform(self, y):
        return y * self.max

    def transform(self, y):
        return y / self.max


class MeanScaler:

    def fit_transform(self, y):
        self.mean = torch.mean(y)
        return y / self.mean, self.mean

    def transform(self, y):
        self.mean = torch.mean(y)
        return y / self.mean

    def inverse_transform(self, y):
        return y * self.mean

    def f_transform(self, y):
        self.mean = torch.mean(y)
        return y / self.mean


class LogScaler:

    def fit_transform(self, y):
        return torch.log1p(y)

    def inverse_transform(self, y):
        return torch.expm1(y)

    def transform(self, y):
        return torch.log1p(y)


def extract_data(name, period, process):
    # 数据路径，提取
    data_path = get_data_path()
    data = pd.read_csv(os.path.join(data_path, name), parse_dates=["date" + str(process)])
    data.drop(data[np.isnan(data['date' + str(process)])].index, inplace=True)
    # 年月日数据提取方法
    data["year"] = data["date" + str(process)].apply(lambda x: x.year)
    data["day_of_week"] = data["date" + str(process)].apply(lambda x: x.dayofweek)
    data["month"] = data["date" + str(process)].dt.month
    data["day"] = data["date" + str(process)].dt.day
    data["day_of_order"] = 1
    # 提取一个数据所属的订单的天数
    # for i in range(1, data.shape[0]):
    #     if data.loc[i, "day"] == data.loc[i - 1, "day"]:
    #         data.loc[i, "day_of_order"] = data.loc[i - 1, "day_of_order"]
    #     else:
    #         data.loc[i, "day_of_order"] = data.loc[i - 1, "day_of_order"] + 1
    for i in range(0, data.shape[0]):
        if data.loc[i, "times"] == 1:
            data.loc[i, "day_of_order"] = 1
            first_day = data.loc[i, "date" + str(process)]
        else:
            # 计算天数的差
            data.loc[i, "day_of_order"] = (data.loc[i, "date" + str(process)] - first_day).days+1
    # 选择数据时间段
    data = data.loc[(data["date" + str(process)].dt.date >= date(period[0], period[1], period[2])) & (
            data["date" + str(process)].dt.date <= date(period[3], period[4], period[5]))]
    # 数据划分为年、月、日、次数
    months = data["month"]
    day_of_w = data["day_of_week"]
    day_of_month = data["day"]
    day_of_order = data["day_of_order"]
    hours = data["hours" + str(process)]  # 数据表里的列索引
    numbers = data["times"]
    # features.shape
    dataset_x = np.c_[np.asarray(months), np.asarray(day_of_order), np.asarray(hours), np.asarray(numbers)]
    # dataset_x = np.c_[np.asarray(months), np.asarray(hours), np.asarray(numbers)]
    # dataset_x = np.c_[np.asarray(hours), np.asarray(numbers)]
    features = dataset_x.shape[1]
    num_periods = len(dataset_x)
    dataset_x = np.asarray(dataset_x).reshape((-1, num_periods, features))
    dataset_y = np.asarray(data["assemble_time" + str(process)]).reshape((-1, num_periods))
    return dataset_x, dataset_y


def data_partition(dataset_x, dataset_y, time_step):
    data_x = []
    data_y = []
    num_row = dataset_x.shape[1]
    for i in range(0, num_row - time_step):
        data_x.append([a for a in dataset_x[0, i:i + time_step]])
        data_y.append([a for a in dataset_y[0, i:i + time_step]])
    # for i in range(0, num_row - time_step, time_step):
    #     data_x.append([a for a in dataset_x[0, i:i + time_step]])
    #     data_y.append([a for a in dataset_y[0, i:i + time_step]])
    data_x, data_y = np.array(data_x), np.array(data_y)
    # 划分训练集和测试集
    rows_sum = data_x.shape[0]
    train_rows = 1500
    test_row = 125
    # shape (batch, time_step, input_size)
    pre_x, pre_y = data_x[:train_rows, :], data_y[:train_rows]
    # advance_test_x, advance_test_y = data_x[train_rows:rows_sum-test_row, :], data_y[train_rows:rows_sum-test_row]
    online_x, online_y = data_x[train_rows:rows_sum-test_row, :], data_y[train_rows:rows_sum-test_row]
    # random_x, random_y = data_x[train_rows-100:train_rows, :], data_y[train_rows-100:train_rows]
    test_x, test_y = data_x[-test_row:, :], data_y[-test_row:]
    # 转化为Tensor
    pre_x, pre_y = torch.tensor(pre_x).float(), torch.tensor(pre_y).float()
    online_x, online_y = torch.tensor(online_x).float(), torch.tensor(online_y).float()
    test_x, test_y = torch.tensor(test_x).float(), torch.tensor(test_y).float()
    # advance_test_x, advance_test_y = torch.tensor(advance_test_x).float(), torch.tensor(advance_test_y).float()
    # random_x, random_y = torch.tensor(random_x).float(), torch.tensor(random_y).float()
    return pre_x, pre_y, online_x, online_y, test_x, test_y


class ModelManager:
    def __init__(self, args, device, save_test_results='test_perf.txt'):
        """
        Create CL experiment manager
        """

        self.args = args
        self.save_test_results = save_test_results
        self.device = device

    def save_model(self, model, processID, path):
        torch.save(model.state_dict(),
                   os.path.join(path, 'p' + str(processID) + 'e' + str(self.args.pre_epoch_nums) + '.pt'))

    def load_models(self, model, processID, path):
        check = torch.load(os.path.join(path, 'p' + str(processID) + 'e' + str(self.args.pre_epoch_nums) + '.pt'),
                           map_location=self.device)

        model.load_state_dict(check)
        model.eval()
        return model

    def create_models(self, num_features, args, device, optimizer_ftml):
        """Create models for CL experiment."""

        train_models = defaultdict(list)
        train_models[args.modelname].append(SRVAE(num_features, args.global_nlayers,args.global_hidden_size, args.n_factors, args.multiple).to(device))
        if optimizer_ftml:
            train_models[args.modelname].append(FTML(train_models[args.modelname][0].parameters(), lr=args.ftml_lr))
            # train_models[args.modelname].append(FTRL(train_models[args.modelname][0].parameters(), alpha=1, beta=1, l1=1, l2=0.1))
            # train_models[args.modelname].append(torch.optim.SGD(train_models[args.modelname][0].parameters(), lr=args.ftml_lr))
        else:
            train_models[args.modelname].append(torch.optim.Adam(train_models[args.modelname][0].parameters(), lr=args.Adam_lr))
            # train_models[args.modelname] = self.load_models(train_models[args.modelname], args.modelname, os.path.join('saved_models'))

        return train_models

    def append_models(self, train_models, num_features, args, device, index, optimizer_ftml):
        """Create models for CL experiment."""
        train_models[index].append(SRVAE(num_features, args.global_nlayers,args.global_hidden_size, args.n_factors, args.multiple).to(device))
        if optimizer_ftml:
            train_models[index].append(FTML(train_models[index][0].parameters(), lr=args.ftml_lr))
        else:
            train_models[index].append(torch.optim.Adam(train_models[index][0].parameters(), lr=args.Adam_lr))

        return train_models


def cuda_usage(loop=5):
    sum_usage = []
    print('检查cuda利用率，大于阈值则选择cpu')
    for i in range(loop):
        cmd = 'nvidia-smi.exe --query-gpu=utilization.gpu --format=csv'
        out = subprocess.getoutput(cmd)
        usage = list(map(float, re.findall(r"\d+\.?\d*", out)))
        sum_usage.append(usage[0])
        time.sleep(0.2)
    print('cuda利用率：', max(sum_usage))
    return max(sum_usage)


def to_data_loader(args, period):
    # 选择归一化方法
    y_scaler = MeanScaler()
    x_scaler = MeanScaler()
    # 数据提取

    raw_x, raw_y = extract_data('data.csv', period, process=args.processID)
    # 数据划分
    pre_x, pre_y, online_x, online_y, test_x, test_y = data_partition(raw_x, raw_y, args.timestep)
    # 预训练数据
    pre_train_x = pre_x
    _, mean_x = x_scaler.fit_transform(pre_train_x)
    pre_train_y, mean_y = y_scaler.fit_transform(pre_y)

    # 在线训练数据
    online_train_x = online_x
    online_train_y = online_y / mean_y
    # 实际测试数据
    test_y = test_y / mean_y
    # 打包pre train数据
    pre_train_data = Data.TensorDataset(pre_train_x, pre_train_y)
    pre_train_loader = Data.DataLoader(dataset=pre_train_data, batch_size=args.pre_batch_size, shuffle=True)
    # 打包online train数据
    online_train_data = Data.TensorDataset(online_train_x, online_train_y)
    online_train_loader = Data.DataLoader(dataset=online_train_data, batch_size=args.online_batch_size, shuffle=False)
    # 打包实际测试集
    torch_test_data = Data.TensorDataset(test_x, test_y)
    test_loader = Data.DataLoader(dataset=torch_test_data, batch_size=args.online_batch_size, shuffle=False)
    # 计算X的特征数
    dim_0, num_rows, num_features = raw_x.shape
    return pre_train_loader, online_train_loader, test_loader, dim_0, num_rows, num_features, mean_x, mean_y, online_train_y


def probability_data_loader(args):
    # 数据路径，提取
    data_path = get_data_path()
    data = pd.read_csv(os.path.join(data_path, 'probability_data.csv'), parse_dates=["times"])
    num_row = 1750
    dynamic_data = np.asarray(data["dynamic"]).reshape((-1, num_row))
    static_data = np.asarray(data["static"]).reshape((-1, num_row))
    real_data = np.asarray(data["real"]).reshape((-1, num_row))
    dynamic_timestep_data = []
    static_timestep_data = []
    real_timestep_data = []

    for i in range(0, num_row - args.timestep):
        dynamic_timestep_data.append([a for a in dynamic_data[0, i:i + args.timestep]])
        static_timestep_data.append([a for a in static_data[0, i:i + args.timestep]])
        real_timestep_data.append([a for a in real_data[0, i:i + args.timestep]])

    dynamic_timestep_data, static_timestep_data,real_timestep_data = np.array(dynamic_timestep_data), np.array(static_timestep_data), np.array(real_timestep_data)
    # 划分训练集和测试集
    train_rows = 1500
    test_row = 125
    test_dynamic, test_static, test_real = dynamic_timestep_data[-test_row:, :], static_timestep_data[-test_row:], real_timestep_data[-test_row:]
    test_dynamic, test_static, test_real = torch.tensor(test_dynamic).float(), torch.tensor(test_static).float(), torch.tensor(test_real).float()
    # 选择归一化方法
    scaler = MeanScaler()
    _, mean = scaler.fit_transform(test_real)
    test_dynamic, test_static, test_real = test_dynamic/mean, test_static/mean, test_real/mean
    # 打包pre train数据
    dynamic_loader = Data.TensorDataset(test_real, test_dynamic)
    dynamic_test_loader = Data.DataLoader(dataset=dynamic_loader, batch_size=args.pre_batch_size, shuffle=True)
    static_loader = Data.TensorDataset(test_real, test_static)
    static_test_loader = Data.DataLoader(dataset=static_loader, batch_size=args.pre_batch_size, shuffle=True)
    return dynamic_test_loader, static_test_loader
