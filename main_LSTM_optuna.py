from __future__ import print_function
import argparse
import os
import optuna
from optuna_dashboard import run_server
from tqdm import tqdm
import util
from DQN import DQN
from LSTMTrain import *
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_train', "-p", action='store_true', default=False)  # 预训练
    parser.add_argument('--online_train', "-o", action='store_true', default=False)  # DQN训练
    parser.add_argument("--modelname", type=str, default='SRVAE')
    parser.add_argument("--processID", type=int, default=1)
    parser.add_argument("--pre_epoch_nums", "-pn", type=int, default=800)  # 1000 保证多个模型的泛化性能
    parser.add_argument("--online_epoch_nums", type=int, default=1000)  # 1000
    parser.add_argument("-Adam_lr", type=float, default=1e-3)  # 大部分4e-3，少部分5e-3
    parser.add_argument("-ftml_lr", type=float, default=1e-3)  # 1e-5
    parser.add_argument("--global_hidden_size", type=int, default=64)  # 128  提取全局信息0
    parser.add_argument("--global_nlayers", type=int, default=1)  # 2
    # parser.add_argument("--n_factors", type=int, default=14)  # 14
    # parser.add_argument("--multiple", type=int, default=1)  # 14
    parser.add_argument("--test_length", type=int, default=30)
    parser.add_argument("--timestep", type=int, default=6)  # 2
    parser.add_argument("--show_plot", "-sp", action="store_true")
    parser.add_argument("--pre_batch_size", type=int, default=256)  # 256
    parser.add_argument("--online_batch_size", type=int, default=5)
    parser.add_argument('--no_cuda', action='store_true', default=False, help='True to disable CUDA')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    # 增量
    parser.add_argument('--ewc_lambda', type=float, default=0, help='EWC task. 0 to disable EWC.')
    # DQN
    parser.add_argument('--DQNepsilon', type=float, default=0.8)
    parser.add_argument('--memory_capacity', type=int, default=1000, help='do not use scientific notation')  # 10000
    parser.add_argument('--DQN_target_update', type=int, default=10)
    parser.add_argument('--optuna', "-optuna", action='store_true', default=False)
    parser.add_argument('--n_trials', "-nt", type=int, default=30)
    # 封装args
    args = parser.parse_args()
    # GPU加速
    args.cuda = not args.no_cuda and torch.cuda.is_available()  # 判断pytorch是否有GPU加速
    # print(torch.cuda.is_available())
    torch.manual_seed(args.seed)  # 为所有GPU设置种子数
    device = torch.device("cuda" if args.cuda else "cpu")  # 代表将torch.Tensor分配到的设备的对象
    print('Start running on {}'.format(device))
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # 模型实例化管理器
    model_manager = util.ModelManager(args, device)

    if args.optuna:
        def objective(trial):
            args.pre_epoch_nums = trial.suggest_int('pre_epoch_nums', 300, 1000, 100)
            args.global_hidden_size = trial.suggest_categorical('global_hidden_size', [32, 64, 128, 256, 512, 1024])
            args.global_nlayers = trial.suggest_int('global_nlayers', 1, 8)
            args.timestep = trial.suggest_int('timestep', 2, 10)
            args.Adam_lr = trial.suggest_discrete_uniform('Adam_lr', 1e-6, 1e-1, 1e-6)

            # 数据相关的超参数变更，重新打包数据
            period = [2018, 1, 1, 2022, 6, 8]
            pre_train_loader, online_train_loader, test_loader, dim_0, num_rows, num_features, mean_x, mean_y, online_train_y = util.to_data_loader(
                args, period)
            print('process[' + str(args.processID) + ']pre training start...')
            #  创建模型(包括模型和优化器)
            pre_model = LSTM(num_features, args.global_nlayers, args.global_hidden_size).to(device)
            pre_optimizer = torch.optim.Adam(pre_model.parameters(), lr=args.Adam_lr)
            # 预训练开始-------------------------------------------------------------------
            train_losses_list = []
            test_losses_list = []

            for epoch in tqdm(range(1, args.pre_epoch_nums + 1)):
                # training
                train_loss = train(pre_model, pre_train_loader, device, pre_optimizer, args)  # scheduler
                # train loss
                train_losses_list.append(train_loss)
                # testing
                test_loss, results, reals = test(pre_model, test_loader, device, args)
                reals = reals * mean_y
                results = results * mean_y
                # test loss
                test_losses_list.append(test_loss)
                # figure
                plot_figure(train_losses_list, test_losses_list, results, reals, epoch, args, online=False)
            return test_loss


        storage = optuna.storages.InMemoryStorage()
        study = optuna.create_study(study_name='pre-train', direction='minimize', storage=storage)
        study.optimize(objective, n_trials=args.n_trials)
        run_server(storage)
        # para_importance = optuna.importance.get_param_importances(study)
        # fig=optuna.visualization.plot_param_importances(study)
        # fig.show()
        print(study.best_params)
        print(study.best_value)
        print(study.best_trial)

    if args.pre_train:
        start = time.time()
        print('process[' + str(args.processID) + ']pre training start...')
        # 数据相关的超参数变更，重新打包数据
        period = [2018, 1, 1, 2022, 5, 8]
        pre_train_loader, online_train_loader, test_loader, dim_0, num_rows, num_features, mean_x, mean_y, online_train_y = util.to_data_loader(
            args, period)
        #  创建模型(包括模型和优化器)
        pre_model = LSTM(num_features, args.global_nlayers, args.global_hidden_size).to(device)
        pre_optimizer = torch.optim.Adam(pre_model.parameters(), lr=args.Adam_lr)
        # 预训练开始-------------------------------------------------------------------
        train_losses_list = []
        test_losses_list = []

        for epoch in tqdm(range(1, args.pre_epoch_nums + 1)):
            # training
            train_loss = train(pre_model, pre_train_loader, device, pre_optimizer, args)  # scheduler
            # train loss
            train_losses_list.append(train_loss)
            # testing
            test_loss, results, reals = test(pre_model, test_loader, device, args)
            reals = reals * mean_y
            results = results * mean_y
            # test loss
            test_losses_list.append(test_loss)
            # figure
            plot_figure(train_losses_list, test_losses_list, results, reals, epoch, args, online=False)
        end = time.time()
        print("LSTM训练耗时", end - start)
        torch.save(pre_model,
                   os.path.join('LSTM_model', 'p' + str(args.processID) + '.pt'))

    if args.online_train:

        # 在线式增量训练数据-------------------------------------------------------
        period = [2018, 1, 1, 2022, 5, 8]
        pre_train_loader, online_train_loader, test_loader, dim_0, num_rows, num_features, mean_x, mean_y, online_train_y = util.to_data_loader(
            args, period)
        print('process[' + str(args.processID) + ']online training start...\n')
        # recreate model (包括模型和优化器)
        online_model_dict = model_manager.create_models(num_features, args, device, optimizer_ftml=True)
        online_model = online_model_dict[args.modelname][0]
        online_optimizer = online_model_dict[args.modelname][1]
        # 强化学习模型实例化
        TARGET_REPLACE_ITER = int(online_train_y.shape[0] / args.online_batch_size) * args.DQN_target_update
        dqn = DQN(device, args, TARGET_REPLACE_ITER)
        # loss存储list
        rewards = []
        test_losses_list = []
        dqn_losses = []
        action_note = []
        # 增量训练开始------------------------------------------------------------------
        for epoch in tqdm(range(1, args.online_epoch_nums + 1)):
            # load model, 0为model，1为优化器
            model_manager.load_models(online_model, args.processID, 'saved_models')  # model, modelname, path
            online_optimizer.state.clear()  # 清空优化器
            # print(online_optimizer.state)
            # 逐渐增大贪婪的阈值，使算法依赖网络输出
            if epoch < int(0.8 * args.online_epoch_nums):
                DQNepsilon = args.DQNepsilon + epoch / int(0.8 * args.online_epoch_nums) * (1 - args.DQNepsilon)
            else:
                DQNepsilon = 1
            # 增量训练
            reward, new_model, mean_dqn_loss, action_note = train_online(online_model, online_train_loader, device,
                                                                         epoch, dqn, args, DQNepsilon, mean_x,
                                                                         training=True)
            # model_manager.save_model(new_model, args.processID, 'saved_models')  # model, modelname, path
            # train loss
            rewards.append(reward)
            dqn_losses.append(mean_dqn_loss)
            # print('reward', round(train_ewc_loss, 4))
            # testing
            test_loss, results, reals = test(new_model, test_loader, device, args)
            results = results * mean_y
            reals = reals * mean_y
            # test loss
            test_losses_list.append(test_loss)
            # plot
            plot_figure(rewards, test_losses_list, results, reals, epoch, args, online=True)

        # save model [model, modelname, path]
        model_manager.save_model(new_model, args.processID, 'online_saved_models')
        torch.save(dqn,
                   os.path.join('dqn_models', 'DQNp' + str(args.processID) + 'e' + str(args.online_epoch_nums) + '.pt'))
        # 保存数据
        results, reals = pd.DataFrame(results.cpu().numpy().reshape(-1, 1)), pd.DataFrame(
            reals.cpu().numpy().reshape(-1, 1))
        rewards, dqn_losses = pd.DataFrame(rewards), pd.DataFrame(dqn_losses)
        df2 = pd.DataFrame()
        df2['rewards'], df2['dqn_loss'] = rewards, dqn_losses
        df2.to_csv('result/dqn_result.csv', index=False)
        # 将选择的action存起来,用于画图
        action_note = pd.DataFrame(action_note)
        df3 = pd.DataFrame()
        df3['action'] = action_note
        df3.to_csv('result/dqn_action.csv', index=False)
