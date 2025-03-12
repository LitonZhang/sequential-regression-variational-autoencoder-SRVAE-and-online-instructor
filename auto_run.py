import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pre', "-p", action='store_true', default=False, help='False to disable pre training')
parser.add_argument('--online', "-o", action='store_true', default=False, help='True to able DQN training')
parser.add_argument('--select', "-s", action='store_true', default=False, help='True to able pre iteration training')
# 封装args
args = parser.parse_args()

if __name__ == "__main__":
    process_id = np.arange(1, 87)
    # 单独预训练
    if args.pre and not args.online:
        for p_id in process_id:
            os.system(f'python main.py --pre_train  --processID {p_id}')
    # 预训练后在线训练DQN
    elif args.pre and args.online:
        for p_id in process_id:
            os.system(f'python main.py --pre_train --online_train '
                      f'--processID {p_id}')  # 1000为online DQN epoch次数
    # 单独在线训练DQN
    elif args.online and not args.pre and not args.select:
        for p_id in process_id:
            os.system(f'python main.py --online_train --processID {p_id}')
    # 单独预训练(选设备)
    elif args.select and not args.online:
        for p_id in process_id:
            os.system(f'python main_select_device.py --pre_train --processID {p_id}')
    # 预训练后在线训练DQN(选设备)
    elif args.select and args.online:
        for p_id in process_id:
            os.system(f'python main_select_device.py --pre_train --online_train '
                      f'--processID {p_id}')  # 1000为online DQN epoch次数
    else:
        print('There is no this situation')
