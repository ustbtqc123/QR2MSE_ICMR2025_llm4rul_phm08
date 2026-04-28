# -*- coding: utf-8 -*-
# @创建时间 : 2024-04-03 17:30
# @作者名称 : tqc
# @文件名称 : main.py
# @开发工具: PyCharm
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch

from data_provider.data_loader import data_provider_phm08, data_provider_one_unit_data
from utils.tools import stat_result, plot_result, plot_result_single_unit, save_args
from utils.train_and_test import train_main, test_all


def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--alpha', type=float, default=.5)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--pre_len', type=int, default=1)
    parser.add_argument('--biaoding_num', type=int, default=125)
    parser.add_argument('--test_size', type=float, default=.2)

    # model
    parser.add_argument('--model_name',
                        default="transformer")  # "GPT4TS" "PatchTST""transformer" "lstm" "cnn" "linear" "DLinear"
    parser.add_argument('--feature_num', type=int, default=14)
    parser.add_argument('--transformer_d_model', type=int, default=64)
    parser.add_argument('--lstm_hidden_size', type=int, default=64)
    parser.add_argument('--lstm_num_layers', type=int, default=1)
    parser.add_argument('--mlp_hidden_size', type=int, default=64)
    parser.add_argument('--PatchTST_d_model', type=int, default=64)
    parser.add_argument('--is_save_model', type=int, default=0)

    # train
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--patience', type=int, default=10)

    # gpt2
    parser.add_argument('--gpt_layers', type=int, default=3)
    parser.add_argument('--block_num', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--token_mixing_factor', type=int, default=1)
    parser.add_argument('--channel_mixing_factor', type=int, default=1)
    
    # stat
    parser.add_argument('--plot_result', type=int, default=0)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    data_path = r'CMAPSSData'

    # model_list = ["transformer", "lstm", "cnn",] # "PatchTST", "DLinear",
    dataset_uno_list = ["FD004"]
    model_list = ["GPT4TS"]  # GPT4TS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = get_args()

    for dataset_uno in dataset_uno_list:
        for model_name in model_list:
            rmse_list = []
            score_list = []
            configs.model_name = model_name
            time_now = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())

            for seed in range(1,2):
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                train_loader, val_loader, test_x, true_rul = \
                    data_provider_phm08(configs, data_path, dataset_uno)

                if configs.plot_result:
                    model = torch.load(f'saved_models/{dataset_uno}.pth')
                    pre_rul = test_all(model, test_x, device)
                    rmse, score = stat_result(true_rul, pre_rul)
                    # plot_unit_id_list = [24,34] # fd001
                    # plot_unit_id_list = [133,150] # fd002
                    # plot_unit_id_list = [92,99] # fd003
                    # plot_unit_id_list = [71,99] # fd004
                    plot_result(true_rul, pre_rul, dataset_uno, 'all', configs.model_name)
                    plot_unit_id_list = list(range(1,101)) # 101 260 101 249
                    for unit_id in plot_unit_id_list:
                        test_x_i, test_y_i = data_provider_one_unit_data(configs, data_path, dataset_uno, unit_id)
                        if len(test_x_i) == 0:
                            continue
                        pre_y_i = test_all(model, test_x_i, device)
                        plot_result_single_unit(test_y_i, pre_y_i, dataset_uno, unit_id, configs.model_name)
                else:
                    model = train_main(configs, train_loader, val_loader, device)
                    pre_rul = test_all(model, test_x, device)
                    rmse, score = stat_result(true_rul, pre_rul)
                    torch.save(model, f'saved_models/{dataset_uno}.pth')

                rmse_list.append(rmse)
                score_list.append(score)
                print(rmse, score)
            if not configs.plot_result:
                args_dict = vars(configs)
                args_dict['time'] = time_now
                args_dict['rmse'] = np.mean(rmse_list)
                args_dict['score'] = np.mean(score_list)
                save_args(args_dict, dataset_uno)

