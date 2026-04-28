# -*- coding: utf-8 -*-
# @创建时间 : 2024-04-03 17:42
# @作者名称 : tqc
# @文件名称 : tools.py
# @开发工具: PyCharm
# 自定义评价指标
import csv
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import math


def RMSE(y_true, y_pre):
    return np.sqrt(mean_squared_error(y_true, y_pre))


def Scoring_2008(y_true, y_pre):
    score_list = []
    for i in range(len(y_true)):
        dk = y_pre[i] - y_true[i]
        if dk <= 0:
            Sk = math.exp(-(dk / 13)) - 1
        else:
            Sk = math.exp(dk / 10) - 1
        score_list.append(Sk)
    sum_score = sum(score_list)
    return sum_score


def stat_result(true_rul, pre_rul):
    true_rul = true_rul.ravel()
    pre_rul = pre_rul.ravel()
    rmse = round(RMSE(true_rul, pre_rul), 4)
    score = round(Scoring_2008(true_rul, pre_rul), 4)
    return rmse, score


def plot_result(true_rul, pre_rul, dataset_uno, model_name, engine, save_path='img_output'):
    plt.rcParams.update({'font.size': 32})  # 设置全局字体大小为16
    rmse, score = stat_result(true_rul, pre_rul)
    plt.figure(figsize=(16, 8))
    # 将两个列表合并成二维列表
    combined_list = list(zip(true_rul, pre_rul))
    # 按照第一个维度进行排序
    sorted_list = sorted(combined_list, key=lambda x: x[0], reverse=True)
    true_rul = [k[0] for k in sorted_list]
    pre_rul = [k[1] for k in sorted_list]
    plt.plot(pre_rul, color='red', label='Prediction', marker='o', linestyle='-', linewidth=1.2)
    plt.plot(true_rul, color='blue', label='Ground Truth', marker='v', linestyle='-', linewidth=1.2)
    # plt.title(f"{dataset_uno}_{model_name}, RMSE: {rmse}, Score2008: {score}", fontsize=12)
    plt.ylabel('RUL')
    plt.xlabel('Unit ID')
    plt.legend(loc='upper right')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f'{dataset_uno}_{model_name}_{engine}_pre.png')
                , bbox_inches='tight', dpi=300)  # 600


def plot_result_single_unit(true_rul, pre_rul, dataset_uno, model_name, engine, save_path='img_output'):
    plt.rcParams.update({'font.size': 32})  # 设置全局字体大小为16
    rmse, score = stat_result(true_rul, pre_rul)
    plt.figure(figsize=(16, 8))
    plt.plot(pre_rul, color='red', label='Prediction', marker='o', linestyle='-', linewidth=1.2)
    plt.plot(true_rul, color='blue', label='Ground Truth', marker='v', linestyle='-', linewidth=1.2)
    # plt.title(f"{dataset_uno}_{model_name}, RMSE: {rmse}, Score2008: {score}", fontsize=12)
    plt.ylabel('RUL')
    plt.xlabel('Time/Cycle')
    plt.legend(loc='upper right')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, f'{dataset_uno}_{model_name}_{engine}_pre.png')
                , bbox_inches='tight', dpi=300)  # 600


def save_args(args_dict, dataset_uno, save_path='result'):
    file_exists = os.path.isfile(os.path.join(save_path, f'{dataset_uno}.csv'))
    headers = list(args_dict.keys())
    values = list(args_dict.values())
    with open(os.path.join(save_path, f'{dataset_uno}.csv'), 'a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)
        # 写入数据行
        writer.writerow(values)


