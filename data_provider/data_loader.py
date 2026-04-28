# -*- coding: utf-8 -*-
# @创建时间 : 2024-03-02 10:16
# @作者名称 : tqc
# @文件名称 : data_loader.py
# @开发工具: PyCharm
import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy import interpolate
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random


def condition_scaler(train_df, test_df, sensor_names):
    train_operations = train_df[['setting_1', 'setting_2', 'setting_3']].values
    test_operations = test_df[['setting_1', 'setting_2', 'setting_3']].values
    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=6, random_state=0, n_init= 'auto')
    kmeans.fit(train_operations)

    train_df['op_cond'] = kmeans.labels_
    test_df['op_cond'] = kmeans.predict(test_operations)

    # apply operating condition specific scaling
    scaler = StandardScaler()
    print(train_df['op_cond'].unique())
    # print(sensor_names)
    for sensor in sensor_names:
    # 检查列的数据类型，如果是int64，则转换为float64
        if train_df[sensor].dtype == 'int64':
            train_df[sensor] = train_df[sensor].astype('float64')
        if test_df[sensor].dtype == 'int64':
            test_df[sensor] = test_df[sensor].astype('float64')
    for condition in train_df['op_cond'].unique():
        scaler.fit(train_df.loc[train_df['op_cond'] == condition, sensor_names])
        train_df.loc[train_df['op_cond'] == condition, sensor_names] = scaler.transform(
            train_df.loc[train_df['op_cond'] == condition, sensor_names])
        test_df.loc[test_df['op_cond'] == condition, sensor_names] = scaler.transform(
            test_df.loc[test_df['op_cond'] == condition, sensor_names])
    return train_df, test_df


def exponential_smoothing(df, var_names, alpha=0.5):
    df = df.copy()
    df[var_names] = df.groupby('unit_id')[var_names] \
        .apply(lambda x: x.ewm(alpha=alpha).mean()).reset_index(level=0, drop=True)

    return df


def data_provider_one_unit_data(configs, data_path, dataset_uno, unit_id):
    data_names = ['unit_id', 'cycle', 'setting_1', 'setting_2', 'setting_3']
    for i in range(1, 22):
        data_names.append('sensor_measurement' + str(i))

    train_df = pd.read_csv(os.path.join(data_path, f"train_{dataset_uno}.txt"),
                           header=None, sep=r'\s+', names=data_names)
    test_df = pd.read_csv(os.path.join(data_path, f"test_{dataset_uno}.txt"),
                          header=None, sep=r'\s+', names=data_names)
    true_rul = pd.read_csv(os.path.join(data_path, f"RUL_{dataset_uno}.txt"), header=None).values
    biaoding_num = 125

    var_indexs = [i + 4 for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
    var_names = train_df.columns[var_indexs]

    # drop null
    train_df = train_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)

    # standardization
    if dataset_uno in ['FD002', 'FD004']:
        train_df, test_df = condition_scaler(train_df, test_df, var_names)

    # normalization
    scaler = MinMaxScaler()
    train_df[var_names] = scaler.fit_transform(train_df[var_names])
    test_df[var_names] = scaler.transform(test_df[var_names])

    # exponential smoothing
    # train_df = exponential_smoothing(train_df, var_names, configs.alpha)
    # test_df = exponential_smoothing(test_df, var_names, configs.alpha)

    # 测试集时间窗数据生成
    grouped_test_df = test_df[test_df['unit_id'] == unit_id]
    test_x = []
    test_y = []
    group_df_data_arr = grouped_test_df[var_names].values
    # 计算时间窗数量
    group_df_label = np.clip(np.arange(len(group_df_data_arr)-1+true_rul[unit_id-1], true_rul[unit_id-1]-1, -1)
                             , 0, configs.biaoding_num)
    # 计算时间窗数量
    num_windows = (len(group_df_data_arr) - configs.seq_len) + 1
    windows = [group_df_data_arr[i:i + configs.seq_len]
               for i in range(0, num_windows)]
    print(group_df_label)
    labels = [group_df_label[i + configs.seq_len - 1]
              for i in range(0, num_windows)]
    test_x.extend(windows)
    test_y.extend(labels)

    test_x = np.array(test_x)
    test_x = torch.from_numpy(test_x).float()
    test_y = np.array(test_y)
    test_y = torch.from_numpy(test_y).float()

    return test_x, test_y


def data_provider_phm08(configs, data_path, dataset_uno):
    random.seed(42)
    data_names = ['unit_id', 'cycle', 'setting_1', 'setting_2', 'setting_3']
    for i in range(1, 22):
        data_names.append('sensor_measurement' + str(i))

    train_df = pd.read_csv(os.path.join(data_path, f"train_{dataset_uno}.txt"),
                           header=None, sep=r'\s+', names=data_names)
    test_df = pd.read_csv(os.path.join(data_path, f"test_{dataset_uno}.txt"),
                          header=None, sep=r'\s+', names=data_names)
    true_rul = pd.read_csv(os.path.join(data_path, f"RUL_{dataset_uno}.txt"), header=None)

    biaoding_num = 125
    true_rul = np.clip(true_rul.values, 0, biaoding_num)

    var_indexs = [i + 4 for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
    var_names = train_df.columns[var_indexs]

    # drop null
    train_df = train_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)

    # standardization
    if dataset_uno in ['FD002', 'FD004']:
        train_df, test_df = condition_scaler(train_df, test_df, var_names)

    # normalization
    scaler = MinMaxScaler()
    train_df[var_names] = scaler.fit_transform(train_df[var_names])
    test_df[var_names] = scaler.transform(test_df[var_names])

    # exponential smoothing
    # train_df = exponential_smoothing(train_df, var_names, configs.alpha)
    # test_df = exponential_smoothing(test_df, var_names, configs.alpha)

    # 训练集时间窗数据生成
    grouped_train_df = train_df.groupby('unit_id')
    val_engine_index = random.sample(list(train_df['unit_id'].unique())
                  , int(train_df['unit_id'].unique().shape[0]*configs.test_size)) #generate 
    # print(len(val_engine_index))
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    for key, group_df in grouped_train_df:
        group_df_data_arr = group_df[var_names].values
        group_df_label = np.clip(group_df['cycle'].values.max() - group_df['cycle'].values
                                 , 0, configs.biaoding_num)
        # 计算时间窗数量
        num_windows = (len(group_df_data_arr) - configs.seq_len) + 1
        windows = [group_df_data_arr[i:i + configs.seq_len]
                   for i in range(0, num_windows)]
        labels = [group_df_label[i + configs.seq_len - 1]
                  for i in range(0, num_windows)]

        if key not in val_engine_index:
          train_x.extend(windows)
          train_y.extend(labels)
        else:
          val_x.extend(windows)
          val_y.extend(labels)

    # 测试集时间窗数据生成
    grouped_test_df = test_df.groupby('unit_id')
    test_x = []
    for key, group_df in grouped_test_df:
        group_df_data_arr = group_df[var_names].values
        # 计算时间窗数量
        if len(group_df_data_arr) > configs.seq_len:
            test_x.append(group_df_data_arr[-configs.seq_len:])
        else:
            data_temp_a = []
            for myi in range(group_df_data_arr.shape[1]):
                x1 = np.linspace(0, configs.seq_len - 1, len(group_df_data_arr))
                x_new = np.linspace(0, configs.seq_len - 1, configs.seq_len)
                tck = interpolate.splrep(x1, group_df_data_arr[:, myi])
                a = interpolate.splev(x_new, tck)
                data_temp_a.append(a.tolist())
            data_temp_a = np.array(data_temp_a)
            data_temp = data_temp_a.T
            test_x.append(data_temp)
    # 转为tensor格式
    train_x = np.array(train_x)
    train_x = torch.from_numpy(train_x).float()

    train_y = np.array(train_y)
    train_y = torch.from_numpy(train_y).float()

    val_x = np.array(val_x)
    val_x = torch.from_numpy(val_x).float()

    val_y = np.array(val_y)
    val_y = torch.from_numpy(val_y).float()

    print(train_y.shape, val_y.shape)
    test_x = np.array(test_x)
    test_x = torch.from_numpy(test_x).float()


    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=True)

    return train_loader, val_loader, test_x, true_rul
