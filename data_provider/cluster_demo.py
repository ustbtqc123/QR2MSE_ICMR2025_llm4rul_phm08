# -*- coding: utf-8 -*-
# @创建时间 : 2024-06-25 16:16
# @作者名称 : tqc
# @文件名称 : cluster_demo.py
# @开发工具: PyCharm
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def cluster_plot():
    data_path = r'../CMAPSSData'
    dataset_uno = 'FD002'

    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # 确保在Windows系统上能够找到字体（可选）
    plt.rcParams['font.family'] = 'sans-serif'

    data_names = ['unit_id', 'cycle', 'setting_1', 'setting_2', 'setting_3']
    for i in range(1, 22):
        data_names.append('sensor_measurement' + str(i))

    train_df = pd.read_csv(os.path.join(data_path, f"train_{dataset_uno}.txt"),
                           header=None, delim_whitespace=True, names=data_names)
    test_df = pd.read_csv(os.path.join(data_path, f"test_{dataset_uno}.txt"),
                          header=None, delim_whitespace=True, names=data_names)

    # drop null
    train_df = train_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)
    train_operations = train_df[['setting_1', 'setting_2', 'setting_3']].values
    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=6, random_state=0)
    kmeans.fit(train_operations)

    # 获取聚类结果
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    # 3. 绘制三维聚类图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制每个聚类的点
    for i in range(6):
        ax.scatter(train_operations[labels == i, 0], train_operations[labels == i, 1],
                   train_operations[labels == i, 2], s=50, c=f'C{i}', label=f'Condition {i + 1}')

    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=50, marker='x', c='red', label='Cluster Center')

    ax.set_xlabel('operational setting 1')
    ax.set_ylabel('operational setting 2')
    ax.set_zlabel('operational setting 3')
    # ax.set_title('3D visualization of K-means clustering')
    ax.legend(loc='upper left')

    plt.show()


def cluster_cmp_plot():
    data_path = r'../CMAPSSData'
    dataset_uno = 'FD002'

    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    # 确保在Windows系统上能够找到字体（可选）
    plt.rcParams['font.family'] = 'sans-serif'

    data_names = ['unit_id', 'cycle', 'setting_1', 'setting_2', 'setting_3']
    for i in range(1, 22):
        data_names.append('sensor_measurement' + str(i))

    train_df = pd.read_csv(os.path.join(data_path, f"train_{dataset_uno}.txt"),
                           header=None, delim_whitespace=True, names=data_names)
    test_df = pd.read_csv(os.path.join(data_path, f"test_{dataset_uno}.txt"),
                          header=None, delim_whitespace=True, names=data_names)

    # drop null
    train_df = train_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)

    train_operations = train_df[['setting_1', 'setting_2', 'setting_3']].values
    test_operations = test_df[['setting_1', 'setting_2', 'setting_3']].values
    # 使用K-means进行聚类
    kmeans = KMeans(n_clusters=6, random_state=0)
    kmeans.fit(train_operations)

    # 获取聚类结果
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    train_df['op_cond'] = labels
    test_df['op_cond'] = kmeans.predict(test_operations)

    var_indexs = [i + 4 for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
    var_names = train_df.columns[var_indexs]

    raw_data = train_df.loc[train_df['unit_id'] == 2, var_names[0]]
    train_df, test_df = condition_scaler(train_df, test_df, var_names)

    scaler = MinMaxScaler()
    train_df[var_names] = scaler.fit_transform(train_df[var_names])
    test_df[var_names] = scaler.transform(test_df[var_names])
    train_df[var_names] = train_df.groupby('unit_id')[var_names]. \
        apply(lambda x: x.ewm(alpha=0.5).mean()).reset_index(level=0, drop=True)

    scaler_data = train_df.loc[train_df['unit_id'] == 2, var_names[0]]
    # scaler_data = moving_average(scaler_data)
    # print(train_df.groupby('unit_id').max())
    plt.figure()
    plt.plot(raw_data)
    plt.xlabel('Time/cycle')
    plt.ylabel('T24')
    plt.figure()
    plt.plot(scaler_data, color=(0, 0, 1, 0.7))
    plt.xlabel('Time/cycle')
    plt.ylabel('T24')
    plt.show()


def condition_scaler(df_train, df_test, sensor_names):
    # apply operating condition specific scaling
    scaler = StandardScaler()
    print(df_train['op_cond'].unique())
    for condition in df_train['op_cond'].unique():
        scaler.fit(df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_train.loc[df_train['op_cond'] == condition, sensor_names] = scaler.transform(
            df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_test.loc[df_test['op_cond'] == condition, sensor_names] = scaler.transform(
            df_test.loc[df_test['op_cond'] == condition, sensor_names])
    return df_train, df_test


def moving_average(interval, window_size=3):
    window = np.ones(int(window_size)) / float(window_size)
    re = np.convolve(interval, window, 'same')
    re = np.hstack((re[:-window_size + 1], interval[-window_size + 1:]))
    return re


if __name__ == '__main__':
    cluster_cmp_plot()

    # import pandas as pd
    #
    # # 生成一组示例数据
    # data = pd.Series([10, 8, 9, 12, 15, 14, 13, 11, 10, 9, 8, 10])
    #
    # # 计算EWMA值，指定alpha参数为0.5
    # ewma_data = data.ewm(alpha=0.8).mean()
    #
    # # 输出原始数据和EWMA数据
    # # print("Original Data:\n", data)
    # # print("\nEWMA Data:\n", ewma_data)
    # plt.plot(data)
    # plt.plot(ewma_data)
    # plt.show()

