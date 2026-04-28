# -*- coding: utf-8 -*-
# @创建时间 : 2024-03-29 20:17
# @作者名称 : tqc
# @文件名称 : train_and_test.py
# @开发工具: PyCharm

import numpy as np
import torch
from torch import nn
from torchinfo import summary
from models.GPT4TS import GPT4TS
from models.traditional_models import TransformerModel, CNN, LSTM


def test_all(model, test_x, device):
    model.eval()

    test_x = test_x.to(device)

    pre_result = model(test_x).cpu().detach().numpy()

    pre_rul = np.array(pre_result).reshape(-1, 1)

    return pre_rul


def val_by_data_loader(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
    val_loss /= len(val_loader)

    return val_loss


def train_main(configs, train_loader, val_loader, device):
    best_model_dict = None
    print(device)

    if configs.model_name == "transformer":
        model = TransformerModel(input_size=configs.feature_num,
                                 d_model=configs.transformer_d_model, output_size=configs.pre_len,
                                 seq_len=configs.seq_len).to(device)
    elif configs.model_name == "cnn":
        model = CNN(configs.feature_num, configs.seq_len, configs.pre_len).to(device)
    elif configs.model_name == "lstm":
        model = LSTM(configs.feature_num, configs.lstm_hidden_size, configs.lstm_num_layers, configs.pre_len).to(device)
    elif configs.model_name == "mlp":
        model = MLP(configs.seq_len, configs.mlp_hidden_size, configs.pre_len)
    elif configs.model_name == "DLinear":
        model = DLinear(configs.feature_num, configs.seq_len, configs.pre_len)
    elif configs.model_name == "PatchTST":
        model = PatchTST(configs.feature_num, configs.seq_len, configs.pre_len, configs.PatchTST_d_model)
    elif configs.model_name == "GPT4TS":
        model = GPT4TS(configs).to(device)
    else:
        model = None

    summary(model, input_size=(configs.batch_size, configs.seq_len, configs.feature_num))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate, weight_decay=1e-2 )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    early_stopping_counter = 0
    best_val_loss = float('inf')

    for epoch in range(configs.num_epochs):
        train_loss = 0.0
        print("Epoch:{}  Lr:{:.2E}".format(epoch+1, optimizer.state_dict()['param_groups'][0]['lr']))

        model.train()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        scheduler.step()    # learning rate decay
        train_loss /= len(train_loader)
        val_loss = val_by_data_loader(model, val_loader, device, criterion)
        print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.
              format(epoch + 1, configs.num_epochs, train_loss, val_loss))

        # 检查是否需要early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            best_model_dict = model.state_dict()
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= configs.patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    model.load_state_dict(best_model_dict)

    return model
