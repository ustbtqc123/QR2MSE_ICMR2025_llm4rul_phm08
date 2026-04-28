# -*- coding: utf-8 -*-
# @创建时间 : 2024-03-04 19:50
# @作者名称 : tqc
# @文件名称 : traditional_models.py
# @开发工具: PyCharm
import math
import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.relu(lstm_out)

        # gathering only the latent end-of-sequence for the linear layer
        lstm_out = lstm_out[:, -1, :]

        fc1_out = self.fc1(lstm_out)
        fc1_out = self.relu(fc1_out)

        fc2_out = self.fc2(fc1_out)
        return fc2_out.squeeze()


class CNN(nn.Module):
    def __init__(self, input_channels, seq_len, output_size):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(32 * (seq_len // 4), output_size)  # seq_len // 4是因为经过两次池化层，每次都会将序列长度减半

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv_out1 = self.conv_layer1(x)
        conv_out2 = self.conv_layer2(conv_out1)
        flattened = conv_out2.reshape(conv_out2.size(0), -1)
        output = self.fc(flattened)
        return output.squeeze()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, output_size, seq_len, num_encoder_layers=1):
        super(TransformerModel, self).__init__()

        self.input_fc = nn.Linear(input_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=4 * d_model,
            dropout=0.1,
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc1 = nn.Linear(seq_len * d_model, output_size)

    def forward(self, x):
        x = self.input_fc(x)  # (256, 24, 128)
        x = self.pos_emb(x)  # (256, 24, 128)
        x = self.encoder(x)
        x = x.flatten(start_dim=1)
        out = self.fc1(x)

        return out.squeeze()
