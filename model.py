import torch
import torch.nn as nn
import math


# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, d_model=128, num_layers=2):
        super().__init__()
        self.fc = nn.Linear(30, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=2, dim_feedforward=512, dropout=0.1),
            num_layers=num_layers)
        self.pred_layer = nn.Linear(d_model, 6)
        self.d_model = d_model

    def forward(self, input):
        output = self.fc(input)
        output = output.transpose(0, 1)
        output = self.encoder(output)
        output = output.transpose(0, 1)
        output = self.pred_layer(output)
        return output
