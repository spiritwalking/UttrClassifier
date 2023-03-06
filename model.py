import torch
import torch.nn as nn
import math


# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_classes, num_layers=2, nhead=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout),
            num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, input, mask=None):
        # src: (seq_len, batch_size)
        src_embed = self.encoder.embedding(input) * math.sqrt(self.d_model)
        src_embed = self.encoder.pos_encoder(src_embed)
        output = self.encoder(src_embed, src_key_padding_mask=mask)
        # output: (seq_len, batch_size, d_model)
        output = output.mean(dim=0)  # (batch_size, d_model)
        output = self.fc(output)  # (batch_size, num_classes)
        return output
