# models/transformer_encoder.py
# Minimal transformer-based landscape encoder stub (PyTorch required).
import torch
import torch.nn as nn

class LandscapeTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (batch, dim) -> (seq_len, batch, d_model) as required by nn.Transformer
        # For landscapes, we treat features as a sequence of length = dim with batch size = batch
        x_proj = self.input_proj(x)  # (batch, d_model)
        seq = x_proj.unsqueeze(0)  # (1, batch, d_model)
        out = self.transformer(seq)  # (1, batch, d_model)
        out = out.squeeze(0)
        return out