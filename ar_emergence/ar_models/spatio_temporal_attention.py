import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implements standard sinusoidal positional encoding for 1D sequences.

    Args:
        d_model (int): Dimension of the model.
        max_len (int): Maximum length of the sequence. Default is 5000.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: Encoded tensor of the same shape as input.
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class SpatioTemporalAttention(nn.Module):
    """
    Two-stage Transformer model for spatio-temporal attention:
        1. Temporal Transformer per spatial grid cell
        2. Spatial Transformer per timestep

    Args:
        time_steps (int): Number of time steps. Default is 120.
        num_channels (int): Number of input channels. Default is 5.
        spatial_cells (int): Number of spatial grid cells. Default is 63.
        d_model (int): Transformer model dimension. Default is 64.
        nhead (int): Number of attention heads. Default is 8.
        num_layers (int): Number of Transformer layers. Default is 2.
        dim_feedforward (int): Feedforward layer size. Default is 256.
        dropout (float): Dropout rate. Default is 0.1.
    """

    def __init__(
        self,
        time_steps: int = 120,
        num_channels: int = 5,
        spatial_cells: int = 63,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_steps = time_steps
        self.spatial_cells = spatial_cells

        self.input_proj = nn.Linear(num_channels, d_model)
        self.temporal_pos_enc = PositionalEncoding(d_model, max_len=time_steps)
        self.spatial_pos_enc = PositionalEncoding(d_model, max_len=spatial_cells)

        temp_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temp_layer, num_layers)

        spat_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_encoder = nn.TransformerEncoder(spat_layer, num_layers)

        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for spatio-temporal attention model.

        Args:
            x (torch.Tensor): Input tensor of shape
                              (batch_size, time_steps, num_channels, spatial_cells).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, spatial_cells).
        """
        bs = x.size(0)

        # Reshape: (B, T, C, P) → (B, P, T, C)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(bs * self.spatial_cells, self.time_steps, -1)

        # Channel projection + temporal positional encoding
        x = self.input_proj(x)
        x = self.temporal_pos_enc(x)
        x = self.temporal_encoder(x)

        # Reshape for spatial encoding
        x = x.view(bs, self.spatial_cells, self.time_steps, -1)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bs * self.time_steps, self.spatial_cells, -1)

        # Spatial encoding
        x = self.spatial_pos_enc(x)
        x = self.spatial_encoder(x)

        # Reshape and pool over time
        x = x.view(bs, self.time_steps, self.spatial_cells, -1)
        x = x.mean(dim=1)

        # Final projection per cell
        out = self.fc(x).squeeze(-1)
        return out


if __name__ == "__main__":
    model = SpatioTemporalAttention().cuda()
    dummy = torch.randn(4, 120, 5, 63).cuda()
    preds = model(dummy)
    print(preds.shape)  # Expected: torch.Size([4, 63])
