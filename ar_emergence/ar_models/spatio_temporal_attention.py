import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for 1D sequences."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class SpatioTemporalAttention(nn.Module):
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
        """
        A two-stage Transformer:
          1) Temporal encoder per grid-cell
          2) Spatial encoder per timestep
        """
        super().__init__()
        self.time_steps = time_steps
        self.spatial_cells = spatial_cells

        # project input channels → d_model
        self.input_proj = nn.Linear(num_channels, d_model)

        # positional encodings
        self.temporal_pos_enc = PositionalEncoding(d_model, max_len=time_steps)
        self.spatial_pos_enc  = PositionalEncoding(d_model, max_len=spatial_cells)

        # temporal Transformer (treats each of the 63 grid-boxes as a separate sequence)
        temp_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(temp_layer, num_layers)

        # spatial Transformer (treats each timestep’s 63 features as a sequence)
        spat_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.spatial_encoder = nn.TransformerEncoder(spat_layer, num_layers)

        # final per-grid-cell regressor
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, time_steps, num_channels, spatial_cells)
        Returns:
            out: (batch_size, spatial_cells)
        """
        bs = x.size(0)
        # → (bs, spatial_cells, time_steps, num_channels)
        x = x.permute(0, 3, 1, 2).contiguous()
        # flatten for temporal encoding: (bs * spatial_cells, time_steps, num_channels)
        x = x.view(bs * self.spatial_cells, self.time_steps, -1)
        # channel projection + positional
        x = self.input_proj(x)                                   # → (bs*P, T, d_model)
        x = self.temporal_pos_enc(x)
        # temporal self-attention
        x = self.temporal_encoder(x)                             # → (bs*P, T, d_model)
        # reshape back: (bs, spatial_cells, time_steps, d_model)
        x = x.view(bs, self.spatial_cells, self.time_steps, -1)

        # now spatial attention at each time step:
        # permute to (bs, time_steps, spatial_cells, d_model) → (bs * T, spatial_cells, d_model)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bs * self.time_steps, self.spatial_cells, -1)
        x = self.spatial_pos_enc(x)
        x = self.spatial_encoder(x)                              # → (bs*T, P, d_model)
        # reshape back to (bs, time_steps, spatial_cells, d_model)
        x = x.view(bs, self.time_steps, self.spatial_cells, -1)

        # reduce over time (e.g. mean pooling)
        x = x.mean(dim=1)                                         # → (bs, P, d_model)

        # predict one scalar per cell
        out = self.fc(x).squeeze(-1)                              # → (bs, P)
        return out


# Example usage
if __name__ == "__main__":
    model = SpatioTemporalAttention().cuda()
    dummy = torch.randn(4, 120, 5, 63).cuda()    # batch of 4 regions
    preds = model(dummy)
    print(preds.shape)  # torch.Size([4, 63])