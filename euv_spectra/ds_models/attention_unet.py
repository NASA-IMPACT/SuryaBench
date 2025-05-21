import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class DoubleConv(nn.Module):
    """(conv → BN → ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=9, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=9, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Encoder: MaxPool → DoubleConv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


# -----------------------------------------------------------------------------
# Attention Gate (additive attention) – paper: "Attention U‑Net" (Oktay et al.)
# -----------------------------------------------------------------------------
class AttentionGate(nn.Module):
    """Additive attention gate that filters skip‑connection features."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal (from decoder), x: skip connection (from encoder)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi  # element‑wise attention


class Up(nn.Module):
    """Decoder: upsample → attention (optional) → concat → DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=False, use_attention=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

        self.use_attention = use_attention
        if use_attention:
            F_g = in_channels // 2  # channels after upsampling
            F_l = in_channels // 2  # channels in the corresponding skip feature map
            F_int = max(F_l // 2, 1)
            self.attention = AttentionGate(F_g, F_l, F_int)
        else:
            self.attention = None

    def forward(self, x1, x2):
        # 1) upsample decoder feature
        x1 = self.up(x1)

        # 2) spatial alignment (padding) – handles odd dims
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # 3) attention gating (optional)
        if self.attention is not None:
            x2 = self.attention(x1, x2)

        # 4) concatenate & fuse
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1×1 conv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.mpavg = nn.ConstantPad2d(12, 0)

        self.hf_conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(2048),  # 4096 -> 2048
            nn.ReLU(),
            nn.Conv2d(
                in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1
            ),  # 2048 -> 1024
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(512),  # 1024 -> 512
            nn.ReLU(),
            nn.Conv2d(
                in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1
            ),  # 512 -> 256
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(128),  # 256 -> 128
            nn.ReLU(),
            nn.Flatten(start_dim=2),
            nn.Linear(128 * 128, 1343),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.mpavg(x)
        x = self.hf_conv1(x)
        return x


# -----------------------------------------------------------------------------
# Attention U‑Net
# -----------------------------------------------------------------------------
class AttentionUNet(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 1, bilinear: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoder (with attention on all skip connections)
        self.up1 = Up(1024, 512 // factor, bilinear, use_attention=True)
        self.up2 = Up(512, 256 // factor, bilinear, use_attention=True)
        self.up3 = Up(256, 128 // factor, bilinear, use_attention=True)
        self.up4 = Up(128, 64 // factor, bilinear, use_attention=True)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path with attentive skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits  # for binary segmentation


# --- Example Usage ---
if __name__ == "__main__":
    B, C, H, W = 1, 13, 4096, 4096  # Example smaller size

    # Create a dummy input tensor
    # Ensure the tensor is on the correct device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    dummy_input = torch.randn(B, C, H, W, device=device, dtype=dtype)

    # Instantiate the UNet model
    # Ensure the model is on the correct device
    model = AttentionUNet(n_channels=C, n_classes=1).to(device)
    model = model.to(dtype=dtype)

    # Perform a forward pass
    try:
        # with torch.no_grad(): # Disable gradient calculation for inference/testing
        # output = model(dummy_input)
        with autocast(device_type="cuda", dtype=dtype):
            output = model(dummy_input)

        # Print input and output shapes
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")

        # Check if output shape matches expected (B, n_classes, H, W)
        # assert output.shape == (B, model.n_classes, H, W)
        print("Model forward pass successful!")

    except RuntimeError as e:
        print(f"Runtime Error (likely OOM): {e}")
        print("Consider reducing batch size or input dimensions (H, W).")

    # You can print the model summary (requires torchinfo)
    try:
        from torchinfo import summary

        summary(model, input_size=(B, C, H, W), dtypes=[dtype])
    except ImportError:
        print("\nInstall torchinfo for model summary: pip install torchinfo")
