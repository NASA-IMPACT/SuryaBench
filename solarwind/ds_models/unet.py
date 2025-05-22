import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Initialize the DoubleConv block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            mid_channels (int, optional): Number of channels in the intermediate
                                         convolution layer. Defaults to out_channels.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        # Define the sequence of layers for the double convolution block
        self.double_conv = nn.Sequential(
            # First convolution layer
            nn.Conv2d(in_channels, mid_channels, kernel_size=9, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),  # Batch normalization
            nn.ReLU(inplace=True),  # ReLU activation
            # Second convolution layer
            nn.Conv2d(mid_channels, out_channels, kernel_size=9, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),  # Batch normalization
            nn.ReLU(inplace=True),  # ReLU activation
        )

    def forward(self, x):
        """Forward pass through the DoubleConv block."""
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        """
        Initialize the Downscaling block (encoder block).

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        # Define the sequence: Max pooling followed by DoubleConv
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # Max pooling with kernel size 2
            DoubleConv(in_channels, out_channels),  # Double convolution block
        )

    def forward(self, x):
        """Forward pass through the Downscaling block."""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        """
        Initialize the Upscaling block (decoder block).

        Args:
            in_channels (int): Number of input channels (from the layer below).
            out_channels (int): Number of output channels.
            bilinear (bool, optional): Whether to use bilinear interpolation for
                                      upsampling. If False, uses ConvTranspose2d.
                                      Defaults to True.
        """
        super().__init__()

        # Define the upsampling method
        if bilinear:
            # Use bilinear interpolation followed by a 1x1 conv
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Use transposed convolution
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Forward pass through the Upscaling block.

        Args:
            x1: Input tensor from the previous layer in the decoder.
            x2: Skip connection tensor from the corresponding encoder layer.

        Returns:
            Tensor after upsampling, concatenation, and double convolution.
        """
        # Upsample x1 (tensor from the layer below)
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]  # Difference in Height
        diffX = x2.size()[3] - x1.size()[3]  # Difference in Width

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        # Apply the double convolution block
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution layer"""

    def __init__(self, in_channels, out_channels):
        """
        Initialize the final output convolution layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output classes (e.g., 1 for binary).
        """
        super(OutConv, self).__init__()
        # Define the 1x1 convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.mpavg = nn.ConstantPad2d(12, 0)

        self.hf_conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(2048),  # 4096 -> 2048
            nn.ReLU(),
            nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1
            ),  # 2048 -> 1024
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(512),  # 1024 -> 512
            nn.ReLU(),
            nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1
            ),  # 512 -> 256
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(128),  # 256 -> 128
            nn.ReLU(),
            nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1
            ),  # 128 -> 64
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(32),  # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1
            ),  # 32 -> 16
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1
            ),  # 8 -> 4
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(2),  # 4 -> 2
            nn.ReLU(),
            nn.Conv2d(
                in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1
            ),  # 2 -> 1
        )

    def forward(self, x):
        """Forward pass through the output convolution."""
        x = self.conv1(x)
        x = self.mpavg(x)
        x = self.hf_conv1(x)
        # x =x.view(-1,1)
        return x


class UNet(nn.Module):
    """
    The U-Net architecture implementation in PyTorch.
    """

    def __init__(self, n_channels=13, n_classes=1, bilinear=False):
        """
        Initializes the UNet model.

        Args:
            n_channels (int, optional): Number of input channels
            n_classes (int, optional): Number of output classes. Defaults to 1.
            bilinear (bool, optional): Whether to use bilinear interpolation
                                       in the decoder. Defaults to False.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # --- Encoder Path ---
        self.inc = DoubleConv(n_channels, 64)  # Initial convolution block
        self.down1 = Down(64, 128)  # First downscaling block
        self.down2 = Down(128, 256)  # Second downscaling block
        self.down3 = Down(256, 512)  # Third downscaling block
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)  # Fourth downscaling block

        # --- Decoder Path ---
        self.up1 = Up(1024, 512 // factor, bilinear)  # First upscaling block
        self.up2 = Up(512, 256 // factor, bilinear)  # Second upscaling block
        self.up3 = Up(256, 128 // factor, bilinear)  # Third upscaling block
        self.up4 = Up(128, 64, bilinear)  # Fourth upscaling block

        # --- Output Layer ---
        self.outc = OutConv(64, n_classes)  # Final output convolution

    def forward(self, x):
        """
        Defines the forward pass of the U-Net.

        Args:
            x (Tensor): Input tensor of shape (batch_size, n_channels, height, width).

        Returns:
            Tensor: Output segmentation map. If n_classes=1, typically passed
                    through Sigmoid activation outside the model during training/inference.
        """
        # --- Encoder ---
        x1 = self.inc(x)  # Initial conv output (for skip connection 1)
        x2 = self.down1(x1)  # Down 1 output (for skip connection 2)
        x3 = self.down2(x2)  # Down 2 output (for skip connection 3)
        x4 = self.down3(x3)  # Down 3 output (for skip connection 4)
        x5 = self.down4(x4)  # Bottleneck output

        # --- Decoder ---
        x = self.up1(x5, x4)  # Upsample 1 + skip 4
        x = self.up2(x, x3)  # Upsample 2 + skip 3
        x = self.up3(x, x2)  # Upsample 3 + skip 2
        x = self.up4(x, x1)  # Upsample 4 + skip 1

        # --- Output ---
        output = self.outc(x)  # Final 1x1 convolution

        return output


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
    model = UNet(n_channels=C, n_classes=1).to(device)
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
