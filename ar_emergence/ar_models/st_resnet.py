import torch
import torch.nn as nn
from torchvision.models.video import r3d_18


class SpatioTemporalResNet(nn.Module):
    """
    A 3D ResNet-based model for spatio-temporal classification tasks.
    Modifies the input convolution to accept multi-channel sequences and outputs class logits.

    Args:
        in_channels (int): Number of input channels. Default is 5.
        num_classes (int): Number of output classes. Default is 63.
        pretrained (bool): Whether to use a pretrained r3d_18 model. Default is False.
    """

    def __init__(
        self, in_channels: int = 5, num_classes: int = 63, pretrained: bool = False
    ):
        super().__init__()
        self.backbone = r3d_18(pretrained=pretrained)

        # Modify the first conv layer to accept `in_channels`
        self.backbone.stem[0] = nn.Conv3d(
            in_channels,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False,
        )

        # Modify the final fully connected layer to output `num_classes`
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SpatioTemporalResNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C, S)
                - B: batch size
                - T: temporal dimension
                - C: number of channels
                - S: spatial flattened dimension

        Returns:
            torch.Tensor: Output tensor of shape (B, num_classes)
        """
        B, T, C, S = x.shape

        # Reshape and permute for Conv3D input: (B, C, T, H=1, W=S)
        x = x.view(B, T, C, 1, S)
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        return self.backbone(x)


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatioTemporalResNet(in_channels=5, num_classes=63).to(device)
    input_tensor = torch.randn(8, 120, 5, 63).to(device)  # (B=8, T=120, C=5, S=63)
    output = model(input_tensor)
    print(output.shape)  # Expected: torch.Size([8, 63])
