import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

class SpatioTemporalResNet(nn.Module):
    def __init__(self, in_channels: int = 5, num_classes: int = 63, pretrained: bool = False):
        super().__init__()
        # Load a 3D ResNet-18 (for video) backbone
        self.backbone = r3d_18(pretrained=pretrained)
        # Modify the first conv layer to accept `in_channels` instead of 3
        self.backbone.stem[0] = nn.Conv3d(
            in_channels,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False
        )
        # Replace the final fully connected layer to output `num_classes`
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, C, S)
               - B: batch size
               - T: temporal length (e.g., 120)
               - C: channels (e.g., 5)
               - S: spatial flattened dimension (e.g., 63)
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        B, T, C, S = x.shape
        # Reshape spatial flatten into H=1, W=S (or adjust if you have a different grid shape)
        x = x.view(B, T, C, 1, S)  # (B, T, C, H=1, W=S)
        # Permute to (B, C, T, H, W) for Conv3D
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # Forward through 3D ResNet backbone
        logits = self.backbone(x)
        return logits

if __name__ == "__main__":
    # Example usage:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatioTemporalResNet(in_channels=5, num_classes=63).to(device)
    input_tensor = torch.randn(8, 120, 5, 63).to(device)  # batch of 8
    output = model(input_tensor)  # output.shape -> (8, 63)
    print(output.shape)  # Should print torch.Size([8, 63])
