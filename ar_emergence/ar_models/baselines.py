import torch
import torch.nn as nn


class TestModel(nn.Module):
    """
    A simple model that predicts the average value across the time dimension
    for each spatial cell.

    Expects input of shape (batch_size, time_steps, num_channels, spatial_cells)
    and returns the mean over time steps for the first channel.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the average model.

        Args:
            x (torch.Tensor): Input tensor of shape
                              (batch_size, time_steps, num_channels, spatial_cells)

        Returns:
            torch.Tensor: Output tensor of shape
                          (batch_size, spatial_cells)
        """
        # Extract the first channel (index 0)
        x = x[:, :, 0, :]
        return x.mean(dim=1)


if __name__ == "__main__":
    # Dummy input tensor of shape (batch_size=32, time_steps=120, num_channels=5, spatial_cells=63)
    x = torch.randn(32, 120, 5, 63)

    average_model = TestModel()
    average_output = average_model(x)

    print("Average Model Output:")
    print(average_output.shape)  # Expected: torch.Size([32, 63])
