import torch
import torch.nn as nn


class TestModel(nn.Module):
    """
    A simple average model that predicts the average of the observed values as the next value.
    """
    def __init__(self):
        super(TestModel, self).__init__()

    def forward(self, x):
        # x is expected to have shape (batch_size, sequence_length, feature_dim)
        # We compute the mean along the sequence dimension
        x = x[:, :, 0, :]
        return x.mean(dim=1)


# Example usage
if __name__ == "__main__":
    # Dummy input tensor of shape (batch_size=2, sequence_length=5, feature_dim=3)
    x = torch.randn(32,120, 5, 63)

    average_model = TestModel()

    persistence_output = persistence_model(x)
    average_output = average_model(x)

    print("Persistence Model Output:")
    print(persistence_output.shape)

    print("\nAverage Model Output:")
    print(average_output.shape)