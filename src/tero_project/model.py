import torch
from torch import nn


class MyAwesomeModel(nn.Module):
    """My awesome model."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cfg.input_channels, cfg.conv1_filters, 3, 1)
        self.conv2 = nn.Conv2d(cfg.conv1_filters, cfg.conv2_filters, 3, 1)
        self.conv3 = nn.Conv2d(cfg.conv2_filters, cfg.conv3_filters, 3, 1)
        self.dropout = nn.Dropout(cfg.dropout_rate)
        self.fc1 = nn.Linear(cfg.conv3_filters, cfg.fc_units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "input_channels": 1,
            "conv1_filters": 32,
            "conv2_filters": 64,
            "conv3_filters": 128,
            "fc_units": 10,
            "dropout_rate": 0.5,
        }
    )
    model = MyAwesomeModel(cfg)
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
