import torch
import typer
from omegaconf import OmegaConf
from .data import corrupt_mnist
from .model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: str = "models/model.pth") -> None:
    """Evaluate the model on the test set."""
    cfg = OmegaConf.create({
        "input_channels": 1,
        "conv1_filters": 32,
        "conv2_filters": 64,
        "conv3_filters": 128,
        "fc_units": 10,
        "dropout_rate": 0.5,
    })
    model = MyAwesomeModel(cfg).to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    typer.run(evaluate)
