import matplotlib.pyplot as plt
import sys
import torch
import hydra
from hydra.utils import instantiate
from loguru import logger
from omegaconf import OmegaConf
from .data import corrupt_mnist
from .model import MyAwesomeModel

logger.remove()  # Remove default handler
logger.add(sys.stderr, format="{message}", level="INFO")  # Match original logging format

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train(cfg) -> None:
    """Train a model on MNIST."""
    # Add file logging to Hydra's output directory
    import hydra.core.hydra_config
    logger.add(f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/train.log", format="{message}", level="INFO")
    
    logger.info("Training day and night")
    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # Use config values
    lr = cfg.training.lr
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs
    seed = cfg.training.seed

    torch.manual_seed(seed)

    logger.info(f"Using: lr={lr}, batch_size={batch_size}, epochs={epochs}, seed={seed}")

    model = MyAwesomeModel(cfg.model).to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                logger.info(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    logger.info("Training complete")
    torch.save(model.state_dict(), "models/model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")


if __name__ == "__main__":
    train()
