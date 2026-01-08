from torch.utils.data import Dataset

from tero_project.data import corrupt_mnist


def test_corrupt_mnist():
    """Test the corrupt_mnist function."""
    train_dataset, test_dataset = corrupt_mnist()
    assert isinstance(train_dataset, Dataset)
    assert isinstance(test_dataset, Dataset)
    assert len(train_dataset) == 30000
    assert len(test_dataset) == 5000
