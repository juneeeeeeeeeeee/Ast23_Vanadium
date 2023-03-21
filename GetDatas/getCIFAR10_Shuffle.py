import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ]
)


def get_data(n_train, batch_size):
    train_set = datasets.CIFAR10(root='./', train=True, download=True, transform=transform)
    pass


def get_test(batch_size):
    test_set = datasets.CIFAR10(root='./', train=False, download=True, transform=transform)
    pass

