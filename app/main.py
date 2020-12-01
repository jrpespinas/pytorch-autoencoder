import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv
import torchvision.transforms as transforms

from model import Autoencoder

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4466),
                         (0.247, 0.243, 0.261))
])

trainTransform = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.4914, 0.4822, 0.4466), (0.247, 0.243, 0.261))
])

train_set = tv.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

data_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

test_set = tv.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    tranform=transform
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
           'frog', 'horse', 'ship', 'truck')

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=4,
    shuffle=False,
    num_workers=2
)
