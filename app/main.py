import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision as tv
import torchvision.transforms as transforms

from model import Autoencoder
from torch.autograd import Variable
from torchvision.utils import save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    transform=transform
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog',
           'frog', 'horse', 'ship', 'truck')

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=4,
    shuffle=False,
    num_workers=2
)


num_epochs = 5
batch_size = 128

model = Autoencoder().to(device)

distance = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in data_loader:
        img, _ = data
        img = Variable(img).to(device)

        # ============ forward ============
        output = model(img)
        loss = distance(output, img)

        # ============ backprop ============
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ============ LOGS ============
    print('epoch [{}/{}], loss:{:.4f}'.format(
        epoch+1, num_epochs, loss))
