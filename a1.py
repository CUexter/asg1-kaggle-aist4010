import os

import torch
from torch import nn, optim
from torch.optim import lr_scheduler as lr
from torchvision import transforms
from torchvision.models import resnet18

from models import VGG, SEResNet, SimpleCNN
from utils.utils import determinism, getDevice, load, loadData, showData, train

seed = 1337
# determinism(seed)

model_name = "resnet18"
os.makedirs(f"backup/{model_name}", exist_ok=True)
train_data_dir = "./train"
val_data_dir = "./val"

device = getDevice()


transList = [
    # transforms.RandomResizedCrop(size=128),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomAutocontrast(),
    # transforms.RandomGrayscale(),
    # transforms.RandomRotation(degrees=10),
]

trans = transforms.Compose(
    [
        transforms.RandomOrder(transList),
        transforms.RandomChoice(transList),
    ]
)

batch_size = 64


train_loader, val_loader = loadData(
    batch_size, train_data_dir, val_data_dir, trans=trans
)

showData(train_loader)


net = resnet18().to(device)
print(net)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters())

epochs = 100


start = load(net, optimizer)


train(
    net,
    model_name,
    epochs,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    device,
    start=start,
)


# print(evaluate_accuracy(net, val_loader))
