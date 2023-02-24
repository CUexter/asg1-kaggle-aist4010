import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.backends import cudnn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets, transforms, utils
from tqdm import tqdm


def determinism(seed):
    random.seed(seed)

    np.random.seed(seed)

    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def imshow(img):
    """function to show image"""
    img = img * 0.5 + 0.5  # unnormalize
    npimg = img.numpy()  # convert to numpy objects
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def showData(data_loader):
    # get random training images with iter function
    dataiter = iter(data_loader)
    images, _ = next(dataiter)

    # call function on our image
    imshow(utils.make_grid(images))


class ApplyTransform(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.dataset)


def loadData(batch_size, train_data_dir, val_data_dir, split_ratio=0.8, trans=None):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = datasets.ImageFolder(train_data_dir)
    val_dataset = datasets.ImageFolder(val_data_dir)
    combined_dataset = ConcatDataset([train_dataset, val_dataset])

    # Calculate the sizes of the training and validation sets based on the split ratio
    num_data = len(combined_dataset)
    num_train = int(num_data * split_ratio)
    num_valid = num_data - num_train

    train_dataset, val_dataset = random_split(
        combined_dataset,
        [num_train, num_valid],
        generator=torch.Generator().manual_seed(42),
    )
    val_dataset = ApplyTransform(val_dataset, transform)
    if trans is not None:
        train_dataset = ApplyTransform(train_dataset, trans)
    train_dataset = ApplyTransform(train_dataset, transform)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=14)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=14)
    return train_loader, val_loader


def load(model, optimizer=None, scheduler=None, path="state", device=None, val=None):
    try:
        state = torch.load(path)
        model.load_state_dict(state["model"])
        if optimizer is not None and state["optimizer"] is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            if state["scheduler"] is not None:
                scheduler.load_state_dict(state["scheduler"])
            else:
                print("scheduler not saved last time, cant load")
        epoch = state["epoch"]
        print(f"Previous data loaded: starting from epoch: {epoch + 1}")
        if device is not None and val is not None:
            val_acc = evaluate_accuracy(model, val, device)
            print(f"Epoch: {epoch + 1} val_accuracy: {val_acc: .3f}")
    except:
        print("cant load")
        epoch = 0
    finally:
        return epoch


def getDevice():
    if torch.cuda.is_available():
        device = torch.device(torch.cuda.current_device())
        print("Running on device: {}".format(torch.cuda.get_device_name(device)))
    else:
        device = torch.device("cpu")
        print("Running on device: {}".format(device))
    return device


def getValLoss(model, val_loader, device, loss_fn):
    model.eval()  # set the model to evaluation mode
    loss = 0

    with torch.no_grad():  # disable gradient calculation for efficiency
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss += loss_fn(outputs, labels)

    return loss / len(val_loader)


def evaluate_accuracy(model, loader, device):
    model.eval()  # set the model to evaluation mode
    total_correct = 0
    total_samples = 0

    with torch.no_grad():  # disable gradient calculation for efficiency
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            _, predictions = torch.max(outputs, 1)
            total_correct += (predictions == labels.to(device)).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    return accuracy


def save(epoch, model, optimizer, scheduler, accuracy, path):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    path = f"backup/{path}/{timestr}-epoch_{epoch}-acc{int(accuracy * 100)}"
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, path)
    torch.save(state, "state")


def train(
    model,
    model_name,
    epochs,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    device,
    start=0,
    scheduler=None,
):
    writer = SummaryWriter(f"runs/{model_name}/")
    e = 0
    val_acc = 0
    try:
        for epoch in range(start, epochs):
            model.train()
            e = epoch
            running_loss = 0.0
            acc = 0
            for i, data in enumerate(tqdm(train_loader), 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step(loss)

                running_loss += loss.item()
                classes = torch.argmax(outputs, dim=1)
                acc += torch.mean((classes == labels).float())

            loss = running_loss / len(train_loader)
            val_loss = getValLoss(model, val_loader, device, loss_fn)
            print(
                f"Epoch: {epoch + 1} loss: {running_loss / len(train_loader): .3f} val_loss: {val_loss : .3f}"
            )
            val_acc = evaluate_accuracy(model, val_loader, device)
            tra_acc = acc / len(train_loader)
            print(
                f"Epoch: {epoch + 1} val_accuracy: {val_acc: .3f} train_accuracy: {tra_acc: .3f}"
            )
            writer.add_scalars(
                "loss", {"train_loss": loss, "val_loss": val_loss}, epoch + 1
            )
            writer.add_scalars(
                "accuracy", {"train_acc": tra_acc, "val_acc": val_acc}, epoch + 1
            )
            save(e, model, optimizer, scheduler, val_acc, model_name)

            if val_acc >= 0.95:
                print(f"Good enough. Stop at epoch: {epoch+1}")
                break
            if val_acc <= 0.95 and loss <= 0.01:
                print(f"Overfit GG")
                break

        print(f"Finished Training at epoch: {e+1}")
    except KeyboardInterrupt:
        print("oh no")
    finally:
        writer.flush()
        writer.close()
