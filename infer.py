import os

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision.transforms import transforms

from utils.utils import evaluate_accuracy, getDevice, load

device = getDevice()

model_name = "resnet18"
model = resnet18().to(device)
start = load(model, path="state")

# Define the path to the unlabeled data directory
unlabeled_data_dir = "./test"
train_data_dir = "./train"
val_data_dir = "./val"

# Define the batch size for the DataLoader
batch_size = 256

# Define the transformations to be applied to the input data
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_dataset = ImageFolder(train_data_dir, transform=transform)
val_dataset = ImageFolder(val_data_dir, transform=transform)

# Calculate the sizes of the training and validation sets based on the split ratio

train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=14)
val_loader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=14)


class testDataset(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = all_imgs

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, self.total_imgs[idx]


# Create an ImageFolder dataset from the unlabeled data directory
unlabeled_dataset = testDataset(unlabeled_data_dir, transform=transform)

# Create a DataLoader for the unlabeled dataset
unlabeled_loader = DataLoader(
    dataset=unlabeled_dataset, batch_size=batch_size, shuffle=False, num_workers=4
)


def infer_unlabeled_data(model, unlabeled_loader):

    # Set the model to evaluation mode
    model.eval()

    filenames = []

    # Turn off autograd to speed up inference
    with torch.no_grad():
        # Create an empty tensor to store the predictions
        predictions = torch.tensor([], dtype=torch.int64, device=device)

        # Iterate over the unlabeled data loader
        for inputs, filename in unlabeled_loader:
            inputs = inputs.to(device)

            # Forward pass through the model to obtain predictions
            outputs = model(inputs)
            predictions = torch.cat(
                (predictions, torch.argmax(F.softmax(outputs, dim=1), dim=1))
            )
            filenames += filename

    return predictions, filenames


predictions, filenames = infer_unlabeled_data(model, unlabeled_loader)
for filename, predict in zip(filenames, predictions):
    print(filename, train_dataset.classes[predict])

print(f"{evaluate_accuracy(model, val_loader, device):.3f}")
