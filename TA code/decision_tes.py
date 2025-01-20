import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
import timm
import os
import numpy as np
from sklearn.metrics import accuracy_score
import PIL
from sklearn.utils import shuffle

import time
from tree import DecisionTree

# Set Random Seed
SEED = 257
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Parameters
BATCH_SIZE = 32
EPOCHS = 25
PATIENCE = 6
NAMING = "b32"

# Data Transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
}

class myDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.transform = transform
        self.images, self.labels = images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = PIL.Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label
    
def load_dataset(path='data/train/'):
    folders = {0: "elephant", 1: "jaguar", 2: "lion", 3: "parrot", 4: "penguin"}
    images = []
    labels = []
    for i in range(5):
        folder = folders[i]
        folder_path = os.path.join(path, folder)
        for file in os.listdir(folder_path):
            images.append(os.path.join(folder_path, file))
            labels.append(i)
    return images, labels

images, labels = load_dataset()
images, labels = shuffle(images, labels, random_state=SEED)
train_len = int(0.8 * len(images))

train_images, test_images = images[:train_len], images[train_len:]
train_labels, test_labels = labels[:train_len], labels[train_len:]
train_dataset = myDataset(train_images, train_labels, transform=data_transforms['train'])
val_dataset = myDataset(test_images, test_labels, transform=data_transforms['val'])

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.model = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=300)

    def forward(self, x):
        x = self.model(x)
        return x

# Helper function to extract features
def get_features_and_labels(model, dataloader, device):
    model.eval()
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            output_features = model(inputs)
            features.append(output_features.cpu().numpy())
            labels.append(targets.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels

# Training and Validation Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Models
conv_model = ConvNet().to(device)
tree_model = DecisionTree(max_depth=7)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

start_time = time.time()
# Extract features from training data
train_features, train_labels = get_features_and_labels(conv_model, train_loader, device)
end_time = time.time()
print(f"conv took {end_time - start_time:.4f} seconds.")

start_time = time.time()
# Train the decision tree classifier
tree_model.fit(train_features, train_labels)
end_time = time.time()
print(f"fit tree took {end_time - start_time:.4f} seconds.")

# Extract features from validation data
val_features, val_labels = get_features_and_labels(conv_model, val_loader, device)

# Predict with the trained decision tree
val_predictions = tree_model.predict(val_features)

# Calculate accuracy
val_accuracy = accuracy_score(val_labels, val_predictions)
print(f"Validation Accuracy: {val_accuracy:.4f}")
