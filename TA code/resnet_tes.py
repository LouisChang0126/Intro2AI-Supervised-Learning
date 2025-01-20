import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
import timm
import os
import numpy as np
import matplotlib.pyplot as plt
import PIL
from tqdm import tqdm
from sklearn.utils import shuffle
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Set Random Seed
SEED = 257
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Parameters
BATCH_SIZE = 32
EPOCHS = 15
PATIENCE = 6
LEARNING_RATE = 1e-4
NAMING = "b32"

data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

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
train_dataset = myDataset(train_images, train_labels, transform=data_transforms)
val_dataset = myDataset(test_images, test_labels, transform=data_transforms)

class SimpleResNet3(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleResNet3, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(4, 4)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(4, 4)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(4, 4)
        
        # 定義全連接層
        self.fc = nn.Linear(64 * 2 * 2, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        x = x.view(-1, 64 * 2 * 2)  # 展平為全連接層
        x = self.fc(x)
        return x

def train(model, train_loader, criterion, optimizer, device, alpha=1.0):
    model.train()
    running_loss = 0.0

    bar = tqdm(enumerate(train_loader), total=len(train_loader))

    for i, (inputs, labels) in bar:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        bar.set_description(f"Training Loss: {loss.item():.5f}")

    return running_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    bar = tqdm(enumerate(val_loader), total=len(val_loader))

    with torch.no_grad():
        for i, data in bar:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            bar.set_description(f"Validate Loss: {loss.item():.5f}")

    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy

# run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = SimpleResNet3().to(device)
criterion = nn.CrossEntropyLoss()

# Optimizer configuration
base_params = [param for name, param in model.named_parameters() if param.requires_grad]

optimizer = optim.Adam(base_params, lr=LEARNING_RATE)

# model.get_parameter_size()

no_improvement_epochs = 0
train_losses = []
val_losses = []
max_acc = 0

for epoch in range(EPOCHS):
    train_loss = train(model, train_loader, criterion, optimizer, device, alpha=1.0)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Acc:  {val_acc:.4f}")

    no_improvement_epochs += 1
    if val_acc > max_acc:
        print(f"Saving model, Best Accuracy: {val_acc:.4f}")
        torch.save(model.state_dict(), f'model_{NAMING}.pth')
        max_acc = val_acc
        no_improvement_epochs = 0

    if no_improvement_epochs >= PATIENCE:
        print("Early stopping")
        break

print(f"Best Accuracy: {max_acc:.4f}")

Epochs = range(epoch + 1)

plt.figure(figsize=(10, 6))
plt.plot(Epochs, train_losses, label='Training Loss')
plt.plot(Epochs, val_losses, label='Validation Loss')

plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig(f'accuracy_plot_{NAMING}.png')
