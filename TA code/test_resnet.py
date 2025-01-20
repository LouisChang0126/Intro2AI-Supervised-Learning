import torch
import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image

# 定義超參數
BATCH_SIZE = 32

# 定義與訓練相同的資料轉換
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 自定義測試資料集類別
class TestDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, os.path.basename(image_path)

class SimpleResNet3(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleResNet3, self).__init__()
        # 定義卷積層，這裡使用簡單的 3 層架構
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
        # 定義全連接層
        self.fc = nn.Linear(256 * 28 * 28, num_classes)

        # 初始化其他層
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(-1, 256 * 28 * 28)  # 展平為全連接層
        x = self.fc(x)
        return x
    
# 加載模型
model = SimpleResNet3(num_classes=5)
model.load_state_dict(torch.load('model.pth'))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 獲取測試資料
test_dir = 'data/test/'
test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]

# 建立資料集和DataLoader
test_dataset = TestDataset(test_images, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 預測並保存結果
results = []

with torch.no_grad():
    for images, image_names in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        
        for image_name, prediction in zip(image_names, predictions.cpu().numpy()):
            results.append({'id': image_name, 'prediction': prediction})

# 保存到CSV
output_file = 'predictions.csv'
df = pd.DataFrame(results)
df.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")
