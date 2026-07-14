import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
from torchvision.datasets.folder import default_loader
from PIL import Image

# -------------------------------
# 1. Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 2. Dataset Class
# -------------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        self.loader = default_loader

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = self.loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image

# -------------------------------
# 3. Transforms & DataLoader
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

data_folder = '/Users/shraawanilattoo/Library/CloudStorage/OneDrive-NorthCarolinaStateUniversity/NCSU/FDS 510/Project Files/Image_data_working/Mouse wound photos-updated/Cropped images adjusted brightness'  # <-- Replace with your path
dataset = ImageFolderDataset(data_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# -------------------------------
# 4. Autoencoder
# -------------------------------
class ConvAutoencoderRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 64 → 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32 → 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16 → 8
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 8 → 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),    # 16 → 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),     # 32 → 64
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# -------------------------------
# 5. Train Autoencoder
# -------------------------------
model = ConvAutoencoderRGB().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 15

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs in dataloader:
        imgs = imgs.to(device)
        output = model(imgs)
        loss = criterion(output, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

# -------------------------------
# 6. Feature Extraction
# -------------------------------
model.eval()
encoded_outputs = []
with torch.no_grad():
    for imgs in DataLoader(dataset, batch_size=64):
        imgs = imgs.to(device)
        encoded = model.encoder(imgs)
        encoded_flat = encoded.view(encoded.size(0), -1)
        encoded_outputs.append(encoded_flat.cpu().numpy())

features = np.vstack(encoded_outputs)

# -------------------------------
# 7. KMeans Clustering
# -------------------------------
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(features)

# -------------------------------
# 8. Extract Day from Filenames
# -------------------------------
image_days = []
for fname in dataset.image_files:
    match = re.search(r'Day\s*(\d+)', fname, re.IGNORECASE)
    if match:
        day = int(match.group(1))
    else:
        day = -1
 # Unknown day
    image_days.append(day)

# -------------------------------
# 9. Build DataFrame for Plotting
# -------------------------------
df = pd.DataFrame({
    'filename': dataset.image_files,
    'cluster': cluster_labels,
    'day': image_days
})

# -------------------------------
# 10. Plot Histograms per Cluster
# -------------------------------
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, n_clusters, figsize=(18, 4), sharey=True)

for i in range(n_clusters):
    sns.histplot(
        df[df['cluster'] == i]['day'],
        bins=16, kde=True, color='blue', ax=axes[i]
    )
    axes[i].set_title(f'Cluster {i}')
    axes[i].set_xlabel('Day')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
