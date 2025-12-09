import torch, torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import os
import random
from matplotlib import pyplot as plt
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

pikachu_dir = "pikachu_images_resized"
other_dir = "other_images_resized"

target_ratio = 9
batch_size = 64
learning_rate = 0.0002
num_epochs = 20

IMG_SIZE = 128


class SpriteDataset(Dataset):
    def __init__(self, pikachu_dir, other_dir, transform=None):
        self.transform = transform

        pikachu_files = [
            os.path.join(pikachu_dir, f)
            for f in os.listdir(pikachu_dir)
            if f.lower().endswith(".png")
        ]

        other_files = [
            os.path.join(other_dir, f)
            for f in os.listdir(other_dir)
            if f.lower().endswith(".png")
        ]

        needed = len(pikachu_files) * target_ratio
        random.shuffle(other_files)
        other_files = other_files[:needed]

        print(f"Using {len(pikachu_files)} pikachu images")
        print(f"Using {len(other_files)} non-pikachu images")

        self.samples = []

        for f in pikachu_files:
            self.samples.append((f, 1))
        for f in other_files:
            self.samples.append((f, 0))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGBA")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor([label], dtype=torch.float32)


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)

dataset = SpriteDataset(pikachu_dir, other_dir, transform)

train_size = int(0.65 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class PikachuNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 128 * 4, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


model = PikachuNet().to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([283 / 2547]))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    total_loss = 0
    model.train()

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = (torch.sigmoid(outputs) >= 0.5).float()

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"| Loss: {total_loss:.4f} "
        f"| Accuracy: {accuracy:.2f}%"
    )

torch.save(model.state_dict(), "pikachu_classifier.pth")
print("Saved model to pikachu_classifier.pth")

model.eval()

sample_paths = random.sample(dataset.samples, 8)
imgs = []
titles = []

for path, label in sample_paths:
    img = Image.open(path).convert("RGBA")
    inp = transform(img).unsqueeze(0).to(device)
    logit = model(inp).item()
    prob = torch.sigmoid(torch.tensor(logit)).item()

    imgs.append(img)
    titles.append(f"True:{label}  Pred:{prob:.2f}")

plt.figure(figsize=(10, 5))
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(imgs[i])
    plt.title(titles[i])
    plt.axis("off")
plt.tight_layout()
plt.show()

wrong_pikachu_dir = "wrong_pikachu"
wrong_other_dir = "wrong_other"
os.makedirs(wrong_pikachu_dir, exist_ok=True)
os.makedirs(wrong_other_dir, exist_ok=True)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for i in range(len(labels)):
            if preds[i] != labels[i]:
                idx = i
                img_array = images[idx].cpu().numpy()
                img_array = (img_array * 0.5 + 0.5) * 255
                img_array = np.transpose(img_array, (1, 2, 0))
                img_array = img_array.astype(np.uint8)
                img_pil = Image.fromarray(img_array, "RGBA")

                if labels[i] == 1:
                    img_pil.save(
                        os.path.join(wrong_pikachu_dir, f"wrong_{total}_{i}.png")
                    )
                else:
                    img_pil.save(
                        os.path.join(wrong_other_dir, f"wrong_{total}_{i}.png")
                    )

accuracy = 100 * correct / total
print(f"\nModel Accuracy: {accuracy:.2f}%")
