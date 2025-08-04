# train_on_shard.py

import torch, torch.nn as nn, pickle, json, os
from torch.utils.data import TensorDataset, DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.makedirs('output', exist_ok=True)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, 128), nn.ReLU(),
            nn.Linear(128, 100)  # CIFAR-100 = 100 classes
        )
    def forward(self, x): return self.fc(x)

# === Load data ===
with open('input/data.pkl', 'rb') as f:
    data, labels = pickle.load(f)

data = torch.tensor(data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.
labels = torch.tensor(labels)
loader = DataLoader(TensorDataset(data, labels), batch_size=32)

# === Train ===
model = SimpleNet()
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters())

for epoch in range(5):  # plus long que 1
    for X, y in loader:
        opt.zero_grad()
        loss_fn(model(X), y).backward()
        opt.step()

# === Save model and accuracy ===
torch.save(model.state_dict(), 'output/model.pt')
acc = (model(data).argmax(1) == labels).float().mean().item()
json.dump({'accuracy': acc}, open('output/metrics.json', 'w'))
