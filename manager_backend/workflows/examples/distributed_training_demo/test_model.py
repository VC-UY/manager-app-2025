import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self): 
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x): return self.fc(x)

model = SimpleNet()
model.load_state_dict(torch.load("outputs/merged_model.pt"))
model.eval()

test = CIFAR10('./data', train=False, transform=transforms.ToTensor())
loader = DataLoader(test, batch_size=32, num_workers=2)

acc = 0
for X, y in loader:
    acc += (model(X).argmax(1) == y).float().sum().item()
print(f"Test accuracy: {acc / len(test)}")
