import torch
import torch.nn as nn
from src.model import create_model
from src.dataset import load_dataset
from src.profiler import ModelProfiler

model = create_model("mnist", 10)
# run estimation
profiler = ModelProfiler(model)
print("Before estimation, finite:", all(torch.isfinite(p).all() for p in model.parameters()))
needs = profiler.estimate_needs("mnist", 32)
print("After estimation, finite:", all(torch.isfinite(p).all() for p in model.parameters()))

# load dataset
train_loader, test_loader = load_dataset("mnist", "./data", 0, 3, "iid", 32)
x, y = next(iter(train_loader))
out = model(x)
loss = nn.CrossEntropyLoss()(out, y)
print("Loss:", loss.item())
print("Loss finite:", torch.isfinite(loss).item())
