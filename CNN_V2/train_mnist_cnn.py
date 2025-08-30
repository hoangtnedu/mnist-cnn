# train_mnist_cnn.py
import os
import random
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

# ============ Reproducibility ============
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False  # allow some cudnn speedups
torch.backends.cudnn.benchmark = True

# ============ Hyperparams ============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "./data"
BATCH_SIZE = 128
EPOCHS = 25
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
STEP_SIZE = 10      # step every 10 epochs
GAMMA = 0.1         # lr decay factor
VAL_SPLIT = 5_000   # 55k train / 5k val

# ============ Model ============
class MNIST_CNN(nn.Module):
    """
    LeNet-style CNN with modern touches:
    - 3x3 convs + BatchNorm + ReLU
    - 2x MaxPool
    - Dropout in classifier
    Achieves ~99.1–99.3% with proper training schedule.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),  # 28x28 -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1, bias=False), # 28x28 -> 28x28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 28 -> 14

            nn.Conv2d(64, 128, 3, padding=1, bias=False), # 14x14 -> 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1, bias=False),# 14x14 -> 14x14
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                 # 128*7*7=6272
            nn.Linear(128*7*7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============ Data ============
def get_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Augmentation (nhẹ, phù hợp MNIST)
    train_tf = transforms.Compose([
        transforms.RandomAffine(
            degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=train_tf)
    test_set   = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=eval_tf)

    # Split 60k -> 55k train / 5k val (dùng cùng transform cho train; val không cần augment)
    g = torch.Generator().manual_seed(SEED)
    train_set, _val_set = random_split(full_train, [60_000-VAL_SPLIT, VAL_SPLIT], generator=g)

    # replace val transform to eval_tf (no aug)
    _val_set.dataset = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=eval_tf)
    # ^ reusing indices is OK because random_split keeps same indices list

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(_val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader

# ============ Train / Eval ============
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, leave=False, desc="Train")
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        pbar.set_postfix(loss=running_loss/total, acc=correct/total)
    return running_loss/total, correct/total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return running_loss/total, correct/total

@torch.no_grad()
def predict_all(model, loader):
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        out = model(x).argmax(dim=1).cpu().numpy()
        preds.append(out)
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(preds)

def main():
    print(f"Device: {DEVICE}")
    train_loader, val_loader, test_loader = get_dataloaders()
    model = MNIST_CNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train_loss: {tr_loss:.4f} acc: {tr_acc:.4f} | "
              f"val_loss: {val_loss:.4f} acc: {val_acc:.4f} | "
              f"lr: {scheduler.get_last_lr()[0]:.5f}")

        # save best on val
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Load best and test
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\n[Test] loss: {test_loss:.4f} | acc: {test_acc:.4f}")

    # Classification report
    y_true, y_pred = predict_all(model, test_loader)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))

    # Optional: save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/mnist_cnn_best.pth")
    print("Saved to checkpoints/mnist_cnn_best.pth")

if __name__ == "__main__":
    main()
