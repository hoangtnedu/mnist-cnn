# train_mnist_cnn.py
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import os
import random

# =========================
# Cấu hình & Siêu tham số
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
DATA_ROOT = "./data"
NUM_WORKERS = 2         # Nếu lỗi trên Windows -> đổi thành 0
SEED = 42
CKPT_PATH = "mnist_cnn_best.pt"

# =========================
# Tiện ích: cố định seed
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Tùy chọn: tăng tái lập (có thể chậm hơn)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# Mô hình CNN đơn giản
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 28x28 -> 28x28
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # 14x14 -> 14x14 (sau pool)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)   # chia 2 kích thước
        self.drop  = nn.Dropout(0.25)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (N,32,14,14)
        x = self.pool(F.relu(self.conv2(x)))  # (N,64,7,7)
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# =========================
# 1 Epoch train/val/test
# =========================
def run_epoch(model, optimizer, criterion, loader, device, train: bool, desc: str):
    model.train(train)
    total_loss, y_true, y_pred = 0.0, [], []

    # Bọc tqdm ở ĐÂY để 'loader' vẫn là DataLoader thật
    for x, y in tqdm(loader, desc=desc):
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        y_true.extend(y.detach().cpu().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc

# =========================
# Chương trình chính
# =========================
def main():
    set_seed(SEED)

    # 1) Dataloader
    transform = transforms.ToTensor()  # [0,1], shape (1,28,28)
    train_full = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform)
    test_set   = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=transform)

    # Tách val: 55k/5k
    g = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(train_full, [55_000, 5_000], generator=g)

    pin_mem = (DEVICE == "cuda")
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin_mem)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin_mem)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin_mem)

    # 2) Model, loss, optimizer
    model = SimpleCNN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 3) Train/Val
    best_val = -1.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, optimizer, criterion, train_loader, DEVICE, True,  f"Epoch {epoch}/{EPOCHS} [train]")
        va_loss, va_acc = run_epoch(model, optimizer, criterion, val_loader,   DEVICE, False, f"Epoch {epoch}/{EPOCHS} [val]  ")
        print(f"Epoch {epoch}: train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), CKPT_PATH)
            print(f"✔ Saved checkpoint: {CKPT_PATH}")

    # 4) Đánh giá trên test
    if os.path.exists(CKPT_PATH):
        model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    te_loss, te_acc = run_epoch(model, optimizer, criterion, test_loader, DEVICE, False, "Test")
    print(f"\nTest loss {te_loss:.4f} | Test acc {te_acc:.4f}")

    # 5) In báo cáo chi tiết
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(DEVICE))
            y_true.extend(y.tolist())
            y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())
    print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    # BẮT BUỘC trên Windows khi dùng DataLoader với num_workers>0
    mp.freeze_support()
    main()
