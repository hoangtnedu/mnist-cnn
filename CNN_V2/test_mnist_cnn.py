# test_mnist_cnn.py
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

# --------- Model (khớp file train_mnist_cnn.py) ----------
class MNIST_CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*7*7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# --------- Utils ----------
@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        x = x.to(device)
        pred = model(x).argmax(dim=1).cpu().numpy()
        preds.append(pred)
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(preds)

def main():
    parser = argparse.ArgumentParser(description="Evaluate MNIST CNN")
    parser.add_argument("--weights", type=str, default="checkpoints/mnist_cnn_best.pth",
                        help="đường dẫn file .pth/.pt của mô hình")
    parser.add_argument("--data-root", type=str, default="./data", help="thư mục MNIST")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-cm", type=str, default="", help="nếu muốn lưu confusion_matrix (.npy)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Dataloader cho tập test (đúng chuẩn normalize của MNIST)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = datasets.MNIST(args.data_root, train=False, download=True, transform=tf)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Model + load weights
    model = MNIST_CNN().to(device)
    state = torch.load(args.weights, map_location=device)
    # hỗ trợ state_dict hoặc nguyên model.state_dict() đã lưu
    if isinstance(state, dict) and "features.0.weight" not in state.keys() and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    # Loss/accuracy nhanh
    criterion = nn.CrossEntropyLoss()
    loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss += criterion(logits, y).item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    print(f"\n[Test] loss: {loss/total:.4f} | acc: {correct/total:.4f}")

    # Classification report (giống ảnh bạn gửi)
    y_true, y_pred = predict_all(model, test_loader, device)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4))

    # Optional: lưu confusion matrix
    if args.save-cm:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
        out = Path(args.save_cm)
        np.save(out.with_suffix(".npy"), cm)
        print(f"Saved confusion_matrix to {out.with_suffix('.npy')}")

if __name__ == "__main__":
    main()
