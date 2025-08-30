# eval_mnist_cnn.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np

from train_mnist_cnn import MNIST_CNN   # import lại kiến trúc model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "./data"

def main():
    # load model
    model = MNIST_CNN().to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/mnist_cnn_best.pth", map_location=DEVICE))
    model.eval()

    # data
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=test_tf)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    # predict
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            preds = logits.argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # report
    print("\nClassification report:")
    print(classification_report(np.array(y_true), np.array(y_pred), digits=4))

if __name__ == "__main__":
    main()
