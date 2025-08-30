import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset & Loader
transform = transforms.ToTensor()
test_set   = datasets.MNIST("./data", train=False, download=True, transform=transform)
test_loader  = DataLoader(test_set, batch_size=64, shuffle=False)

# Định nghĩa lại kiến trúc CNN giống lúc train
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop  = nn.Dropout(0.25)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

# Load model
model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load("mnist_cnn_best.pt", map_location=DEVICE))
model.eval()

# Đánh giá
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in test_loader:
        logits = model(x.to(DEVICE))
        y_true.extend(y.tolist())
        y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())

print("Classification report:\n", classification_report(y_true, y_pred, digits=4))
