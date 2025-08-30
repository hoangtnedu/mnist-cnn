import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Định nghĩa lại kiến trúc CNN giống khi train
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

# Load model đã train
model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load("mnist_cnn_best.pt", map_location=DEVICE))
model.eval()

# Load tập test MNIST
transform = transforms.ToTensor()
test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)

# Lấy 1 ảnh bất kỳ từ test set (ví dụ index 0)
idx = 0
img, label = test_set[idx]

# Hiển thị ảnh
plt.imshow(img.squeeze(), cmap="gray")
plt.title(f"Ground Truth: {label}")
plt.show()

# Chuẩn bị dữ liệu để dự đoán
x = img.unsqueeze(0).to(DEVICE)  # thêm batch dimension
with torch.no_grad():
    logits = model(x)
    pred = torch.argmax(logits, dim=1).item()

print(f"✅ Mô hình dự đoán: {pred}, Ground truth: {label}")
