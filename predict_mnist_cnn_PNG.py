import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
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

# ========= (1) Dự đoán ảnh trong tập test =========
transform = transforms.ToTensor()
test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)

idx = 0  # thử ảnh đầu tiên trong test set
img, label = test_set[idx]

plt.imshow(img.squeeze(), cmap="gray")
plt.title(f"Ground Truth (MNIST test): {label}")
plt.show()

x = img.unsqueeze(0).to(DEVICE)  # thêm batch dimension
with torch.no_grad():
    logits = model(x)
    pred = torch.argmax(logits, dim=1).item()
print(f"✅ Dự đoán MNIST test: {pred}, Ground truth: {label}")


# ========= (2) Dự đoán ảnh PNG/JPG bên ngoài =========
def predict_external_image(img_path):
    # Đọc ảnh
    img = Image.open(img_path).convert("L")  # chuyển grayscale
    img = img.resize((28, 28))               # resize 28x28
    plt.imshow(img, cmap="gray")
    plt.title("Ảnh bên ngoài")
    plt.show()

    # Chuyển thành tensor
    transform_ext = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # chuẩn hóa [-1,1]
    ])
    x = transform_ext(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()
    return pred

# Ví dụ: thay "digit3.png" bằng file ảnh bạn vẽ
try:
    external_path = "digit3.png"
    prediction = predict_external_image(external_path)
    print(f"✅ Dự đoán ảnh ngoài: {prediction}")
except FileNotFoundError:
    print("⚠️ Không tìm thấy file ảnh ngoài (digit3.png). Hãy vẽ số và lưu vào thư mục này.")
