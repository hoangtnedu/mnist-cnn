import torch
from torchvision import datasets, transforms

# Nơi lưu dữ liệu
root = "./data"

# Tiền xử lý: chuyển thành tensor
transform = transforms.ToTensor()

# Tải tập train và test
train_set = datasets.MNIST(root=root, train=True, download=True, transform=transform)
test_set  = datasets.MNIST(root=root, train=False, download=True, transform=transform)

print("Train size:", len(train_set))   # 60000
print("Test size:", len(test_set))     # 10000
