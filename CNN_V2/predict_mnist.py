# predict_mnist.py
import argparse, os
from typing import Tuple
import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# ===== Model (khớp với code train của bạn) =====
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
            nn.MaxPool2d(2),  # 28 -> 14

            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # 128*7*7=6272
            nn.Linear(128*7*7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ===== Tiền xử lý ảnh =====
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)

def make_square_and_resize(im: Image.Image, size: int = 28) -> Image.Image:
    # pad cho vuông rồi resize giữ tỉ lệ
    w, h = im.size
    if w == h:
        return im.resize((size, size), Image.BILINEAR)
    pad_left = pad_top = 0
    if w > h:
        pad = (w - h) // 2
        im = ImageOps.expand(im, border=(0, pad, 0, w - h - pad), fill=0)
    else:
        pad = (h - w) // 2
        im = ImageOps.expand(im, border=(pad, 0, h - w - pad, 0), fill=0)
    return im.resize((size, size), Image.BILINEAR)

def load_and_preprocess(path: str, auto_invert: bool = True) -> torch.Tensor:
    im = Image.open(path).convert("L")  # 1 kênh
    # auto invert nếu nền trắng (mean pixel cao) để thành chữ trắng–nền đen như MNIST
    if auto_invert:
        mean_val = np.asarray(im).mean()
        # nếu nền sáng (thường > 127), đảo màu
        if mean_val > 127:
            im = ImageOps.invert(im)
    im = make_square_and_resize(im, 28)
    tfm = transforms.Compose([
        transforms.ToTensor(),                  # [0,1]
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    x = tfm(im).unsqueeze(0)  # [1,1,28,28]
    return x

# ===== Dự đoán =====
@torch.no_grad()
def predict(image_path: str, ckpt_path: str, device: str = None, topk: int = 3):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # model
    model = MNIST_CNN().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    x = load_and_preprocess(image_path).to(device)  # [1,1,28,28]
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top_idx = probs.argsort()[::-1][:topk]
    print(f"Ảnh: {image_path}")
    print(f"Dự đoán: {top_idx[0]} (p = {probs[top_idx[0]]:.4f})")
    print("Top-k:")
    for i in top_idx:
        print(f"  {i}: {probs[i]:.4f}")

def main():
    parser = argparse.ArgumentParser(description="MNIST single-image inference")
    parser.add_argument("image", help="Đường dẫn ảnh cần nhận dạng (png/jpg/bmp, ...)")
    parser.add_argument("--ckpt", default="checkpoints/mnist_cnn_best.pth",
                        help="Đường dẫn checkpoint .pth đã train")
    parser.add_argument("--device", default=None, help="'cuda' hoặc 'cpu' (mặc định tự chọn)")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {args.ckpt}")
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {args.image}")

    predict(args.image, args.ckpt, device=args.device, topk=args.topk)

if __name__ == "__main__":
    main()
