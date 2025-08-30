# predict_mnist_cnn.py
import argparse
from pathlib import Path
import json
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

# ========== Model (khớp đúng kiến trúc lúc train) ==========
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
            nn.Flatten(),              # 128*7*7 = 6272
            nn.Linear(128*7*7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ========== Utils ==========
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN_STD = ((0.1307,), (0.3081,))

def load_model(ckpt_path: str) -> nn.Module:
    model = MNIST_CNN().to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    # nếu state_dict bị prefix "module."
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model

def maybe_invert(img_pil: Image.Image, force_invert: bool = False) -> Image.Image:
    """
    MNIST chuẩn: số TRẮNG trên nền ĐEN.
    - Nếu ảnh của bạn là số đen trên nền trắng => nên invert.
    - Mặc định dùng heuristic: nếu ảnh có độ sáng trung bình cao -> invert.
    """
    if force_invert:
        return Image.fromarray(255 - np.array(img_pil))
    arr = np.array(img_pil, dtype=np.float32)
    if arr.mean() > 127:  # nền sáng -> đảo
        arr = 255 - arr
    return Image.fromarray(arr.astype(np.uint8))

def preprocess_image(path: Path, force_invert: bool = False) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((28, 28), Image.BILINEAR)
    img = maybe_invert(img, force_invert=force_invert)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*MEAN_STD),
    ])
    x = tf(img).unsqueeze(0)  # [1,1,28,28]
    return x.to(DEVICE)

@torch.no_grad()
def predict_one(model: nn.Module, image_path: Path, force_invert: bool = False):
    x = preprocess_image(image_path, force_invert)
    logits = model(x)
    prob = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred = int(prob.argmax())
    top5_idx = prob.argsort()[-5:][::-1]
    top5 = {int(i): float(prob[i]) for i in top5_idx}
    return pred, float(prob[pred]), top5

def list_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in exts and p.is_file():
            yield p

# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="Predict MNIST digit from image(s)")
    parser.add_argument("--ckpt", type=str, default="checkpoints/mnist_cnn_best.pth",
                        help="đường dẫn checkpoint .pth")
    parser.add_argument("--image", type=str, default=None,
                        help="đường dẫn 1 ảnh PNG/JPG để dự đoán")
    parser.add_argument("--folder", type=str, default=None,
                        help="dự đoán toàn bộ ảnh trong thư mục")
    parser.add_argument("--invert", action="store_true",
                        help="bắt buộc đảo màu (khi ảnh là số đen trên nền trắng)")
    parser.add_argument("--save-json", type=str, default="",
                        help="nếu set, lưu kết quả ra file JSON này")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    model = load_model(args.ckpt)

    results = []

    if args.image:
        p = Path(args.image)
        pred, conf, top5 = predict_one(model, p, force_invert=args.invert)
        print(f"[{p.name}] -> predict: {pred} (confidence: {conf:.4f})")
        print(f"Top-5 probs: {top5}")
        results.append({"image": str(p), "pred": pred, "confidence": conf, "top5": top5})

    elif args.folder:
        folder = Path(args.folder)
        assert folder.is_dir(), f"Không tìm thấy thư mục: {folder}"
        for p in list_images(folder):
            pred, conf, top5 = predict_one(model, p, force_invert=args.invert)
            print(f"[{p.name}] -> {pred} | conf: {conf:.4f} | top5: {top5}")
            results.append({"image": str(p), "pred": pred, "confidence": conf, "top5": top5})
        print(f"Tổng ảnh dự đoán: {len(results)}")

    else:
        print("⚠️ Bạn cần truyền --image <path> hoặc --folder <dir> để dự đoán.")
        return

    if args.save_json:
        out = Path(args.save_json)
        out.write_text(json.dumps(results, indent=2))
        print(f"✓ Đã lưu kết quả vào: {out.resolve()}")

if __name__ == "__main__":
    main()
