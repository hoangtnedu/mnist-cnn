import ssl, gzip, struct, hashlib, time
from pathlib import Path
import urllib.request, urllib.error
import certifi

# Thư mục lưu dữ liệu
DATA_DIR = Path("data/mnist/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Link mirror
MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]

FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afc8b62a52d9c1d73c3e9c",
}

# SSL + User-Agent
context = ssl.create_default_context(cafile=certifi.where())
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
opener.addheaders = [("User-Agent", "Mozilla/5.0 (MNIST-downloader)")]
urllib.request.install_opener(opener)

def md5sum(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""): 
            h.update(chunk)
    return h.hexdigest()

def fetch(fname: str) -> Path:
    out = DATA_DIR / fname
    if out.exists() and md5sum(out) == FILES[fname]:
        print(f"✔ Đã có: {fname}"); return out
    for base in MIRRORS:
        try:
            url = base + fname
            print(f"↓ Tải: {url}")
            urllib.request.urlretrieve(url, out)
            if md5sum(out) != FILES[fname]:
                raise RuntimeError("MD5 mismatch")
            print(f"✔ Thành công: {fname}")
            return out
        except Exception as e:
            print(f"✗ Lỗi: {e} → thử mirror khác...")
    raise RuntimeError("Không tải được " + fname)

def gunzip(gz: Path) -> Path:
    raw = DATA_DIR / gz.name.replace(".gz", "")
    if raw.exists(): return raw
    with gzip.open(gz, "rb") as fin, raw.open("wb") as fout: 
        fout.write(fin.read())
    print(f"⇪ Giải nén: {gz.name} → {raw.name}")
    return raw

def parse_images(p: Path):
    with p.open("rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        return num, rows, cols

def parse_labels(p: Path):
    with p.open("rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        return num

if __name__ == "__main__":
    gz_files = [fetch(fn) for fn in FILES]
    for g in gz_files: gunzip(g)

    ntr, r, c   = parse_images(DATA_DIR / "train-images-idx3-ubyte")
    nltr        = parse_labels(DATA_DIR / "train-labels-idx1-ubyte")
    nte, r2, c2 = parse_images(DATA_DIR / "t10k-images-idx3-ubyte")
    nlte        = parse_labels(DATA_DIR / "t10k-labels-idx1-ubyte")

    print("\n=== TÓM TẮT ===")
    print(f"Train images : {ntr} mẫu, {r}x{c}")
    print(f"Train labels : {nltr} mẫu")
    print(f"Test images  : {nte} mẫu, {r2}x{c2}")
    print(f"Test labels  : {nlte} mẫu")
