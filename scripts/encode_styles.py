#!/usr/bin/env python3
"""여러 스타일 이미지를 인코딩하여 AdaIN용 mean/std로 저장."""

import sys
from pathlib import Path

# scripts/에서 실행 시 server 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from model.function import calc_mean_std
from model import net

SERVER_ROOT = Path(__file__).resolve().parent.parent
BASE = SERVER_ROOT / "styles"
_people = BASE / "ghibli_people"
ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_has_people = (
    _people.exists()
    and any(p.suffix.lower() in ALLOWED for p in _people.iterdir())
)
STYLES_DIR = _people if _has_people else BASE / "ghibli"
OUTPUT_FILE = SERVER_ROOT / "styles" / "ghibli_style.pt"
MODELS_DIR = SERVER_ROOT / "models"
STYLE_SIZE = 384

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    if not STYLES_DIR.exists():
        print(f"폴더 없음: {STYLES_DIR}")
        print("python scripts/crawl_ghibli_images.py 를 먼저 실행하세요.")
        return

    image_paths = [
        p
        for p in STYLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS
    ]
    if not image_paths:
        print(f"이미지 없음: {STYLES_DIR}")
        return

    print(f"로드: {len(image_paths)}개 이미지")

    # VGG 로드
    vgg_path = MODELS_DIR / "vgg_normalised.pth"
    if not vgg_path.exists():
        vgg_path = MODELS_DIR / "vgg_normalized.pth"
    if not vgg_path.exists():
        print("VGG 모델 없음. python scripts/download_models.py 먼저 실행.")
        return

    vgg_full = net.vgg
    vgg_full.load_state_dict(torch.load(vgg_path, map_location="cpu"))
    vgg = nn.Sequential(*list(vgg_full.children())[:31])
    vgg.eval()

    transform = transforms.Compose([
        transforms.Resize(STYLE_SIZE),
        transforms.ToTensor(),
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vgg.to(device)

    means = []
    stds = []

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = vgg(x)
        m, s = calc_mean_std(feat)
        means.append(m)
        stds.append(s)

    if not means:
        print("유효한 이미지 없음.")
        return

    # 전체 평균으로 블렌딩
    style_mean = torch.stack(means).mean(dim=0)
    style_std = torch.stack(stds).mean(dim=0)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"style_mean": style_mean.cpu(), "style_std": style_std.cpu()},
        OUTPUT_FILE,
    )
    print(f"저장: {OUTPUT_FILE}")
    print("서버 재시작 시 이 파일을 자동으로 사용합니다.")


if __name__ == "__main__":
    main()
