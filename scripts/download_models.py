#!/usr/bin/env python3
"""Download AdaIN pretrained models from naoto0804/pytorch-AdaIN."""

import urllib.request
from pathlib import Path

RELEASE_BASE = "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0"
MODELS = ["decoder.pth", "vgg_normalised.pth"]
SERVER_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = SERVER_ROOT / "models"


def main():
    MODELS_DIR.mkdir(exist_ok=True, parents=True)

    for model_name in MODELS:
        model_path = MODELS_DIR / model_name
        if model_path.exists():
            print(f"Already exists: {model_path}")
            continue

        url = f"{RELEASE_BASE}/{model_name}"
        print(f"Downloading {model_name} from {url}...")

        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"Saved: {model_path}")
        except Exception as e:
            print(f"Failed to download {model_name}: {e}")
            print(
                "\nManual download: Visit https://github.com/naoto0804/pytorch-AdaIN/releases/tag/v0.0.0"
            )
            print(f"Place decoder.pth and vgg_normalised.pth in {MODELS_DIR}")
            raise

    print("\nDone. Models ready.")


if __name__ == "__main__":
    main()
