#!/usr/bin/env python3
"""지브리 캐릭터/인물 이미지 크롤링 - 사람 대상 스타일 변환용."""

import shutil
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = SERVER_ROOT / "styles" / "ghibli_people"

SEARCH_QUERIES = [
    ("studio ghibli character portrait", 10),
    ("spirited away chihiro art", 10),
    ("howl moving castle character art", 10),
    ("ghibli girl face illustration", 10),
    ("totoro character art", 5),
]

TARGET_COUNT = 30


def main():
    print(f"지브리 인물 이미지 크롤링 → {OUTPUT_DIR}")
    print("(bing-image-downloader 사용)")
    print("-" * 50)

    try:
        from bing_image_downloader import downloader
    except ImportError:
        print("설치 필요: pip install bing-image-downloader")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUTPUT_DIR.parent

    for query, limit in SEARCH_QUERIES:
        print(f"\n다운로드: {query} (최대 {limit}개)")
        try:
            downloader.download(
                query,
                limit=limit,
                output_dir=str(base),
                adult_filter_off=False,
                force_replace=False,
            )
        except Exception as e:
            print(f"  실패: {e}")
            continue

    # bing은 output_dir/query폴더명/ 에 저장 (공백 포함) -> ghibli_people로 모음
    for d in base.iterdir():
        if d.is_dir() and d.name not in ("ghibli_people", "ghibli"):
            for f in d.iterdir():
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                    dst = OUTPUT_DIR / f.name
                    n = 0
                    while dst.exists():
                        n += 1
                        dst = OUTPUT_DIR / f"{f.stem}_{n}{f.suffix}"
                    shutil.move(str(f), str(dst))
                    print(f"  이동: {dst.name}")
            shutil.rmtree(d, ignore_errors=True)

    count = len([
        f for f in OUTPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ])
    print(f"\n완료: {count}개 이미지 → {OUTPUT_DIR}")
    if count > 0:
        print("\n인코딩하려면:")
        print("  python scripts/encode_styles.py")


if __name__ == "__main__":
    main()
