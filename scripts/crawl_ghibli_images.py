#!/usr/bin/env python3
"""지브리 스타일 이미지 크롤링 - DuckDuckGo 이미지 검색에서 다운로드."""

import os
import re
import time
from pathlib import Path

import requests
from duckduckgo_search import DDGS

SERVER_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = SERVER_ROOT / "styles" / "ghibli"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 검색어 (영어/한국어)
SEARCH_QUERIES = [
    "studio ghibli art landscape",
    "studio ghibli painting style",
    "ghibli anime scenery",
    "miyazaki art style",
    "spirited away artwork",
]


def sanitize_filename(url: str, index: int) -> str:
    """URL에서 안전한 파일명 추출."""
    ext = ".jpg"
    if ".png" in url.lower():
        ext = ".png"
    elif ".webp" in url.lower():
        ext = ".webp"
    return f"ghibli_{index:03d}{ext}"


def download_image(url: str, filepath: Path, timeout: int = 10) -> bool:
    """이미지 다운로드. 성공 시 True."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        if len(resp.content) < 5000:
            return False
        filepath.write_bytes(resp.content)
        return True
    except Exception:
        return False


def main():
    print(f"지브리 스타일 이미지 크롤링 → {OUTPUT_DIR}")
    print("-" * 50)

    collected_urls = set()
    for query in SEARCH_QUERIES:
        print(f"\n검색: {query}")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(query, max_results=30))
        except Exception as e:
            print(f"  검색 실패: {e}")
            continue

        for r in results:
            url = r.get("image") or r.get("href") or ""
            if url and url not in collected_urls:
                collected_urls.add(url)

        time.sleep(1)

    print(f"\n총 {len(collected_urls)}개 URL 수집. 다운로드 시작...")

    success = 0
    for i, url in enumerate(collected_urls):
        if success >= 20:
            break
        filepath = OUTPUT_DIR / sanitize_filename(url, i)
        if filepath.exists():
            success += 1
            print(f"  [skip] {filepath.name} (이미 있음)")
            continue
        if download_image(url, filepath):
            success += 1
            print(f"  [ok] {filepath.name}")
        else:
            print(f"  [fail] {url[:60]}...")
        time.sleep(0.5)

    print(f"\n완료: {success}개 이미지 저장 → {OUTPUT_DIR}")
    if success > 0:
        first = list(OUTPUT_DIR.glob("ghibli_*"))[0]
        print(f"\n기본 스타일로 설정하려면:")
        print(f"  cp {first} styles/default.jpg")
    print(f"\n인코딩하려면: python scripts/encode_styles.py")


if __name__ == "__main__":
    main()
