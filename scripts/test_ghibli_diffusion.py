#!/usr/bin/env python3
"""
Ghibli Diffusion (nitrosocke/Ghibli-Diffusion) 테스트 스크립트.
- img2img: example.jpg를 지브리 스타일로 변환
- text-to-image (--text-only): 프롬프트만으로 이미지 생성
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

SERVER_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = SERVER_ROOT / "example.jpg"
DEFAULT_OUTPUT = SERVER_ROOT / "example_ghibli_diffusion.jpg"
DEFAULT_TEXT_OUTPUT = SERVER_ROOT / "example_ghibli_text2img.jpg"


def run_text2img(args):
    """프롬프트만 사용한 텍스트→이미지 생성."""
    print("Ghibli Diffusion (text-to-image) 로딩 중...")
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError:
        print("diffusers 패키지가 필요합니다: pip install diffusers transformers accelerate")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        "nitrosocke/Ghibli-Diffusion",
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)

    print("이미지 생성 중... (프롬프트만 사용)")
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    )
    out_image = result.images[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_image.save(args.output, quality=95)
    print(f"저장: {args.output}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Ghibli Diffusion 테스트 (img2img / text-to-image)")
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="입력 이미지 없이 프롬프트만으로 이미지 생성 (text-to-image)",
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"입력 이미지 (기본: {DEFAULT_INPUT}, --text-only 시 무시)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="출력 이미지 경로 (미지정 시 img2img는 example_ghibli_diffusion.jpg, text-only는 example_ghibli_text2img.jpg)",
    )
    parser.add_argument(
        "--prompt",
        default="ghibli style beautiful Caribbean beach tropical (sunset)",
        help="생성 프롬프트",
    )
    parser.add_argument(
        "--negative-prompt",
        default="soft blurry",
        help="네거티브 프롬프트",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="img2img 이미지 변환 강도 0.0~1.0 (기본: 0.75)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="추론 스텝 수 (기본: 20)",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.0,
        help="CFG 스케일 (기본: 7.0)",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = DEFAULT_TEXT_OUTPUT if args.text_only else DEFAULT_OUTPUT

    if args.text_only:
        return run_text2img(args)

    if not args.input.exists():
        print(f"오류: 입력 파일 없음: {args.input}")
        return 1

    print("Ghibli Diffusion (img2img) 로딩 중... (처음 실행 시 모델 다운로드, 수 분 소요)")
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
    except ImportError:
        print("diffusers 패키지가 필요합니다: pip install diffusers transformers accelerate")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "nitrosocke/Ghibli-Diffusion",
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe = pipe.to(device)

    print(f"입력 이미지 로딩: {args.input}")
    init_image = Image.open(args.input).convert("RGB")
    w, h = init_image.size
    if max(w, h) > 512:
        ratio = 512 / max(w, h)
        init_image = init_image.resize(
            (int(w * ratio), int(h * ratio)),
            Image.Resampling.LANCZOS,
        )

    print("변환 중... (20~60초 소요)")
    result = pipe(
        prompt=args.prompt,
        image=init_image,
        negative_prompt=args.negative_prompt,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    )
    out_image = result.images[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_image.save(args.output, quality=95)
    print(f"저장: {args.output}")
    return 0


if __name__ == "__main__":
    exit(main() or 0)
