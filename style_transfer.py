"""AdaIN style transfer inference engine."""

import io
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from model.function import (
    adaptive_instance_normalization,
    adaptive_instance_normalization_with_stats,
    coral,
)
from model import net


class StyleTransferEngine:
    """GPU-accelerated AdaIN style transfer engine."""

    def __init__(
        self,
        models_dir: str | Path = "models",
        content_size: int = 384,
        style_size: int = 384,
        alpha: float = 1.0,
        preserve_color: bool = False,
        device: str | None = None,
    ):
        self.models_dir = Path(models_dir)
        self.content_size = content_size
        self.style_size = style_size
        self.alpha = alpha
        self.preserve_color = preserve_color
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._decoder = None
        self._vgg = None
        self._content_transform = None
        self._style_transform = None
        self._default_style: torch.Tensor | None = None
        self._cached_style_stats: tuple[torch.Tensor, torch.Tensor] | None = None

    def load_models(self) -> None:
        """Load VGG and decoder weights."""
        vgg_path = self.models_dir / "vgg_normalised.pth"
        decoder_path = self.models_dir / "decoder.pth"

        # Support American spelling as fallback
        if not vgg_path.exists():
            vgg_path = self.models_dir / "vgg_normalized.pth"

        if not vgg_path.exists():
            raise FileNotFoundError(
                f"VGG model not found. Run scripts/download_models.py first.\n"
                f"Expected at: {vgg_path}"
            )
        if not decoder_path.exists():
            raise FileNotFoundError(
                f"Decoder model not found. Run scripts/download_models.py first.\n"
                f"Expected at: {decoder_path}"
            )

        self._decoder = net.decoder
        # Load full VGG first (pretrained has all layers), then take first 31 (up to relu4_1)
        vgg_full = net.vgg
        vgg_full.load_state_dict(torch.load(vgg_path, map_location="cpu"))
        self._vgg = nn.Sequential(*list(vgg_full.children())[:31])

        self._decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))

        self._vgg.to(self.device)
        self._decoder.to(self.device)
        self._vgg.eval()
        self._decoder.eval()

        self._content_transform = self._build_transform(self.content_size)
        self._style_transform = self._build_transform(self.style_size)

        # Pre-encoded style (ghibli_style.pt 등) 로드 시도
        self._load_cached_style_if_exists()

    def _build_transform(self, size: int, crop: bool = False):
        transform_list = []
        if size > 0:
            transform_list.append(transforms.Resize(size))
        if crop:
            transform_list.append(transforms.CenterCrop(size))
        transform_list.append(transforms.ToTensor())
        return transforms.Compose(transform_list)

    def _load_cached_style_if_exists(self) -> None:
        """ghibli_style.pt 등 미리 인코딩된 스타일 로드."""
        styles_dir = self.models_dir.parent / "styles"
        candidates = ["ghibli_style.pt", "cached_style.pt"]
        for name in candidates:
            path = styles_dir / name
            if path.exists():
                try:
                    data = torch.load(path, map_location="cpu")
                    mean = data["style_mean"].to(self.device)
                    std = data["style_std"].to(self.device)
                    self._cached_style_stats = (mean, std)
                    return
                except Exception:
                    pass

    def set_default_style(self, style_image: Image.Image | bytes) -> None:
        """Preload a default style image for fast inference."""
        if isinstance(style_image, bytes):
            style_image = Image.open(io.BytesIO(style_image)).convert("RGB")
        style_tensor = self._style_transform(style_image).unsqueeze(0)
        self._default_style = style_tensor.to(self.device)

    def stylize(
        self,
        content_image: Image.Image | bytes,
        style_image: Image.Image | bytes | None = None,
        alpha: float | None = None,
    ) -> bytes:
        """
        Apply style transfer to content image.
        Returns JPEG bytes of stylized image.
        """
        if self._vgg is None or self._decoder is None:
            self.load_models()

        alpha = alpha if alpha is not None else self.alpha
        assert 0.0 <= alpha <= 1.0

        # Load content
        if isinstance(content_image, bytes):
            content_image = Image.open(io.BytesIO(content_image)).convert("RGB")
        content = self._content_transform(content_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            content_f = self._vgg(content)

            # 1) 미리 인코딩된 스타일 (ghibli_style.pt) 사용 - VGG 스타일 인코딩 생략
            if self._cached_style_stats is not None:
                style_mean, style_std = self._cached_style_stats
                feat = adaptive_instance_normalization_with_stats(
                    content_f, style_mean, style_std
                )
            else:
                # 2) 스타일 이미지 또는 default.jpg 사용 - 매번 VGG 인코딩
                if style_image is not None:
                    if isinstance(style_image, bytes):
                        style_image = Image.open(
                            io.BytesIO(style_image)
                        ).convert("RGB")
                    style = self._style_transform(style_image).unsqueeze(
                        0
                    ).to(self.device)
                    if self.preserve_color:
                        style = coral(style[0], content[0]).unsqueeze(0)
                elif self._default_style is not None:
                    style = self._default_style
                    if self.preserve_color:
                        style = coral(style[0], content[0]).unsqueeze(0)
                else:
                    raise ValueError(
                        "No style. Run scripts/encode_styles.py or add styles/default.jpg"
                    )
                style_f = self._vgg(style)
                feat = adaptive_instance_normalization(content_f, style_f)

            feat = feat * alpha + content_f * (1 - alpha)
            output = self._decoder(feat)

        # Clamp and convert to PIL
        output = output.cpu().clamp(0, 1)
        output_image = transforms.ToPILImage()(output[0])

        # Return as JPEG bytes
        buffer = io.BytesIO()
        output_image.save(buffer, format="JPEG", quality=95)
        return buffer.getvalue()
