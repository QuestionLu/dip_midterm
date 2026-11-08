"""功能 1：影像載入與基礎解析。

支援 PNG / JPG / JPEG / BMP（另外附帶 WebP、TIFF），
回傳統一為 RGB uint8 的 numpy 陣列，以及可顯示的中繼資料。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
MAX_SIDE = 2048  # 處理用影像的長邊上限，避免卷積運算過慢

# PIL 模式 -> (通道數, 位元深度)
_MODE_INFO = {
    "1": (1, 1),
    "L": (1, 8),
    "P": (1, 8),
    "RGB": (3, 8),
    "RGBA": (4, 8),
    "LA": (2, 8),
    "CMYK": (4, 8),
    "I;16": (1, 16),
    "I": (1, 32),
}


class ImageLoadError(ValueError):
    """影像無法載入時拋出，訊息可直接顯示給使用者。"""


@dataclass
class ImageInfo:
    filename: str
    fmt: str
    width: int
    height: int
    mode: str
    channels: int
    bit_depth: int
    file_size_bytes: int
    processed_size: Optional[Tuple[int, int]] = None  # 若被縮放，記錄 (w, h)

    @property
    def file_size_text(self) -> str:
        n = self.file_size_bytes
        if n < 1024:
            return f"{n} B"
        if n < 1024 ** 2:
            return f"{n / 1024:.1f} KB"
        return f"{n / 1024 ** 2:.2f} MB"

    def to_markdown(self) -> str:
        rows = [
            ("檔名", self.filename),
            ("格式", self.fmt),
            ("影像解析度", f"{self.width} × {self.height} px（寬 × 高）"),
            ("總像素數", f"{self.width * self.height:,}"),
            ("色彩維度", f"{self.mode}（{self.channels} 通道，{self.bit_depth}-bit／通道）"),
            ("檔案大小", self.file_size_text),
        ]
        if self.processed_size is not None:
            rows.append(
                ("處理用尺寸", f"{self.processed_size[0]} × {self.processed_size[1]} px"
                 f"（長邊超過 {MAX_SIDE}px，已等比縮小以加速運算）")
            )
        else:
            rows.append(("處理用尺寸", "與原圖相同"))
        body = "\n".join(f"| {k} | {v} |" for k, v in rows)
        return f"| 項目 | 內容 |\n|---|---|\n{body}"


def load_image(path: str) -> Tuple[np.ndarray, ImageInfo]:
    """讀入影像並回傳 (RGB uint8 陣列 HxWx3, ImageInfo)。

    - 依 EXIF 方向資訊自動轉正
    - 灰階 / 調色盤 / 含透明度影像一律轉為 RGB（透明處以白底合成）
    - 長邊超過 MAX_SIDE 時等比縮小（原始解析度仍記錄於 ImageInfo）
    """
    if not path or not os.path.isfile(path):
        raise ImageLoadError("找不到影像檔案，請重新上傳。")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTS:
        raise ImageLoadError(
            f"不支援的檔案格式「{ext or '無副檔名'}」，請使用 PNG、JPG/JPEG 或 BMP。"
        )

    try:
        img = Image.open(path)
        img.load()
    except Exception as exc:  # 檔案損毀、副檔名與內容不符等
        raise ImageLoadError(f"無法解析影像檔案（可能已損毀）：{exc}") from exc

    fmt = img.format or ext.lstrip(".").upper()
    orig_w, orig_h = img.size
    mode = img.mode
    channels, bit_depth = _MODE_INFO.get(mode, (len(img.getbands()), 8))

    img = ImageOps.exif_transpose(img)

    # --- 統一轉為 RGB uint8 ---
    if mode in ("I;16", "I"):
        arr16 = np.asarray(img, dtype=np.float64)
        scale = 257.0 if mode == "I;16" else max(arr16.max(), 1.0) / 255.0
        gray = np.clip(arr16 / scale, 0, 255).astype(np.uint8)
        rgb_img = Image.fromarray(gray, mode="L").convert("RGB")
    elif mode in ("RGBA", "LA") or (mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        rgb_img = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        rgb_img = img.convert("RGB")

    w, h = rgb_img.size
    processed_size = None
    if max(w, h) > MAX_SIDE:
        ratio = MAX_SIDE / max(w, h)
        processed_size = (max(1, round(w * ratio)), max(1, round(h * ratio)))
        rgb_img = rgb_img.resize(processed_size, Image.LANCZOS)

    arr = np.ascontiguousarray(np.asarray(rgb_img, dtype=np.uint8))
    info = ImageInfo(
        filename=os.path.basename(path),
        fmt=fmt,
        width=orig_w,
        height=orig_h,
        mode=mode,
        channels=channels,
        bit_depth=bit_depth,
        file_size_bytes=os.path.getsize(path),
        processed_size=processed_size,
    )
    return arr, info
