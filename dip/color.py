"""功能 2：色彩空間轉換與直方圖。

YUV 採用 BT.601 係數（類比 YUV 定義），並加上 +128 偏移以便以 8-bit 顯示：

    Y =  0.299 R + 0.587 G + 0.114 B
    U = -0.14713 R - 0.28886 G + 0.436 B      (= 0.492 (B - Y))
    V =  0.615 R - 0.51499 G - 0.10001 B      (= 0.877 (R - Y))

U、V 理論範圍分別為 ±111.2 與 ±156.8，顯示時加 128 並截斷到 [0, 255]。
"""
from __future__ import annotations

from typing import Dict

import numpy as np

# 列向量 [Y, U, V] = M @ [R, G, B]
_RGB2YUV = np.array(
    [
        [0.299, 0.587, 0.114],
        [-0.14713, -0.28886, 0.436],
        [0.615, -0.51499, -0.10001],
    ],
    dtype=np.float64,
)
_YUV2RGB = np.linalg.inv(_RGB2YUV)


def split_rgb(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """回傳 R、G、B 三個灰階通道圖 (H, W) uint8。"""
    return {"R": rgb[..., 0].copy(), "G": rgb[..., 1].copy(), "B": rgb[..., 2].copy()}


def rgb_to_yuv(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (H,W,3) -> 浮點 YUV (H,W,3)，U、V 帶正負號（未加偏移）。"""
    return rgb.astype(np.float64) @ _RGB2YUV.T


def yuv_to_rgb(yuv: np.ndarray) -> np.ndarray:
    """浮點 YUV (未加偏移) -> RGB uint8。"""
    out = yuv @ _YUV2RGB.T
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def split_yuv(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    """回傳 Y、U、V 三個可顯示的 8-bit 通道圖 (U、V 已加 128 偏移)。"""
    yuv = rgb_to_yuv(rgb)
    y = np.clip(np.rint(yuv[..., 0]), 0, 255).astype(np.uint8)
    u = np.clip(np.rint(yuv[..., 1] + 128.0), 0, 255).astype(np.uint8)
    v = np.clip(np.rint(yuv[..., 2] + 128.0), 0, 255).astype(np.uint8)
    return {"Y": y, "U": u, "V": v}


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """以 BT.601 亮度公式轉灰階 (H,W) uint8（即 Y 分量）。"""
    return split_yuv(rgb)["Y"]


def histogram(channel: np.ndarray) -> np.ndarray:
    """計算 0–255 共 256 個 bin 的像素數量（長度 256 的整數陣列）。"""
    if channel.dtype != np.uint8:
        raise ValueError("histogram() 只接受 uint8 影像通道")
    return np.bincount(channel.ravel(), minlength=256).astype(np.int64)


def channel_stats(channel: np.ndarray) -> Dict[str, float]:
    """回傳通道的最小值、最大值、平均、標準差。"""
    c = channel.astype(np.float64)
    return {
        "min": float(c.min()),
        "max": float(c.max()),
        "mean": float(c.mean()),
        "std": float(c.std()),
    }
