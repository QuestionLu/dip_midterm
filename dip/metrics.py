"""量化品質指標：MSE 與 PSNR。

    MSE  = 1/(M·N·C) · Σ_{i,j,c} ( I(i,j,c) − K(i,j,c) )²
    PSNR = 10 · log10( MAX² / MSE ) = 20 · log10( MAX ) − 10 · log10( MSE )     [dB]

其中 MAX 為像素可能的最大值（8-bit 影像為 255）。
MSE = 0（兩張影像完全相同）時 PSNR 為無限大。
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"影像尺寸不一致：{a.shape} vs {b.shape}")
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff * diff))


def psnr(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    """回傳 PSNR（dB）。完全相同時回傳 math.inf。"""
    err = mse(a, b)
    if err == 0.0:
        return math.inf
    return 10.0 * math.log10((max_val ** 2) / err)


def psnr_per_channel(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """彩色影像逐通道（R、G、B）的 PSNR。"""
    if a.ndim != 3:
        return {}
    return {name: psnr(a[..., i], b[..., i]) for i, name in enumerate("RGB")}


def format_db(value: float) -> str:
    return "∞ dB（兩圖完全相同）" if math.isinf(value) else f"{value:.2f} dB"


def abs_diff_image(a: np.ndarray, b: np.ndarray, gain: float = 4.0) -> np.ndarray:
    """絕對差異圖（放大 gain 倍以便觀察），回傳 uint8。"""
    d = np.abs(a.astype(np.float64) - b.astype(np.float64)) * gain
    return np.clip(d, 0, 255).astype(np.uint8)
