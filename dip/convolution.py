"""功能 3：自訂卷積核空間濾波。

離散二維卷積定義（kernel 大小 (2a+1) × (2b+1)，中心為原點）：

    y[i, j] = Σ_{u=-a..a} Σ_{v=-b..b}  k[u, v] · x[i - u, j - v]

與「相關（correlation）」的差別只在於 kernel 是否旋轉 180°。
本實作不呼叫 cv2.filter2D / scipy，而以 numpy「平移相加」方式手刻：
對 kernel 的每個權重，把整張（已補邊）影像平移後乘上該權重再累加，
時間複雜度 O(kh·kw·H·W)，每一步皆為向量化運算。
"""
from __future__ import annotations

import re
from fractions import Fraction
from typing import List, Tuple

import numpy as np

MAX_KERNEL_SIZE = 15

BORDER_MODES = {
    "補零 (zero padding)": "zeros",
    "複製邊緣 (replicate)": "replicate",
    "鏡射 (reflect)": "reflect",
}
NORMALIZE_MODES = {
    "不正規化": "none",
    "除以核總和 (Σk = 1)": "sum",
    "除以絕對值總和 (Σ|k| = 1)": "abs",
}

# 預設核，文字格式與使用者輸入一致（可直接填入輸入框）
PRESETS = {
    "原圖 (Identity 3×3)": "0 0 0\n0 1 0\n0 0 0",
    "均值模糊 (Box 3×3)": "1 1 1\n1 1 1\n1 1 1",
    "高斯平滑 (Gaussian 3×3)": "1 2 1\n2 4 2\n1 2 1",
    "高斯平滑 (Gaussian 5×5)": "1 4 6 4 1\n4 16 24 16 4\n6 24 36 24 6\n4 16 24 16 4\n1 4 6 4 1",
    "銳化 (Sharpen 3×3)": "0 -1 0\n-1 5 -1\n0 -1 0",
    "強銳化 (Sharpen 3×3 強)": "-1 -1 -1\n-1 9 -1\n-1 -1 -1",
    "邊緣偵測 (Laplacian 3×3)": "-1 -1 -1\n-1 8 -1\n-1 -1 -1",
    "Sobel X（垂直邊緣）": "-1 0 1\n-2 0 2\n-1 0 1",
    "Sobel Y（水平邊緣）": "-1 -2 -1\n0 0 0\n1 2 1",
    "浮雕 (Emboss 3×3)": "-2 -1 0\n-1 1 1\n0 1 2",
}

# 預設搭配的正規化方式（高斯／均值要除以總和；其餘不正規化）
PRESET_DEFAULT_NORMALIZE = {
    "均值模糊 (Box 3×3)": "除以核總和 (Σk = 1)",
    "高斯平滑 (Gaussian 3×3)": "除以核總和 (Σk = 1)",
    "高斯平滑 (Gaussian 5×5)": "除以核總和 (Σk = 1)",
}
# 浮雕核總和為 1，輸出以 0 為中心，加 128 偏移較易觀察
PRESET_DEFAULT_OFFSET = {"浮雕 (Emboss 3×3)": 128, "Sobel X（垂直邊緣）": 128, "Sobel Y（水平邊緣）": 128}


class KernelError(ValueError):
    """卷積核輸入有誤，訊息可直接顯示給使用者。"""


def parse_kernel(
    text: str, allow_nonsquare: bool = False, max_size: int = MAX_KERNEL_SIZE
) -> np.ndarray:
    """把使用者輸入的文字解析為 float64 二維陣列，並做完整防呆檢查。

    格式：每列一行（或以分號分隔），元素以空白／逗號分隔；
    支援分數（例如 1/9）與科學記號。
    """
    if text is None or not text.strip():
        raise KernelError("卷積核是空的，請輸入數值矩陣（例如 3×3）。")

    s = text.replace(";", "\n")
    s = re.sub(r"\]\s*,?\s*\[", "\n", s)  # 支援 [[1,2],[3,4]] 單行寫法
    s = re.sub(r"[\[\]\(\)]", " ", s)

    rows: List[List[float]] = []
    for r_idx, line in enumerate(
        (ln for ln in s.splitlines() if ln.strip()), start=1
    ):
        tokens = [t for t in re.split(r"[,\s]+", line.strip()) if t]
        row = []
        for c_idx, tok in enumerate(tokens, start=1):
            try:
                val = float(Fraction(tok))
            except ZeroDivisionError:
                raise KernelError(f"第 {r_idx} 列第 {c_idx} 個元素「{tok}」的分母為 0。")
            except (ValueError, OverflowError):
                raise KernelError(
                    f"第 {r_idx} 列第 {c_idx} 個元素「{tok}」不是有效數字。"
                    "（可輸入整數、小數或分數，如 -1、0.5、1/9）"
                )
            if abs(val) > 1e6:
                raise KernelError(f"第 {r_idx} 列第 {c_idx} 個元素數值過大（|值| 需 ≤ 1,000,000）。")
            row.append(val)
        rows.append(row)

    n_rows = len(rows)
    lengths = {len(r) for r in rows}
    if len(lengths) != 1:
        detail = "、".join(f"第{i}列 {len(r)} 個" for i, r in enumerate(rows, start=1))
        raise KernelError(f"各列元素數量不一致（{detail}），無法構成矩陣。")
    n_cols = lengths.pop()

    if n_rows == 0 or n_cols == 0:
        raise KernelError("卷積核是空的，請輸入數值矩陣（例如 3×3）。")
    if n_rows != n_cols and not allow_nonsquare:
        raise KernelError(
            f"輸入為 {n_rows}×{n_cols} 的非方陣。請改輸入方陣（如 3×3、5×5），"
            "或在進階設定勾選「允許非方陣核」。"
        )
    if n_rows % 2 == 0 or n_cols % 2 == 0:
        raise KernelError(
            f"卷積核尺寸需為奇數×奇數（目前為 {n_rows}×{n_cols}），否則沒有中心點。"
        )
    if max(n_rows, n_cols) > max_size:
        raise KernelError(f"卷積核過大（{n_rows}×{n_cols}），上限為 {max_size}×{max_size}。")

    return np.array(rows, dtype=np.float64)


def normalize_kernel(kernel: np.ndarray, mode: str) -> Tuple[np.ndarray, List[str]]:
    """依模式正規化卷積核，回傳 (新核, 警告訊息列表)。

    - none：不處理
    - sum ：除以 Σk，使整體亮度不變（高斯、均值核常用）。
            若 Σk = 0（邊緣／浮雕核）會造成除以零，改為略過並警告。
    - abs ：除以 Σ|k|，對總和為 0 的核仍可運作。
    """
    warnings: List[str] = []
    if mode == "none":
        return kernel, warnings
    if mode == "sum":
        s = float(kernel.sum())
        if abs(s) < 1e-12:
            warnings.append(
                "核總和為 0，無法以「核總和」正規化（會除以零），已略過正規化。"
                "可改選「除以絕對值總和」。"
            )
            return kernel, warnings
        return kernel / s, warnings
    if mode == "abs":
        s = float(np.abs(kernel).sum())
        if s < 1e-12:
            warnings.append("卷積核全為 0，無法正規化，輸出將為全黑。")
            return kernel, warnings
        return kernel / s, warnings
    raise KernelError(f"未知的正規化模式：{mode}")


def convolve2d(
    channel: np.ndarray, kernel: np.ndarray, *, flip: bool = True, border: str = "zeros"
) -> np.ndarray:
    """對單一通道 (H, W) 做二維卷積，回傳 float64 (H, W)，輸出大小與輸入相同（same）。

    flip=True  : 真正的卷積（kernel 旋轉 180°）
    flip=False : 相關（與 cv2.filter2D 行為一致）
    border     : zeros / replicate / reflect
    """
    x = np.asarray(channel, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("convolve2d 只接受二維影像通道")
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] % 2 == 0 or k.shape[1] % 2 == 0:
        raise KernelError("卷積核必須是奇數×奇數的二維陣列。")
    kh, kw = k.shape
    h, w = x.shape
    if kh > h or kw > w:
        raise KernelError(f"卷積核（{kh}×{kw}）大於影像（{h}×{w}），請縮小卷積核。")

    pad_mode = {"zeros": "constant", "replicate": "edge", "reflect": "reflect"}.get(border)
    if pad_mode is None:
        raise ValueError(f"未知的邊界處理方式：{border}")

    if flip:
        k = k[::-1, ::-1]
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((ph, ph), (pw, pw)), mode=pad_mode)

    out = np.zeros((h, w), dtype=np.float64)
    for u in range(kh):
        for v in range(kw):
            weight = k[u, v]
            if weight != 0.0:
                out += weight * xp[u : u + h, v : v + w]
    return out


def apply_filter(
    img: np.ndarray,
    kernel: np.ndarray,
    *,
    flip: bool = True,
    border: str = "zeros",
    offset: float = 0.0,
) -> np.ndarray:
    """對灰階 (H,W) 或彩色 (H,W,3) uint8 影像逐通道卷積，加上偏移後四捨五入並截斷到 [0,255]。"""
    if img.ndim == 2:
        out = convolve2d(img, kernel, flip=flip, border=border)
    else:
        out = np.stack(
            [convolve2d(img[..., c], kernel, flip=flip, border=border) for c in range(img.shape[2])],
            axis=-1,
        )
    return np.clip(np.rint(out + offset), 0, 255).astype(np.uint8)
