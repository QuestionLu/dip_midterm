"""產生內建的合成範例圖（不含任何真人照片），用於展示色彩、直方圖與濾波效果。

執行：  python scripts/make_examples.py
人臉偵測請另外放入自己的照片（例如 0 張臉、1 張臉、多人合照）到 examples/ 資料夾。
"""
import os

import cv2
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
os.makedirs(OUT, exist_ok=True)


def color_pattern(h=360, w=480):
    """彩色漸層 + 色塊 + 幾何圖形：觀察 R/G/B、Y/U/V 分量。"""
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.float64)
    img[..., 0] = xx / (w - 1) * 255
    img[..., 1] = yy / (h - 1) * 255
    img[..., 2] = (1 - xx / (w - 1)) * 200
    img = img.astype(np.uint8)
    blocks = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255), (0, 0, 0)]
    bw = w // len(blocks)
    for i, c in enumerate(blocks):
        img[: h // 6, i * bw : (i + 1) * bw] = c
    cv2.circle(img, (w // 2, h // 2 + 20), 70, (250, 200, 30), -1)
    cv2.rectangle(img, (40, h // 2), (170, h - 40), (20, 90, 200), -1)
    cv2.putText(img, "DIP", (w - 170, h - 50), cv2.FONT_HERSHEY_DUPLEX, 2.0, (255, 255, 255), 3)
    return img


def edge_pattern(h=360, w=480):
    """棋盤格 + 斜線 + 文字：觀察邊緣偵測、銳化與模糊。"""
    img = np.full((h, w, 3), 235, np.uint8)
    s = 30
    for r in range(0, h // 2, s):
        for c in range(0, w // 2, s):
            if (r // s + c // s) % 2 == 0:
                img[r : r + s, c : c + s] = 30
    for k in range(8):
        cv2.line(img, (w // 2 + 10, 20 + k * 40), (w - 20, 60 + k * 40), (40, 40, 40), 1 + k // 3)
    cv2.putText(img, "Edge Test 123", (20, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (10, 10, 10), 2)
    cv2.circle(img, (w // 4, 3 * h // 4 - 20), 40, (0, 0, 0), 2)
    return img


def noise_pattern(h=360, w=480, seed=7):
    """含高斯雜訊的柔和影像：觀察高斯平滑去雜訊與 PSNR 變化。"""
    rng = np.random.default_rng(seed)
    base = color_pattern(h, w).astype(np.float64)
    base = cv2.GaussianBlur(base, (0, 0), 6)
    noisy = base + rng.normal(0, 18, base.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def save(name, rgb, params=None):
    path = os.path.join(OUT, name)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), params or [])
    print("saved", path)


if __name__ == "__main__":
    save("pattern_color.png", color_pattern())
    save("pattern_edges.jpg", edge_pattern(), [cv2.IMWRITE_JPEG_QUALITY, 92])
    save("pattern_noise.bmp", noise_pattern())
