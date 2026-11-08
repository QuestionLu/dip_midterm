"""濾波效果的客觀評估（PSNR），產生報告第 3 章可直接使用的圖表與數據。

執行：  python scripts/evaluate_psnr.py        輸出至 report_assets/

實驗 A：各預設核對同一張影像的 PSNR（原圖為基準 → 濾波後影像偏離原圖多少）。
實驗 B：去雜訊。以「乾淨影像」為基準，比較 雜訊影像 與 各種平滑濾波結果 的 PSNR，
        PSNR 越高代表越接近乾淨影像；並觀察高斯 σ 過大時的過度平滑（PSNR 反降）。
"""
import csv
import math
import os
import sys

import cv2
import numpy as np
from matplotlib.figure import Figure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import make_examples  # noqa: E402
from dip import convolution as C, metrics  # noqa: E402

OUT = os.path.join(ROOT, "report_assets")
os.makedirs(OUT, exist_ok=True)

# 配色：採驗證過的類別色前三格（藍、橘、青綠），文字一律使用中性墨色
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"


def style(ax):
    ax.grid(alpha=0.25, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def gaussian_kernel(size, sigma):
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = np.outer(g, g)
    return k / k.sum()


def run_preset(rgb, name):
    k = C.parse_kernel(C.PRESETS[name])
    norm = C.NORMALIZE_MODES[C.PRESET_DEFAULT_NORMALIZE.get(name, "不正規化")]
    k, _ = C.normalize_kernel(k, norm)
    off = C.PRESET_DEFAULT_OFFSET.get(name, 0)
    out = C.apply_filter(rgb, k, border="replicate", offset=off)
    return out


# ---------------------------------------------------------------- 實驗 A
def experiment_a():
    src = make_examples.edge_pattern()
    rows = []
    for name in C.PRESETS:
        out = run_preset(src, name)
        rows.append((name, metrics.mse(src, out), metrics.psnr(src, out)))
    with open(os.path.join(OUT, "psnr_presets.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "MSE", "PSNR_dB"])
        for n, m, p in rows:
            w.writerow([n, f"{m:.2f}", "inf" if math.isinf(p) else f"{p:.2f}"])

    finite = [r for r in rows if not math.isinf(r[2])]
    finite.sort(key=lambda r: r[2])
    fig = Figure(figsize=(8.2, 4.6), dpi=150)
    ax = fig.subplots()
    labels = [r[0] for r in finite]
    ys = np.arange(len(finite))
    ax.barh(ys, [r[2] for r in finite], color=BLUE, height=0.6)
    for y, r in zip(ys, finite):
        ax.text(r[2] + 0.4, y, f"{r[2]:.1f} dB", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([_ascii(n) for n in labels], fontsize=9)
    ax.set_xlabel("PSNR vs. original (dB)")
    ax.set_title("Experiment A: PSNR of each preset kernel (edge test image, replicate border)", fontsize=10.5)
    ax.set_xlim(0, max(r[2] for r in finite) + 6)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "psnr_presets.png"))
    return rows


def _ascii(name):
    """圖表用英文標籤（避免伺服器無中文字型而出現方塊字）。"""
    table = {
        "原圖 (Identity 3×3)": "Identity 3x3",
        "均值模糊 (Box 3×3)": "Box blur 3x3",
        "高斯平滑 (Gaussian 3×3)": "Gaussian 3x3",
        "高斯平滑 (Gaussian 5×5)": "Gaussian 5x5",
        "銳化 (Sharpen 3×3)": "Sharpen 3x3",
        "強銳化 (Sharpen 3×3 強)": "Strong sharpen 3x3",
        "邊緣偵測 (Laplacian 3×3)": "Laplacian edge 3x3",
        "Sobel X（垂直邊緣）": "Sobel X (+128)",
        "Sobel Y（水平邊緣）": "Sobel Y (+128)",
        "浮雕 (Emboss 3×3)": "Emboss 3x3 (+128)",
    }
    return table.get(name, name)


# ---------------------------------------------------------------- 實驗 B
def experiment_b():
    base = make_examples.color_pattern()
    clean = cv2.GaussianBlur(base, (0, 0), 6)
    rng = np.random.default_rng(7)
    noisy = np.clip(clean.astype(np.float64) + rng.normal(0, 18, clean.shape), 0, 255).astype(np.uint8)
    base_psnr = metrics.psnr(clean, noisy)

    sigmas = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    g_rows = []
    for s in sigmas:
        size = min(15, 2 * math.ceil(3 * s) + 1)
        out = C.apply_filter(noisy, gaussian_kernel(size, s), border="reflect")
        g_rows.append((f"Gaussian sigma={s}", size, metrics.psnr(clean, out), out))
    b_rows = []
    for size in (3, 5, 7):
        k = np.full((size, size), 1.0 / size ** 2)
        out = C.apply_filter(noisy, k, border="reflect")
        b_rows.append((f"Box {size}x{size}", size, metrics.psnr(clean, out), out))

    with open(os.path.join(OUT, "denoise_psnr.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["filter", "kernel_size", "PSNR_to_clean_dB"])
        w.writerow(["noisy (no filter)", "-", f"{base_psnr:.2f}"])
        for n, sz, p, _ in g_rows + b_rows:
            w.writerow([n, f"{sz}x{sz}", f"{p:.2f}"])

    # 圖 1：PSNR vs sigma
    fig = Figure(figsize=(7.6, 4.2), dpi=150)
    ax = fig.subplots()
    ax.plot(sigmas, [r[2] for r in g_rows], color=BLUE, lw=2, marker="o", ms=6, mec="white", mew=1.5, label="Gaussian filter")
    ax.axhline(base_psnr, color=ORANGE, lw=2, ls="--", label=f"Noisy input ({base_psnr:.1f} dB)")
    best = max(g_rows, key=lambda r: r[2])
    ax.annotate(f"best: {best[2]:.1f} dB\n({best[0].split('=')[1]})", (float(best[0].split('=')[1]), best[2]),
                textcoords="offset points", xytext=(10, -34), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED))
    ax.set_xlabel("Gaussian sigma (kernel size = min(15, 2*ceil(3*sigma)+1))")
    ax.set_ylabel("PSNR vs. clean image (dB)")
    ax.set_title("Experiment B: denoising quality vs. smoothing strength", fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "denoise_sigma.png"))

    # 圖 2：視覺比較
    best_out = best[3]
    box5 = [r for r in b_rows if r[1] == 5][0]
    panels = [("Clean (reference)", clean, None), ("Noisy input", noisy, base_psnr),
              (f"Gaussian {best[0].split('=')[1]} ({best[1]}x{best[1]})", best_out, best[2]),
              ("Box 5x5", box5[3], box5[2])]
    fig = Figure(figsize=(10, 2.9), dpi=150)
    axes = fig.subplots(1, 4)
    for ax, (t, im, p) in zip(axes, panels):
        ax.imshow(im)
        ax.set_axis_off()
        ax.set_title(t + ("" if p is None else f"\nPSNR {p:.2f} dB"), fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "denoise_visual.png"))
    return base_psnr, g_rows, b_rows


if __name__ == "__main__":
    rows = experiment_a()
    print("== Experiment A (edge pattern, replicate border) ==")
    for n, m, p in rows:
        print(f"{n:<28s} MSE={m:9.2f}  PSNR={'inf' if math.isinf(p) else f'{p:.2f}'} dB")
    base_psnr, g, b = experiment_b()
    print("\n== Experiment B (PSNR to clean image) ==")
    print(f"noisy input                  PSNR={base_psnr:.2f} dB")
    for n, sz, p, _ in g + b:
        print(f"{n:<22s} {sz:>2d}x{sz:<2d}      PSNR={p:.2f} dB")
    print("\nsaved to", OUT)
