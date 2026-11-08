"""數位影像處理期中實作：互動式影像處理與人臉分析工具（Gradio Web App）。

執行：  python app.py       然後開啟 http://127.0.0.1:7860
"""
from __future__ import annotations

import glob
import os
import time
import traceback

import gradio as gr
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from dip import color, convolution as C, face, io_utils, metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NO_IMAGE = "⚠️ 請先在上方「① 上傳影像」載入一張影像。"

VIEW_ORIGINAL = "Original（原始影像）"
VIEW_RGB = "RGB 分離（R / G / B 通道）"
VIEW_YUV = "YUV 分離（Y / U / V 分量）"
VIEW_MODES = [VIEW_ORIGINAL, VIEW_RGB, VIEW_YUV]

SCOPE_COLOR = "彩色（R、G、B 三通道分別濾波）"
SCOPE_GRAY = "灰階（先轉成亮度 Y 再濾波）"

DEFAULT_PRESET = "銳化 (Sharpen 3×3)"


def _gray3(ch: np.ndarray) -> np.ndarray:
    """單通道轉三通道（僅供顯示）。"""
    return np.stack([ch, ch, ch], axis=-1)


# ----------------------------------------------------------------------------
# 功能 1：影像載入
# ----------------------------------------------------------------------------
def on_upload(path):
    """回傳 (共用影像狀態, 中繼資料 Markdown, 預覽圖)。"""
    if not path:
        return None, "尚未載入影像。請上傳 PNG、JPG/JPEG 或 BMP 檔案。", None
    try:
        rgb, info = io_utils.load_image(path)
    except io_utils.ImageLoadError as exc:
        return None, f"❌ {exc}", None
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return None, f"❌ 發生未預期的錯誤：{exc}", None
    return rgb, info.to_markdown(), rgb


# ----------------------------------------------------------------------------
# 功能 2：色彩空間與直方圖
# ----------------------------------------------------------------------------
_HIST_COLORS = {"R": "#d62728", "G": "#2ca02c", "B": "#1f77b4", "Y": "#555555", "U": "#9467bd", "V": "#ff7f0e", "Gray": "#777777"}


def _style_axis(ax, title):
    ax.set_xlim(0, 255)
    ax.set_xticks([0, 64, 128, 192, 255])
    ax.set_xlabel("Pixel intensity (0-255)")
    ax.set_ylabel("Pixel count")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))


def _histogram_figure(panels, overlay_rgb=None):
    """panels: [(名稱, uint8 通道)]；overlay_rgb: 在第一個面板疊加 R/G/B 曲線（Original 模式）。"""
    n = len(panels)
    fig = Figure(figsize=(4.4 * n if n > 1 else 8.0, 3.6), dpi=100)
    axes = fig.subplots(1, n, squeeze=False)[0]
    xs = np.arange(256)
    for ax, (name, ch) in zip(axes, panels):
        h = color.histogram(ch)
        ax.bar(xs, h, width=1.0, color=_HIST_COLORS.get(name, "#555555"), alpha=0.85, linewidth=0, label=name)
        title = f"{name} histogram"
        if overlay_rgb is not None:
            for cname, cch in overlay_rgb.items():
                ax.plot(xs, color.histogram(cch), color=_HIST_COLORS[cname], lw=1.0, alpha=0.9, label=cname)
            title = "Original: gray (Y) bars + R/G/B curves"
            ax.legend(fontsize=8, loc="upper right")
        _style_axis(ax, title)
    fig.tight_layout()
    return fig


def _stats_table(panels):
    rows = []
    for name, ch in panels:
        s = color.channel_stats(ch)
        rows.append(f"| {name} | {s['min']:.0f} | {s['max']:.0f} | {s['mean']:.2f} | {s['std']:.2f} |")
    return "| 通道 | 最小值 | 最大值 | 平均 | 標準差 |\n|---|---|---|---|---|\n" + "\n".join(rows)


def _image_updates(items):
    """items: [(影像, 標籤)] 最多 3 張 -> 3 個 gr.Image 的更新（多餘的隱藏）。"""
    updates = []
    for i in range(3):
        if i < len(items):
            arr, label = items[i]
            updates.append(gr.update(value=arr, label=label, visible=True))
        else:
            updates.append(gr.update(value=None, visible=False))
    return updates


def render_color_view(rgb, mode):
    """回傳 (圖1, 圖2, 圖3, 直方圖, 統計表)。"""
    if rgb is None:
        return (*_image_updates([]), None, NO_IMAGE)
    try:
        if mode == VIEW_RGB:
            chans = color.split_rgb(rgb)
            panels = [(k, chans[k]) for k in "RGB"]
            items = [(_gray3(chans[k]), f"{k} 通道 (channel)") for k in "RGB"]
            fig = _histogram_figure(panels)
        elif mode == VIEW_YUV:
            chans = color.split_yuv(rgb)
            panels = [(k, chans[k]) for k in "YUV"]
            items = [
                (_gray3(chans[k]), f"{k} 分量" + ("（亮度）" if k == "Y" else "（色度，+128 偏移）")) for k in "YUV"
            ]
            fig = _histogram_figure(panels)
        else:
            gray = color.to_gray(rgb)
            panels = [("Gray", gray)] + [(k, v) for k, v in color.split_rgb(rgb).items()]
            items = [(rgb, "Original（原始影像）")]
            fig = _histogram_figure([("Gray", gray)], overlay_rgb=color.split_rgb(rgb))
        return (*_image_updates(items), fig, _stats_table(panels))
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return (*_image_updates([]), None, f"❌ 發生未預期的錯誤：{exc}")


# ----------------------------------------------------------------------------
# 功能 3：自訂卷積濾波與 PSNR
# ----------------------------------------------------------------------------
def on_preset(name):
    text = C.PRESETS[name]
    norm = C.PRESET_DEFAULT_NORMALIZE.get(name, "不正規化")
    offset = C.PRESET_DEFAULT_OFFSET.get(name, 0)
    return text, norm, offset


def _psnr_comment(db: float) -> str:
    if np.isinf(db):
        return "兩張影像完全相同。"
    if db >= 40:
        return "差異極小（> 40 dB），肉眼幾乎無法分辨。"
    if db >= 30:
        return "差異輕微（30–40 dB）。"
    if db >= 20:
        return "差異明顯（20–30 dB）。"
    return "差異很大（< 20 dB），影像已被大幅改變（例如邊緣偵測、浮雕）。"


def _empty_filter_outputs(msg):
    return None, None, None, "", msg


def run_filter(rgb, kernel_text, norm_label, offset, border_label, use_flip, scope, allow_nonsquare):
    if rgb is None:
        return _empty_filter_outputs(NO_IMAGE)
    try:
        kernel = C.parse_kernel(kernel_text, allow_nonsquare=bool(allow_nonsquare))
        kernel_used, warnings = C.normalize_kernel(kernel, C.NORMALIZE_MODES[norm_label])
        if not np.any(kernel):
            warnings.append("卷積核全為 0，濾波結果將是全黑（加上偏移量時為單一灰階）。")

        if scope == SCOPE_GRAY:
            src = color.to_gray(rgb)
        else:
            src = rgb

        t0 = time.perf_counter()
        out = C.apply_filter(
            src, kernel_used, flip=bool(use_flip), border=C.BORDER_MODES[border_label], offset=float(offset)
        )
        elapsed = (time.perf_counter() - t0) * 1000.0

        value = metrics.psnr(src, out)
        err = metrics.mse(src, out)
        per_ch = metrics.psnr_per_channel(src, out) if src.ndim == 3 else {}

        disp_src = _gray3(src) if src.ndim == 2 else src
        disp_out = _gray3(out) if out.ndim == 2 else out
        diff = metrics.abs_diff_image(disp_src, disp_out, gain=4.0)

        lines = [
            f"## PSNR = {metrics.format_db(value)}",
            f"- MSE = **{err:.2f}**　（PSNR = 10·log₁₀(255² / MSE)，原圖為基準、濾波後為比較對象）",
            f"- 評語：{_psnr_comment(value)}",
        ]
        if per_ch:
            lines.append("- 各通道 PSNR：" + "　".join(f"{k} = {metrics.format_db(v)}" for k, v in per_ch.items()))
        lines.append(
            f"- 影像 {src.shape[1]}×{src.shape[0]}，卷積核 {kernel.shape[0]}×{kernel.shape[1]}，"
            f"運算 {elapsed:.0f} ms，{'真正卷積（旋轉 180°）' if use_flip else '相關運算（不旋轉）'}，邊界：{border_label}"
        )
        for w in warnings:
            lines.append(f"- ⚠️ {w}")
        status = "\n".join(lines)

        k_txt = np.array2string(kernel_used, precision=4, suppress_small=True, floatmode="maxprec", max_line_width=120)
        kernel_md = (
            f"**實際使用的卷積核**（Σk = {kernel_used.sum():.4g}，Σ|k| = {np.abs(kernel_used).sum():.4g}）\n"
            f"```\n{k_txt}\n```"
        )
        return disp_src, disp_out, diff, kernel_md, status
    except (C.KernelError, ValueError) as exc:
        return _empty_filter_outputs(f"❌ 輸入錯誤：{exc}")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return _empty_filter_outputs(f"❌ 發生未預期的錯誤：{exc}")


# ----------------------------------------------------------------------------
# 功能 4：人臉偵測
# ----------------------------------------------------------------------------
def on_backend_change(label):
    is_yunet = face.BACKENDS[label] == "yunet"
    return gr.update(visible=is_yunet), gr.update(visible=not is_yunet)


def run_face(rgb, backend_label, score_thr, nms_thr, scale_factor, min_neighbors, min_size):
    if rgb is None:
        return None, NO_IMAGE, []
    try:
        res = face.detect_faces(
            rgb,
            face.BACKENDS[backend_label],
            score_threshold=score_thr,
            nms_threshold=nms_thr,
            scale_factor=scale_factor,
            min_neighbors=int(min_neighbors),
            min_size=int(min_size),
        )
    except face.FaceDetectionError as exc:
        return None, f"❌ {exc}", []
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return None, f"❌ 發生未預期的錯誤：{exc}", []

    rows = [[i, f.x, f.y, f.w, f.h, round(f.score, 3)] for i, f in enumerate(res.faces, start=1)]
    if res.count == 0:
        summary = (
            f"## 偵測到 0 張人臉\n- 偵測器：{res.backend}，耗時 {res.elapsed_ms:.0f} ms\n"
            "- 可嘗試降低信心門檻（YuNet）或降低 minNeighbors（Haar）。"
        )
    else:
        avg = sum(f.score for f in res.faces) / res.count
        mx = max(f.score for f in res.faces)
        summary = (
            f"## 偵測到 {res.count} 張人臉\n"
            f"- 偵測器：{res.backend}，耗時 {res.elapsed_ms:.0f} ms\n"
            f"- 信心度：平均 {avg:.3f}，最高 {mx:.3f}\n"
            f"- 說明：{res.score_note}"
        )
    return res.annotated, summary, rows


# ----------------------------------------------------------------------------
# 介面
# ----------------------------------------------------------------------------
def build_demo() -> gr.Blocks:
    example_files = sorted(
        p for ext in ("png", "jpg", "jpeg", "bmp") for p in glob.glob(os.path.join(BASE_DIR, "examples", f"*.{ext}"))
    )

    with gr.Blocks(title="DIP 期中專題｜影像處理與人臉分析工具") as demo:
        gr.Markdown(
            "# 🖼️ 互動式影像處理與人臉分析工具\n"
            "數位影像處理期中實作：影像載入 → 色彩空間與直方圖 → 自訂卷積濾波與 PSNR → 人臉偵測"
        )
        image_state = gr.State(None)  # 以 RGB uint8 陣列在各分頁共用目前影像

        # ---- 功能 1 ----
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(
                    label="① 上傳影像（PNG / JPG / JPEG / BMP）",
                    file_types=[".png", ".jpg", ".jpeg", ".bmp"],
                    file_count="single",
                    type="filepath",
                    height=110,
                )
                preview = gr.Image(label="預覽", interactive=False, height=260)
            with gr.Column(scale=1):
                gr.Markdown("### 影像中繼資料")
                info_md = gr.Markdown("尚未載入影像。請上傳 PNG、JPG/JPEG 或 BMP 檔案。")
        if example_files:
            gr.Examples(examples=[[p] for p in example_files], inputs=upload, label="範例影像（點擊載入）")

        with gr.Tabs():
            # ---- 功能 2 ----
            with gr.Tab("② 色彩空間與直方圖"):
                view_mode = gr.Radio(VIEW_MODES, value=VIEW_ORIGINAL, label="顯示模式（切換時直方圖同步更新）")
                with gr.Row():
                    view_imgs = [
                        gr.Image(label=f"影像 {i + 1}", interactive=False, height=320, visible=False) for i in range(3)
                    ]
                gr.Markdown("#### 直方圖（橫軸：像素強度 0–255；縱軸：像素數量）")
                hist_plot = gr.Plot(show_label=False)
                stats_md = gr.Markdown()

            # ---- 功能 3 ----
            with gr.Tab("③ 卷積濾波與 PSNR"):
                with gr.Row():
                    with gr.Column(scale=1):
                        preset = gr.Dropdown(list(C.PRESETS), value=DEFAULT_PRESET, label="預設卷積核（可選後再手動修改）")
                        kernel_text = gr.Textbox(
                            value=C.PRESETS[DEFAULT_PRESET],
                            lines=6,
                            label="卷積核矩陣（每列一行；元素以空白或逗號分隔；支援分數，如 1/9）",
                            placeholder="0 -1 0\n-1 5 -1\n0 -1 0",
                        )
                        norm = gr.Dropdown(list(C.NORMALIZE_MODES), value="不正規化", label="正規化 (Normalization)")
                        offset = gr.Slider(0, 255, value=0, step=1, label="輸出偏移量（浮雕／Sobel 建議 128）")
                        with gr.Accordion("進階設定", open=False):
                            border = gr.Dropdown(list(C.BORDER_MODES), value="補零 (zero padding)", label="邊界處理")
                            flip = gr.Checkbox(value=True, label="真正卷積（kernel 旋轉 180°；取消勾選則為相關運算）")
                            scope = gr.Radio([SCOPE_COLOR, SCOPE_GRAY], value=SCOPE_COLOR, label="濾波對象")
                            nonsquare = gr.Checkbox(value=False, label="允許非方陣核（奇數×奇數，如 3×5）")
                        run_btn = gr.Button("套用濾波並計算 PSNR", variant="primary")
                    with gr.Column(scale=2):
                        status_md = gr.Markdown("按下「套用濾波並計算 PSNR」後顯示結果。")
                        kernel_md = gr.Markdown()
                with gr.Row():
                    out_orig = gr.Image(label="濾波前（原圖）", interactive=False)
                    out_filt = gr.Image(label="濾波後", interactive=False)
                    out_diff = gr.Image(label="差異圖 |原圖 − 濾波後| × 4", interactive=False)

            # ---- 功能 4 ----
            with gr.Tab("④ 人臉偵測"):
                with gr.Row():
                    with gr.Column(scale=1):
                        backend = gr.Radio(list(face.BACKENDS), value=list(face.BACKENDS)[0], label="偵測器")
                        with gr.Group(visible=True) as yunet_group:
                            score_thr = gr.Slider(0.1, 0.99, value=0.7, step=0.01, label="信心度門檻 (score threshold)")
                            nms_thr = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="NMS 門檻")
                        with gr.Group(visible=False) as haar_group:
                            scale_factor = gr.Slider(1.05, 1.5, value=1.1, step=0.01, label="scaleFactor")
                            min_neighbors = gr.Slider(1, 10, value=5, step=1, label="minNeighbors")
                            min_size = gr.Slider(10, 200, value=30, step=5, label="最小人臉尺寸 (px)")
                        face_btn = gr.Button("偵測人臉", variant="primary")
                        face_md = gr.Markdown("按下「偵測人臉」後顯示結果。")
                    with gr.Column(scale=2):
                        face_img = gr.Image(label="偵測結果（綠框＝人臉邊界框，標籤＝編號與信心度）", interactive=False)
                face_table = gr.Dataframe(
                    headers=["#", "x", "y", "w", "h", "信心度 (Score)"],
                    datatype=["number"] * 6,
                    interactive=False,
                    label="人臉清單（x, y 為左上角座標，單位 px）",
                )

        # ---- 事件連結 ----
        color_outputs = [*view_imgs, hist_plot, stats_md]
        upload.change(on_upload, upload, [image_state, info_md, preview], api_name="load_image").then(
            render_color_view, [image_state, view_mode], color_outputs, api_name="color_view_after_load"
        )
        view_mode.change(render_color_view, [image_state, view_mode], color_outputs, api_name="color_view")

        filter_inputs = [image_state, kernel_text, norm, offset, border, flip, scope, nonsquare]
        filter_outputs = [out_orig, out_filt, out_diff, kernel_md, status_md]
        preset.change(on_preset, preset, [kernel_text, norm, offset], api_name="choose_preset").then(
            run_filter, filter_inputs, filter_outputs, api_name="filter_after_preset"
        )
        run_btn.click(run_filter, filter_inputs, filter_outputs, api_name="filter")

        backend.change(on_backend_change, backend, [yunet_group, haar_group], api_name="switch_backend")
        face_btn.click(
            run_face,
            [image_state, backend, score_thr, nms_thr, scale_factor, min_neighbors, min_size],
            [face_img, face_md, face_table],
            api_name="detect_faces",
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
