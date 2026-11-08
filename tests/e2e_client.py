"""端對端測試：對執行中的 Gradio 應用透過 API 呼叫（需先 `python app.py`）。

執行：  python tests/e2e_client.py [URL]
"""
import os
import sys

from gradio_client import Client, handle_file

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.join(os.path.dirname(HERE), "examples")

client = Client(URL, verbose=False)
fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  -> {detail}" if detail and not cond else ""))
    if not cond:
        fails.append(name)


# 未載入影像時各功能的提示
res = client.predict(None, "Original（原始影像）", api_name="/color_view")
check("未載入影像: 色彩分頁提示", "請先" in res[4], res[4])

# 功能 1
info = client.predict(handle_file(os.path.join(EX, "pattern_edges.jpg")), api_name="/load_image")[0]
check("功能1: 中繼資料含解析度", "480 × 360" in info and "JPEG" in info, info)
check("功能1: 中繼資料含檔案大小/通道", "KB" in info and "3 通道" in info, info)
for fn in ["pattern_color.png", "pattern_noise.bmp"]:
    i = client.predict(handle_file(os.path.join(EX, fn)), api_name="/load_image")[0]
    check(f"功能1: 載入 {fn}", "480 × 360" in i, i)

# 錯誤檔案
import tempfile

with tempfile.TemporaryDirectory() as d:
    p = os.path.join(d, "bad.png")
    open(p, "wb").write(b"junk")
    i = client.predict(handle_file(p), api_name="/load_image")[0]
    check("功能1: 損毀檔案顯示錯誤", "❌" in i, i)
    # 載入失敗後重新載入正常圖
    client.predict(handle_file(os.path.join(EX, "pattern_edges.jpg")), api_name="/load_image")[0]

# 功能 2
for mode, n_expect in [("Original（原始影像）", 1), ("RGB 分離（R / G / B 通道）", 3), ("YUV 分離（Y / U / V 分量）", 3)]:
    i1, i2, i3, plot, stats = client.predict(mode, api_name="/color_view")
    n_imgs = sum(1 for x in (i1, i2, i3) if isinstance(x, dict) and x.get("visible"))
    check(f"功能2: {mode[:3]} 圖片數量={n_expect}", n_imgs == n_expect, str(n_imgs))
    check(f"功能2: {mode[:3]} 有直方圖與統計", plot is not None and "平均" in stats)

# 功能 3
def filt(text, norm="不正規化", offset=0, border="補零 (zero padding)", flip=True, scope="彩色（R、G、B 三通道分別濾波）", ns=False):
    return client.predict(text, norm, offset, border, flip, scope, ns, api_name="/filter")


sharpen = "0 -1 0\n-1 5 -1\n0 -1 0"
o, f, d, k, s = filt(sharpen)
check("功能3: 銳化可執行並顯示 PSNR", f is not None and "PSNR =" in s and " dB" in s, s)
o, f, d, k, s = filt("0 0 0\n0 1 0\n0 0 0")
check("功能3: Identity → PSNR ∞", "∞" in s, s)
o, f, d, k, s = filt("1 2 1\n2 4 2\n1 2 1", norm="除以核總和 (Σk = 1)")
check("功能3: 高斯正規化", "Σk = 1" in k, k)

bad_cases = {
    "非方陣": ("1 2 3\n4 5 6", "非方陣"),
    "偶數尺寸": ("1 2\n3 4", "奇數"),
    "非數字": ("1 x 3\n4 5 6\n7 8 9", "不是有效數字"),
    "各列長度不一": ("1 2 3\n4 5\n7 8 9", "不一致"),
    "空輸入": ("", "空"),
    "分母為0": ("1/0 1 1\n1 1 1\n1 1 1", "分母為 0"),
}
for name, (txt, kw) in bad_cases.items():
    o, f, d, k, s = filt(txt)
    check(f"功能3 防呆: {name}", f is None and "❌" in s and kw in s, s)

o, f, d, k, s = filt("-1 -1 -1\n-1 8 -1\n-1 -1 -1", norm="除以核總和 (Σk = 1)")
check("功能3 防呆: 核總和為0的正規化不崩潰", f is not None and "除以零" in s, s)
o, f, d, k, s = filt("1 2 3\n4 5 6\n7 8 9\n1 2 3\n4 5 6", ns=True)
check("功能3: 允許非方陣 5×3", f is not None and "5×3" in s, s)
big = "\n".join(" ".join(["1"] * 15) for _ in range(15))
o, f, d, k, s = filt(big, norm="除以核總和 (Σk = 1)")
check("功能3: 15×15 極端核可執行", f is not None, s)
o, f, d, k, s = filt("0 0 0\n0 0 0\n0 0 0")
check("功能3: 全零核警告", f is not None and "全為 0" in s, s)
o, f, d, k, s = filt(sharpen, scope="灰階（先轉成亮度 Y 再濾波）")
check("功能3: 灰階模式", f is not None and "PSNR" in s, s)
o, f, d, k, s = filt("-2 -1 0\n-1 1 1\n0 1 2", offset=128)
check("功能3: 浮雕+偏移128", f is not None, s)

# 預設核選單會自動帶入並執行
txt, norm, off = client.predict("浮雕 (Emboss 3×3)", api_name="/choose_preset")
check("功能3: 選預設核自動帶入偏移", off == 128 and "-2" in txt, f"{txt!r} {off}")

# 功能 4
FACE_ARGS = dict(nms=0.3, scale=1.1, neigh=5, minsize=30)
img, md, table = client.predict("YuNet（深度學習，輕量）", 0.7, 0.3, 1.1, 5, 30, api_name="/detect_faces")
check("功能4: 無人臉圖 → 0 張", "偵測到 0 張" in md, md)
img, md, table = client.predict("Haar Cascade（經典特徵）", 0.7, 0.3, 1.1, 5, 30, api_name="/detect_faces")
check("功能4: Haar 無人臉圖 → 0 張", "偵測到 0 張" in md, md)

try:
    from skimage import data
    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "face.png")
        a = data.astronaut()
        Image.fromarray(a).save(p)
        client.predict(handle_file(p), api_name="/load_image")
        for be in ["YuNet（深度學習，輕量）", "Haar Cascade（經典特徵）"]:
            img, md, table = client.predict(be, 0.7, 0.3, 1.1, 5, 30, api_name="/detect_faces")
            check(f"功能4: {be[:4]} 偵測到人臉並回傳信心度", "偵測到 0 張" not in md and len(table["data"]) >= 1, md)
            print("   ", md.replace("\n", " | "), table["data"])
except ImportError:
    print("SKIP 人臉影像測試（需 scikit-image）")

print()
print("全部通過" if not fails else f"失敗 {len(fails)} 項: {fails}")
sys.exit(1 if fails else 0)
