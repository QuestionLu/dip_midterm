---
title: DIP Midterm Image Tool
emoji: 🖼️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12.12"
app_file: app.py
pinned: false
---

# 互動式影像處理與人臉分析工具（數位影像處理期中實作）

以 Gradio 開發的 Web App，包含四大核心功能：

| # | 功能 | 實作位置 |
|---|------|----------|
| 1 | 影像載入（PNG/JPG/BMP）與中繼資料（解析度、色彩維度、檔案大小） | `dip/io_utils.py` |
| 2 | Original / RGB 分離 / YUV 分離檢視，並同步繪製 0–255 直方圖 | `dip/color.py`、`app.py` |
| 3 | 自訂卷積核（手刻卷積）、正規化、濾波前後比較、PSNR | `dip/convolution.py`、`dip/metrics.py` |
| 4 | 人臉偵測（YuNet 深度學習 / Haar Cascade），框出人臉、人數與信心度 | `dip/face.py` |

## 執行

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py            # 開啟 http://127.0.0.1:7860
```

需要 Python 3.10 以上（Gradio 6 的需求）。模型檔已放在 `models/`，不需另外下載。

## 測試

```bash
pip install -r requirements-dev.txt
pytest -v                              # 60 項核心演算法測試
python app.py &                        # 啟動後執行端對端測試（透過 API 實際操作四個功能與邊界案例）
python tests/e2e_client.py
python scripts/evaluate_psnr.py        # 產生 PSNR 評估圖表到 report_assets/
```

## 專案結構

```
app.py                 Gradio 介面與事件連結（含輸入防呆與錯誤提示）
dip/io_utils.py        影像載入、EXIF 轉正、透明度合成、中繼資料
dip/color.py           RGB 分離、BT.601 YUV 轉換、直方圖、統計
dip/convolution.py     卷積核解析/驗證/正規化、手刻 2D 卷積（平移相加法）
dip/metrics.py         MSE、PSNR
dip/face.py            YuNet 與 Haar 偵測、框與信心度標註
models/                YuNet ONNX 與 Haar XML
examples/              合成測試圖（可自行加入人臉照片）
tests/                 pytest 單元測試與端對端 API 測試
scripts/               範例圖產生、PSNR 評估實驗
report_assets/         評估實驗輸出的圖與數據
```

## 公式

- 卷積：`y[i,j] = Σ_u Σ_v k[u,v] · x[i−u, j−v]`（kernel 旋轉 180° 後與影像做相關）
- YUV（BT.601）：`Y = 0.299R + 0.587G + 0.114B`，`U = 0.492(B−Y)`，`V = 0.877(R−Y)`（顯示時 U、V 加 128）
- `MSE = 1/(MNC) · Σ (I − K)²`，`PSNR = 10·log₁₀(255² / MSE)`（dB；MSE=0 時為 ∞）

## 授權與來源

- YuNet 模型：OpenCV Zoo `face_detection_yunet_2023mar.onnx`（MIT）
- Haar 模型：OpenCV `haarcascade_frontalface_default.xml`
