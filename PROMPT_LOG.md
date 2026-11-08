# AI 協作紀錄（AI Audit Trail）— 進行中

- **AI 模型**：Claude Sonnet 5.5（`claude-sonnet-5-5`），於 Claude 的單一連續對話中完成，未中途換對話。
- 日期：2026-11-08 起
- 完整對話可由 Claude 介面匯出；此檔為整理後的時間序摘要，報告 Chapter 4 以此為基礎擴寫。

## 階段 0：規劃（2026-11-08 16:22）
**原始 Prompt**：貼上整份期中試題（四大功能、PDF 繳交規範、Rubric 圖片），詢問「該從哪裡開始」，並詢問用 Windows 桌機或 Linux 筆電開發。

**AI 產出**：
- 以 Rubric 權重（功能 35／部署 20／Prompt Log 20／報告 15／UX 10）拆解開發順序。
- 技術選型：Gradio + Hugging Face Spaces + GitHub。
- 指出易失分點：卷積須手刻並對照 `cv2.filter2D`、卷積與相關的差別（kernel 翻轉）、正規化除以零、非方陣/偶數核防呆、PSNR 在 MSE=0 時為 ∞、YUV 顯示的 128 偏移、Haar 無原生信心度。
- 硬體建議：Windows 桌機即可（純 CPU、重點是跨平台寫法）。

## 階段 1：補充限制條件（16:27）
**Prompt**：截止為明天 24:00；桌機沒有鏡頭（讀圖片即可）；附上課程章節清單截圖，詢問需要哪些教材。

**AI 決策**：不需要鏡頭；不等教材先開工；人臉偵測以 YuNet 為主、Haar 為輔（MediaPipe 版本相容性風險高，不採用）。

## 階段 2：實作與測試（AI 自主執行）
| 步驟 | 內容 | 結果 / 除錯 |
|---|---|---|
| 2.1 | 建立 `dip/` 模組（io、color、convolution、metrics、face） | 首次 pytest 60 項全過 |
| 2.2 | 以 `gr.Image(type="filepath")` 做上傳 | **問題**：端對端測試上傳損毀檔時，Gradio 在進入我們的程式前就拋出 `ComponentProcessingError`，無法顯示友善訊息。**修正**：改用 `gr.File`（限定副檔名）+ 獨立預覽圖，由 `load_image()` 自行驗證並回傳中文錯誤訊息 |
| 2.3 | 事件綁定使用 `show_api=False` | **問題**：Gradio 6 的事件監聽器不支援此參數 → `TypeError`。**修正**：改用 `api_name` |
| 2.4 | 端對端測試腳本 | **問題**：`gr.State` 不會出現在 API 回傳值，解包數量錯誤。**修正**：調整測試對回傳值的解讀 |
| 2.5 | 環境／模型取得 | YuNet 模型從 `raw.githubusercontent` 取得只拿到 Git LFS 指標檔（378 bytes）→ 改由 `media.githubusercontent` 取得完整 232 KB ONNX，並驗證可載入 |
| 2.6 | 介面截圖檢視 | 版面與功能正常 |
| 2.7 | PSNR 評估實驗 | 實驗 A（各預設核 PSNR）、實驗 B（以乾淨影像為基準的去雜訊，高斯 σ 掃描）；修正圖例遮住基準線 |

## 階段 3：使用者在本機（Windows）執行後回報的問題
| 時間 | 使用者回報 | AI 診斷與修正 |
|---|---|---|
| 11/9 00:00 | PowerShell 執行 `.venv\Scripts\activate` 出現「已停用指令碼執行」 | 這是 PowerShell ExecutionPolicy 限制，與程式無關。建議直接用 `.venv\Scripts\python` 呼叫，或僅對當前視窗暫時放行 |
| 11/9 00:02 | `pytest` 結果 54 passed、6 failed，皆為人臉偵測：`Can't read ONNX file: E:\高科大\數位影像處理\...\models\...onnx`、`Can't open file ... haarcascade ...xml` | **根因**：專案路徑含中文，Windows 版 OpenCV（C++ 端）無法開啟非 ASCII 路徑；先前在 Linux 沙盒測試無法重現。**修正**：`dip/face.py` 改由 Python 讀檔成位元組，YuNet 用 `FaceDetectorYN.create("onnx", buffer, ...)`、Haar 用 `FileStorage(..., FILE_STORAGE_MEMORY)` 從記憶體載入，與資料夾名稱無關。修正後沙盒 60 項測試全過，偵測結果與修正前一致 |

| 11/9 00:24 | 使用者提供「色彩分頁」截圖：RGB / YUV 模式只顯示一張大圖＋下方縮圖，而不是三個通道並排 | **根因**：`gr.Gallery` 在單張（Original）模式會進入「預覽」狀態，切換到三張時仍維持預覽狀態；沙盒截圖流程不同所以未發現。**修正**：改為 3 個固定的 `gr.Image` 並排，依模式更新內容與顯示/隱藏；同步修改端對端測試。另將直方圖標題移出元件標籤，避免遮住圖內標題 |

**反思**：開發環境（Linux、純 ASCII 路徑）與使用者環境（Windows、中文路徑）不同，導致測試無法涵蓋此類問題。之後的驗收以使用者端實際執行結果為準。

## 階段 4：（待續）
- 在本機執行、放入自己的測試照片、部署到 HF Spaces
- 截圖、撰寫報告 Chapter 1–4
- 之後的除錯對話請持續接在同一個對話中，並補在此檔
