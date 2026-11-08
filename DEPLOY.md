# 部署步驟（GitHub + Hugging Face Spaces）

## 1. 本機先跑起來
```bash
cd dip_midterm
python -m venv .venv
.venv\Scripts\activate          # Windows；macOS/Linux 用 source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                       # 應該 60 passed
python app.py                   # 開瀏覽器 http://127.0.0.1:7860
```

## 2. 放進自己的測試照片
把照片放到 `examples/`（建議：0 張臉、1 張臉、多人合照、側臉/遮擋各一張）。
`app.py` 會自動把 `examples/` 內的 png/jpg/bmp 列為範例。**請用自己或已授權的照片。**

## 3. 推上 GitHub
```bash
git init
git add .
git commit -m "DIP midterm: interactive image tool"
git branch -M main
git remote add origin https://github.com/<你的帳號>/<repo名稱>.git
git push -u origin main
```
Repo 設為 **Public**。

## 4. 部署到 Hugging Face Spaces
1. 到 https://huggingface.co/new-space ，SDK 選 **Gradio**，Visibility 選 **Public**。
2. 方法 A（網頁）：把專案所有檔案拖進 Space 的 Files 頁面。
   方法 B（git）：
   ```bash
   git remote add space https://huggingface.co/spaces/<帳號>/<space名稱>
   git push space main
   ```
   （密碼處貼 HF 的 Access Token，需有 write 權限）
3. `README.md` 最上方的 YAML 區塊就是 Space 設定，**不要刪**。
4. 等 Build 完成（約 2–5 分鐘），開啟連結實測四個功能。
5. 無痕視窗開啟連結，確認未登入也看得到（Public）。

## 5. 繳交前檢查
- [ ] Space 連結未登入可開啟、四個功能都能操作
- [ ] GitHub 連結為 Public，clone 後 `pip install -r requirements.txt && python app.py` 能直接跑
- [ ] 報告 PDF 命名 `學號_姓名_期中專題.pdf`
