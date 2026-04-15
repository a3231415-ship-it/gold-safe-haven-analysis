name: Auto-Run Data Analysis

# 設定觸發條件
on:
  push:
    branches:
      - main
  workflow_dispatch: # 允許在 GitHub 網頁上手動點擊執行

# 設定執行任務
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest

    # 給予 GitHub Actions 寫入權限，才能把畫好的圖推回儲存庫
    permissions:
      contents: write

    steps:
      # 1. 抓取你儲存庫裡的程式碼
      - name: Checkout repository
        uses: actions/checkout@v3

      # 2. 安裝 Python 環境
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      # 3. 安裝我們寫程式需要的套件 (包含 seaborn)
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install yfinance pandas matplotlib seaborn

      # 4. 執行你剛剛寫的那隻 Python 檔案
      - name: Run Analysis Script
        run: python src/analysis.py

      # 5. 將 Python 產生的圖片和網頁 (docs資料夾) 自動推回 GitHub
      - name: Commit and push changes
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action Bot"
          # 只把 docs 資料夾裡的異動加進去
          git add docs/
          # 如果有產生新檔案才 commit，避免沒有更新時報錯
          git diff --quiet && git diff --staged --quiet || (git commit -m "Auto-update charts and HTML" && git push)
