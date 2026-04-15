import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 建立用來存放網頁與圖片的 docs 資料夾
os.makedirs('docs', exist_ok=True)

print("正在獲取歷史金融數據...")
# 1. 定義標的與抓取資料
tickers = ["GC=F", "DX-Y.NYB", "^TNX", "^VIX"]
data = yf.download(tickers, start="2022-01-01")['Close']
data.columns = ['DXY', 'Gold', 'US10Y', 'VIX']

# 2. 資料清理：使用向前填補法 (Forward Fill) 處理缺失值
data = data.ffill()

print("資料下載完成，正在繪製圖表...")
# 3. 繪製圖表一：黃金與 VIX 恐慌指數
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(data.index, data['Gold'], color='gold', label='Gold Price (USD)')
ax1.set_ylabel('Gold Price (USD)', color='black')
ax1.tick_params(axis='y', labelcolor='black')

ax2 = ax1.twinx()
ax2.plot(data.index, data['VIX'], color='red', alpha=0.5, label='VIX Index')
ax2.set_ylabel('VIX Index', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Time Series: Gold Price vs VIX (Safe Haven Dynamics)')
fig.tight_layout()
plt.savefig('docs/gold_vix_trend.png')
plt.close()

# 4. 繪製圖表二：總經變數相關性熱力圖
plt.figure(figsize=(8, 6))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title('Correlation Matrix: Macro Variables')
plt.tight_layout()
plt.savefig('docs/correlation_heatmap.png')
plt.close()

print("圖表繪製完成，正在生成 HTML 網頁...")
# 5. 自動生成網頁 (HTML)
html_content = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>美國、以色列與伊朗戰爭分析 - 黃金避險資產</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; padding: 20px; max-width: 1000px; margin: 0 auto; color: #333; }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
        h2 { color: #2980b9; margin-top: 30px; }
        img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; padding: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .footer { margin-top: 50px; font-size: 0.9em; color: #7f8c8d; text-align: center; }
    </style>
</head>
<body>
    <h1>中東地緣政治衝突對黃金避險資產之影響分析</h1>
    <p>本網頁由 GitHub Actions 自動抓取最新 Yahoo Finance 數據並重新繪製。這展示了戰爭風險升溫時，黃金價格與傳統總經指標的動態關係。</p>

    <h2>一、 時間序列分析：黃金價格與市場恐慌指數 (VIX)</h2>
    <p>此圖表檢驗研究假設一：當重大地緣政治事件發生，VIX 指數飆升時，黃金作為避險資產是否出現顯著的價格推升。</p>
    <img src="gold_vix_trend.png" alt="Gold and VIX Trend">

    <h2>二、 相關性分析：黃金、美元指數與美債殖利率</h2>
    <p>此圖表檢驗研究假設二：在極端風險下，黃金與美元 (DXY)、美國十年期公債殖利率 (US10Y) 傳統的負相關性是否發生改變。</p>
    <img src="correlation_heatmap.png" alt="Correlation Heatmap">

    <div class="footer">
        <p>自動化資料更新時間：由 GitHub Actions 觸發執行</p>
    </div>
</body>
</html>
"""

with open('docs/index.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("專案執行完畢！檔案已存入 docs/ 資料夾。")
