# Homework 1：MERT-v2 + MLP

以凍結的 `m-a-p/MERT-v2-30s` 擷取音訊特徵，分別訓練 Dataset A（年代）與 Dataset B（發行市場）的六分類 MLP。這是初始化 codebase：尚未下載音訊／模型權重，也未執行訓練、推論或測試，沒有訓練結果或 checkpoint。

## 目錄

```text
models.py                 MERT encoder、標準化與 MLP
train/                    訓練腳本目錄
  train.py                標準 Cross-Entropy 訓練
  train_contrastive.py    Supervised Contrastive (SupCon) + CE 訓練
train.py                  根目錄捷徑訓練腳本
test.py                   兩個任務的 test 推論與作業 JSON（不是單元測試）
dataset.py                gdown 下載、安全解壓縮、manifest 轉換與音訊讀取
features.py               依音訊內容與模型版本建立特徵快取
utils.py                  標籤、random seed、metrics、混淆矩陣
notebooks/01_workflow.ipynb 未執行的操作範例與結果檢視
examples/manifest.csv      欄位示意，並非真實樣本
pyproject.toml            uv 套件設定
uv.lock                   跨平台依賴鎖定
requirements.txt          由 uv 匯出，供助教 pip 安裝
.env.example              自行填入 Hugging Face token
```

## 安裝

需要 Python 3.11／3.12；建議使用 3.11。固定採用模型官方示範版本 Torch 2.6.0、torchaudio 2.6.0、Transformers 4.53.2。環境已用 uv 建立於 `.venv/`。

```bash
# 如 uv 尚未在 PATH，安裝：https://docs.astral.sh/uv/getting-started/installation/
uv sync --frozen
cp .env.example .env
# 編輯 .env，把 HF_TOKEN 填入自己的 token
```

本次初始化所用 uv 位於工作區 `work/bin/uv`。在本機新終端機，可先執行：

```bash
export PATH="/Users/cengchenghao/Documents/Codex/2026-09-19/files-mentioned-by-the-user-homework/work/bin:$PATH"
cd /Users/cengchenghao/Documents/Codex/2026-09-19/files-mentioned-by-the-user-homework/outputs/mert-mlp
```

供助教使用的替代安裝方式：`pip install -r requirements.txt`，後續將 `uv run python` 改成 `python`。更新依賴後重新匯出：

```bash
uv export --frozen --no-dev --no-hashes --no-emit-project --output-file requirements.txt
```

預設 `--device auto` 選 CUDA，沒有 CUDA 時使用 CPU。macOS 使用 CPU；沒有預設啟用 MPS。MERT 約 632M 參數，特徵擷取比 MLP 訓練耗費資源，預設 extraction batch size 為 1。顯存需求未實測；若 CUDA 記憶體不足，使用 CPU 或先減少音訊長度。改變長度需重新訓練以保持前處理一致。

## 下載資料與建立 manifests

```bash
uv run python dataset.py download --output data/raw
```

此指令以 gdown 下載使用者指定的 [Drive 資料夾](https://drive.google.com/drive/folders/1C8RymiLbr-EGmkxh2Ap5TIybnJYqNsb4)，解壓縮兩個 ZIP，保留原始檔。公開目錄目前列出 `dataset_A.zip`、`dataset_B.zip`、`prediction_format_example_NOT_ANSWERS.json`；ZIP 合計約 3 GB。配額或權限問題會直接回報失敗，不會略過遺漏檔案。

**資料結構已完成核對：**
下載解壓縮後，官方檔案包含：
- `data/raw/dataset_A/manifest.csv` 與 `data/raw/dataset_A/audio/*.wav` (A 共有 1026 train / 132 validation / 132 test)
- `data/raw/dataset_B/manifest.csv` 與 `data/raw/dataset_B/audio/*.wav` (B 共有 798 train / 102 validation / 102 test)
- 欄位包含 `sample_id`, `split`, `label`, `audio_path`, `duration_seconds`, `sample_rate`, `sha256`。

可執行以下一鍵指令，自動從 raw 資料產生繳交所需的六個標準 split CSV：

```bash
uv run python dataset.py prepare-all
```

產生清單：
```text
data/manifests/A_train.csv
data/manifests/A_validation.csv
data/manifests/A_test.csv
data/manifests/B_train.csv
data/manifests/B_validation.csv
data/manifests/B_test.csv
```

每個 manifest 的格式：
```csv
sample_id,path,label
<官方ID>,<相對於data-root的WAV路徑>,<官方標籤>
```

- A 標籤：`1960s, 1970s, 1980s, 1990s, 2000s, 2010s`。
- B 標籤：`US, UK, Brazil, Spain, Germany, Italy`。
- 數量核對完全符合：A 為 1026 / 132 / 132；B 為 798 / 102 / 102。

## 架構與前處理

```text
WAV → mono → 24 kHz → 中央最多30秒
    → frozen MERT-v2 最後一層 → masked temporal mean pooling
    → training-set standardization → L2 normalization
    → Linear(1024,256) → GELU → Dropout(0.3) → Linear(256,6)
```

聲道取平均；只在原取樣率不同時 resample。作業已提供中央 30 秒，因此預設不再隨機裁切。沒有額外音訊 augmentation。前處理經官方 feature extractor；輸出 frame mask 用於 pooling，排除 padding。使用 logits 配合 cross-entropy，分類排名等同 softmax 機率排名。

特徵快取 key 包含音訊 SHA256、Hugging Face commit、長度、取樣率與前處理版本。首次 train.py 才會從 Hugging Face 下載權重。checkpoint 固定儲存解析後的 commit；test.py 使用同一 commit 重新載入凍結 encoder。模型需要 `trust_remote_code=True`，載入指定 repo 的自訂模型程式。

每個任務的 mean/std **只在 train features 估計**，並存成 MLP buffer。Validation 只用於 early stopping／選擇最佳 validation loss，不做梯度更新。Test 完全不參與 model selection。此版本只訓練 MLP，不做 MERT fine-tuning。

## 執行訓練

訓練腳本已自動支援官方 raw manifests 與標準化 manifests。以下兩種指令皆可執行：

### 方式一：直接使用官方 raw manifest（最簡潔，自動切分 train/validation）
```bash
uv run python train.py --task A --output-dir runs/A
uv run python train.py --task B --output-dir runs/B
```
*(腳本會自動讀取 `data/raw/dataset_A/manifest.csv` 與 `data/raw/dataset_B/manifest.csv` 並依 `split` 欄位篩選)*

### 方式二：使用個別 manifest（相容官方標準切分檔）
```bash
uv run python train/train.py --task A --data-root data/raw \
  --train-manifest data/manifests/A_train.csv \
  --val-manifest data/manifests/A_validation.csv --output-dir runs/A

uv run python train/train.py --task B --data-root data/raw \
  --train-manifest data/manifests/B_train.csv \
  --val-manifest data/manifests/B_validation.csv --output-dir runs/B
```

### 方式三：使用對比學習訓練（Supervised Contrastive Learning, SupCon）
```bash
uv run python train/train_contrastive.py --task A --output-dir runs/A_contrastive \
  --contrastive-weight 0.5 --temperature 0.1

uv run python train/train_contrastive.py --task B --output-dir runs/B_contrastive \
  --contrastive-weight 0.5 --temperature 0.1
```
*(以 SupCon 損失將同一年代/類別的歌曲在超球面上拉近、不同類別推遠。儲存的 `best.pt` Checkpoint 完全相容於 `test.py` 推論)*

預設 AdamW、lr=0.001、weight decay=0.0001、batch size=64、最多100 epochs、patience=15、seed=42。輸出 `best.pt`、`config.json`、`history.json`、`validation_metrics.json`、`validation_confusion.png`。混淆矩陣是 counts，列為真實類別、欄為預測類別；Top-1／Top-3 是 0 到 1 的比例。固定 seed 仍可能因裝置／底層運算差異產生數值差異。

已存在 `best.pt` 的 run 目錄不會被覆寫；請另選 `--output-dir`。目前不提供 optimizer resume。

## 未來執行推論與提交

```bash
uv run python test.py --data-root /path/on/grader/device \
  --manifest-a data/manifests/A_test.csv \
  --manifest-b data/manifests/B_test.csv \
  --checkpoint-a runs/A/best.pt --checkpoint-b runs/B/best.pt \
  --template data/raw/prediction_format_example_NOT_ANSWERS.json \
  --output predictions/STUDENT_ID.json
```

如範例檔實際位於子目錄，調整 `--template` 路徑。它僅檢查 A/B 的 sample ID 集合完全一致，**不讀取範例答案來預測**；建議提交時保留此參數。輸出格式完全依作業第18頁：

```json
{
  "dataset_A": {"<A_sample_id>": ["2010s", "2000s", "1990s"]},
  "dataset_B": {"<B_sample_id>": ["UK", "US", "Germany"]}
}
```

上面僅為 schema 示意。實際標籤由模型分數排序，每個 manifest ID 恰好一次、各三個不重複標籤。

上傳 source、兩個 `best.pt`、`requirements.txt`、README、manifests；不要上傳 `.env`、`.venv`、released dataset、模型 cache 或特徵 cache。這個凍結版本的 checkpoint 只含 MLP 與標準化參數，重現需要連線 Hugging Face 下載所記錄 commit 的 MERT，或事先備好對應 cache；HF token 由執行者提供。

## Notebook 與作業剩餘工作

```bash
uv sync --frozen --group notebooks
uv run --group notebooks jupyter lab
```

Notebook 保留未執行狀態，不會自動啟動訓練。作業另有 audio language model／prompt 比較、報告 PDF、錯誤分析與最終提交等要求；本次依你的範圍只初始化 MERT+MLP codebase，沒有實作 audio language model 實驗，也沒有產生或聲稱完成實驗結果。

## 參考來源

- 使用者提供的 Homework 1.pdf：任務、split 數量、標籤、評估與提交格式。
- [MERT-v2-30s model card](https://huggingface.co/m-a-p/MERT-v2-30s)：載入方式與輸出介面。權重授權 CC BY-NC 4.0。
- [MERT 官方程式庫](https://github.com/yizhilll/MERT)：MERT 研究背景。
- [gdown](https://github.com/wkentaro/gdown)：公開 Google Drive 下載。
- [uv 文件](https://docs.astral.sh/uv/)：環境與套件管理。
