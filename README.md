# 無人機搜救專案：三個 Python 檔案逐塊解說

這三個檔案組成「先找候選位置，再確認是不是人，最後輸出標記圖片」的兩階段偵測流程：

```text
stage1_train.py：訓練候選區域模型
        ↓
stage2_train.py：訓練人／非人分類模型
        ↓
cascade_infer.py：載入兩個模型，對新圖片推論
```

## 1. `stage1_train.py`：訓練第一階段候選區域模型

這個模型在 960×540 影像上尋找「可能有目標」的格子。輸出是一張 17×30 的分數圖，每格的分數越高，越值得交給第二階段檢查。

### 1.1 檔案說明與匯入

檔案最上方的說明文字交代資料格式、影像尺寸和相依套件。`argparse` 負責讀命令列參數；`random`、`numpy`、`torch` 用於隨機抽樣、陣列處理和模型訓練；`Path` 處理路徑；OpenCV 與 Pillow 負責影像讀取和轉換；`Dataset`、`DataLoader`、`Subset` 用來建立與切分資料；`torchvision.transforms.functional` 將圖片轉成 Tensor。

`from __future__ import annotations` 讓型別註記延後解析，不改變模型訓練流程。

### 1.2 固定尺寸設定

```python
IMAGE_H, IMAGE_W = 540, 960
GRID_H, GRID_W = 17, 30
CELL_SIZE = 32
```

模型畫布高為 544 像素：原圖高 540，底部補到 544，剛好切成 17 列、每格 32 像素；寬 960 則切成 30 格。

### 1.3 `enhance_lab_dog()`：影像增強

這個函式要求輸入是三通道 `uint8` RGB 圖片，接著：

1. 把 RGB 轉成 Lab 色彩空間，分開處理亮度與色彩。
2. 使用 CLAHE 增強不同區域的亮度對比。
3. 對亮度做兩種程度的模糊，取兩者差異取得細微邊緣反應。
4. 把部分邊緣反應加回亮度，並把像素值限制在 0 到 255。
5. 合併 Lab 通道並轉回 RGB。

如此能強化對比與邊緣，且維持 RGB 三通道輸入。

### 1.4 `prepare_stage1_tensor()`：準備模型輸入

函式先確認圖片尺寸為 960×540，接著套用影像增強。它建立 960×544 畫布，並以灰色像素值 114 填滿底部多出的 4 列。最後用 `TF.to_tensor()` 轉成模型使用的浮點 Tensor，形狀是 `[3, 544, 960]`，數值範圍為 0 到 1。

### 1.5 `Stage1Dataset`：讀取影像和格子標籤

資料夾預期為：

```text
stage1/{train 或 test}/images
stage1/{train 或 test}/grid
```

初始化時檢查資料夾存在、找到影像，並確認每張圖片都有同檔名的 `.npy` 標籤。

每次 `__getitem__()` 取資料時，程式會讀取 RGB 圖片並檢查尺寸；呼叫前處理函式；讀取並確認標籤是 17×30；將大於 0 的標籤轉為 1、其他轉為 0，增加通道維度成為 `[1, 17, 30]`。若啟用資料增強，則以 50% 機率同時水平翻轉圖片和標籤，確保目標格子仍對應正確位置。

### 1.6 模型區塊：`SqueezeExcitation`、`DWBlock`、`CandidateNet`

`SqueezeExcitation` 是通道注意力模組：先對各特徵通道做全域平均，再經過兩層 1×1 卷積和 Sigmoid 產生通道權重，最後用權重調整輸入特徵。

`DWBlock` 使用深度可分離卷積。第一層逐通道擷取空間特徵，第二層 1×1 卷積混合通道資訊；中間加入 BatchNorm 和 ReLU6。這種結構通常比一般卷積計算量小。

`CandidateNet` 將卷積層、數個 `DWBlock` 和通道注意力串接起來。步幅為 2 的卷積會逐步縮小特徵圖，最後 1×1 卷積把 64 個特徵通道轉成 1 個 logits 通道，輸出形狀為 `[batch, 1, 17, 30]`。logits 尚未轉成機率。

### 1.7 `focal_loss()`：訓練損失

函式先用 Sigmoid 把 logits 轉成機率，分別計算正格子和負格子的 focal loss。`alpha` 調整正負類別權重；`gamma` 降低容易判斷樣本的影響，讓訓練多關注較難判斷的格子。最後依正樣本數正規化，降低大量背景格對 loss 的支配。

### 1.8 `evaluate()`：驗證與測試

`@torch.no_grad()` 停用梯度計算，並將模型切換成評估模式。程式累計 focal loss，使用 `sigmoid(logits) >= threshold` 判定預測為正的格子，計算真實正格子中有多少被找到。回傳平均 loss 和 **cell recall**；這個指標重視找回目標格子，不會因為大量背景格預測正確而虛高。

### 1.9 `main()`：訓練流程

1. 讀取命令列參數，例如資料路徑、輸出路徑、epochs、batch size、learning rate、驗證比例與門檻。
2. 確認驗證比例介於 0 和 1 之間。
3. 設定 Python、NumPy、PyTorch 的隨機種子，選擇 CUDA GPU 或 CPU。
4. 載入 train 資料並切出訓練集、驗證集；驗證集使用不做翻轉增強的資料集版本。test 資料保留作最後評估。
5. 建立 DataLoader，將資料分批讀入；GPU 模式才啟用 `pin_memory`。
6. 建立模型、AdamW optimizer、CosineAnnealingLR scheduler 和 GradScaler。GradScaler 在 GPU 上配合混合精度訓練。
7. 每個 epoch 對每批資料清除舊梯度、計算 loss、反向傳播、更新參數，再調整學習率。
8. 每個 epoch 結束後計算驗證 loss 和 recall；recall 創新高時，將模型權重與輸入尺寸、格子尺寸、門檻、epoch 等資訊存進 checkpoint。
9. 訓練完成後重新載入最佳 checkpoint，在 test 資料上評估並輸出結果。

檔尾的 `if __name__ == "__main__":` 表示只有直接執行此檔時才會呼叫 `main()`；被其他檔案匯入時不會自動開始訓練。

## 2. `stage2_train.py`：訓練人／非人分類模型

第一階段找出可能區域後，第二階段查看每個裁切小圖，判斷它屬於 `person` 還是 `not_person`。

### 2.1 匯入與資料格式

程式從 `stage1_train.py` 匯入 `enhance_lab_dog()`，讓兩階段使用相同影像增強方式。其他重要匯入包括 ResNet-18 和 PyTorch 資料載入、訓練工具。

資料夾預期為：

```text
stage2/{train 或 test}/{person 或 not_person}/影像檔
```

### 2.2 `Stage2Dataset`：讀取分類圖片

初始化時用 `ImageFolder` 掃描類別資料夾，並確認 `person` 和 `not_person` 兩類都存在。`ImageFolder` 依資料夾名稱建立類別索引。

每次取資料時，程式讀取 RGB 圖、套用共用影像增強；訓練時以 50% 機率水平翻轉；再轉成 Tensor，並回傳影像和整數標籤。

此程式沒有強制縮放圖片；雖然模型說明與 checkpoint 註記預期 crop 尺寸為 64×64，實際資料尺寸須由資料準備流程確保。ResNet-18 本身可接受不同空間尺寸，但訓練資料與推論資料的處理方式應一致。

### 2.3 `Stage2Classifier`：ResNet-18 分類器

程式建立沒有預訓練權重的 ResNet-18，將最後的全連接層換成輸出兩類分數的線性層。輸出是 logits；訓練時交給 CrossEntropyLoss，推論時再用 Softmax 轉成機率。

### 2.4 `stratified_split()`：分層切分驗證集

函式按類別把資料索引分組，分別打亂並切出驗證資料，再合併索引。這樣驗證集通常能保留每個類別的樣本。函式回傳訓練索引和驗證索引。

### 2.5 `evaluate()`：分類評估指標

評估函式計算 CrossEntropyLoss、整體 accuracy，以及以 person 為正類的 precision、recall 和 F1：

- **TP**：實際為 person，也預測為 person。
- **FP**：實際不是 person，卻預測為 person。
- **FN**：實際為 person，卻預測成其他類別。
- **Precision**：預測為 person 的樣本中，實際為 person 的比例。
- **Recall**：實際為 person 的樣本中，成功找出的比例。
- **F1**：precision 和 recall 的調和平均。

`person_idx` 從資料集類別索引取得，所以程式不必假設 person 一定是第 0 類或第 1 類。

### 2.6 `main()`：分類模型訓練流程

1. 讀取訓練參數並檢查驗證比例。
2. 固定隨機種子，選擇 GPU 或 CPU。
3. 載入 train 資料，按類別分層切分訓練集與驗證集；載入獨立 test 資料。
4. 根據訓練集各類別數量計算 class weights，讓較少的類別在 loss 中有較大的權重。
5. 建立 DataLoader、ResNet-18、加權 CrossEntropyLoss、AdamW、CosineAnnealingLR 和 GradScaler。
6. 逐 epoch 訓練，並計算驗證 accuracy、person precision、recall 和 F1。
7. 驗證 F1 提高時儲存 checkpoint，包含模型權重、類別索引、person 索引、epoch 和 F1。
8. 訓練完後載入最佳 checkpoint，在 test 集評估並列印結果。

保存 `class_to_idx` 和 `person_idx` 很重要：推論時必須知道模型輸出的哪個位置代表 person。

## 3. `cascade_infer.py`：串接兩階段模型做推論

這個檔案不訓練模型，而是載入兩個訓練好的 checkpoint，對新影像先找候選，再分類候選，最後把判定結果畫回圖片。

### 3.1 匯入與尺寸常數

從 `stage1_train.py` 匯入候選模型和第一階段前處理；從 `stage2_train.py` 匯入分類模型。

- `STAGE2_CROP_SOURCE = 96`：1920×1080 原始影像上的候選裁切基準為 96×96。
- `STAGE2_INPUT = 64`：候選 crop 送入分類模型前縮成 64×64。
- `CELL_SIZE = 32`：第一階段格子的邊長。

### 3.2 `crop_with_padding()`：邊界安全裁切

函式依中心點和寬高計算裁切範圍。若範圍超出圖片邊界，就用 `BORDER_REFLECT_101` 反射邊緣補足，最後回傳固定大小 crop。這讓靠近影像邊緣的候選也能被分類。

### 3.3 `merged_candidates()`：從高分格產生候選位置

1. 保留 Stage 1 分數高於 threshold 的格子。
2. 依分數由高到低排序，以格子距離抑制相鄰重複候選，最多保留 `max-proposals` 個。
3. 直接用高分格的位置當裁切中心，避免大型連通熱區的質心偏離高分格或人員位置。

每個候選保留自己的 Stage 1 格子分數；輸出圖以黃色框顯示 Stage 1 候選，並標出 Stage 1 與 Stage 2 分數。Stage 2 判定為人員的候選會再以綠色框標示。

### 3.4 `main()`：完整推論流程

1. 讀取輸入圖片、兩個模型 checkpoint、輸出位置、兩階段門檻、候選間距、候選數上限，以及是否保存候選 crop 等參數。
2. 用 OpenCV 讀圖（BGR），轉成 RGB；記錄原始尺寸，再縮放成 960×540 供第一階段使用。
3. 載入 stage1 checkpoint，建立 `CandidateNet` 並切換到評估模式；再載入 stage2 checkpoint，建立分類器並讀出 person 類別索引。
4. 將 stage1 圖片轉成 Tensor，模型輸出經 Sigmoid 轉成機率圖，再交給 `merged_candidates()` 產生候選位置。
5. 依原圖尺寸換算 crop 大小。1920×1080 圖片使用 96×96；其他尺寸依比例調整。
6. 對每個候選：
   - 把候選格中心換算成 960×540 座標，再映射回原圖座標。
   - 從原始 RGB 圖裁切，必要時補邊。
   - 縮放成 64×64，套用同一影像增強，轉成 Tensor。
   - 用 stage2 分類，Softmax 後讀取 person 類別機率。
   - 若指定 `--save-crops`，保存候選 crop。
   - 輸出圖先以黃色框標出候選與兩階段分數；person 機率低於門檻就跳過，高於門檻則增加人數並以綠色框標示。
7. 建立輸出資料夾，把標記完成的 RGB 圖轉回 BGR 後儲存，並輸出候選數、人數及檔案路徑。

同樣地，檔尾的 `if __name__ == "__main__":` 確保只有直接執行此檔時才啟動推論。

## 閱讀與加註解時可留意

- Stage 1 threshold 決定哪些格子形成候選；person threshold 決定候選最後是否判為人。
- Stage 1 前處理把影像由 960×540 補成 960×544，模型輸出 17×30 格子分數圖。
- Stage 2 的分類索引由 `ImageFolder` 建立，推論需從 checkpoint 使用 `person_idx` 找到 person 分數。
- `stage1_train.py` 預設輸出 `runs/stage1/best_enhanced.pt`；`cascade_infer.py` 的 stage1 預設 checkpoint 也是這個路徑。Stage 2 預設輸出為 `runs/stage2/best_stage2.pt`。
- 加註解時優先寫清楚區塊用途、資料形狀、函式輸入輸出、座標換算與評估指標；避免逐字重述每行程式。
