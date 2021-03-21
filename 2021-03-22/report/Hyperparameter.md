# Hyperparameter
## Parameter
可以從 training data 取得，model 透過數據<b>自動學習</b>出參數。其為預測模型的一部分。

* weight、bias

<br>

## Hyperparameter
訓練 model 前<b>手動設置的</b>，其值無法從數據估計得到。其為使訓練模型時表現得更出色。

* learning rate、batch size、layers、hidden units

<br>

### 如何找出超參數的最佳數值

1. 經驗法則 : 對於模型、資料有一定熟悉度，但是並不適用全部的情況。

2. 反覆測試 : 給定範圍值，給電腦自動調校，較耗時但是能夠得到最佳數值。

<br>

### Hyperparameter 調校順序
#### Machine Learning
機器學習，將每個參數等距以格子法選取任意個數的點 (圖一)。

然後，分別使用不同點對應的參數組合進行系統化訓練，最後根據驗證集上的表現好壞，來決定最佳參數。

<img src='img/ML-Hyperparameter.png'>   圖一

<hr>

#### Deep Learning
深度神經網絡模型中，比較好的做法是隨機選擇 (圖二)。

原因為:
1. 我們無法預先知道那些參數對網絡有較大的影響。
2. 再者，隨機選取的方式，則可確保我們儘量測試到更多可能的參數。

<img src='img/DL-Hyperparameter01.png'> 圖二

通常使用 由粗到細的採樣(coarse to fine sampling scheme)，經過隨機採樣之後，我們可能得到某些區域模型的表現較好 (圖三左)

再對此區域做更密集的隨機採樣 (圖三右)

<img src='img/DL-Hyperparameter02.png'> 圖三
