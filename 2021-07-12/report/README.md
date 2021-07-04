# Algorithm architecture of YOLO v1: Real Time Object Detection.

You Only Look Once: Unified, Real-Time Object Detection

<br>

## Table of contents

* [Introduce](#introduce)
    * [Object Detection](#Object-Detection)
    * [Object localization and classification](#Object-localization-and-classification)
    * [Comparison to Other Real-Time Systems](#Comparison-to-Other-Real-Time-Systems)
        * [IOU](#IOU)
        * [AP、mAP](#APmAP)
* [Algorithm architecture](#Algorithm-architecture)
    * [Unified Detection](#Unified-Detection)
        * [The Model](#the-model)
        * [Confidence](#confidence)
    * [The Architecture](#The-Architecture)
        * [Activation function](#Activation-function)
    * [Training](#training)
        * [Loss Function](#loss-function)
    * [NMS (Non-max suprresed)](#NMS-Non-max-suprresed)
* [Conclusion](#Conclusion)
* [Reference](#reference)

<br>

## Introduce

### Object Detection

Algorithm of The YOLO Detection System 其流程主要分為三個步驟 :
1. 將影像大小調整至448\*448
2. 執行卷積神經網路進行物件偵測與分類
3. 透過 NMS (Non-max suprresed) 方式框出影像中物件之位置，輸出最終結果

![img1](./img/ObjectDetection.png)

### Object localization and classification

Object detection 運作步驟 :
1. 偵測目標位置(產生物件框)
2. 對目標物件進行分類

其演算法架構可分為 one-stage, two-stage.

* two-stage: 將步驟1, 2分開執行，輸入之影像先藉由物件偵測產生物件框後，再透過 classification 進行分類。performance 通常較好，若偵測出的物件過多，除非有很強的GPU平行運算，否則運算時間將會慢許多。

    e.g. RCNN

* one-stage: 輸入之影像透過神經網路同時進行物件偵測與辨識。運算速度較 two-stage 快，但 performance 相對沒有很好，不過後續研究結構的複雜化使其 performance 愈來愈好甚至超越 two-stage。

    e.g. YOLO

![img2](./img/stage.png)

### Comparison to Other Real-Time Systems

YOLO - FPS: 45, mAP: 63.4

於 Real-Time Detectors 雖然每秒幀數(FPS)表現普通，不過其對所有辨識種類的平均辨識率(mAP)為最高。

於 Less Than Real-Time 其mAP表現不遜色於其他，且FPS為最高。

| Comparison                      | Error Analysis                     |
| :-----------------------------: | :--------------------------------: |
| ![img3.1](./img/comparison.jpg) | ![img3.2](./img/ErrorAnalysis.jpg) |

#### IOU

IOU = 交集a / 聯集a，其值介於0~1之間。

一般判斷辨識率以IOU >= 0.5 為基準。

| bounding box              | IOU                      |
| :-----------------------: | :----------------------: |
| ![img4.1](./img/bird.png) | ![img4.2](./img/IOU.png) |

#### AP、mAP

precision: 所有被系統預測為鴨子的結果中，真的是鴨子的比例。

recall: 所有真的鴨子，被系統預測正確的比例。

| predict                        | result                      |
| :----------------------------: | :-------------------------: |
| ![img5.1](./img/predict.png)   | ![img5.2](./img/result.png) |
| <b>precision</b>               | <b>recall</b>               |
| ![img5.3](./img/precision.png) | ![img5.4](./img/recall.png) |

AP (average precision): 系統預測該類別時(鴨子)的平均辨識率。

mAP (mean average precision): 系統對於所有辨識種類(鴨子、貓、狗、人、車...等等)的平均辨識率。

<br>

## Algorithm architecture

### Unified Detection

#### The Model

YOLO會將影像分成 S\*S 格(grid)，每個 grid 有兩個 bounding box 做物件偵測，其一開始偵測到的物件有 7\*7\*2 = 98個，接著每個 grid 會辨識該物件框所框出之物件所屬的類別，最後採用 NMS 將多餘的 bounding box 濾除。

若 grid cell 包含<b>被偵測的物件中心</b>，此 grid cell 須負責偵測該物件。

其最後輸出 tensor 的維度 : S \* S \* (B \* 5 \+ C)

* S : 網格數量
* B : 每個 grid 預測物件的 bounding box 數 (YOLO v1 set B=2)
* 5 : 物件中心 (x, y)、寬高 (w, h)、confidence(是否為物件)
* C : 類別數量(兩個 bounding box 的類別機率)

![img6](./img/detections.png)

#### Confidence

Grid cell 包含目標的機率與IOU相乘。

Pr(Object) : bounding box 裡可能是物件的 probabilities

Pr(Class | Object) : 偵測為物件後，該物件所屬類別的 probabilities

![img7](./img/confidence.png)

### The Architecture

輸入尺寸調整至448\*448，以增加提取解析度。

神經網路參考GoogleNet，24層 Conv Layers、2層F.C。

不同的是 YOLO 使用 1\*1 卷積(降維)對 3\*3 卷積核運算做壓縮，以減少計算參數。取代 GoogleNet 的 Inception modules。

最後輸出 tensor 為 7 \* 7 \*(2 \* 5 \+ 20) = 7 \* 7 \* 30

![img8.1](./img/model1.png)

![img8.2](./img/model2.png)

![img8.3](./img/yolov1_output.png)

Bounding box 四個位置值為正規化數值 :

(x, y, w, h) = bbox(x, y, w, h) / 原影像(x, y, w, h)

C = 20，使用 PASCAL VOC 資料集，有20種類別。

#### Activation function

Activation function 採用 leaky rectified linear activation (leaky ReLU):

ReLU 會使部分神經元輸出為0，以解決 Overfitting，但有些神經元可能無法被激活(Dead ReLU Problem)，因此採用 Leaky ReLU 不增加計算複雜度，提升模型的學習能力。

ReLU 是將所有負值皆設為零；Leaky ReLU 則是將負值乘上非零斜率。

![img9](./img/leakyrelu.png)

除了輸出層使用 linear activation，其他皆使用 leaky ReLU。

### Training

前20層 Conv Layers 是以大型 dataset(ImageNet) 進行 pretrain(特徵提取)，因此不修正此處權重。

Pretrain 完成後，再接上隨機權重的4層 Conv Layers(分類器)、2層 F.C。

| Inference                           |
| :---------------------------------: |
| ![img10.1](./img/inference.jpg)     |
| Detection Procedure                 |
| ![img10.2](./img/yolov1_detect.png) |

最後輸出層進行 detection procedure 時，以Grid 包含兩個 bbox 的 confidence 乘上 Pr(Class)，形成評估 bbox 的指數。

#### Loss Function

採用平方誤差和 (sum-squared error) 做 loss function。

誤差有分類誤差(class error)、邊界框定位誤差(localization error)。

沒有物件的邊界框其 confidence 很低，會將最後指標推向幾乎等於0，導致誤差梯度過大，使整個損失函數被沒有物件的邊界框主導，造成損失不穩定且難以訓練好。

因此誤差除了分類與邊界框定位外，還將有無包含物件的邊界框分開計算，且給予不同權重。

![img11.2](./img/lossfunction.png)

![img11.2](./img/lossfunction2.png)

w、h 取平方根 : bbox 的大小對 bias 的影響比例不同，因此取平方根以降低 bias。

![img11.3](./img/bbox-bias.jpg)

### NMS (Non-max suprresed)

物件偵測時一個物件可能被很多物件框選中，因此採用 NMS 將多餘的物件框濾除。

1. 將 confidence 很低的 bbox 去除，並選出 confidence 最高的 bbox 加入"確定是物件集合" (selected objects)
2. 其他 bbox 與選出的 bbox 計算IOU，若 bbox 的IOU結果大於設定好之閾值，其 confidence 會設定為0

Repeat 1、2 步驟直到沒有 bbox 的 confidence > 0，selected objects 為最後結果，NMS結束。

![img12](./img/NMS.png)

<br>

## Conclusion

YOLO v1 的速度較 two-stage 模型快上好幾倍(45 fps)，且 mAP(63.4) 也比 R-CNN 好很多。

但其也有不少缺點 :

1. 每個格子只預測兩個框，且一個框只有一個分類，因此對於群體的小物件偵測能力不佳 (e.g. 一群鳥)。
2. 由訓練資料學習辨識與邊界框，對於新的、長寬比不常見之物件難以偵測。其他演算法 e.g. SSD (sol: data augmentation)。
3. 經過多個降維，在特徵解析度粗糙的 feature map 上預測邊界框，其泛化能力差(對新數據的適應能力)。
4. 於loss function上，邊界框定位誤差為影響預測效果的主因，bounding box 的大小在 loss 的反應上不佳，小的 bbox 對 IOU 影響較大。

<br>

## Reference
* [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)
* [深度學習: 物件偵測上的模型結構變化](https://chih-sheng-huang821.medium.com/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-%E7%89%A9%E4%BB%B6%E5%81%B5%E6%B8%AC%E4%B8%8A%E7%9A%84%E6%A8%A1%E5%9E%8B%E7%B5%90%E6%A7%8B%E8%AE%8A%E5%8C%96-e23fd928ee59)
* [影像辨識常見的IOU、AP、mAP是什麼意思?](http://yy-programer.blogspot.com/2020/06/iouapmap.html)
* [YOLO v1 物件偵測~論文整理](https://medium.com/%E7%A8%8B%E5%BC%8F%E5%B7%A5%E4%BD%9C%E7%B4%A1/yolo-v1-%E7%89%A9%E4%BB%B6%E5%81%B5%E6%B8%AC-%E8%AB%96%E6%96%87%E6%95%B4%E7%90%86-935bfd51d5e0)
* [深度學習YOLO V1 深刻解讀YOLO V1(圖解)](https://blog.csdn.net/c20081052/article/details/80236015)
* [物件偵測 S4: YOLO v1 簡介](https://yuweichiu.github.io/%E4%BA%BA%E5%AD%B8%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92/p0005-Object-Detection-S4-YOLO-v1/)
* [常用啟用函式](https://www.itread01.com/content/1546354994.html)
* [機器/深度學習: 物件偵測 Non-Maximum Suppression (NMS)](https://chih-sheng-huang821.medium.com/%E6%A9%9F%E5%99%A8-%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-%E7%89%A9%E4%BB%B6%E5%81%B5%E6%B8%AC-non-maximum-suppression-nms-aa70c45adffa)
* [圖解一階段物件偵測算法_Part01 - YOLOv1](https://www.youtube.com/watch?v=sq_OfIhb5Oc)