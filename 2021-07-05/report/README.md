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

![img0](./img/ObjectDetection.png)

### Object localization and classification

Object detection 運作步驟 :
1. 偵測目標位置(產生物件框)
2. 對目標物件進行分類

其演算法架構可分為 one-stage, two-stage.

* two-stage: 將步驟1, 2分開執行，輸入之影像先藉由物件偵測產生物件框後，再透過 classification 進行分類。performance 通常較好，若偵測出的物件過多，除非有很強的GPU平行運算，否則運算時間將會慢許多。

    e.g. RCNN

* one-stage: 輸入之影像透過神經網路同時進行物件偵測與辨識。運算速度較 two-stage 快，但 performance 相對沒有很好，不過後續研究結構的複雜化使其 performance 愈來愈好甚至超越 two-stage。

    e.g. YOLO

![img1](./img/stage.png)

### Comparison to Other Real-Time Systems

YOLO - FPS: 45, mAP: 63.4

於 Real-Time Detectors 雖然每秒幀數(FPS)表現普通，不過其對所有辨識種類的平均辨識率(mAP)為最高。

於 Less Than Real-Time 其mAP表現不遜色於其他，且FPS為最高。

| Comparison                    | Error Analysis                   |
| :---------------------------: | :------------------------------: |
| ![img2](./img/comparison.jpg) | ![img2](./img/ErrorAnalysis.jpg) |

#### IOU

IOU = 交集a / 聯集a，其值介於0~1之間。

一般判斷辨識率以IOU >= 0.5 為基準。

| bounding box            | IOU                    |
| :---------------------: | :--------------------: |
| ![img3](./img/bird.png) | ![img4](./img/IOU.png) |

#### AP、mAP

precision: 所有被系統預測為鴨子的結果中，真的是鴨子的比例。

recall: 所有真的鴨子，被系統預測正確的比例。

| predict                      | result                    |
| :--------------------------: | :-----------------------: |
| ![img5](./img/predict.png)   | ![img6](./img/result.png) |
| <b>precision</b>             | <b>recall</b>             |
| ![img7](./img/precision.png) | ![img8](./img/recall.png) |

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

![img9](./img/detections.png)

#### Confidence

Grid cell 包含目標的機率與IOU相乘。

Pr(Object) -> bounding box 裡可能是物件的 probabilities

Pr(Class | Object) -> 偵測為物件後，該物件所屬類別的 probabilities

![img9](./img/confidence.png)

### The Architecture

輸入尺寸調整至448\*448，以增加提取解析度。

神經網路參考GoogleNet，24層 Conv Layers、2層F.C。

不同的是 YOLO 使用 1\*1 卷積(降維)對 3\*3 卷積核運算做壓縮，以減少計算參數。取代 GoogleNet 的 Inception modules。

最後輸出 tensor 為 7 \* 7 \*(2 \* 5 \+ 20) = 7 \* 7 \* 30

![img10](./img/model1.png)

![img10](./img/model2.png)

![img10](./img/yolov1_output.png)

Bounding box 四個位置值為正規化數值 :

(x, y, w, h) = bbox(x, y, w, h) / 原影像(x, y, w, h)

C = 20，使用 PASCAL VOC 資料集，有20種類別。

#### Activation function

Activation function 採用 leaky rectified linear activation (leaky ReLU):

ReLU 會使部分神經元輸出為0，以解決 Overfitting，但有些神經元可能無法被激活(Dead ReLU Problem)，因此採用 Leaky ReLU 不增加計算複雜度，提升模型的學習能力。

ReLU 是將所有負值皆設為零；Leaky ReLU 則是將負值乘上非零斜率。

![img10](./img/leakyrelu.png)

除了輸出層使用 linear activation，其他皆使用 leaky ReLU。

### Training

前20層 Conv Layers 是以大型 dataset(ImageNet) 進行 pretrain(特徵提取)，因此不修正此處權重。

Pretrain 完成後，再接上隨機權重的4層 Conv Layers(分類器)、2層 F.C。

| Inference                         |
| :-------------------------------: |
| ![img11](./img/inference.jpg)     |
| Detection Procedure               |
| ![img11](./img/yolov1_detect.png) |

最後輸出層進行 detection procedure 時，以Grid 包含兩個 bbox 的 confidence 乘上 Pr(Class)，形成評估 bbox 的指數。

#### Loss Function

### NMS (Non-max suprresed)

<br>

## Conclusion

<br>

## Reference
* [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)
* [深度學習: 物件偵測上的模型結構變化](https://chih-sheng-huang821.medium.com/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-%E7%89%A9%E4%BB%B6%E5%81%B5%E6%B8%AC%E4%B8%8A%E7%9A%84%E6%A8%A1%E5%9E%8B%E7%B5%90%E6%A7%8B%E8%AE%8A%E5%8C%96-e23fd928ee59)
* [影像辨識常見的IOU、AP、mAP是什麼意思?](http://yy-programer.blogspot.com/2020/06/iouapmap.html)
* [YOLO v1 物件偵測~論文整理](https://medium.com/%E7%A8%8B%E5%BC%8F%E5%B7%A5%E4%BD%9C%E7%B4%A1/yolo-v1-%E7%89%A9%E4%BB%B6%E5%81%B5%E6%B8%AC-%E8%AB%96%E6%96%87%E6%95%B4%E7%90%86-935bfd51d5e0)
* [深度學習YOLO V1 深刻解讀YOLO V1(圖解)](https://blog.csdn.net/c20081052/article/details/80236015)
* [物件偵測 S4: YOLO v1 簡介](https://yuweichiu.github.io/%E4%BA%BA%E5%AD%B8%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92/p0005-Object-Detection-S4-YOLO-v1/)
* [常用啟用函式](https://www.itread01.com/content/1546354994.html)
* [圖解一階段物件偵測算法_Part01 - YOLOv1](https://www.youtube.com/watch?v=sq_OfIhb5Oc)