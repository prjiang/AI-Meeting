# 神經網路優化技巧

## 對資料進行最佳化

加快模型收斂速度，使神經網路的訓練更快。

### Data normalization

Data normalization 加快神經網路的訓練。

* normalization = (data - mean) / std
* data -> mean = 0, var = 1
* data 經過 sigmoid、tanh 等啟動函數後，得到的導數較大
* 導數愈大，調整幅度大，愈快逼近目標；導數愈小，調整幅度愈小

Data normalization :

```python
# Dataset
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))     # transforms.Normalize((mean), (std))
])
```

計算 mean、std :

```python
loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                        batch_size=len(train_dataset)
                                    )
data = next(iter(loader))
data_mean = data[0].mean()      # tensor(0.1307)
data_std = data[0].std()        # tensor(0.3081)
data_mean, data_std
```

### Data augmentation

增加資料的多樣性，使資料分布更均勻。

* PyTorch 中有預設
* 常用 Augementor、imgaug 等套件

<br>

## 對超參數的最佳化

| Hyperparameters                       | Description                   |
| :------------------------------------ | :---------------------------- |
| batch_size (批次)                     | 影響梯度值 (收斂速度)         |
| epoch (輪數)                          | 使準確率上升                  |
| learning rate                         | 影響收斂速度 (梯度下降的步長) |
| torch.optim.lr_scheduler (動態學習率) | 隨著輪數改變學習率            |

* batch_size 愈大，隨機梯度值會愈平均，使收斂速度愈快。若批次過大，GPU 的記憶體不夠大，會造成執行錯誤。
* epoch 增加，準確率停止增加，表示收斂結束，epoch 需停止。
* lr 過大，會無法收斂至最小值 (跳過最小值的點)；lr 過小，造成無法收斂 (梯度下降緩慢)。
* 動態lr 固定輪數調整lr。梯度下降快，lr 大；梯度平緩，lr 小。

`torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)`

* `optimizer:` 模型的優化器
* `step_size:` 每隔幾個 epoch 更新一次
* `gamma:` 每次更新乘上該值，預設0.1

<br>

## 最佳化方法的最佳化

模型準確率沒有提高、收斂速度過慢，可以嘗試更換最佳化方法，以提升準確率、加快收斂速度。

PyTorch 中可以直接選取 :

* SGD
* Adagrad
* RMSProp
* Adam = RMSProp + Momentum

`Reference:` [Optimization for Deep Learning](https://github.com/prjiang/AI-Seminar/blob/main/2021-03-15/report/Optimization%20for%20Deep%20Learning.pdf)

<br>

## 損失函數的最佳化

### Loss function

* MSE (mean square error)
* Binary Cross Entropy
* Cross Entropy

### Regularization

[Regularization](https://medium.com/chung-yi/ml%E5%85%A5%E9%96%80-%E5%8D%81%E4%BA%94-regularization-solving-overfitting-9d000e3dd561) 對 loss function 進行正規化，以解決 overfitting 的情形。

* L1 regularization : [Lasso Regression](https://ithelp.ithome.com.tw/articles/10227654)
* L2 regularization : [Ridge Regression](https://medium.com/chung-yi/ml%E5%85%A5%E9%96%80-%E4%BA%8C%E5%8D%81%E4%BA%8C-ridge-regression-f638e1887a7e)
* `compare:` [[Machine Learning]Lasso Regression & Ridge Regression](https://dotblogs.com.tw/dash_analysis/2017/11/03/161734)

<br>

## 模型本身的最佳化

* Dropout : 拿掉一些神經元 (避免 overfitting)
* batch normalization : layers 的正規化 (解決梯度消失問題)
    * `Reference:` [Batch Normalization 介紹](https://medium.com/ching-i/batch-normalization-%E4%BB%8B%E7%B4%B9-135a24928f12)
* pre-trained model (transfer learning) : 預訓練
    * 預訓練模型 : 使用別人設計的神經網路
    * 預訓練參數 : 載入別人訓練一段時間的參數

<br>

## 善用硬體加速

* 使用單一主機的多個GPU `torch.nn.DataParallel(model)`
    * 使用多GPU 不一定能省記憶體，記憶體會複製至多個GPU上
    * 使用多GPU 不一定能增加模型準確度，但可以加快訓練速度
* 使用 Mixed Precision (混合精度)
    * PyTorch 預設使用 fp32 (佔用較多記憶體)
    * PyTorch v1.6 支援amp (Auto Mixed Precision) : 部分參數使用 fp32，部分參數使用 fp16
        * 可以<b>節省GPU 的記憶體</b>，且加快神經網路執行速度
    * 影響模型準確率 (影響不大)

amp 使用方法 :

```python
# 優化scaler
scaler = torch.cuda.amp.GradScaler()

# 加入訓練過程中
with torch.cuda.amp.autocast():
    output = model(data)
    loss = criterion(output, target)
scaler.scale(loss).backward()       # 針對scaler 做反向傳遞
scaler.step(optimizer)              # 針對optimizer scaler 做梯度下降 
scaler.update()                     # 更新權重
```