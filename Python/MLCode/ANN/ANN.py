#%% 引用模組
import pandas as pd
import numpy as np
# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, TargetEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics
import keras_tuner
#%% 讀資料
data_file = 'C:/研究所/自學/各模型/DATA/Heart Disease.csv'
df = pd.read_csv(data_file)
original_df = df.copy()
new_df = df.copy()
#%% 資料檢視
# 查看前5筆
df.head()
#查看資料型態類別有無缺失值
df.info()
#查看資料平均、標準差四分位
df.describe()
#列出資料中的所有columns
df.columns
#%% 設定使用參數
random_state = 42
test_size = 0.3
cv = 5
n_iter = 100
scoring = 'f1'
threshold = 0.5
#%% 分割出特徵及目標變數資料
x = new_df.drop(['Heart_Disease'], axis = 1)
y = new_df['Heart_Disease']
#%% 分割出續練測試資料資料
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, 
    test_size = test_size, 
    random_state = random_state)
#%% 資料預處理
'''
資料正規化標準化 : 在樹類模型影響不大。但對於其他影像較大尤其深度學習
    1.收斂速度變慢、梯度消失、爆炸 : 當使用非線性激活函數(sigmoid、tanh)，特徵範圍差距大
    時，會導致梯度變得非常小，發生梯度消失問題。另外特徵的範圍過大，梯度可能會變得大來更新
    權重，導致發生梯度爆炸問題。而梯度消失、爆炸會讓模型收斂速度變慢。
    2.神經網絡的敏感性 : 模型難以適應這同尺度的特徵，可能過度關注那些值較大的特徵
我們在做模型訓練前，會做資料的標準化，讓所有的資料縮放到相同的尺度可以幫助模型得訓練速度及加快梯度下降。
模型無法直接用文字做訓練，類別資料雖對樹模型不影響結果，但如LINER REGRESSION、
    NN等需要把資料標準化，提高模型型的準確性且減少模型訓練時間
轉換資料中文字、無序資料至數值數字
TargetEncoder : EX:A,B,C三類分別有或無，計算有 : A,B,C，無 : A,B,C的比例
在做資料數值轉換時，要先切割好訓練、測試資料
'''
# 創建targetencoder
encoder = TargetEncoder()
# 要轉換的列
columns_to_encode = [
    'General_Health', 'Checkup', 'Exercise', 'Skin_Cancer', 
    'Other_Cancer', 'Depression', 'Diabetes', 'Arthritis', 
    'Sex', 'Age_Category', 'Smoking_History',
    'Alcohol_Consumption', 'Fruit_Consumption',
    'Green_Vegetables_Consumption', 'FriedPotato_Consumption']
for col in columns_to_encode:
    xtrain[col] = encoder.fit_transform(
        xtrain[[col]], ytrain)
    xtest[col] = encoder.transform(xtest[[col]])
# 把數值資料轉換平均:0、標準差1(Z分佈)
scaler = StandardScaler()
for col in xtrain.columns:
    xtrain[col] = scaler.fit_transform(xtrain[[col]])
    xtest[col] = scaler.transform(xtest[[col]])
# 把目標變數轉為0,1
label_encoder = LabelEncoder()
ytrain = label_encoder.fit_transform(ytrain)
ytest = label_encoder.fit_transform(ytest)
x_train_test = [xtrain, xtest]
for col in x_train_test:
    col.rename(columns={
        'Height_(cm)' : 'Height',
        'Weight_(kg)' : 'Weight'},
        inplace = True)    
# 查看資料是否為平衡資料
'''
不平衡資料經調整權重之後，可以更好的訓練模型，泛化外來資料。
在最後結果指標的評估，假如使用Accuracy，只關心正確預測的有多少。而使用Precision、Recall會把錯誤預測的考慮進去，而F1為Precision、Recall綜合之指標，F1 Score對少數樣本的預測效果敏感，當少數樣本預測不好時會顯著下降。並且可以配合AUC(不受類別不平衡影響)一同評估模型。
'''
[sum(ytrain == 0), sum(ytrain == 1)]
# 因資料為不平衡資料，需改變權重
weights = compute_class_weight(
    class_weight = "balanced", 
    classes = np.array(
    [0, 1]), 
    y = ytrain)
class_weight = {
    0 : weights[0], 
    1 : weights[1]}
#%% 類神經ANN(Tensorflow)
'''
1.在NN中包含了Input Layer，中間的Hidden Layer，及輸出的Output Layer。Input Layer及Output Layer可以有很多個神經元，但只會有一層，
Hidden Layer可以有很多層。每一個Input Layer都會與下一層的所有Hidden Layer連接，最後一層的Hidden Layer會與所有Output Layer連接，這被
稱做Fully Connected。在每一個Input Layer連結每一個Hidden Layer會通過Weight及Bias計算，再進入Active Function，例如:Relu、Tanh、
Sigmoid，對應出Active Function中的某部分，並畫出多條線。之後每一個Active Function連結至每一個Output Layer都會再經過一個Weight及
Active Function，這時多條線透過計算及對應，合併並且調整、轉換成一條最Fit Data的線。最終Output Layer呈現的結果通常是任何數，如是
Regression可直接使用，如是Classification會經過SoftMax、Sigmoid轉換成0至1之間的數。過程中所有的Weight及Bias，都是通過Backpropagation
決定，第一個Weight及Bias由隨機生成做計算，得出結果後在Regression問題中可以使用MSE等，Classification可使用Cross Entropy，看預測出的結果
跟實際差多少，在反過來調整Weight及Bias去縮小差距。為了找到最佳的Weight及Bias，可使用的方式是Gradient Decent，通過不同Weight及Bias所計算
出資料的MSE或Cross Entropy，可以畫出一個點及此點的梯度線，向梯度線的下方移動，希望可以獲得一組全局最佳參數，一條梯度為0的線，但我們無法得知全
局最佳故通常為局部最佳參數。Gradient Decent中又可以分為三種Batch Gradient Decent、Stochastic Gradient Decent、Mini Batch 
Gradient Decent。Batch Gradient Decent是會用所有資料去計算MSE或Cross Entropy，優點是比較精確，缺點是太耗時。Stochastic Gradient
Decent會隨機所有資料只用一筆資料，優點是速度快，缺點是不精準。而Mini Batch Gradient Decent則是會隨機一小批，結合了Batch Gradient 
Decent、Stochastic Gradient Decent優缺點。

2.keras.Sequential([
    keras.layers.Dense(30, input_shape = [30], activation = 'relu'),
    keras.layers.Dense(3, activation = 'relu'),
    keras.layers.Dense(1, activation = 'sigmoid')])
Sequential:一堆層、Dense:每個神經元與下一層所有神經元連結(fully connected)
Dense(第一個HIDDEN LAYER有幾個神經元, input_shape = [input形狀]，
Dense(第二個HIDDEN LAYER有幾個神經元, activation = 使用的激活函數),                                                   
最後一筆Dense(幾個OUTPUT LAYER，轉換結果的ACTIVE FUNCTION)]

3.hp : 定義超參數的類型INT、FLOAT
    num_hidden_layers = hp.Int(
        'num_hidden_layers', 
        min_value=1, 
        max_value=5, step=1) : 名字為num_hidden_layers的參數，從最小1至最大5，
    每次選擇步長1，最終傳一個整數給num_hidden_layers 
    hp.Float() : 浮點數
    hp.Choice('activation', values=['relu', 'tanh', 'sigmoid']) : 
        名字叫activation的參數，由values中選一個
        
4.在FOR迴圈中，當我num_hidden_layers決定有幾個隱藏層後，就會跑幾次迴圈，每次回圈都會
    重新選擇units、activation、dropout
    
5.metrics  
[keras.metrics.BinaryCrossentropy(name='cross entropy'),
keras.metrics.MeanSquaredError(name='Brier score'),
keras.metrics.TruePositives(name='tp'),
keras.metrics.FalsePositives(name='fp'),
keras.metrics.TrueNegatives(name='tn'),
keras.metrics.FalseNegatives(name='fn'), 
keras.metrics.BinaryAccuracy(name='accuracy'),
keras.metrics.Precision(name='precision'),
keras.metrics.Recall(name='recall'),
keras.metrics.AUC(name='auc'),
keras.metrics.AUC(name='prc', curve='PR'),
keras.metrics.F1Score(name='f1_score')]

6.
model.compile:
    optimizer : 像是用甚麼方式找最佳參數(EX:SGD)
    loss : 可以理解為MSE、cross entropy，透過優化LOSS找到更好的WEIGHT、BIAS
    metrics : 當模型在訓練時，可以隨時監控的指標
BayesianOptimization:
    objective = 'accuracy' : 比較哪組參數比較好的標準(val_accuracy有做ES可改) 
    max_trials = 50 : 總共會找幾次參數
    num_initial_points = 10 : 初始參數筆數
    directory = '...' : 把最佳參數及模型儲存在主目錄
    project_name = '...' : 把最佳參數及模型儲存在子目錄
tuner.search
    epochs : 做幾次的BACK PROPAGATION
    validation_split : 把資料切出多少比例做為驗證集(因ANN中無K-FOLD的參數，可以
                       設定split當每組參數的比較)
    batch_size : 把所有資料丟進去訓練類神經網路，每次用batch_size設定的筆資料計
        算LOSS，約做 資料數/BATCH SIZE次 的BACKPROPAGATION優化WEIGHT BIAS，循環
        Epochs次
Early Stopping:
    monitor : 通過甚麼指標監測是否執行ES    
    patience : 經過幾次沒有改善執行ES           
    restore_best_weights : 把最加的結果存下來 

7.總共會進行max_trials設定的50組不同參數，每組參數會進行進行epochs設定的100次，每次epochs中進行總資料數/BATCH_SIZE次的BACK PROPAGATION。每次的BACK PROPAGATION會計算選的loss
並使用選的optimizer去優化WEIGHT、BIASES，結束100次EPOCHS後會開始建立代理模型，在每一組不同參數結束後加到代理模型中，直到num_initial_points設定得10筆之後，會基於代理模型中所有筆資料去找更好的參數。並加入代理模型，直到max_trials設定的50次。
.get_best_models(num_models = 1)[0]、.get_best_hyperparameters(num_trials = 1)[0]
從50個參數模型選出最好的。

8.預防OVERFITTING
Batch Normalization : 如同我們在做模型訓練前，會做資料的標準化，在每一層輸出都做一次特徵
    標準化，不但可以加快梯度下降，而且在一定程度緩解了深層網絡中梯度消失的問題，也很好解決
    OVERFITTING。(甚至可以取代DROPOUT)
Early stopping
    
9.classification:
    output layer active function : 二元sigmoid、多元softmax
    hidden layer active function : 不可sigmoid
    
10.
沙漏型 : 神經元先減少後增加
    優點 : 可以壓縮信息，然後重建，適用於自編碼器
    缺點 : 中間層可能成為信息瓶頸
金字塔型 : 神經元逐層減少
    優點 : 逐步提取更高級特徵，減少參數數量(減少Over fitting)
    缺點 : 可能損失一些低級特徵
反金字塔型 : 神經元逐層增加
    優點 : 允許網絡學習更多複雜特徵
    缺點 : 可能導致Over fitting
統一型 : 所有隱藏層神經元數量相同
    優點 : 簡單，易於調整
    缺點 : 可能不夠靈活來適應複雜任務
'''
#%% ANN(Tensorflow)
tf.random.set_seed(random_state)
# tf建立全局隨機種子，後續不用在設定
Xtrain, Xval, Ytrain, Yval = train_test_split(
    xtrain, ytrain, 
    test_size = 0.1, 
    random_state = random_state)
train_data = tf.data.Dataset.from_tensor_slices((
    Xtrain, Ytrain))
train_data = train_data.cache().shuffle(1000).batch(
    batch_size = 64).prefetch(
        buffer_size = tf.data.AUTOTUNE)
val_data = tf.data.Dataset.from_tensor_slices((
    Xval, Yval))
val_data = val_data.cache().shuffle(1000).batch(
    batch_size = 64).prefetch(
        buffer_size = tf.data.AUTOTUNE)

normal_ann = keras.Sequential([
    keras.layers.Input(
        shape = (xtrain.shape[1],)),
    keras.layers.Dense(
        64, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(
        32, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(
        16, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(
        8, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(
        4, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(
        1, activation = 'sigmoid')])

normal_ann.compile(  
    optimizer = keras.optimizers.Adam(
        learning_rate = 1e-2),
    loss = keras.losses.BinaryCrossentropy(),
    metrics = [keras.metrics.AUC(name ='auc')])

normal_ann.fit(
    train_data,
    epochs = 1000,
    batch_size = 100,
    validation_data = val_data,
    class_weight = class_weight)

ytrain_best_normal_ann_proba = normal_ann.predict(xtrain)
ytest_best_normal_ann_proba = normal_ann.predict(xtest)
# >0.5 : 1, <0.5 : 0
ytrain_normal_ann_predicted_labels = (ytrain_best_normal_ann_proba > threshold).astype(int)
ytest_normal_ann_predicted_labels = (ytest_best_normal_ann_proba > threshold).astype(int)
train_accuracy = metrics.accuracy_score(
    ytrain, ytrain_normal_ann_predicted_labels)
test_accuracy = metrics.accuracy_score(
    ytest, ytest_normal_ann_predicted_labels)
train_f1 = metrics.f1_score(
    ytrain, ytrain_normal_ann_predicted_labels)
test_f1 = metrics.f1_score(
    ytest, ytest_normal_ann_predicted_labels)
train_auc = metrics.roc_auc_score(
    ytrain, ytrain_best_normal_ann_proba)
test_auc = metrics.roc_auc_score(
    ytest, ytest_best_normal_ann_proba)
print(f' train_normal_ann Accuracy Score: {train_accuracy:.5f}')
print(f' test_normal_ann Accuracy Score: {test_accuracy:.5f}')
print(f' train_normal_ann F1 score: {train_f1:.5f}')
print(f' test_normal_ann F1 score: {test_f1:.5f}')
print(f' train_normal_ann AUC score: {train_auc:.5f}')
print(f' test_normal_ann AUC score: {test_auc:.5f}')
#%%
# from keras.callbacks import ModelCheckpoint
# checkpoint_callback = ModelCheckpoint(
#     'best_model.keras',
#     monitor='val_auc',
#     save_best_only=True,
#     mode='max',
#     verbose=1)

# normal_ann.fit(
#     Xtrain,
#     Ytrain,
#     epochs = 10,
#     batch_size = 64,
#     validation_data = (Xval_array, Yval),
#     class_weight = class_weight,
#     callbacks=[checkpoint_callback])

# best_model = keras.models.load_model('best_model.keras')

from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_auc',
    mode = 'max',
    factor = 0.5,
    patience = 5,
    min_lr = 1e-6)

from keras.layers import Normalization
normalizer = Normalization(axis = -1)
xtrain_array = np.array(xtrain)
normalizer.adapt(xtrain_array)

import time
start_time = time.time()
early_stopping = keras.callbacks.EarlyStopping(
    monitor = 'val_auc', 
    mode = 'max',
    patience = 10,           
    restore_best_weights = True)

def tuning_ann(hp):
    model = keras.Sequential()
    model.add(normalizer)
#    model.add(keras.layers.Input(
#        shape = (xtrain.shape[1],)))

    nums_layers = hp.Int('nums_layers', 1, 3, 1)
    max_units = 64  
    previous_units = 0
    for i in range(nums_layers):
        if i == 0:
            units = hp.Int(f'layer_{i+1}_units', 8, max_units, 1)
        else : 
            units = hp.Int(f'layer_{i+1}_units', 8, previous_units, 1)
            
        model.add(keras.layers.Dense(units))
       
        activation_choice = hp.Choice(f'layer_{i+1}_activation', values=['relu', 'leaky_relu'])
        if activation_choice == 'leaky_relu':
            model.add(keras.layers.LeakyReLU())
        else:
            model.add(keras.layers.ReLU()) 
        model.add(keras.layers.BatchNormalization())
        previous_units = units 
             
    model.add(keras.layers.Dense(
        1, activation = 'sigmoid'))
    choiced_optimizer = hp.Choice(
        'optimizer', values = ['adam', 'sgd', 'rmsprop', 'adamw'])
    choiced_learning_rate = hp.Float(
        'learning_rate', 1e-5, 1e-3, sampling = 'log')
    
    if choiced_optimizer == 'adam':
        optimizer = keras.optimizers.Adam(
            learning_rate = choiced_learning_rate)
        
    elif choiced_optimizer == 'sgd':
        optimizer = keras.optimizers.SGD(
            learning_rate = choiced_learning_rate)
        
    elif choiced_optimizer == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(
            learning_rate = choiced_learning_rate)
        
    elif choiced_optimizer == 'adamw':
        optimizer = keras.optimizers.AdamW(
            learning_rate = choiced_learning_rate)
        
    model.compile(  
        optimizer = optimizer,
        loss = 'binary_crossentropy',
        metrics = [keras.metrics.AUC(name = 'auc')])
    
    return model

bayes_tuning_ann = keras_tuner.BayesianOptimization(
    tuning_ann,
    objective = keras_tuner.Objective('val_auc', direction = 'max'),
    max_trials = 5,
    num_initial_points = 10)  
# 把資料FIT進去找
bayes_tuning_ann.search(
    train_data,
    epochs = 5,
    batch_size = 64,
    class_weight = class_weight,
    validation_data = val_data,          
    callbacks = [early_stopping, reduce_lr])
end_time = time.time()
end_time-start_time

ytrain_best_normal_ann_proba = normal_ann.predict(xtrain)
ytest_best_normal_ann_proba = normal_ann.predict(xtest)
# >0.5 : 1, <0.5 : 0
ytrain_normal_ann_predicted_labels = (ytrain_best_normal_ann_proba > threshold).astype(int)
ytest_normal_ann_predicted_labels = (ytest_best_normal_ann_proba > threshold).astype(int)
train_accuracy = metrics.accuracy_score(
    ytrain, ytrain_normal_ann_predicted_labels)
test_accuracy = metrics.accuracy_score(
    ytest, ytest_normal_ann_predicted_labels)
train_f1 = metrics.f1_score(
    ytrain, ytrain_normal_ann_predicted_labels)
test_f1 = metrics.f1_score(
    ytest, ytest_normal_ann_predicted_labels)
train_auc = metrics.roc_auc_score(
    ytrain, ytrain_best_normal_ann_proba)
test_auc = metrics.roc_auc_score(
    ytest, ytest_best_normal_ann_proba)
print(f' train_normal_ann Accuracy Score: {train_accuracy:.5f}')
print(f' test_normal_ann Accuracy Score: {test_accuracy:.5f}')
print(f' train_normal_ann F1 score: {train_f1:.5f}')
print(f' test_normal_ann F1 score: {test_f1:.5f}')
print(f' train_normal_ann AUC score: {train_auc:.5f}')
print(f' test_normal_ann AUC score: {test_auc:.5f}')
