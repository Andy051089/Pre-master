#%% 引用模組
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
os.environ['KERAS_BACKEND'] = 'torch'
import keras
import keras_tuner
import matplotlib.pyplot as plt
#%% LSTM、GRU
'''
1.RNN是想要預測有時間順序，前後有關係的資料。當第一筆資料進入第一個Input時，通過Weight及Bias後進入Active Function，在經過另
一個Weight放置Hidden State，所有層要進入Hidden State的Weight為相同，之後會與第二筆資料進入第二個Input，通過Weight及Bias
後，一起進入第二個Active Function，以此循環直至傳至最後的一個資料。在最後一筆資料中，通過Active Function，會在經過Weight
及Bias輸出Output。另根據需求不同，也可於過程中有Output。而RNN在找尋最佳參數使用Gradient Decent時，經過多次疊加的Weight，
如果是大於1，經相乘之後，數值會太大。如果是小於1，經相乘之後，數值會太小。使得Gradient Decent會一直無法找到最佳參數，稱為梯度爆
炸或消失。

2.LSTM或GRU大大降低了梯度爆炸或消失的問題。LSTM添加了Long Path。當第一筆資料進入Input Gate時，通過計算此筆資料重不重要，如果
重要會把此筆資料的參數存至Memory Cell，並從Output Gate進入Hidden State，如果不重要就會忘記此筆資料的參數。當第二筆資料進入
Input Gate後，一樣通過計算此筆資料重不重要，如果重要一樣會把此筆資料的參數存至Memory Cell。同一時間原本在Memory Cell中所留下
的之前參數會經過Forget Gate，計算是否還是重要，如重要則此筆參數不會變更，如不重要則忘記，並且在Hidden State中也會消失。最終在
做預測時，也是使用Hidden State中的參數做計算。而GRU是把LSTM中Input Gate及Forget Gate合併在一起稱Update Gate，並且移除
了Memory Cell，當參數有更新變化決，直接進入Hidden State。

3.輸入資料型態(x, y, z) : x為總共幾筆資料，y為時間步數(使用幾天的資料是預測下一天的)，z為幾個特徵。所以第一筆資料(1, 10, 6)為
有10筆每筆(ROWS)每筆皆有6項特徵(COLUMNS)

4.keras.layers.LSTM(25,return_sequences = True)
    25 : 要經過幾個LSTM，第個輸入(10, 6)，每筆都會經過25次LSTM，通過計算在輸出時(x, y, z) : x為總共幾筆資料，y為時間步數
    ，z為通過幾個LSTM所計算產生的Hidden State。
    return_sequences : True為輸出包含每個時間步的隱藏狀態、False為輸出不包含每個時間步的隱藏狀態
    TimeDistributed : 會把每筆資料間的時間順序靠慮進去
'''
#%% 設定使用參數
random_state = 42
test_size = 0.3
cv = 5
n_iter = 100
scoring = 'f1'
threshold = 0.5
keras.utils.set_random_seed(random_state)
#%% 讀資料
data_file = 'C:/研究所/自學/各模型/DATA/NFLX.csv'
df = pd.read_csv(data_file)
data = df[
    ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
# 建立每10天預測下一天的Close資料
def create_dataset(data, time_step = 10):
    x, y = [], []
    for i in range(len(data) - time_step):
        x.append(data.iloc[i:(i + time_step)].values)
        y.append(data.iloc[i + time_step]['Close'])
    return np.array(x), np.array(y)
x, y = create_dataset(data)
# 分割資料訓練驗證測試
xtrain_val, xtest, ytrain_val, ytest = train_test_split(
    x, y, 
    test_size = 0.2, 
    random_state = random_state,
    shuffle = False)
xtrain, xval, ytrain, yval = train_test_split(
    xtrain_val, ytrain_val, 
    test_size = 0.1, 
    random_state = random_state,
    shuffle = False)
# 標準化
x_scaler = StandardScaler()
x_train = x_scaler.fit_transform(
    xtrain.reshape(-1, xtrain.shape[-1])).reshape(xtrain.shape)
x_test = x_scaler.transform(
    xtest.reshape(-1, xtest.shape[-1])).reshape(xtest.shape)
x_val = x_scaler.transform(
    xval.reshape(-1, xval.shape[-1])).reshape(xval.shape)
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(
    ytrain.reshape(-1, 1)).reshape(ytrain.shape)
y_test = y_scaler.transform(
    ytest.reshape(-1, 1)).reshape(ytest.shape)
y_val = y_scaler.transform(
    yval.reshape(-1, 1)).reshape(yval.shape)
# 建立模型
normal_gru = keras.Sequential([
    keras.layers.Input(
        shape = (
            x_train.shape[1],
            x_train.shape[2])),
    keras.layers.GRU(
        50, 
        return_sequences = True),
    keras.layers.BatchNormalization(),
    keras.layers.GRU(
        25, 
        return_sequences = True),
    keras.layers.BatchNormalization(),   
    keras.layers.GRU(
        14, 
        return_sequences = True),
    keras.layers.BatchNormalization(),
    keras.layers.TimeDistributed(keras.layers.Dense(1)),
    keras.layers.GRU(
        7, 
        return_sequences = False),
    keras.layers.BatchNormalization(),  
    keras.layers.Dense(
        14, 
        activation = 'relu'),
    keras.layers.BatchNormalization(), 
    keras.layers.Dense(
        7, 
        activation = 'relu'),
    keras.layers.BatchNormalization(), 
    keras.layers.Dense(
        1,
        activation = 'linear')])   
normal_gru.compile(  
    optimizer = keras.optimizers.Adam(learning_rate = 1e-3),
    loss = 'mean_squared_error',
    metrics = [keras.metrics.RootMeanSquaredError()])
normal_gru.fit(
    x_train,
    y_train,
    epochs = 250,
    batch_size = 64,
    validation_data = (x_val, y_val))
train_predict = normal_gru.predict(x_train)
test_predict = normal_gru.predict(x_test)
train_loss, train_rmse = normal_gru.evaluate(x_train, y_train)
test_loss,  test_rmse = normal_gru.evaluate(x_test, y_test)
print(f' normal_gru train loss : {train_loss}:.5f')
print(f' normal_gru test loss : {test_loss}:.5f')
print(f' normal_gru train rmse : {train_rmse}:.5f')
print(f' normal_gru train rmse : {test_rmse}:.5f')
# 標準化資料轉換為原始數據
test_predict_inverse = y_scaler.inverse_transform(test_predict)
# 把實際和預測值畫圖
# 圖大小
plt.figure(figsize = (12, 6))
# 繪製實際值
plt.plot(
    ytest, 
    label = 'Actual Values', 
    color = 'blue')
# 繪製預測值
plt.plot(
    test_predict_inverse, 
    label = 'Predicted Values', 
    color = 'red', 
    linestyle = '--')
# 添加標題和標籤
plt.title('Actual vs Predicted Values')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()

def tuning_gru(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(
        shape = (x_train.shape[1], x_train.shape[2])))
    model.add(keras.layers.GRU(
            units = hp.Int('first_unit', 
            min_value = 50, max_value = 200, step = 50),
        activation = hp.Choice(
            'activation_input', 
            values = ['relu', 'tanh']),
        return_sequences = True))  
    model.add(keras.layers.BatchNormalization())
    
    num_gru_layers = hp.Int('num_gru_layers', 1, 5, step = 1)
    for i in range(num_gru_layers):
        model.add(keras.layers.GRU(
            units = hp.Int(f'gru_units_{i}', 50, 200, step = 50),
            activation = hp.Choice(f'gru_activation_{i}', 
                values = ['relu', 'tanh']),
            return_sequences = True))
        model.add(keras.layers.BatchNormalization())
        
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(1)))    
    model.add(keras.layers.GRU(
        units = hp.Int('last_unit', 
            min_value = 50, max_value = 200, step = 50),
        activation = hp.Choice(
            'last_activation', 
            values = ['relu', 'tanh']),
        return_sequences = False))  
    model.add(keras.layers.BatchNormalization())        

    nums_dense_layers = hp.Int('nums_dense_layers', 1, 3, step = 1)    
    for i in range(nums_dense_layers):
        model.add(keras.layers.Dense(
            units = hp.Int(f'dense_units_{i}', 2, 100, step = 30),
            activation = hp.Choice(f'dense_activation_{i}', 
                                   values = ['relu', 'tanh'])))
        model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1, activation = 'linear'))

    choiced_optimizer = hp.Choice(
        'optimizer', values = ['adam', 'sgd', 'rmsprop', 'adamw'])
    choiced_learning_rate = hp.Float(
        'learning_rate', 1e-5, 1e-4, sampling = 'log')
    
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
        loss = 'mean_squared_error',
        metrics = [keras.metrics.RootMeanSquaredError()])
    return model

early_stopping = keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    mode = 'min',    
    patience = 50,           
    restore_best_weights = True)

# 創建BayesianOptimization
bayes_tuning_gru = keras_tuner.BayesianOptimization(
    tuning_gru,
    objective = 'val_loss',
    max_trials = 50,
    num_initial_points = 10)  
# 把資料FIT進去找
bayes_tuning_gru.search(
    x_train,
    y_train,
    epochs = 500,
    batch_size = 64,
    validation_data = (x_val, y_val),
    callbacks = [early_stopping])   

best_bayes_tuning_gru_model = bayes_tuning_gru.get_best_models(
    num_models = 1)[0]
best_bayes_tuning_gru_model.summary()
# 最佳超參數
best_bayes_tuning_gru_hp = bayes_tuning_gru.get_best_hyperparameters(
    num_trials = 1)[0]        
ytrain_gru_pred = best_bayes_tuning_gru_model.predict(x_train)
ytest_gru_pred = best_bayes_tuning_gru_model.predict(x_test)
train_loss, train_rmse = best_bayes_tuning_gru_model.evaluate(
    x_train, y_train)
test_loss,  test_rmse = best_bayes_tuning_gru_model.evaluate(
    x_test, y_test)
print(f' tuning_gru train loss : {train_loss}:.5f')
print(f' tuning_gru test loss : {test_loss}:.5f')
print(f' tuning_gru train rmse : {train_rmse}:.5f')
print(f' tuning_gru test rmse : {test_rmse}:.5f')

# 標準化資料轉換為原始數據
test_predict_inverse = y_scaler.inverse_transform(ytest_gru_pred)
# 把實際和預測值畫圖
# 圖大小
plt.figure(figsize = (12, 6))
# 繪製實際值
plt.plot(
    ytest,
    label = 'Actual Values',
    color = 'blue')
# 繪製預測值
plt.plot(
    test_predict_inverse, 
    label = 'Predicted Values', 
    color = 'red', 
    linestyle = '--')
# 添加標題和標籤
plt.title('Actual vs Predicted Values')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()