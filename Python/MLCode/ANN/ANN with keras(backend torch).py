#%% 引用模組
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, TargetEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics
import os
os.environ['KERAS_BACKEND'] = 'torch'
import keras
import keras_tuner
import pickle
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
#%% Keras(使用torch跑)
keras.utils.set_random_seed(random_state)
Xtrain, Xval, Ytrain, Yval = train_test_split(
    xtrain, ytrain, 
    test_size = 0.1, 
    random_state = random_state)

Xtrain_array = Xtrain.to_numpy()
Xval_array = Xval.to_numpy()

normal_ann = keras.Sequential([
    keras.layers.Input(shape = (xtrain.shape[1],)),
    keras.layers.Dense(500, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(400, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(250, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(200, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(1, activation = 'sigmoid')])

normal_ann.compile(  
    optimizer = keras.optimizers.Adam(learning_rate = 1e-2),
    loss = keras.losses.BinaryCrossentropy(),
    metrics = [keras.metrics.AUC(name ='auc')])

normal_ann.fit(
    Xtrain,
    Ytrain,
    epochs = 100,
    batch_size = 500,
    validation_data = (Xval_array, Yval),
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
    ytrain, ytrain_normal_ann_predicted_labels)
test_auc = metrics.roc_auc_score(
    ytest, ytest_normal_ann_predicted_labels)
print(f' train_normal_ann Accuracy Score: {train_accuracy:.5f}')
print(f' test_normal_ann Accuracy Score: {test_accuracy:.5f}')
print(f' train_normal_ann F1 score: {train_f1:.5f}')
print(f' test_normal_ann F1 score: {test_f1:.5f}')
print(f' train_normal_ann AUC score: {train_auc:.5f}')
print(f' test_normal_ann AUC score: {test_auc:.5f}')
#%% tuning_ann
# 建模型
keras.utils.set_random_seed(random_state)
Xtrain, Xval, Ytrain, Yval = train_test_split(
    xtrain, ytrain, 
    test_size = 0.1, 
    random_state = random_state)

Xtrain_array = Xtrain.to_numpy()
Xval_array = Xval.to_numpy()

def tuning_ann(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(
        shape = (xtrain.shape[1],)))
    
    for i in range(2):
        activation_choices = ['relu', 'leaky_relu']
        
        model.add(keras.layers.Dense(
            units = hp.Int(
                f'hidden_unit_{i+1}', 
                min_value = 8, 
                max_value = 64, 
                step = 8)))
        
        activation_choice = hp.Choice(
            f'activation_{i+1}', values = activation_choices)
        if activation_choice == 'leaky_relu' : 
            model.add(keras.layers.LeakyReLU())
        else:
            model.add(keras.layers.Activation(activation_choice))
        
    model.add(keras.layers.BatchNormalization())
        
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
        metrics = [keras.metrics.AUC(name ='auc')])
    
    return model

early_stopping = keras.callbacks.EarlyStopping(
    monitor = 'val_auc', 
    mode = 'max',
    patience = 50,           
    restore_best_weights = True)
# 創建BayesianOptimization
bayes_tuning_ann = keras_tuner.BayesianOptimization(
    tuning_ann,
    objective = 'val_auc',
    max_trials = 50,
    num_initial_points = 10)  
# 把資料FIT進去找
bayes_tuning_ann.search(
    Xtrain_array,
    Ytrain,
    epochs = 500,
    batch_size = 200,
    class_weight = class_weight,
    validation_data = (Xval, Yval),          
    callbacks = [early_stopping])            
# 最佳模型(不需再重新FIT一次資料)num_models = 1 : 所有裡面最佳的
best_bayes_tuning_ann_model = bayes_tuning_ann.get_best_models(num_models = 1)[0]
# 最佳超參數
best_bayes_tuning_ann_hp = bayes_tuning_ann.get_best_hyperparameters(num_trials = 1)[0]
# 把模型及參數存起來
file_name = "C:/Users/88691/Desktop/自學/AI/practice/best_bayes_tuning_ann_model.pkl"
pickle.dump(best_bayes_tuning_ann_model, open(file_name, "wb"))
best_bayes_tuning_ann_model = pickle.load(open(file_name, "rb"))
file_name = "C:/Users/88691/Desktop/自學/AI/practice/best_bayes_tuning_ann_hp.pkl"
pickle.dump(best_bayes_tuning_ann_hp, open(file_name, "wb"))
best_bayes_tuning_ann_hp = pickle.load(open(file_name, "rb"))
final_best_hp = best_bayes_tuning_ann_hp.values.items()
ytrain_best_tuning_ann_proba = best_bayes_tuning_ann_model.predict(xtrain)
ytest_best_tuning_ann_proba = best_bayes_tuning_ann_model.predict(xtest)
# >0.5 : 1, <0.5 : 0
ytrain_tuning_ann_predicted_labels = (ytrain_best_tuning_ann_proba > threshold).astype(int)
ytest_tuning_ann_predicted_labels = (ytest_best_tuning_ann_proba > threshold).astype(int)
train_accuracy = metrics.accuracy_score(
    ytrain, ytrain_tuning_ann_predicted_labels)
test_accuracy = metrics.accuracy_score(
    ytest, ytest_tuning_ann_predicted_labels)
train_f1 = metrics.f1_score(
    ytrain, ytrain_tuning_ann_predicted_labels)
test_f1 = metrics.f1_score(
    ytest, ytest_tuning_ann_predicted_labels)
train_auc = metrics.roc_auc_score(
    ytrain, ytrain_best_tuning_ann_proba)
test_auc = metrics.roc_auc_score(
    ytest, ytest_best_tuning_ann_proba)
print(f' train_tuning_ann Accuracy Score: {train_accuracy:.5f}')
print(f' test_tuning_ann Accuracy Score: {test_accuracy:.5f}')
print(f' train_tuning_ann F1 score: {train_f1:.5f}')
print(f' test_tuning_ann F1 score: {test_f1:.5f}')
print(f' train_tuning_ann AUC score: {train_auc:.5f}')
print(f' test_tuning_ann AUC score: {test_auc:.5f}')