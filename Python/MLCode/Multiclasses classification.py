#%% 引用模組
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder, StandardScaler, LabelEncoder, label_binarize
import numpy as np
from sklearn import metrics
import xgboost as xgb
import os
os.environ['KERAS_BACKEND'] = 'torch'
import keras
#%% 讀資料、看資料
file = 'C:\研究所\自學\各模型\DATA\healthcare_dataset.csv'
data = pd.read_csv(file)
new_df = data.copy()
data.isna().sum()
data.info()
data.columns
data['Doctor'].value_counts()
data['Name'].value_counts()
#%% 設定常用參數
test_size = 0.3
random_state = 42
threshold = 0.5
cv = 5
#%% 資料處理、分資料
new_df = new_df.drop(
    ['Room Number'], axis = 1)
# 轉為時間格式，並相減算出天數
new_df['Discharge Date'] = pd.to_datetime(
    new_df['Discharge Date']) 
new_df['Date of Admission'] = pd.to_datetime(
    new_df['Date of Admission']) 
new_df['Days'] = (
    new_df['Discharge Date'] - new_df['Date of Admission']).dt.days
new_df = new_df.drop(
    ['Date of Admission', 'Discharge Date'], axis = 1)
new_df['Age Group'] = pd.cut(
    new_df['Age'], 
    bins = [
        10, 20, 30, 40, 50, 60, 70, 80, 90],
    labels = [
        '10-20', '21-30', '31-40', '41-50', '51-60', 
        '61-70', '71-80', '81-90'])
new_df = new_df.drop(['Age'], axis = 1)
x = new_df.drop(['Test Results'], axis = 1)
y = new_df['Test Results']
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y,
    test_size = test_size,
    random_state = random_state)
xtrain.columns
cols_to_targetencode = [
    'Name', 'Gender', 'Blood Type', 'Medical Condition', 'Doctor',
    'Hospital', 'Insurance Provider', 'Admission Type',
    'Medication', 'Age Group']
target_encoder = TargetEncoder()
for col in cols_to_targetencode:
    xtrain[col] = target_encoder.fit_transform(
        xtrain[[col]], ytrain)
    xtest[col] = target_encoder.transform(xtest[[col]])
standardscaler = StandardScaler()
for col in xtrain.columns:
    xtrain[col] = standardscaler.fit_transform(xtrain[[col]])
    xtest[col] = standardscaler.transform(xtest[[col]])
labelencoder = LabelEncoder()
ytrain = labelencoder.fit_transform(ytrain)
ytest = labelencoder.fit_transform(ytest)
# 查看每類個數
np.unique(ytrain, return_counts = True)
#%%
'''
Decision Tree、Random Forest參數沒有太多變化
XGBoost的objective要改變
ANN中loss、metrics要修改，Output units更改為有幾個類別
在模型評估上Proba要的是每一類分別的機率。並且原本0.1.2一個Column要展開成3個Columns，分別0.1。
宏平均(Macro-average) : 對所有類別的指標（如精確率、召回率、F1分數）進行平均，不考慮每個類別的樣本數量。
加權平均(Weighted-average) : 根據每個類別的樣本數量對指標進行加權平均
AUC計算時加上multi_class = 'ovr' : 每一類別會和剩下其他類別一起比較計算
'''
normal_xgb = xgb.XGBClassifier(
    objective = 'multi:softmax', 
    random_state = random_state,
    tree_method = "hist", 
    device = "cuda",
    n_estimators = 1000,
    learning_rate = 0.1,
    max_depth = 10,
    subsample = 0.6,
    colsample_bytree = 0.6,
    cv = cv)
normal_xgb.fit(
    xtrain, ytrain)
ytrain_normal_xgb_pred = normal_xgb.predict(xtrain)
ytest_normal_xgb_pred = normal_xgb.predict(xtest)
ytrain_normal_xgb_pred_proba = normal_xgb.predict_proba(xtrain)
ytest_normal_xgb_pred_proba = normal_xgb.predict_proba(xtest)
ytrain_for_auc = label_binarize(
    ytrain, classes = [0, 1, 2])
ytest_for_auc = label_binarize(
    ytest, classes = [0, 1, 2])
train_classification_report = metrics.classification_report(
    ytrain, ytrain_normal_xgb_pred)
test_classification_report = metrics.classification_report(
    ytest, ytest_normal_xgb_pred)
train_auc = metrics.roc_auc_score(
   ytrain_for_auc, 
   ytrain_normal_xgb_pred_proba, 
   multi_class = 'ovr')
test_auc = metrics.roc_auc_score(
   ytest_for_auc, 
   ytest_normal_xgb_pred_proba, 
   multi_class = 'ovr')
print(f' train_normal_tree_classification_report : {train_classification_report}')
print(f' test_normal_tree_classification_report : {test_classification_report}')
print(f' train_normal_tree AUC score : {train_auc:.5f}')
print(f' test_normal_tree AUC score : {test_auc:.5f}')
#%%
keras.utils.set_random_seed(random_state)
Xtrain, Xval, Ytrain, Yval = train_test_split(
    xtrain, ytrain, 
    test_size = 0.2, 
    random_state = random_state)

Xtrain_array = Xtrain.to_numpy()
Xval_array = Xval.to_numpy()

normal_ann = keras.Sequential([
    keras.layers.Input(shape = (xtrain.shape[1],)),
    keras.layers.Dense(50, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(25, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(14, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(7, activation = 'relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(3, activation = 'softmax')])

normal_ann.compile(  
    optimizer = keras.optimizers.Adam(learning_rate = 1e-2),
    loss = keras.losses.SparseCategoricalCrossentropy(),
    metrics = [keras.metrics.SparseCategoricalAccuracy(
        name = "sparse_categorical_accuracy")])

normal_ann.fit(
    Xtrain,
    Ytrain,
    epochs = 50,
    batch_size = 64,
    validation_data = (Xval_array, Yval))

ytrain_best_normal_ann_proba = normal_ann.predict(xtrain)
ytest_best_normal_ann_proba = normal_ann.predict(xtest)
ytrain_normal_ann_predicted_labels = ytrain_best_normal_ann_proba.argmax(axis  = 1)
ytest_normal_ann_predicted_labels = ytest_best_normal_ann_proba.argmax(axis = 1)
ytrain_for_auc = label_binarize(
    ytrain, classes = [0, 1, 2])
ytest_for_auc = label_binarize(
    ytest, classes = [0, 1, 2])
train_classification_report = metrics.classification_report(
    ytrain, ytrain_normal_ann_predicted_labels)
test_classification_report = metrics.classification_report(
    ytest, ytest_normal_ann_predicted_labels)
train_auc = metrics.roc_auc_score(
   ytrain_for_auc, 
   ytrain_best_normal_ann_proba, 
   multi_class = 'ovr')
test_auc = metrics.roc_auc_score(
   ytest_for_auc, 
   ytest_best_normal_ann_proba, 
   multi_class = 'ovr')
print(f' train_normal_ann_classification_report : {train_classification_report}')
print(f' test_normal_ann_classification_report : {test_classification_report}')
print(f' train_normal_ann AUC score : {train_auc:.5f}')
print(f' test_normal_ann AUC score : {test_auc:.5f}')