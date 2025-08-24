#%% 引用模組
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, TargetEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, optim
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
#%% 類神經ANN(Pytorch)
xtrain_array = xtrain.to_numpy()
xtrain_tensor = torch.FloatTensor(xtrain_array)
ytrain_tensor = torch.FloatTensor(ytrain).unsqueeze(1)
train_val_dataset = TensorDataset(xtrain_tensor, ytrain_tensor)

xtest_array = xtest.to_numpy()
xtest_tensor = torch.FloatTensor(xtest_array)

train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    train_val_dataset, 
    [train_size, val_size])

train_dataloader = DataLoader(
    train_dataset, 
    batch_size = 64,
    num_workers=4, prefetch_factor=2)

val_dataloader = DataLoader(
    val_dataset, 
    batch_size = 64,num_workers=4, prefetch_factor=2)

model = nn.Sequential(
    nn.Linear(in_features = 18, out_features = 64),
    nn.ReLU(),
    nn.BatchNorm1d(64),
    nn.Linear(in_features = 64, out_features = 32),
    nn.ReLU(),
    nn.BatchNorm1d(32),
    nn.Linear(in_features = 32, out_features = 16),
    nn.ReLU(),
    nn.BatchNorm1d(16),
    nn.Linear(in_features = 16, out_features = 8),
    nn.ReLU(),
    nn.BatchNorm1d(8),
    nn.Linear(in_features = 8, out_features = 4),
    nn.ReLU(),
    nn.BatchNorm1d(4),
    nn.Linear(in_features = 4, out_features = 1),
    nn.Sigmoid())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(
    model.parameters(), lr = 1e-3)
num_epochs = 10
# 訓練循環
for epoch in range(num_epochs):
    model.train()
    train_preds = []
    train_targets = []
    for batch_inputs, batch_targets in train_dataloader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()     
        train_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        train_targets.extend(batch_targets.cpu().numpy())
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for batch_inputs, batch_targets in val_dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)  
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            val_targets.extend(batch_targets.cpu().numpy())
    train_auc = metrics.roc_auc_score(train_targets, train_preds)
    val_auc = metrics.roc_auc_score(val_targets, val_preds)
    print(f' Epoch [{epoch + 1} / {num_epochs}]')
    print(f' Train AUC : {train_auc:.4f}, Val AUC : {val_auc:.4f}')
    
with torch.no_grad():
    model.eval()  
    xtrain_tensor = xtrain_tensor.to(device)
    xtrain_outputs = model(xtrain_tensor)
    xtrain_proba = torch.sigmoid(xtrain_outputs)
    xtrain_pred = (xtrain_proba > 0.5).int() 
    xtrain_pred = xtrain_pred.cpu().numpy()
    xtrain_proba = xtrain_proba.cpu().numpy()
    
    xtest_tensor = xtest_tensor.to(device)
    xtest_outputs = model(xtest_tensor)
    xtest_proba = torch.sigmoid(xtest_outputs)
    xtest_pred = (xtest_proba > 0.5).int()  
    xtest_pred = xtest_pred.cpu().numpy()  
    xtest_proba = xtest_proba.cpu().numpy()
    
train_accuracy = metrics.accuracy_score(
    ytrain, xtrain_pred)
test_accuracy = metrics.accuracy_score(
    ytest, xtest_pred)
train_f1 = metrics.f1_score(
    ytrain, xtrain_pred)
test_f1 = metrics.f1_score(
    ytest, xtest_pred)
train_auc = metrics.roc_auc_score(
    ytest, xtest_pred)
test_auc = metrics.roc_auc_score(
    ytest, xtest_proba)
print(f' train_tuning_ann Accuracy Score: {train_accuracy:.5f}')
print(f' test_tuning_ann Accuracy Score: {test_accuracy:.5f}')
print(f' train_tuning_ann F1 score: {train_f1:.5f}')
print(f' test_tuning_ann F1 score: {test_f1:.5f}')
print(f' train_tuning_ann AUC score: {train_auc:.5f}')
print(f' test_tuning_ann AUC score: {test_auc:.5f}')