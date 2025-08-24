#%% 引用模組
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, TargetEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
#%% 讀資料
data_file = 'C:/研究所/自學/各模型/DATA/Heart Disease.csv'
df = pd.read_csv(data_file)
new_df = df.copy()
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
    xtest[col] = encoder.transform(
        xtest[[col]])
# 把數值資料轉換平均:0、標準差1(Z分佈)
scaler = StandardScaler()
for col in xtrain.columns:
    xtrain[col] = scaler.fit_transform(
        xtrain[[col]])
    xtest[col] = scaler.transform(
        xtest[[col]])
# 把目標變數轉為0,1
label_encoder = LabelEncoder()
ytrain = label_encoder.fit_transform(ytrain)
ytest = label_encoder.fit_transform(ytest)
x_train_test = [xtrain, xtest]
for col in x_train_test:
    col.rename(columns = {
        'Height_(cm)' : 'Height',
        'Weight_(kg)' : 'Weight'},
        inplace = True)
#%%特徵選取
'''
特徵選取是降維的一種方式，挑選出對預測結果有顯著影響的特徵，以提高模型效能、減少訓練時間及防止過擬合等。
*高線性及高非線性關係，不一定代表高預測貢獻性。可能存在低線性、低非線性關係，但高貢獻性的情況(當與其他特徵結合時，可能在預測中發揮關鍵作用)
1.過濾法(統計方式:相關係數、卡方檢驗、信息增益等）
    1.透過對單個X對Y的關係
    2.計算速度快
    3.適合大資料庫
    4.減少過擬合風險
2.包裝法(基於模型:逐步向前選取、逐步向後選取、遺傳演算法)
    1.同時考慮多個X對Y的關係
    2.對非線性問題較適合
    3.當特徵本就很少時，容易過擬合
    4.透過訓練模型所的，計算成本較大
'''
#過濾法
#計算相互資訊
mi_scores = mutual_info_classif(xtrain, ytrain)
feature_names = xtrain.columns.tolist()
mi_feature_pairs = list(zip(feature_names, mi_scores))
#由高到低排序
sorted_mi_features = sorted(mi_feature_pairs, key=lambda x: x[1], reverse=True)
#選取前幾個
'''
0:特徵和目標變量之間沒有互信息，即它們是獨立
較大的值表示特徵和目標變量之間有更強的統計依賴關係
'''
k = 20
selected_features = [feature for feature, _ in sorted_mi_features[:k]]
xtrain_filtered = xtrain.loc[:, selected_features]
xtest_filtered = xtest.loc[:, selected_features]

#包裝法
'''
1.逐步選擇法
從空集合開始，每次加入對模型性能提升最大的特徵，直到性能不再顯著改善
2.逐步向後選擇
從所有特徵的集合開始，每次移除對模型性能影響最小的特徵，直到性能開始顯著下降
3.雙向選擇
結合向前和向後的方法，既可以加入新特徵，也可以移除現有特徵
4.遺傳演算法
利用遺傳演算法的選擇、交叉、突變等操作來尋找最佳特徵組合，模擬自然選擇的過程來優化特徵選取
    例如:假如10個X，想要3個X。那就會隨機搭配多組3個X的特徵組合，選擇最好的。
5.遞迴特徵消除(透過訓練模型選特徵)
基於模型的重要性（如線性迴歸係數或支持向量機的權重），每次移除最不重要的特徵，然後重新訓練模型，重複這個過程直到達到預定的特徵數目
    例如:假如10個X，想要3個X。那就會從10個開始移除1個最不重要的，直到剩3個
    
遞迴特徵消除 VS 逐步向後選擇
不同處:
    遞迴特徵消除:在每一步中，反覆訓練模型，決定要刪除哪個特徵
    逐步向後選擇:依據統計顯著性或模型的性能變化，逐步移除特徵，不一定在每一步都重新訓練模型
'''
#遞迴特徵消除
model = RandomForestClassifier(n_estimators = 100, 
                               random_state = random_state)
n_features_to_select = 10
rfe = RFE(model, n_features_to_select=n_features_to_select)
rfe.fit(xtrain, ytrain)
selected_features = rfe.support_
feature_names = xtrain.columns
selected_feature_names = feature_names[selected_features]