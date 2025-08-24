#%% 引用模組
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, TargetEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import pickle
from skopt import BayesSearchCV, space
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
    col.rename(columns = {
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
#%% 隨機森林
'''
1.監督式學習。當使用決策樹仍有overfitting、較大的variance時，可以透過bagging的方式解決，random forest就是常見的bagging方式。透過生成很
    多顆各自獨立的決策樹，最終結果是計算所有樹分類結果較多者。在每顆樹的生成開始時，都會把訊練資料做自抽法至與訓練資料大小相同的資料筆數，隨機抽
    取N個特徵，並在樹的每個分支使用不同特徵條件，並重複n_estimators設定的次數。
2.
criterion : 通過不同計算方式決定樹分支條件
max_depth : 決定樹最大的層樹
max_features : 決定每次分裂時考慮多少特徵
n_estimators : 總共建多少棵樹
min_samples_leaf : 每個葉最少需要多少樣本
min_samples_split : 葉必須要有多少樣本才會分
n_jobs = -1 : 把所有可用的CPU都用
pre_dispatch : 把2倍CPU的工作量，分給所有CPU
'''
# 一般隨機森林
normal_forest = RandomForestClassifier(
    random_state = random_state,
    class_weight = class_weight,
    criterion = 'entropy',
    max_depth = 10,
    n_estimators = 100,
    max_features = 15,
    min_samples_leaf = 10,
    min_samples_split = 10,
    ccp_alpha = 0.05)
normal_forest.fit(
    xtrain, ytrain)
ytrain_normal_forest_pred = normal_forest.predict(xtrain)
ytest_normal_forest_pred = normal_forest.predict(xtest)
# 把訓練測試資料FIT模型做預測看結果的機率
ytrain_normal_forest_pred_proba = normal_forest.predict_proba(xtrain)[:,1]
ytest_normal_forest_pred_proba = normal_forest.predict_proba(xtest)[:,1]
train_accuracy = normal_forest.score(
    xtrain, ytrain)
test_accuracy = normal_forest.score(
    xtest, ytest)
train_f1 = metrics.f1_score(
    ytrain, ytrain_normal_forest_pred)
test_f1 = metrics.f1_score(
    ytest, ytest_normal_forest_pred)
train_auc = metrics.roc_auc_score(
    ytrain, ytrain_normal_forest_pred_proba)
test_auc = metrics.roc_auc_score(
    ytest, ytest_normal_forest_pred_proba)
print(f' train_normal_forest Accuracy Score: {train_accuracy:.5f}')
print(f' test_normal_forest Accuracy Score: {test_accuracy:.5f}')
print(f' train_normal_forest F1 score: {train_f1:.5f}')
print(f' test_normal_forest F1 score: {test_f1:.5f}')
print(f' train_normal_forest AUC score: {train_auc:.5f}')
print(f' test_normal_forest AUC score: {test_auc:.5f}')
# 建模型
tuning_forest = RandomForestClassifier(
    random_state = random_state,
    class_weight = class_weight)
# 建超參數範圍
forest_param_dist = {
    'criterion' : space.Categorical(['entropy', 'gini']),
    'max_depth' : space.Integer(1, 30),
    'n_estimators' : space.Integer(100, 1000),
    'max_features' : space.Integer(1, 18),
    'min_samples_leaf' : space.Integer(1, 10),
    'min_samples_split' : space.Integer(1,100)}
# 建立BayesSearchCV
bayes_tuning_forest_search =  BayesSearchCV(
    estimator = tuning_forest, 
    search_spaces = forest_param_dist,
    n_iter = n_iter,
    random_state = random_state,
    n_jobs = -1,     
    cv = cv,
    pre_dispatch = '2*n_jobs',   
    verbose = 2,
    scoring = scoring)    
#把資料FIT進去找最佳超參數
bayes_tuning_forest_search.fit(
    xtrain, ytrain)
#最佳參數
best_bayes_tuning_forest_params = bayes_tuning_forest_search.best_params_
# 把最佳超參數存下來
file_name = "C:/Users/88691/Desktop/自學/AI/practice/best_tuning_forest_params.pkl"
pickle.dump(best_bayes_tuning_forest_params, open(file_name, "wb"))
best_bayes_tuning_forest_params = pickle.load(open(file_name, "rb"))
# 把最佳超參數FIT進隨機森林
bayes_tuning_forest = RandomForestClassifier(
    **best_bayes_tuning_forest_params, 
    random_state = random_state,
    class_weight = class_weight)
# bayes_tuning_forest = RandomForestClassifier(
    # criterion = 'gini',
    # max_depth = 27,
    # max_features = 6,
    # min_samples_leaf = 10,
    # min_samples_split = 87,
    # random_state = random_state,
    # class_weight = class_weight,
    # n_estimators = 156)
# 把資料FIT進最佳超參數的隨機森林
best_tuning_forest_model = bayes_tuning_forest.fit(
    xtrain, ytrain)
# 把訓練測試資料FIT模型做預測看結果
ytrain_best_tuning_forest_pred = best_tuning_forest_model.predict(xtrain)
ytest_best_tuning_forest_pred = best_tuning_forest_model.predict(xtest)
# 把訓練測試資料FIT模型做預測看結果的機率
ytrain_best_tuning_forest_pred_proba = best_tuning_forest_model.predict_proba(xtrain)[:,1]
ytest_best_tuning_forest_pred_proba = best_tuning_forest_model.predict_proba(xtest)[:,1]
train_accuracy = best_tuning_forest_model.score(
    xtrain, ytrain)
test_accuracy = best_tuning_forest_model.score(
    xtest, ytest)
train_f1 = metrics.f1_score(
    ytrain, ytrain_best_tuning_forest_pred)
test_f1 = metrics.f1_score(
    ytest, ytest_best_tuning_forest_pred)
train_auc = metrics.roc_auc_score(
    ytrain, ytrain_best_tuning_forest_pred_proba)
test_auc = metrics.roc_auc_score(
    ytest, ytest_best_tuning_forest_pred_proba)
print(f' train_tuning_forest Accuracy Score: {train_accuracy:.5f}')
print(f' test_tuning_forest Accuracy Score: {test_accuracy:.5f}')
print(f' train_tuning_forest F1 score: {train_f1:.5f}')
print(f' test_tuning_forest F1 score: {test_f1:.5f}')
print(f' train_tuning_forest AUC score: {train_auc:.5f}')
print(f' test_tuning_forest AUC score: {test_auc:.5f}')