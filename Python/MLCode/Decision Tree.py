#%% 引用模組
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, TargetEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics, tree
from skopt import BayesSearchCV, space
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
#%% 決策樹
'''
1.監督式學習。把資料透過不同特徵條件做分類，在做每一次分類分支決定時，把所有特徵條件都做一次分支，並計算每個特徵條件分支後
    criterion(gini, entropy, information gain)，由計算結果去決定哪個條件分類的比較好。但如果把資料全部分完，雖然訓練資料有很好
    的結果，但沒辦法泛化，會有overfitting的問題，可以透過調整max_depth、min_samples_leaf、min_samples_split，去決定數的發展，
    或是分到最後之後，通過計算每個葉的alpha(ccp_alpha)，去做修剪樹的動作。
2.
criterion : 通過不同計算方式決定樹分支條件
max_depth : 決定樹最大的層樹
max_features : 決定每次分裂時考慮多少特徵
ccp_alpha : 通過計算如何去修剪樹
min_samples_leaf : 每個葉最少需要多少樣本
min_samples_split : 葉必須要有多少樣本才會分
splitter : 直接使用best，通過criterion所得到的方式做最好的選擇
n_jobs = -1 : 把所有可用的CPU都用
pre_dispatch : 把2倍CPU的工作量，分給所有CPU
'''
# 一般決策樹
normal_tree = tree.DecisionTreeClassifier(
    random_state = random_state,
    criterion = 'entropy',
    max_depth = 10,
    max_features = 18,
    min_samples_leaf = 10,
    min_samples_split = 10,
    ccp_alpha = 0.1,
    class_weight = class_weight)
normal_tree.fit(
    xtrain, ytrain)
ytrain_normal_tree_pred = normal_tree.predict(xtrain)
ytest_normal_tree_pred = normal_tree.predict(xtest)
ytrain_normal_tree_pred_proba = normal_tree.predict_proba(xtrain)[:,1]
ytest_normal_tree_pred_proba = normal_tree.predict_proba(xtest)[:,1]
train_accuracy = normal_tree.score(
    xtrain, ytrain)
test_accuracy = normal_tree.score(
    xtest, ytest)
train_f1 = metrics.f1_score(
    ytrain, ytrain_normal_tree_pred)
test_f1 = metrics.f1_score(
    ytest, ytest_normal_tree_pred)
train_auc = metrics.roc_auc_score(
    ytrain, ytrain_normal_tree_pred_proba)
test_auc = metrics.roc_auc_score(
    ytest, ytest_normal_tree_pred_proba)
print(f' train_normal_tree Accuracy Score: {train_accuracy:.5f}')
print(f' test_normal_tree Accuracy Score: {test_accuracy:.5f}')
print(f' train_normal_tree F1 score: {train_f1:.5f}')
print(f' test_normal_tree F1 score: {test_f1:.5f}')
print(f' train_normal_tree AUC score: {train_auc:.5f}')
print(f' test_normal_tree AUC score: {test_auc:.5f}')
# 創建模型
tuning_tree = tree.DecisionTreeClassifier(
    random_state = random_state,
    class_weight = class_weight)
# 建立參數範圍
bayes_tuning_tree_param = {
    'criterion': space.Categorical(['entropy', 'gini']),
    'max_depth': space.Integer(1, 15),  
    'max_features': space.Integer(1, 18),
    'min_samples_leaf': space.Integer(1, 19),  
    'min_samples_split': space.Integer(2, 19), 
    'ccp_alpha': space.Real(0.0, 0.1, prior = 'uniform')}
# 建立BayesSearchCV
bayes_tuning_tree_search =  BayesSearchCV(
    estimator = tuning_tree, 
    search_spaces = bayes_tuning_tree_param,
    n_iter = n_iter,
    random_state = random_state,
    n_jobs = -1,     
    cv = cv,
    pre_dispatch = '2*n_jobs',   
    verbose = 2,
    scoring = scoring)
# 把資料FIT進去找最佳超參數
bayes_tuning_tree_search.fit(
    xtrain, ytrain)
# 最佳超參數
best_bayes_tuning_tree_params = bayes_tuning_tree_search.best_params_
# 把最佳超參數存下來
file_name = "C:/Users/88691/Desktop/自學/AI/practice/best_tuning_tree_params.pkl"
pickle.dump(best_bayes_tuning_tree_params, open(file_name, "wb"))
best_bayes_tuning_tree_params = pickle.load(open(file_name, "rb"))
# 把最佳超參數FIT進決策樹
bayes_tuning_tree = tree.DecisionTreeClassifier(
    **best_bayes_tuning_tree_params, 
    random_state = random_state,
    class_weight = class_weight)
# bayes_tuning_tree = tree.DecisionTreeClassifier(
#     ccp_alpha = 0.0,
#     criterion = 'entropy',
#     max_depth = 8,
#     max_features = 14,
#     min_samples_leaf = 7,
#     min_samples_split = 5,
#     random_state = random_state,
#     class_weight = class_weight)
# 把資料FIT進最佳超參數的決策樹
best_tuning_tree_model = bayes_tuning_tree.fit(
    xtrain, ytrain)
#把訓練測試資料FIT模型做預測看結果
ytrain_best_tuning_tree_pred = best_tuning_tree_model.predict(xtrain)
ytest_best_tuning_tree_pred = best_tuning_tree_model.predict(xtest)
#把訓練測試資料FIT模型做預測看結果的機率
ytrain_best_tuning_tree_pred_proba = best_tuning_tree_model.predict_proba(xtrain)[:,1]
ytest_best_tuning_tree_pred_proba = best_tuning_tree_model.predict_proba(xtest)[:,1]
train_accuracy = best_tuning_tree_model.score(
    xtrain, ytrain)
test_accuracy = best_tuning_tree_model.score(
    xtest, ytest)
train_f1 = metrics.f1_score(
    ytrain, ytrain_best_tuning_tree_pred)
test_f1 = metrics.f1_score(
    ytest,ytest_best_tuning_tree_pred)
train_auc = metrics.roc_auc_score(
    ytrain, ytrain_best_tuning_tree_pred_proba)
test_auc = metrics.roc_auc_score(
    ytest, ytest_best_tuning_tree_pred_proba)
print(f' train_tuning_tree Accuracy Score: {train_accuracy:.5f}')
print(f' test_tuning_tree Accuracy Score: {test_accuracy:.5f}')
print(f' train_tuning_tree F1 score: {train_f1:.5f}')
print(f' test_tuning_tree F1 score: {test_f1:.5f}')
print(f' train_tuning_tree AUC score: {train_auc:.5f}')
print(f' test_tuning_tree AUC score: {test_auc:.5f}')