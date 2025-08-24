#%% 引用模組
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, TargetEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
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
test_size = 0.2
cv = 2
n_iter = 2
scoring = 'f1'
threshold = 0.5
#%% 分割出特徵及目標變數資料
x = new_df.drop(['Heart_Disease'], 
                axis = 1)
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
    class_weight = 'balanced', 
    classes = np.array(
    [0, 1]), 
    y = ytrain)
class_weight = {
    0 : weights[0], 
    1 : weights[1]}
scale_pos_weight = sum(ytrain == 0) / sum(ytrain == 1)
#%% Stack
'''
一種集成的方式，分為兩層。第一層可創建選擇多個模型，把原始資料多個X及Y分別給多個模型訓練，每個模型都會預測出一個Y。再把第一層多個型預測的Y及原始資料多個X結合變成第二層的X，並且與原始資料的Y，訓練模型，最終得出結果。
'''
# 第一層基礎模型
estimators = [
    ('dt', DecisionTreeClassifier(random_state = random_state,
                                  class_weight = class_weight)),
    ('rf', RandomForestClassifier(n_estimators = 10, 
                                  random_state = random_state,
                                  class_weight = class_weight,)),
    ('xgb', XGBClassifier(n_estimators = 80, 
                          random_state = random_state,
                          scale_pos_weight = scale_pos_weight))]
# 第二層基礎模型
final_estimators = {
    'RandomForest': RandomForestClassifier(n_estimators = 100, 
                                           random_state = random_state,
                                           class_weight = class_weight),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
    'GaussianNB': GaussianNB(),
    'MLPClassifier': MLPClassifier(random_state = random_state,)}

for name, final_estimator in final_estimators.items():
    stack = StackingClassifier(
        estimators = estimators, 
        final_estimator = final_estimator,
        passthrough = True,
        cv = cv,
        verbose = 2,
        n_jobs = -1)
    
    stack.fit(xtrain, ytrain)
    ytrain_stack_pred = stack.predict(xtrain)
    ytest_stack_pred = stack.predict(xtest)
    ytrain_stack_pred_proba = stack.predict_proba(xtrain)[:,1]
    ytest_stack_pred_proba = stack.predict_proba(xtest)[:,1]
    
    train_accuracy = stack.score(
        xtrain, ytrain)
    test_accuracy = stack.score(
        xtest, ytest)
    train_f1 = metrics.f1_score(
        ytrain, ytrain_stack_pred)
    test_f1 = metrics.f1_score(
        ytest, ytest_stack_pred)
    train_auc = metrics.roc_auc_score(
        ytrain, ytrain_stack_pred_proba)
    test_auc = metrics.roc_auc_score(
        ytest, ytest_stack_pred_proba)
    print(f' final estimator {name} train Accuracy: {train_accuracy:.5f}')
    print(f' final estimator {name} test Accuracy: {test_accuracy:.5f}')
    print(f' final estimator {name} train F1 score: {train_f1:.5f}')
    print(f' final estimator {name} test F1 score: {test_f1:.5f}')
    print(f' final estimator {name} train AUC score: {train_auc:.5f}')
    print(f' final estimator {name} test AUC score: {test_auc:.5f}')
#%%
base_estimators = [
    ('dt', DecisionTreeClassifier(random_state = random_state,
                                  class_weight = class_weight)),
    ('rf', RandomForestClassifier(random_state = random_state,
                                  class_weight = class_weight)),
    ('xgb', XGBClassifier(random_state = random_state,
                          scale_pos_weight = scale_pos_weight))]

final_estimators = {
    'RandomForest': RandomForestClassifier(random_state = random_state,
                                           class_weight = class_weight),
    'GradientBoosting': GradientBoostingClassifier(random_state = random_state),
    'GaussianNB': GaussianNB()}

base_param_space = {
    'dt__criterion': space.Categorical(['entropy', 'gini']),
    'dt__max_depth': space.Integer(1, 15),  
    'dt__max_features': space.Integer(1, 18),
    'dt__min_samples_leaf': space.Integer(1, 19),  
    'dt__min_samples_split': space.Integer(2, 19), 
    
    'rf__criterion': space.Categorical(['entropy', 'gini']),
    'rf__max_depth': space.Integer(1, 30),
    'rf__n_estimators': space.Integer(50, 1000),
    'rf__max_features': space.Integer(1, 18),
    'rf__min_samples_leaf': space.Integer(1, 10),
    'rf__min_samples_split': space.Integer(1, 100),
    
    'xgb__learning_rate': space.Real(0.01, 0.3, prior='uniform'),
    'xgb__n_estimators': space.Integer(50, 1000),
    'xgb__max_depth': space.Integer(3, 10),        
    'xgb__subsample': space.Real(0.5, 1, prior='uniform'),
    'xgb__colsample_bytree': space.Real(0.5, 1, prior='uniform'),
    'xgb__min_child_weight': space.Integer(0, 10)}

final_param_spaces = {
    'RandomForest': {
        'final_estimator__criterion': space.Categorical(['entropy', 'gini']),
        'final_estimator__max_depth': space.Integer(1, 30),
        'final_estimator__n_estimators': space.Integer(100, 1000),
        'final_estimator__max_features': space.Integer(1, 18),
        'final_estimator__min_samples_leaf': space.Integer(1, 10),
        'final_estimator__min_samples_split': space.Integer(1,100)},
    
    'GradientBoosting': {
        'final_estimator__n_estimators': space.Integer(50, 1000),
        'final_estimator__learning_rate': space.Real(0.01, 0.3, prior='uniform'),
        'final_estimator__max_depth': space.Integer(1, 10),
        'final_estimator__min_samples_split': space.Integer(1, 100),
        'final_estimator__min_samples_leaf': space.Integer(1, 10),
        'final_estimator__subsample': space.Real(0.5, 1, prior='uniform'),
        'final_estimator__max_features': space.Integer(1, 18)},
    
    'GaussianNB': {
        'final_estimator__var_smoothing': space.Real(1e-10, 1e-8, prior='uniform')}}

for name, final_estimator in final_estimators.items():
    stack = StackingClassifier(
        estimators = base_estimators, 
        final_estimator = final_estimator, 
        passthrough = True,
        cv = cv,
        n_jobs = -1,
        verbose = 2)
    
    param_space = {**base_param_space, **final_param_spaces[name]}
   
    bayes_search = BayesSearchCV(
        estimator = stack, 
        search_spaces = param_space,
        n_iter = n_iter,
        cv = cv,
        n_jobs = -1,
        random_state = random_state,
        verbose = 2,
        scoring = scoring)
    
    bayes_search.fit(xtrain, ytrain)
    best_stack = bayes_search.best_estimator_
    ytrain_stack_pred = best_stack.predict(xtrain)
    ytest_stack_pred = best_stack.predict(xtest)
    ytrain_stack_pred_proba = best_stack.predict_proba(xtrain)[:,1]
    ytest_stack_pred_proba = best_stack.predict_proba(xtest)[:,1]
    train_accuracy = best_stack.score(xtrain, ytrain)
    test_accuracy = best_stack.score(xtest, ytest)
    train_f1 = metrics.f1_score(ytrain, ytrain_stack_pred)
    test_f1 = metrics.f1_score(ytest, ytest_stack_pred)
    train_auc = metrics.roc_auc_score(ytrain, ytrain_stack_pred_proba)
    test_auc = metrics.roc_auc_score(ytest, ytest_stack_pred_proba)
    print(f' final estimator {name} train Accuracy: {train_accuracy:.5f}')
    print(f' final estimator {name} test Accuracy: {test_accuracy:.5f}')
    print(f' final estimator {name} train F1 score: {train_f1:.5f}')
    print(f' final estimator {name} test F1 score: {test_f1:.5f}')
    print(f' final estimator {name} train AUC score: {train_auc:.5f}')
    print(f' final estimator {name} test AUC score: {test_auc:.5f}')