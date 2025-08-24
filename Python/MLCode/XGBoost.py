#%% 引用模組
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, TargetEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics
from skopt import BayesSearchCV, space
import xgboost as xgb
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
#%% XGBoost
'''
1.監督式學習。當有較大的bias時，boosting則會是其中一個方式，與random forest的差別為下一顆生產的樹，會更加專注上一顆樹分類不好的資料，並加重
    權重，使得這一顆生產的樹可以改善之前分類不好的資料。
2.在每一顆樹的生成時，只使用subsample設定的比例做為生成樹的資料。也只使用colsample_bytree設定比例，作為可以被選為作為分支的特徵條件。在一顆
    樹的生成時，每一次的分支，會計算每個不同特徵條件下，實際與預測結果計算得出的一個值(gain值)，來決定使用哪個特徵條件做分支，直至max_depth設
    定的最大層數，完成一顆完整樹的生成。下一顆樹會計算上一顆完整樹的實際與預測差距，加重分類不好資料權重，並專注於分類不好的資料。直至
    n_estimators設定的顆樹為止。最終預測結果由初始值與learning rate及每顆樹所分類到的葉上值計算所得。另外在預防overfitting上，可以通過計
    算min_child_weight(每個葉最小的樣本數)、gamma(設定葉上最小的GAIN值)來限制葉是否繼續分支。或是透過正規化reg_alpha(L1)
    、reg_lambda(L2)降低一點bias來大大提高variance。
3.EARLY STOPPING:
    當訓練模型過程中，評估指標並沒有改善時，停止訓練。
    1.避免浪費資源
    2.一直讓模型訓練下去時，模型為了要更好的學習訓練資料時，會產生Over fitting的問題
    在XGBoost中，可能已經找到最佳的n_estimators，使用EarlyStopping。
    
    把原本訓練資料再次切格成訓練驗證資料，當生成的樹在驗證資料中設定
    early_stopping_rounds的次數，並沒改善的評估分數，就會停止生成樹，決定最佳n_estimators。
    
4.
objective : 決定XGBoost執行甚麼任務，最終結果數值轉換
learning_rate : 此超參數於樹的生成及最終做預測計算皆有影響
n_estimators : 總共產生幾棵樹
colsample_bytree : 生成每顆樹使用多少比例的特徵
scale_pos_weight : 不平衡資料需要調整
reg_alpha : 正規劃L1
reg_lambda : 正規劃L2
gamma : 決定一個葉是否繼續做分支
min_child_weight : 決定一個葉是否繼續做分支
subsample : 使用多少比例的資料生成每顆樹
scale_pos_weight : 把樣本數多/樣本數少 = 權重小/權重大
tree_method = "hist"、device = "cuda" : 使用GPU做運算
設定在建XGB n_estimators : 最大生成幾顆
early_stopping_rounds : 多少次之後仍無改善就停止
FIT處eval_set : 設定EARLY STOPPING的評估驗證資料資料
.get_booster().best_iteration : 把最終的n_estimators拿出

1.一定要切割在切割，沒有驗證集不能做
2.只要不寫early stopping，寫上其他都不會執行，結果都一樣
3.有沒有寫EVAL沒差，EVAL寫logloss，結果不變
4.有沒有寫n estermate，有差
5.寫在XGB還是BSCV，有差
'''
# 把原本的訓練資料再分一次成訓練驗證資料
X_train, X_val, y_train, y_val = train_test_split(
    xtrain, ytrain, 
    test_size = 0.1, 
    random_state = random_state)
# 把分割再分割的訓練驗證資料重新算不平衡比例
weights_for_es = compute_class_weight(
    class_weight = "balanced", 
    classes = np.array([0, 1]), 
    y = y_train)
scale_pos_weight = weights_for_es[1] / weights_for_es[0]
scale_pos_weight1 = sum(y_train == 0) / sum(y_train == 1)
# 一般XGBOOST
scale_pos_weight_original = sum(ytrain == 0) / sum(ytrain == 1)
normal_xgb = xgb.XGBClassifier(
    objective = 'binary:logistic', 
    random_state = random_state,
    scale_pos_weight = scale_pos_weight_original,
    tree_method = "hist", 
    device = "cuda",
    n_estimators = 1000,
    learning_rate = 0.1,
    max_depth = 10,
    subsample = 0.6,
    colsample_bytree = 0.6,
    gamma = 2,
    min_child_weight = 2,
    reg_alpha = 0.1,
    reg_lambda = 0.1)
normal_xgb.fit(
    xtrain, ytrain)
ytrain_normal_xgb_pred = normal_xgb.predict(xtrain)
ytest_normal_xgb_pred = normal_xgb.predict(xtest)
ytrain_normal_xgb_pred_proba = normal_xgb.predict_proba(xtrain)[:,1]
ytest_normal_xgb_pred_proba = normal_xgb.predict_proba(xtest)[:,1]
train_accuracy = normal_xgb.score(
    xtrain, ytrain)
test_accuracy = normal_xgb.score(
    xtest, ytest)
train_f1 = metrics.f1_score(
    ytrain, ytrain_normal_xgb_pred)
test_f1 = metrics.f1_score(
    ytest, ytest_normal_xgb_pred)
train_auc = metrics.roc_auc_score(
    ytrain, ytrain_normal_xgb_pred_proba)
test_auc = metrics.roc_auc_score(
    ytest, ytest_normal_xgb_pred_proba)
print(f' train_normal_xgb Accuracy Score: {train_accuracy:.5f}')
print(f' test_normal_xgb Accuracy Score: {test_accuracy:.5f}')
print(f' train_normal_xgb F1 score: {train_f1:.5f}')
print(f' test_normal_xgb F1 score: {test_f1:.5f}')
print(f' train_normal_xgb AUC score: {train_auc:.5f}')
print(f' test_normal_xgb AUC score: {test_auc:.5f}')

# 創建模型
tuning_xgb = xgb.XGBClassifier(
    objective = 'binary:logistic', 
    random_state = random_state,
    scale_pos_weight = scale_pos_weight,
    tree_method = "hist", 
    device = "cuda",     
    n_estimators = 1000,   
    early_stopping_rounds = 10)  
# 參數範圍
bayes_tuning_xgb_param = {
    'learning_rate' : space.Real(0.01, 0.3, prior = 'uniform'),
    'max_depth' : space.Integer(3, 10),        
    'subsample' : space.Real(0.5, 1, prior = 'uniform'),
    'colsample_bytree' : space.Real(0.5, 1, prior = 'uniform'),
    'gamma' : space.Real(0, 10, prior = 'uniform'),
    'min_child_weight' : space.Integer(0, 10),
    'reg_lambda' : space.Real(0, 1, prior = 'uniform'),
    'reg_alpha' : space.Real(0, 1, prior = 'uniform')}
# 建立BayesSearchCV
bayes_tuning_xgb_search = BayesSearchCV(
    estimator = tuning_xgb, 
    search_spaces = bayes_tuning_xgb_param,
    n_iter = 10,
    random_state = random_state,
    n_jobs = -1, 
    cv = 5,
    pre_dispatch = '2*n_jobs',
    verbose = 2,
    scoring = scoring)
# 把資料FIT進去找最佳超參數
bayes_tuning_xgb_search.fit(
    X_train, y_train,
    eval_set = [(X_val, y_val)])
# 最佳參數
best_bayes_tuning_xgb_params = bayes_tuning_xgb_search.best_params_
best_bayes_tuning_xgb_estimator = bayes_tuning_xgb_search.best_estimator_
# for ES 機棵樹
best_tuning_xgb_nestimator = best_bayes_tuning_xgb_estimator.get_booster().best_iteration
# 把最佳超參數存下來
file_name = "C:/Users/88691/Desktop/自學/AI/practice/best_bayes_tuning_xgb_params.pkl"
pickle.dump(best_bayes_tuning_xgb_params, open(file_name, "wb"))
best_bayes_tuning_xgb_params = pickle.load(open(file_name, "rb"))
file_name = "C:/Users/88691/Desktop/自學/AI/practice/best_bayes_tuning_xgb_estimator.pkl"
pickle.dump(best_bayes_tuning_xgb_estimator, open(file_name, "wb"))
best_bayes_tuning_xgb_estimator = pickle.load(open(file_name, "rb"))
# 最終MODEL用原本的訓練測試資料，計算scale_pos_weight
all_pos_weight = weights[1] / weights[0]
# 把最佳超參數FIT進XGB
bayes_tuning_xgb = xgb.XGBClassifier(
    objective = 'binary:logistic', 
    **best_bayes_tuning_xgb_params,
    n_estimators = best_tuning_xgb_nestimator, 
    random_state = random_state,
    scale_pos_weight = all_pos_weight)
# bayes_tuning_xgb = xgb.XGBClassifier(
#     objective = 'binary:logistic', 
#     random_state = random_state,
#     scale_pos_weight = all_pos_weight,
#     colsample_bytree = 0.8670140089927842,
#     gamma = 9.393697376027717,
#     learning_rate = 0.05744608180517957,
#     max_depth = 4,
#     min_child_weight = 8,
#     reg_alpha = 0.37257977798325786,
#     reg_lambda = 0.4590245141508057,
#     subsample = 0.7673825800605678,
#     n_estimators = best_xgb_nestimator)
# 把資料FIT進最佳超參數的xgb
best_tuning_xgb_model = bayes_tuning_xgb.fit(
    X_train, y_train)
# 把訓練測試資料FIT模型做預測看結果
ytrain_best_tuning_xgb_pred = best_tuning_xgb_model.predict(xtrain)
ytest_best_tuning_xgb_pred = best_tuning_xgb_model.predict(xtest)
# 把訓練測試資料FIT模型做預測看結果的機率
ytrain_best_tuning_xgb_pred_proba = best_tuning_xgb_model.predict_proba(xtrain)[:,1]
ytest_best_tuning_xgb_pred_proba = best_tuning_xgb_model.predict_proba(xtest)[:,1]
train_accuracy = best_tuning_xgb_model.score(
    xtrain, ytrain)
test_accuracy = best_tuning_xgb_model.score(
    xtest, ytest)
train_f1 = metrics.f1_score(
    ytrain, ytrain_best_tuning_xgb_pred)
test_f1 = metrics.f1_score(
    ytest, ytest_best_tuning_xgb_pred)
train_auc = metrics.roc_auc_score(
    ytrain, ytrain_best_tuning_xgb_pred_proba)
test_auc = metrics.roc_auc_score(
    ytest, ytest_best_tuning_xgb_pred_proba)
print(f' train_tuning_tree Accuracy Score: {train_accuracy:.5f}')
print(f' test_tuning_tree Accuracy Score: {test_accuracy:.5f}')
print(f' train_tuning_tree F1 score: {train_f1:.5f}')
print(f' test_tuning_tree F1 score: {test_f1:.5f}')
print(f' train_tuning_tree AUC score: {train_auc:.5f}')
print(f' test_tuning_tree AUC score: {test_auc:.5f}')