import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn import metrics
from pycaret.classification import setup, create_model, tune_model, compare_models, interpret_model
import matplotlib.pyplot as plt
#讀取檔案，設定標籤及特徵資料
data = pd.read_csv('C:/AI/practice/diabetes.csv')
data.columns
data['Diabetes_binary'].value_counts()
X = data.drop(['Diabetes_binary'], axis = 1)
y = data[['Diabetes_binary']]

#設定相同超參數
random_state = 42
test_size = 0.3
n_iter = 500
cv = 5
n_jobs = -1
scoring = 'roc_auc'

#分割訓練及測試資料
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, 
                                                test_size = test_size, 
                                                random_state = random_state)

#創建決策樹及設定RandomizedSearchCV超參數範圍											
diabete_tree = tree.DecisionTreeClassifier(random_state = random_state)
tree_param_dist = {'criterion' : ['entrophy', 'gini'],
		   'max_depth' : range(1, 16),
		   'splitter' : ['best', 'random'],
		   'max_features' : range(1, 22),
		   'min_samples_leaf' : range(50, 101) }

#使用RandomizedSearchCV得出最佳超參數
random_search = RandomizedSearchCV(estimator = diabete_tree, 
                                   param_distributions = tree_param_dist,
                                   n_iter = n_iter,
                                   random_state = random_state,
                                   scoring = scoring,
                                   n_jobs = n_jobs, 
                                   cv = cv)
random_search.fit(Xtrain, ytrain)
best_tree_random_params = random_search.best_params_
print(f'最佳參數:{best_tree_random_params}')

#使用最佳超參數決策樹模型下Accuracy Score、F1 score、AUC score
random_diabete_tree = tree.DecisionTreeClassifier(**best_tree_random_params,
						  random_state = random_state)
random_diabete_tree.fit(Xtrain, ytrain)
ytrain_random_tree_pred = random_diabete_tree.predict(Xtrain)
ytest_random_tree_pred = random_diabete_tree.predict(Xtest)
ytrain_random_tree_pred_proba = random_diabete_tree.predict_proba(Xtrain)[:,1]
ytest_random_tree_pred_proba = random_diabete_tree.predict_proba(Xtest)[:,1]
print(f' 決策樹RandomizedSearchCV最佳超參數下訓練資料Accuracy Score: {random_diabete_tree.score(Xtrain, ytrain)}')
print(f' 決策樹RandomizedSearchCV最佳超參數下測試資料Accuracy Score: {random_diabete_tree.score(Xtest, ytest)}')
print(f' 決策樹RandomizedSearchCV最佳超參數下訓練資料F1 score: {metrics.f1_score(ytrain, ytrain_random_tree_pred)}')
print(f' 決策樹RandomizedSearchCV最佳超參數下測試資料F1 score: {metrics.f1_score(ytest, ytest_random_tree_pred)}')
print(f' 決策樹RandomizedSearchCV最佳超參數下訓練資料AUC score: {metrics.roc_auc_score(ytrain, ytrain_random_tree_pred_proba)}')
print(f' 決策樹RandomizedSearchCV最佳超參數下測試資料AUC score: {metrics.roc_auc_score(ytest, ytest_random_tree_pred_proba)}')

#創建XGBoost及設定RandomizedSearchCV超參數範圍	
xgb = XGBClassifier(objective = 'binary:logistic', random_state = random_state)
xgb_param_dist = {'n_estimators' : range(100, 501),
		  'learning_rate' : np.arange(0.01, 0.3, 0.01),
		  'max_depth' : range(1, 16),
		  'colsample_bytree' : np.arange(0.1, 1.1, 0.1)}

#使用RandomizedSearchCV得出最佳超參數   
random_search = RandomizedSearchCV(estimator = xgb, 
                                   param_distributions = xgb_param_dist,
                                   n_iter = n_iter,
                                   random_state = random_state,
                                   scoring = scoring,
                                   n_jobs = n_jobs, 
                                   cv = cv)
random_search.fit(Xtrain, ytrain)
best_xgb_random_params = random_search.best_params_
print(f'最佳參數:{best_xgb_random_params}')

#使用最佳超參數XGBoost模型下Accuracy Score、F1 score、AUC score
random_diabete_xgb = XGBClassifier(**best_xgb_random_params,
				   random_state = random_state)
random_diabete_xgb.fit(Xtrain, ytrain)
ytrain_random_xgb_pred = random_diabete_xgb.predict(Xtrain)
ytest_random_xgb_pred = random_diabete_xgb.predict(Xtest)
ytrain_random_xgb_pred_proba = random_diabete_xgb.predict_proba(Xtrain)[:,1]
ytest_random_xgb_pred_proba = random_diabete_xgb.predict_proba(Xtest)[:,1]
print(f' xgb RandomizedSearchCV最佳超參數下訓練資料Accuracy Score: {random_diabete_xgb.score(Xtrain, ytrain)}')
print(f' xgb RandomizedSearchCV最佳超參數下測試資料Accuracy Score: {random_diabete_xgb.score(Xtest, ytest)}')
print(f' xgb RandomizedSearchCV最佳超參數下訓練資料F1 score: {metrics.f1_score(ytrain, ytrain_random_xgb_pred)}')
print(f' xgb RandomizedSearchCV最佳超參數下測試資料F1 score: {metrics.f1_score(ytest, ytest_random_xgb_pred)}')
print(f' xgb RandomizedSearchCV最佳超參數下訓練資料AUC score: {metrics.roc_auc_score(ytrain, ytrain_random_xgb_pred_proba)}')
print(f' xgb RandomizedSearchCV最佳超參數下測試資料AUC score: {metrics.roc_auc_score(ytest, ytest_random_xgb_pred_proba)}')

#設定pycaret不同模型相同超參數
fold = 5
optimize = 'AUC'
sort = 'AUC'

#分割訓練及測試資料
train, test = train_test_split(data,
                               test_size = test_size,
                               random_state = random_state)

#設定使用的訓練資料及對應標籤並創建想要比較的不同模型
setup(data = train, target = 'Diabetes_binary')
dt = create_model('dt', random_state = random_state)
xgboost = create_model('xgboost', random_state = random_state)

#尋找模型最佳超參數
dt_tune, tuner = tune_model(estimator = dt,
                            fold = fold,
                            n_iter = n_iter,
                            return_tuner = True,
                            optimize = optimize)
xgboost_tune, tuner = tune_model(estimator = xgboost,
				 fold = fold,
				 n_iter = n_iter,
				 return_tuner = True,
				 optimize = optimize)
								
#比較最佳超參數下不同模型
best_model = compare_models([dt_tune, xgboost_tune],
			    sort = sort)

#使用pycaret得出最佳超參數、最佳模型下Accuracy Score、F1 score、AUC score
py_final = best_model.fit(Xtrain, ytrain)
pycaret_best_ytrain_pred = pycaret_final.predict(Xtrain)
pycaret_best_ytest_pred = pycaret_final.predict(Xtest)
pycaret_ytrain_pred_proba = pycaret_final.predict_proba(Xtrain)[:,1]
pycaret_ytest_pred_proba = pycaret_final.predict_proba(Xtest)[:,1]
print(f' pycaret下xgb訓練資料Accuracy Score: {pycaret_final.score(Xtrain, ytrain)}')
print(f' pycaret下xgb測試資料Accuracy Score: {pycaret_final.score(Xtest, ytest)}')
print(f' pycaret下xgb訓練資料F1 score: {metrics.f1_score(ytrain, pycaret_best_ytrain_pred)}')
print(f' pycaret下xgb測試資料F1 score: {metrics.f1_score(ytest, pycaret_best_ytest_pred)}')
print(f' pycaret下xgb訓練資料AUC score: {metrics.roc_auc_score(ytrain, pycaret_ytrain_pred_proba)}')
print(f' pycaret下xgb測試資料AUC score: {metrics.roc_auc_score(ytest, pycaret_ytest_pred_proba)}')

#可解釋AI圖
fig = plt.figure()
interpret_model(random_diabete_xgb)
fig
fig.show()
