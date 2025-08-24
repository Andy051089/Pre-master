import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt

data = pd.read_csv('C:/AI/practice/LinerRegression/Salary_dataset.csv')
data.columns


X = data[['YearsExperience']]
y = data[['Salary']]

random_state = 42
test_size = 0.3
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, 
                                                test_size = test_size, 
                                                random_state = random_state)


reg = linear_model.SGDRegressor(random_state = random_state)
reg_param_dist = {'loss' : ['squared_error'],
                  'learning_rate' : ['optimal'],
                  'penalty' : ['l2', 'l1', 'elasticnet', 'None'],
                  'max_iter' : range(1000, 2001),
                  'alpha' : np.arange(0.0001,0.1,0.01),
                  'fit_intercept' : [True, False]}

random_search = RandomizedSearchCV(estimator = reg, 
                                   param_distributions = reg_param_dist,
                                   n_iter = 500,
                                   random_state = random_state,
                                   scoring = 'neg_mean_squared_error',
                                   n_jobs = -1, 
                                   cv = 5)
random_search.fit(Xtrain, ytrain)
best_reg_random_params = random_search.best_params_
print(f' 最佳參數:{best_reg_random_params}')

salary = linear_model.SGDRegressor(**best_reg_random_params, random_state = random_state)
salary.fit(Xtrain, ytrain)
ytrain_pred = salary.predict(Xtrain)
ytest_pred = salary.predict(Xtest)

print(f' 訓練資料MSE: {np.sqrt(mean_squared_error(ytrain, ytrain_pred))}')
print(f' 測試資料MSE: {np.sqrt(mean_squared_error(ytest, ytest_pred))}')
plt.scatter(Xtrain, ytrain, color = 'black')
plt.scatter(Xtest, ytest, color = 'red')
plt.plot(Xtest, ytest_pred, color = 'blue', linewidth = 1)
plt.show()