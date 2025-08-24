import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


data = pd.read_csv('C:/AI/practice/LinerRegression/Salary_dataset.csv')
data.columns


X = data[['YearsExperience']]
y = data[['Salary']]

random_state = 42
test_size = 0.3
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, 
                                                test_size = test_size, 
                                                random_state = random_state)

salary = linear_model.LinearRegression()
salary.fit(Xtrain, ytrain)
ytrain_pred = salary.predict(Xtrain)
ytest_pred = salary.predict(Xtest)

print(f' 訓練資料Accuracy Score: {salary.score(Xtrain, ytrain)}')
print(f' 測試資料Accuracy Score: {salary.score(Xtest, ytest)}')
print(f' 訓練資料MSE: {mean_squared_error(ytrain_pred, ytrain)}')
print(f' 測試資料MSE: {mean_squared_error(ytest_pred, ytest)}')