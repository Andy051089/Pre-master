import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, TargetEncoder, StandardScaler
from sklearn.impute import KNNImputer

file = 'C:\研究所\自學\各模型\DATA\heart_disease_uci.csv'
data = pd.read_csv(file)
new_df = data.copy()
data.info()
data.isna().sum()
new_df.columns
new_df = new_df.drop(['id'], axis = 1)
new_df['age'].describe()
new_df['Age Group'] = pd.cut(
    new_df['age'],
    bins = [20, 30, 40, 50, 60, 70, 80],
    labels = ['20-30', '31-40', '41-50', '51-60', '61-70',
              '71-80'])
new_df = new_df.drop(['age'], axis = 1)
x = new_df.drop(['num'], axis = 1)
y = new_df['num']
test_size = 0.2
random_state = 42
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y,
    test_size = test_size,
    random_state = random_state)
xtrain.columns
cols_to_targetencode = [
    'sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope',
    'ca', 'thal', 'Age Group']
targetencoder = TargetEncoder()
for col in cols_to_targetencode:
    xtrain[col] = targetencoder.fit_transform(xtrain[[col]], ytrain)
    xtest[col] = targetencoder.transform(xtest[[col]])
standardscaler = StandardScaler()
for col in xtrain.columns:
    xtrain[col] = standardscaler.fit_transform(xtrain[[col]])
    xtest[col] = standardscaler.transform(xtest[[col]])
labelencoder = LabelEncoder()
ytrain = labelencoder.fit_transform(ytrain)
ytest = labelencoder.fit_transform(ytest)
# 找最相近鄰的兩筆，進行填補
knnimputer = KNNImputer(n_neighbors = 2)
xtrain_imputed = knnimputer.fit_transform(xtrain)
xtest_imputed = knnimputer.transform(xtest)
xtrain_imputed = pd.DataFrame(
    xtrain_imputed, 
    columns = xtrain.columns, 
    index = xtrain.index)
xtest_imputed = pd.DataFrame(
    xtest_imputed, 
    columns = xtest.columns, 
    index = xtest.index)
