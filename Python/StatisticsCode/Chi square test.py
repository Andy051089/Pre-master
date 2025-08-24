import pandas as pd
from scipy.stats import chi2_contingency
import pingouin as pg


csv_file = 'C:/Users/88691/Desktop/自學/STATISTICS/檢定程式/Chi square test/LungCapData.csv'
data = pd.read_csv(csv_file) 


data.head()

table = pd.crosstab(data['Gender'], data['Smoke'])
chi2_contingency(table)

pg.chi2_independence(data, x = 'Gender', y = 'Smoke')