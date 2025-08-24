import pandas as pd
from scipy.stats import ttest_ind, levene

csv_file = 'C:/Users/88691/Desktop/自學/STATISTICS/檢定程式/Two sample t-test/LungCapData.csv'
Data = pd.read_csv(csv_file)

Data.head()

No_smoke = Data.loc[Data['Smoke'] == 'no']['LungCap']
Yes_smoke = Data.loc[Data['Smoke'] == 'yes']['LungCap']


ttest_ind(Yes_smoke, No_smoke,
          equal_var = False, 
          alternative = 'greater')

levene(Yes_smoke, No_smoke, center = 'mean')