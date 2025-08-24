import pandas as pd
from scipy.stats import ttest_rel
import pingouin as pg

csv_files = 'C:/Users/88691/Desktop/自學/STATISTICS/檢定程式/Paired t-test/BloodPressure.csv'
BP = pd.read_csv(csv_files)
BP.head()
ttest_rel(BP['Before'], BP['After'], alternative = 'less')



Answer = pg.ttest(BP['Before'], BP['After'], 
         paired = True, 
         confidence = 0.95, 
         alternative = 'two-sided')


