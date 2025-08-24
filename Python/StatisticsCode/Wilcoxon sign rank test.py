import pandas as pd
import pingouin as pg
from scipy.stats import wilcoxon

csv_file = 'C:/Users/88691/Desktop/自學/STATISTICS/檢定程式/Wilcoxon signed rank test/BloodPressure.csv'
Bp = pd.read_csv(csv_file)

#1
help(wilcoxon)
wilcoxon(Bp['Before'], Bp['After'],
         alternative = 'greater',
         method = 'exact')

#2
pg.wilcoxon(Bp['Before'], Bp['After'],
         alternative = 'greater',
         method = 'exact')

