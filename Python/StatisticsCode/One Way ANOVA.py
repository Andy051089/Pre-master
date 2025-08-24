import pandas as pd
from scipy import stats
import pingouin as pg

csv_file = 'C:/Users/88691/Desktop/自學/STATISTICS/檢定程式/One Way ANOVA/DietWeightLoss.csv'
data = pd.read_csv(csv_file)
data.head()
diet_a = data.loc[data.Diet == 'A']['WeightLoss']
diet_b = data.loc[data.Diet == 'B']['WeightLoss']
diet_c = data.loc[data.Diet == 'C']['WeightLoss']
diet_d = data.loc[data.Diet == 'D']['WeightLoss']

stats.f_oneway(diet_a, diet_b, diet_c, diet_d)

pg.anova(data = data, dv = 'WeightLoss', between = 'Diet', detailed = 'True')
ans = pg.pairwise_tukey(data = data, dv = 'WeightLoss', between = 'Diet', effsize = 'cohen')