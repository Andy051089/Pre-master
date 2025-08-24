import pandas as pd
import pingouin as pg
import scikit_posthocs as sp

csv_file = 'C:/Users/88691/Desktop/自學/STATISTICS/檢定程式/Kruskal Wallis/DietWeightLoss.csv'
data = pd.read_csv(csv_file)

pg.kruskal(data = data, 
           dv = 'WeightLoss', 
           between = 'Diet', 
           detailed = True)

sp.posthoc_dunn(a = data, 
           val_col = 'WeightLoss', 
           group_col = 'Diet',
           p_adjust = 'bonferroni')