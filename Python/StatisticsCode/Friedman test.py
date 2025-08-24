import pandas as pd
import pingouin as pg
import scikit_posthocs as sp

csv_file = 'C:/Users/88691/Desktop/自學/STATISTICS/檢定程式/Friedman/Friedman.csv'
data = pd.read_csv(csv_file)

pg.friedman(data = data,
            dv = 'Rating',
            within = 'Tea',
            subject = 'ID')
#%%
a = data.loc[data['Tea'] == 1]['Rating'].reset_index(drop=True)
b = data.loc[data['Tea'] == 2]['Rating'].reset_index(drop=True)
c = data.loc[data['Tea'] == 3]['Rating'].reset_index(drop=True)

df = pd.DataFrame(data = {
    'T1' : a,
    'T2' : b,
    'T3' : c})

sp.posthoc_nemenyi_friedman(df)
#%%
data_pivot = data.pivot(index='ID', columns='Tea', values='Rating')
sp.posthoc_nemenyi_friedman(data_pivot)
#%%
a = pg.anova(data = data, dv = 'Rating', between = ['Tea', 'ID'])
print(a)
ans = pg.pairwise_tukey(data = data, dv = 'Rating', 
                        between = ['Tea', 'ID'], 
                        effsize = 'cohen')

data.head()
help(pg.pairwise_tukey)
pg.anova2(data = data, 
          dv = 'Rating',
          between = ['Tea', 'ID'], 
          detailed = True)

pg.p