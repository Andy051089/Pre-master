import pandas as pd
import pingouin as pg

data = {
    'patient': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
    'time': ['T1', 'T2', 'T3', 'T1', 'T2', 'T3', 'T1', 'T2', 'T3', 'T1', 'T2', 'T3'],
    'value': [5.1, 5.5, 5.8, 6.2, 6.3, 6.4, 5.9, 6.1, 6.0, 6.3, 6.4, 6.5]
}

df = pd.DataFrame(data)

result = pg.rm_anova(dv='value', within='time', subject='patient', 
                     data=df, detailed=True, correction = True)

pg.print_table(result)


ans = pg.pairwise_tests(dv='value', within='time', subject='patient', 
                         data=df, padjust='bonferroni')