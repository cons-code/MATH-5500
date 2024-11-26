# %%
import pandas as pd
import seaborn as sns

# %%
df = pd.read_csv('Data/combined.csv', encoding='latin-1')

sns.pairplot(df[['Salary', 'GP', 'PTS', '+/-', 
                 'Grit', 'TOI', 'PIM', 'Is_Offense']])

# %%
numeric = df.select_dtypes(include='number')
cols = numeric.columns.drop('Salary')

for col in cols:
    print(col, df['Salary'].corr(df[col]))

# %%
cols = numeric.columns.drop('PTS')
for col in cols:
    print(col, df['PTS'].corr(df[col]))

# %%
cols = numeric.columns.drop('Salary')
msk = numeric['Salary'].ge(1)

for col in cols:
    print(col, df['Salary'][msk].corr(df[col][msk]))