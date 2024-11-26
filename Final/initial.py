# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

from src.nhl_linreg import nhl_linreg

# %%
df = pd.read_csv('Data/combined.csv', encoding='latin-1')
sns.pairplot(df[['Salary', 'GP', 'PTS', '+/-', 
                 'Grit', 'TOI', 'PIM', 'Is_Offense']])
plt.show();
# %%
# R^2_a of 0.52
msk = df['Position'].str.contains('D')
Y = df['Salary'][msk]
cols = ['GP', '+/-', 'Grit', 'TOI']

X = df[cols][msk]
X = sm.add_constant(X)
model1 = sm.OLS(Y, X)
results1 = model1.fit()
print(results1.summary())
print(results1.rsquared_adj)
diag1 = nhl_linreg(results1)

vif1, fig1, ax1 = diag1()
print(vif1['VIF Factor'].mean())
# %%
# R^2_a of 0.51
cols = ['GP', 'PTS', '+/-', 'Grit', 'TOI', 'PIM'] #, 'iHF' , 'iRB', 'iSF', 'PIM']
Y = df['Salary']
X = df[cols]
X = sm.add_constant(X)

model2 = sm.OLS(np.log(Y), X)
results2 = model2.fit()
print(results2.summary())
print(results2.rsquared_adj)
diag2 = nhl_linreg(results2)

vif2, fig2, ax2 = diag2()
print(vif2['VIF Factor'].mean())

# %%
Y = df['Salary']
cols = ['GP', 'PTS', '+/-', 'Grit', 'TOI', 'PIM', 'Is_Offense']
X = df[cols]
X = sm.add_constant(X)

model3 = sm.OLS(Y, X)
results3 = model3.fit()
print(results3.summary())
print(results3.rsquared_adj)
diag3 = nhl_linreg(results3)
vif3, fig3, ax3 = diag3()
print(vif3['VIF Factor'].mean()) # 5.8

# %%
msk = df['Is_Offense'].eq(1)
Y = df['Salary'][msk]
cols = ['GP', 'PTS', '+/-', 'Grit', 'TOI', 'PIM']
X = df[cols][msk]
X = sm.add_constant(X)

model4a = sm.OLS(Y, X)
results4a = model4a.fit()
print(results4a.summary())
print(results4a.rsquared_adj)
diag4a = nhl_linreg(results4a)
vif4a, fig4a, ax4a = diag4a()
print(vif4a['VIF Factor'].mean()) # 11.36

# %%
msk = df['Is_Offense'].eq(1)
Y = df['Salary'][msk]
cols = ['GP', 'PTS', '+/-', 'Grit', 'TOI', 'PIM']
X = df[cols][msk]
X = sm.add_constant(X)

model4b = sm.OLS(np.log(Y), X)
results4b = model4b.fit()
print(results4b.summary())
print(results4b.rsquared_adj)
diag4b = nhl_linreg(results4b)
vif4b, fig4b, ax4b = diag4b()
print(vif4b['VIF Factor'].mean()) # 11.36

# %%
msk = df['Is_Offense'].eq(1)
Y = df['Salary'][msk]
cols = ['GP', 'PTS', 'TOI', 'PIM']
X = df[cols][msk]
X = sm.add_constant(X)

model4c = sm.OLS(np.log(Y), X)
results4c = model4c.fit()
print(results4c.summary())
print(results4c.rsquared_adj)
diag4c = nhl_linreg(results4c)
vif4c, fig4c, ax4c = diag4c()
print(vif4c['VIF Factor'].mean()) # 11.36

# %%
Y = df['Salary']
cols = ['GP', 'G', 'A', 'TOI', 'PIM']
X = df[cols]
X = sm.add_constant(X)

model5 = sm.OLS(np.log(Y), X)
results5 = model5.fit()
print(results5.summary())
print(results5.rsquared_adj)
diag5 = nhl_linreg(results5)
vif5, fig5, ax5 = diag5()
print(vif5['VIF Factor'].mean()) # 5.59

# %%
msk = df['GP'].eq(df['GP'].max())
Y = df['Salary'][msk]
cols = ['G', 'A', 'TOI', 'PIM']
X = df[cols][msk]
X = sm.add_constant(X)

model6 = sm.OLS(Y, X)
results6 = model6.fit()
print(results6.summary())
print(results6.rsquared_adj)
diag6 = nhl_linreg(results6)
vif6, fig6, ax6 = diag6()
print(vif6['VIF Factor'].mean()) # 11.734

# %%
msk = df['GP'].ge(70)
Y = df['Salary'][msk]
cols = ['G', 'A', 'TOI', 'PIM']
X = df[cols][msk]
X = sm.add_constant(X)

model7 = sm.OLS(Y, X)
results7 = model7.fit()
print(results7.summary())
print(results7.rsquared_adj)
diag7 = nhl_linreg(results7)
vif7, fig7, ax7 = diag7()
print(vif7['VIF Factor'].mean()) # 8.174

# %%
msk = df['GP'].ge(70)
Y = df['Salary'][msk]
cols = ['G', 'A', 'TOI', 'PIM', 'Is_Offense']
X = df[cols][msk]
X = sm.add_constant(X)

model8 = sm.OLS(Y, X)
results8 = model8.fit()
print(results8.summary())
print(results8.rsquared_adj)
diag8 = nhl_linreg(results8)
vif8, fig8, ax8 = diag8()
print(vif8['VIF Factor'].mean()) # 18.58

# %%
Y = df['Salary']
cols = ['GP', 'PTS', '+/-', 'Grit', 'PIM']
X = df[cols]
X = sm.add_constant(X)
model9 = sm.OLS(np.log(Y), X)
results9 = model9.fit()
print(results9.summary())
print(results9.rsquared_adj)
diag9 = nhl_linreg(results9)
vif9, fig9, ax9 = diag9()
print(vif9['VIF Factor'].mean()) # 3.43

# %%
scaler = StandardScaler()
cols = ['GP', 'PTS', '+/-', 'Grit', 'PIM']
keep = df[cols + ['Salary']]
keep['Salary'] = np.log(keep['Salary'])

scaler.fit(keep)
out = scaler.transform(keep)
scaled = pd.DataFrame(out, columns=keep.columns)

Y = scaled['Salary']
cols = ['GP', 'PTS', '+/-', 'Grit', 'PIM']
X = scaled[cols]
X = sm.add_constant(X)
model10 = sm.OLS(Y, X)
results10 = model10.fit()
print(results10.summary())
print(results10.rsquared_adj)
diag10 = nhl_linreg(results10)
vif10, fig10, ax10 = diag10()
print(vif10['VIF Factor'].mean()) # 2.9

# %%
Y = df['PTS']
X = df[['TOI', '+/-', 'Grit', 'Salary', 'PIM']]
X['TOI'] = X['TOI'].div(60)
X = sm.add_constant(X)
model11 = sm.OLS(Y, X)
results11 = model11.fit()
print(results11.summary())
print(results11.rsquared_adj)
diag11 = nhl_linreg(results11)
vif11, fig11, ax11 = diag11()
print(vif11['VIF Factor'].mean())

# %%
Y = df['PTS']
X = df[['TOI', '+/-', 'Grit', 'Salary', 'PIM', 'Is_Offense']]
X['TOI'] = X['TOI'].div(60)
X = sm.add_constant(X)
model11a = sm.OLS(Y, X)
results11a = model11a.fit()
print(results11a.summary())
print(results11a.rsquared_adj)
diag11a = nhl_linreg(results11a)
vif11a, fig11a, ax11a = diag11a()
print(vif11a['VIF Factor'].mean())

# %%
Y = df['Salary']
cols = ['PTS', 'TOI', '+/-', 'Grit', 'PIM']
X = df[cols]
X = sm.add_constant(X)
model12 = sm.OLS(np.log(Y), X)
results12 = model12.fit()
print(results12.summary())
print(results12.rsquared_adj)
diag12 = nhl_linreg(results12)
vif12, fig12, ax12 = diag12()
print(vif12['VIF Factor'].mean()) # 4.33

# %%
# %%
Y = df['Salary']
cols = ['PTS', 'TOI', '+/-']
X = df[cols]
X = sm.add_constant(X)
model12a = sm.OLS(np.log(Y), X)
results12a = model12a.fit()
print(results12a.summary())
print(results12a.rsquared_adj)
diag12a = nhl_linreg(results12a)
vif12a, fig12a, ax12a = diag12a()
print(vif12a['VIF Factor'].mean()) # 2.4

# %%
Y = df['Salary']
cols = ['PTS', 'TOI']
X = df[cols]
X = sm.add_constant(X)
model12b = sm.OLS(np.log(Y), X)
results12b = model12b.fit()
print(results12b.summary())
print(results12b.rsquared_adj)
diag12b = nhl_linreg(results12b)
vif12b, fig12b, ax12b = diag12b()
print(vif12b['VIF Factor'].mean()) # 2.4

# %%
Y = df['Salary']
cols = ['PTS', 'TOI', 'Is_Offense']
X = df[cols]
X = sm.add_constant(X)
model12c = sm.OLS(np.log(Y), X)
results12c = model12c.fit()
print(results12c.summary())
print(results12c.rsquared_adj)
diag12c = nhl_linreg(results12c)
vif12c, fig12c, ax12c = diag12c()
print(vif12c['VIF Factor'].mean())

# %%
clean = df.dropna(subset=['DftRd'])
Y = clean['Salary']
cols = ['PTS', 'TOI', 'DftRd']
X = clean[cols]
X = sm.add_constant(X)
model12d = sm.OLS(np.log(Y), X)
results12d = model12d.fit()
print(results12d.summary())
print(results12d.rsquared_adj)
diag12d = nhl_linreg(results12d)
vif12d, fig12d, ax12d = diag12d()
print(vif12d['VIF Factor'].mean())

# %%
Y = df['Salary']
cols = ['TOI', 'Is_Offense']
X = df[cols]
X = sm.add_constant(X)
model12e = sm.OLS(np.log(Y), X)
results12e = model12e.fit()
print(results12e.summary())
print(results12e.rsquared_adj)
diag12e = nhl_linreg(results12e)
vif12e, fig12e, ax12e = diag12e()
print(vif12e['VIF Factor'].mean())

# %%
clean = df.dropna(subset=['Ovrl'])
Y = clean['Salary']
cols = ['PTS', 'TOI/GP', 'Ovrl']
X = clean[cols]
X = sm.add_constant(X)
model12f = sm.OLS(np.log(Y), X)
results12f = model12f.fit()
print(results12f.summary())
print(results12f.rsquared_adj)
diag12f = nhl_linreg(results12f)
vif12f, fig12f, ax12f = diag12f()
print(vif12f['VIF Factor'].mean())

# %%
clean = df.dropna(subset=['iCF'])
Y = clean['Salary']
cols = ['PTS', 'TOI/GP', 'iCF']
X = clean[cols]
X = sm.add_constant(X)
model12g = sm.OLS(np.log(Y), X)
results12g = model12g.fit()
print(results12g.summary())
print(results12g.rsquared_adj)
diag12g = nhl_linreg(results12g)
vif12g, fig12g, ax12g = diag12g()
print(vif12g['VIF Factor'].mean())

# %%
clean = df.dropna(subset=['iSCF'])
Y = clean['Salary']
cols = ['PTS', 'TOI/GP', 'iSCF']
X = clean[cols]
X = sm.add_constant(X)
model12h = sm.OLS(np.log(Y), X)
results12h = model12g.fit()
print(results12h.summary())
print(results12h.rsquared_adj)
diag12h = nhl_linreg(results12h)
vif12h, fig12h, ax12h = diag12h()
print(vif12h['VIF Factor'].mean())

# %%
Y = df['Salary']
cols = ['PTS', 'TOI/GP', 'Is_Offense', 'iHF', 'S.Slap', 'iPenT']
X = df[cols]
X = sm.add_constant(X)
model12i = sm.OLS(np.log(Y), X)
results12i = model12i.fit()
print(results12i.summary())
print(results12i.rsquared_adj)
diag12i = nhl_linreg(results12i)
vif12i, fig12i, ax12i = diag12i()
print(vif12i['VIF Factor'].mean())

# %%
clean = df.dropna(subset=['CF'])
Y = clean['Salary']
cols = ['PTS', 'TOI/GP', 'Is_Offense', 'iHF', 'S.Slap', 'iPENT', 'CF']
X = clean[cols]
X = sm.add_constant(X)
model12j = sm.OLS(np.log(Y), X)
results12j = model12j.fit()
print(results12j.summary())
print(results12j.rsquared_adj)
diag12j = nhl_linreg(results12j)
vif12j, fig12j, ax12j = diag12j()
print(vif12j['VIF Factor'].mean())

# %%
clean = df.dropna(subset=['CF'])
Y = clean['Salary']
cols = ['A', 'TOI/GP', 'Is_Offense', 'iHF', 'S.Slap', 'iPENT', 'CF']
X = clean[cols]
X = sm.add_constant(X)
model12k = sm.OLS(np.log(Y), X)
results12k = model12j.fit()
print(results12k.summary())
print(results12k.rsquared_adj)
diag12k = nhl_linreg(results12k)
vif12k, fig12k, ax12k = diag12k()
print(vif12k['VIF Factor'].mean())

# %%
Y = df['G']
cols = ['G.Bkhd', 'G.Dflct', 'G.Slap', 'G.Snap', 
        'G.Tip', 'G.Wrap', 'G.Wrst']
X = df[cols]
X = sm.add_constant(X)
model13 = sm.OLS(Y, X)
results13 = model13.fit()
print(results13.summary())
diag13 = nhl_linreg(results13)
vif13, fig13, ax13 = diag13()