# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence

from src.nhl_linreg import nhl_linreg
# %%
df = pd.read_csv('Data/combined.csv', encoding='latin-1')

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
print(vif3['VIF Factor'].mean()) # 7.22

# %%
e_i = results3.resid
y_hat = results3.fittedvalues
k_hat = results3.scale

influence = OLSInfluence(results3)
rstudent = influence.get_resid_studentized_external()

# %%
fig, ax = plt.subplots()
ax.scatter(y_hat, np.abs(rstudent))
ax.set_title('Y-hat vs. |studentized residuals|')
plt.show();

# %%
fig, ax = plt.subplots()
ax.scatter(df['GP'], np.abs(rstudent))
ax.set_title('Games Played vs. |studentized residuals|')
plt.show();

# %%
fig, ax = plt.subplots()
ax.scatter(df['PTS'], np.abs(rstudent))
ax.set_title('Points vs. |studentized residuals|')
plt.show();

# %%
fig, ax = plt.subplots()
ax.scatter(df['+/-'], np.abs(rstudent))
ax.set_title('+/- vs. |studentized residuals|')
plt.show();

# %%
fig, ax = plt.subplots()
ax.scatter(df['Grit'], np.abs(rstudent))
ax.set_title('Grit vs. |studentized residuals|')
plt.show();

# %%
fig, ax = plt.subplots()
ax.scatter(df['TOI'], np.abs(rstudent))
ax.set_title('Time on ice vs. |studentized residuals|')
plt.show();

# %%
fig, ax = plt.subplots()
ax.scatter(df['PIM'], np.abs(rstudent))
plt.show();

# %%
fig, ax = plt.subplots()
ax.scatter(df['Is_Offense'], np.abs(rstudent))
plt.show();

# %%
# trying to improve GP
# standard deviation function
sd_fun_data = pd.DataFrame({
    'y': np.abs(e_i),
    'x': df['GP']
})
Y = sd_fun_data['y']
X = sd_fun_data['x']
model_sdf = sm.OLS(Y, X)
results_sdf = model_sdf.fit()
sigma_hat = results_sdf.fittedvalues
w_i_sd  = 1/(sigma_hat ** 2)

# %%
Y = df['Salary']
cols = ['GP', 'PTS', '+/-', 'Grit', 'TOI', 'PIM', 'Is_Offense']
X = df[cols]
X = sm.add_constant(X)

model_wls = sm.WLS(Y, X, weights = w_i_sd)
results_wls = model_wls.fit()
print(results_wls.summary())