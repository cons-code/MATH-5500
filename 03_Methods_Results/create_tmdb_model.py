import argparse
# import requests
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
# import sklearn
from ast import literal_eval

import warnings
warnings.filterwarnings("ignore")

#from src.tmdb_frame import tmdb_frame
from src.tmdb_model import tmdb_model
from src.tmdb_linreg import tmdb_linreg

# Import data from CSV file.
data = pd.read_csv('../02_Data/2B_data-output_10.27.1700.csv', index_col=0)
# data = pd.read_csv('3A_data-output.csv', index_col=0)

data.head(6)

X = data.drop(columns = ['opening_revenue', 'is_spring', 'is_summer', 'is_fall', 'is_holiday'])
# X = data.drop(columns = ['id', 'release', 'release_date', 'opening_revenue','origin_country', 'production_companies', 'production_countries', 'is_spring',
#        'is_summer', 'is_fall', 'is_holiday'])

result_df = pd.DataFrame()
i = 0
cut = X.columns.to_list()
lim = len(cut) + 1

while i < 400:
    subset = X[cut]
    subset = sm.add_constant(subset)
    Y = pd.DataFrame(np.sqrt(data['opening_revenue']))
    model = sm.OLS(Y, subset)
    results = model.fit()
    rsquared_adj = results.rsquared_adj
    columns = subset.columns.to_list()
    d = pd.DataFrame({'model' : i, 'feature_count' : len(subset.columns), 'adj_rsquared' : rsquared_adj, 'feature_space' : [columns]}, index = [i])
    result_df = pd.concat([result_df, d])
    
    pvals = pd.DataFrame(results.pvalues).sort_values(by = 0, ascending = False).reset_index()
    pvals = pvals[pvals['index'] != 'const']
    cut = pvals['index'].to_list()[1:]
    i = i + 1

result_df.sort_values(by = 'adj_rsquared', ascending = False).reset_index().head(6)

final_cut = result_df.sort_values(by = 'adj_rsquared', ascending = False).reset_index().feature_space[0]
# final_cut.remove('const')

# Initialize TMDB model class.
mdl = tmdb_model()

cdf = data

"""
sns.pairplot(cdf[['opening_revenue', 'budget', 'runtime', \
                  'previous_film_release', 'previous_film_rating']])

sns.pairplot(cdf[['opening_revenue', 'cast_xp_median', 'cast_rating_max', 'writer_xp_median', \
                  'writer_xp_sum', 'writer_rating_median', 'writer_rating_max']])

sns.pairplot(cdf[['opening_revenue', 'cast_xp_median', 'writer_xp_median', 'writer_xp_sum']])

sns.pairplot(cdf[['opening_revenue', 'writer_rating_median', 'writer_rating_max']])

sns.pairplot(cdf[['opening_revenue', 'writers_room', 'sound_room', 'crew']])
"""

# Generate joint covariates.
month_id = ['is_jan', 'is_feb', 'is_mar', 'is_apr', 'is_may', 'is_jun', \
            'is_jul', 'is_aug', 'is_sep', 'is_oct', 'is_nov', 'is_dec']
genre_id = ['is_adventure', 'is_fantasy', 'is_animation', 'is_drama', 'is_horror', 'is_action', \
            'is_comedy', 'is_history', 'is_thriller', 'is_crime', 'is_documentary', 'is_sci_fi', \
            'is_mystery', 'is_music', 'is_romance', 'is_family']
season_id = ['is_winter', 'is_spring', 'is_summer', 'is_fall']

joint_id = []

cdf['is_winter'] = np.sum((X['is_jan'], X['is_feb'], X['is_dec']))
cdf['is_spring'] = np.sum((X['is_mar'], X['is_apr'], X['is_may']))
cdf['is_summer'] = np.sum((X['is_jun'], X['is_jul'], X['is_aug']))
cdf['is_fall'] = np.sum((X['is_sep'], X['is_oct'], X['is_nov']))

# Generate month-genre interactions.
for i in range(len(month_id)):

    for j in range(len(genre_id)):

        cdf[str(month_id[i] + '_*_' + genre_id[j])] = (cdf[month_id[i]] * cdf[genre_id[j]])
        joint_id.append(str(month_id[i] + '_*_' + genre_id[j]))

# Generate genre-genre interactions.
for i in range(len(genre_id)):

    for j in range((i+1), len(genre_id)):

        cdf[str(genre_id[i] + '_*_' + genre_id[j])] = (cdf[genre_id[i]] * cdf[genre_id[j]])
        joint_id.append(str(genre_id[i] + '_*_' + genre_id[j]))

# Generate additional interactions.
cdf['is_series_*_is_sci_fi'] = (cdf['is_series'] * cdf['is_sci_fi'])

test_cut = ['budget', 'num_production_companies', 'crew', 'cast_xp_median', 'is_series', 'previous_film_release', 'previous_film_rating', 'first_time_directors', \
            'is_feb', 'is_mar', 'is_sep', 'is_oct', 'is_dec', \
            'is_horror', 'is_romance', 'is_comedy', 'is_family', 'is_animation', 'is_sci_fi', 'is_drama', \
            'is_oct_*_is_horror', 'is_dec_*_is_family', \
            'is_comedy_*_is_romance', 'is_animation_*_is_family', \
            'is_series_*_is_sci_fi']

# Initialize TMDB model class.
mdl = tmdb_model()

cdf = data

# Generate pair plot of continuous covariates.
# sns.pairplot(cdf[['opening_revenue', 'budget', 'num_production_companies', 'runtime', 'month', 'pct_indie']])

# Initialize TMDB model class.
mdl = tmdb_model()

cdf = data

# Generate OLS model.
Y = ['opening_revenue']
X = test_cut

print('num predictors: ' + str(len(X)))

cov_transform = {'opening_revenue': 'sqrt'}
# cov_transform = None

#cov_joint = [['cast_xp_sum', 'director_xp']]
cov_joint = None

_, res = mdl.get_ols_model(cdf, Y, X, cov_transform=cov_transform, cov_joint=cov_joint)

# Generate linear regression diagnostics for OLS model.
print(res.summary())

"""
pvals = pd.DataFrame(res.pvalues).sort_values(by = 0, ascending = False).reset_index()
pvals = pvals[pvals['index'] != 'const']

for i in range(pvals.shape[0]):

    print(pvals['index'][i] + ': ' + str(pvals[0][i]))
"""

diag = tmdb_linreg(res)
vif, fig, ax = diag()
