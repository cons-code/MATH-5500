import argparse

import numpy as np
import pandas as pd
import seaborn as sns

from src.tmdb_frame import tmdb_frame
from src.tmdb_model import tmdb_model
from src.tmdb_linreg import tmdb_linreg


#%% Main method & associated utilities

# Parse input arguments.
def parse_inputs():

    # Initialize argument parser object.
    parser = argparse.ArgumentParser(description='Tool to generate TMDB model from specified data file.')

    # Add file parameters.
    parser.add_argument('--fpath', type=str, default='./data', help='Relative filepath to input directory.')
    parser.add_argument('--fname', type=str, default='final_cut.csv', help='Name of input CSV file containing TMDB data.')
    parser.add_argument('--n', type=int, default=500, help='Number of movies in TMDB data.')

    # Parse input arguments.
    args = parser.parse_args()

    return args


# Define main method.
def main():

    # Create GMTI model object from input arguments.
    args = parse_inputs()

    fpath = args.fpath
    fname = args.fname

    n = args.n

    # Initialize TMDB model class.
    mdl = tmdb_model()

    # Generate cleaned TMDB dataframe.
    df = tmdb_frame(fpath, fname, n=n)
    cdf = df.get_clean_pd_df()

    # Generate pair plot of continuous covariates.
    sns.pairplot(cdf[['opening_revenue', 'budget', 'num_production_companies', 'runtime', 'month', 'pct_indie']])

    # Generate OLS model.
    Y = ['opening_revenue']
    X = ['budget', 'runtime', 'writers_room_size', 'num_production_companies', \
         'pct_indie', 'is_indie', \
         'cast_xp_sum', 'director_xp', 'director_rating_median', \
         'prev_film_release', 'prev_film_rating', \
         'is_feb', 'is_mar', 'is_oct', 'is_dec', \
         'is_drama', 'is_horror', 'is_comedy', 'is_western', 'is_crime', 'is_documentary', 'is_sci_fi', 'is_music', 'is_family']

    cov_transform = {'opening_revenue': 'sqrt'}
    # cov_transform = None
    
    cov_joint = [['prev_film_release', 'prev_film_rating'], ['is_horror', 'is_oct'], ['cast_xp_sum', 'director_xp']]
    # cov_joint = None

    _, res = mdl.get_ols_model(cdf, Y, X, cov_transform=cov_transform, cov_joint=cov_joint)

    # Generate linear regression diagnostics for OLS model.
    print(res.summary())

    diag = tmdb_linreg(res)
    vif, fig, ax = diag()


#%% Main method execution

if __name__ == '__main__':

    main()

    