import pandas as pd
import numpy as np

from utils.tmdb_utils import utils


#%% TMDB frame class & associated utilities
class tmdb_frame(utils):

    # Initialize TMDB dataframe parameters.
    def __init__(self, fpath: str, fname: str, n: int=500):

        # Inherit parent utilities class.
        super().__init__()
        
        # Initialize film parameter arrays.
        self.title = np.zeros((n,), dtype=object)               # Movie title
        self.title_id = np.zeros((n,), dtype=int)               # TMDB numerical identifier for movie title
        self.imdb_id = np.zeros((n,), dtype=int)                # IMDB numerical identifier for movie title

        self.runtime = np.zeros((n,), dtype=int)                # Total runtime (min)

        self.is_series = np.zeros((n,), dtype=bool)             # Part of collection (T/F)
        self.is_indie = np.zeros((n,), dtype=bool)              # Independent film (T/F)

        self.num_studios = np.zeros((n,), dtype=int)            # Number of production companies
        self.budget = np.zeros((n,), dtype=int)                 # Production budget ($USD)

        # Initialize release date parameters.
        self.is_spring = np.zeros((n,), dtype=bool)             # Released during the spring season (i.e., March-April) (T/F)
        self.is_summer = np.zeros((n,), dtype=bool)             # Released during the summer season (i.e., May-August) (T/F)
        self.is_fall = np.zeros((n,), dtype=bool)               # Released during the fall season (i.e., September-October) (T/F)
        self.is_holiday = np.zeros((n,), dtype=bool)            # Released during the holiday season (i.e., November-December) (T/F)

        # Initialize opening weekend parameters.
        self.opening_revenue = np.zeros((n,), dtype=int)        # Opening weekend revenue ($USD)
        self.num_opening_theaters = np.zeros((n,), dtype=int)   # Number of opening weekend theaters

        # Initialize critical reception parameters.
        self.popularity = np.zeros((n,), dtype=float)           # TDMB proprietary popularity rating

        # Define TMDB items from dataframe.
        df = self.parse_csv(fpath, fname)

        for i in range(n):
            
            # Define film parameters.
            self.title[i] = df.loc[i, 'release']
            self.title_id[i] = int(df.loc[i, 'id'])
            self.imdb_id[i] = int(df.loc[i, 'imdb_id'].replace('tt', ''))

            self.runtime[i] = int(df.loc[i, 'runtime'])

            _, _, self.is_series[i] = self.unpack_collection(df, i)

            _, production_co_id = self.unpack_production_co(df, i)
            self.is_indie[i] = self.check_indie(production_co_id)

            self.num_studios[i] = production_co_id.size
            self.budget[i] = int(df.loc[i, 'budget'])

            # Define release date parameters.
            season_id = self.unpack_release_date(df, i)

            if (season_id == 1):

                self.is_spring[i] = True
            
            elif (season_id == 2):

                self.is_summer[i] = True
            
            elif (season_id == 3):

                self.is_fall[i] = True
            
            elif (season_id == 4):

                self.is_holiday[i] = True

            # Define opening weekend parameters.
            self.opening_revenue[i] = int(df.loc[i, 'opening_revenue'])
            self.num_opening_theaters[i] = int(df.loc[i, 'opening_theaters'])

            # Define critical reception parameters.
            self.popularity[i] = float(df.loc[i, 'popularity'])
            

    # Create Pandas dataframe of TMDB data.
    def get_pd_df(self) -> pd.DataFrame:

        # Initialize Pandas dataframe.
        df = pd.DataFrame()

        # Define film parameters.
        df['title'] = self.title
        df['title_id'] = self.title_id
        df['imdb_id'] = self.imdb_id

        df['runtime'] = self.runtime

        df['is_series'] = self.is_series.astype(int)
        df['is_indie'] = self.is_indie.astype(int)

        df['num_production_companies'] = self.num_studios
        df['budget'] = self.budget

        # Define release date parameters.
        df['is_spring'] = self.is_spring.astype(int)
        df['is_summer'] = self.is_summer.astype(int)
        df['is_fall'] = self.is_fall.astype(int)
        df['is_holiday'] = self.is_holiday.astype(int)

        # Define opening weekend parameters.
        df['opening_revenue'] = self.opening_revenue
        df['opening_theaters'] = self.num_opening_theaters

        # Define critical reception parameters.
        df['popularity'] = self.popularity

        return df


    # Create Pandas dataframe of cleaned TMDB data.
    def get_clean_pd_df(self) -> pd.DataFrame:

        # Create base copy of Pandas dataframe.
        df = self.get_pd_df()

        # Clean opening revenue parameter in dataframe.
        cdf = df.copy(deep=True)
        cdf = cdf[cdf['opening_revenue'].gt(0)]

        return cdf
        
