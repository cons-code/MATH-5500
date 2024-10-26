import pandas as pd
import numpy as np

from utils.tmdb_utils import utils


#%% TMDB frame class & associated utilities
class tmdb_frame(utils):

    # Initialize TMDB dataframe parameters.
    def __init__(self, fpath: str, fname: str, n: int=500):

        # Inherit parent utilities class.
        super().__init__()
        
        # Initialize film parameters.
        self.title = np.zeros((n,), dtype=object)                   # Movie title
        self.title_id = np.zeros((n,), dtype=int)                   # TMDB numerical identifier for movie title
        #self.imdb_id = np.zeros((n,), dtype=int)                    # IMDB numerical identifier for movie title

        self.runtime = np.zeros((n,), dtype=int)                    # Total runtime (min)

        self.is_series = np.zeros((n,), dtype=bool)                 # Part of collection (T/F)
        self.prev_film_release = np.zeros((n,), dtype=bool)         # Not first movie in collection (T/F)
        self.prev_film_rating = np.zeros((n,), dtype=float)         # Rating of previous movie in collection (T/F)
        self.is_indie = np.zeros((n,), dtype=bool)                  # Independent film (T/F)
        self.is_based_on_novel = np.zeros((n,), dtype=bool)         # Based on novel (T/F)

        self.num_studios = np.zeros((n,), dtype=int)                # Number of production companies
        self.pct_indie = np.zeros((n,), dtype=float)                # Percentage of independent production companies
        self.budget = np.zeros((n,), dtype=int)                     # Production budget ($USD)

        # Initialize production parameters.
        self.writer_xp_median = np.zeros((n,), dtype=float)         # Writer credits median
        self.writer_xp_sum = np.zeros((n,), dtype=int)              # Writer credits sum

        self.writer_rating_median = np.zeros((n,), dtype=float)     # Writer rating median
        self.writer_rating_max = np.zeros((n,), dtype=int)          # Writer rating maximum

        self.director_xp = np.zeros((n,), dtype=int)                # Director credits

        self.director_rating_median = np.zeros((n,), dtype=float)   # Director rating median
        self.director_rating_max = np.zeros((n,), dtype=int)        # Director rating maximum

        self.cast_xp_median = np.zeros((n,), dtype=float)           # Cast credits median
        self.cast_xp_sum = np.zeros((n,), dtype=int)                # Cast credits sum

        self.cast_rating_max = np.zeros((n,), dtype=int)            # Cast rating maximum

        self.sound_room_size = np.zeros((n,), dtype=int)            # Number of sound room employees
        self.writers_room_size = np.zeros((n,), dtype=int)          # Number of writer's room employees

        # Initialize genre parameters.
        self.is_adventure = np.zeros((n,), dtype=bool)              # Genre of adventure (genre_id: 12) (T/F)
        self.is_drama = np.zeros((n,), dtype=bool)                  # Genre of drama (genre_id: 18) (T/F)
        self.is_horror = np.zeros((n,), dtype=bool)                 # Genre of horror (genre_id: 27) (T/F)\
        self.is_action = np.zeros((n,), dtype=bool)                 # Genre of action (genre_id: 28) (T/F)
        self.is_comedy = np.zeros((n,), dtype=bool)                 # Genre of comedy (genre_id: 35) (T/F)
        self.is_history = np.zeros((n,), dtype=bool)                # Genre of history (genre_id: 36) (T/F)
        self.is_western = np.zeros((n,), dtype=bool)                # Genre of Western (genre_id: 37) (T/F)
        self.is_thriller = np.zeros((n,), dtype=bool)               # Genre of Thriller (genre_id: 53) (T/F)
        self.is_crime = np.zeros((n,), dtype=bool)                  # Genre of crime (genre_id: 80) (T/F)
        self.is_documentary = np.zeros((n,), dtype=bool)            # Genre of documentary (genre_id: 99) (T/F)
        self.is_sci_fi = np.zeros((n,), dtype=bool)                 # Genre of science fiction (genre_id:878) (T/F)
        self.is_mystery = np.zeros((n,), dtype=bool)                # Genre of myster (genre_id:9648) (T/F)
        self.is_music = np.zeros((n,), dtype=bool)                  # Genre of music (genre_id: 10402) (T/F)
        self.is_romance = np.zeros((n,), dtype=bool)                # Genre of romance (genre_id: 10749) (T/F)
        self.is_family = np.zeros((n,), dtype=bool)                 # Genre of family (genre_id: 10751) (T/F)

        # Initialize release date parameters.
        #self.month = np.zeros((n,), dtype=int)                      # Release month (i.e., 1-12)

        self.is_jan = np.zeros((n,), dtype=bool)                    # Released during the month of January
        self.is_feb = np.zeros((n,), dtype=bool)                    # Released during the month of February
        self.is_mar = np.zeros((n,), dtype=bool)                    # Released during the month of March
        self.is_apr = np.zeros((n,), dtype=bool)                    # Released during the month of April
        self.is_may = np.zeros((n,), dtype=bool)                    # Released during the month of May
        self.is_jun = np.zeros((n,), dtype=bool)                    # Released during the month of June
        self.is_jul = np.zeros((n,), dtype=bool)                    # Released during the month of July
        self.is_aug = np.zeros((n,), dtype=bool)                    # Released during the month of August
        self.is_sep = np.zeros((n,), dtype=bool)                    # Released during the month of September
        self.is_oct = np.zeros((n,), dtype=bool)                    # Released during the month of October
        self.is_nov = np.zeros((n,), dtype=bool)                    # Released during the month of November
        self.is_dec = np.zeros((n,), dtype=bool)                    # Released during the month of December

        #self.is_spring = np.zeros((n,), dtype=bool)                 # Released during the spring season (i.e., March-April) (T/F)
        #self.is_summer = np.zeros((n,), dtype=bool)                 # Released during the summer season (i.e., May-August) (T/F)
        #self.is_fall = np.zeros((n,), dtype=bool)                   # Released during the fall season (i.e., September-October) (T/F)
        #self.is_holiday = np.zeros((n,), dtype=bool)                # Released during the holiday season (i.e., November-December) (T/F)

        # Initialize opening weekend parameters.
        self.opening_revenue = np.zeros((n,), dtype=int)            # Opening weekend revenue ($USD)

        # Define TMDB items from dataframe.
        df = self.parse_csv(fpath, fname)

        for i in range(n):
            
            # Define film parameters.
            self.title[i] = df.loc[i, 'release']
            self.title_id[i] = int(df.loc[i, 'id'])
            #self.imdb_id[i] = int(df.loc[i, 'imdb_id'].replace('tt', ''))

            self.runtime[i] = int(df.loc[i, 'runtime'])

            _, _, self.is_series[i] = self.unpack_collection(df, i)
            self.prev_film_release[i] = bool(df.loc[i, 'previous_film_release'])
            self.prev_film_rating[i] = float(df.loc[i, 'previous_film_rating'])

            _, production_co_id = self.unpack_production_co(df, i)
            self.pct_indie[i] = self.get_pct_indie(production_co_id)
            self.is_indie[i] = self.check_indie(production_co_id)
            self.is_based_on_novel[i] = bool(df.loc[i, 'based_on_novel'])

            self.num_studios[i] = production_co_id.size
            self.budget[i] = int(df.loc[i, 'budget'])

            # Define production parameters.
            self.writer_xp_median[i] = float(df.loc[i, 'writer_xp_median'])
            self.writer_xp_sum[i] = int(df.loc[i, 'writer_xp_sum'])

            self.writer_rating_median[i] = float(df.loc[i, 'writer_rating_median'])
            self.writer_rating_max[i] = int(df.loc[i, 'writer_rating_max'])

            self.director_xp[i] = int(df.loc[i, 'director_xp'])

            self.director_rating_median[i] = float(df.loc[i, 'director_rating_median'])
            self.director_rating_max[i] = int(df.loc[i, 'director_rating_max'])

            self.cast_xp_median[i] = float(df.loc[i, 'cast_xp_median'])
            self.cast_xp_sum[i] = int(df.loc[i, 'cast_xp_sum'])

            self.cast_rating_max[i] = int(df.loc[i, 'cast_rating_max'])

            self.sound_room_size[i] = int(df.loc[i, 'sound_room'])
            self.writers_room_size[i] = int(df.loc[i, 'writers_room'])

            # Define genre parameters.
            self.is_adventure[i] = bool(df.loc[i, '12'])
            self.is_drama[i] = bool(df.loc[i, '18'])
            self.is_horror[i] = bool(df.loc[i, '27'])
            self.is_action[i] = bool(df.loc[i, '28'])
            self.is_comedy[i] = bool(df.loc[i, '35'])
            self.is_history[i] = bool(df.loc[i, '36'])
            self.is_western[i] = bool(df.loc[i, '37'])
            self.is_thriller[i] = bool(df.loc[i, '53'])
            self.is_crime[i] = bool(df.loc[i, '80'])
            self.is_documentary[i] = bool(df.loc[i, '99'])
            self.is_sci_fi[i] = bool(df.loc[i, '878'])
            self.is_mystery[i] = bool(df.loc[i, '9648'])
            self.is_music[i] = bool(df.loc[i, '10402'])
            self.is_romance[i] = bool(df.loc[i, '10749'])
            self.is_family[i] = bool(df.loc[i, '10751'])

            
            # Define release date parameters.
            #self.month[i], season_id = self.unpack_release_date(df, i)

            self.is_jan[i] = bool(df.loc[i, 'release_month_1'])
            self.is_feb[i] = bool(df.loc[i, 'release_month_2'])
            self.is_mar[i] = bool(df.loc[i, 'release_month_3'])
            self.is_apr[i] = bool(df.loc[i, 'release_month_4'])
            self.is_may[i] = bool(df.loc[i, 'release_month_5'])
            self.is_jun[i] = bool(df.loc[i, 'release_month_6'])
            self.is_jul[i] = bool(df.loc[i, 'release_month_7'])
            self.is_aug[i] = bool(df.loc[i, 'release_month_8'])
            self.is_sep[i] = bool(df.loc[i, 'release_month_9'])
            self.is_oct[i] = bool(df.loc[i, 'release_month_10'])
            self.is_nov[i] = bool(df.loc[i, 'release_month_11'])
            self.is_dec[i] = bool(df.loc[i, 'release_month_12'])

            #if (season_id == 1):

             #   self.is_spring[i] = True
            
            # elif (season_id == 2):

            #     self.is_summer[i] = True
            
            # elif (season_id == 3):

            #     self.is_fall[i] = True
            
            # elif (season_id == 4):

            #     self.is_holiday[i] = True

            # Define opening weekend parameters.
            self.opening_revenue[i] = int(df.loc[i, 'opening_revenue'])
            

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
        df['prev_film_release'] = self.prev_film_release.astype(int)
        df['prev_film_rating'] = self.prev_film_rating

        df['is_indie'] = self.is_indie.astype(int)
        df['is_based_on_novel'] = self.is_based_on_novel.astype(int)

        df['pct_indie'] = self.pct_indie
        df['num_production_companies'] = self.num_studios
        df['budget'] = self.budget

        # Define production parameters.
        df['writer_xp_median'] = self.writer_xp_median
        df['writer_xp_sum'] = self.writer_xp_sum

        df['writer_rating_median'] = self.writer_rating_median
        df['writer_rating_max'] = self.writer_rating_max

        df['director_xp'] = self.director_xp

        df['director_rating_median'] = self.director_rating_median
        df['director_rating_max'] = self.director_rating_max

        df['cast_xp_median'] = self.cast_xp_median
        df['cast_xp_sum'] = self.cast_xp_sum

        df['cast_rating_max'] = self.cast_rating_max

        df['writers_room_size'] = self.writers_room_size
        df['sound_room_size'] = self.sound_room_size

        # Define genre parameters.
        df['is_adventure'] = self.is_adventure.astype(int)
        df['is_drama'] = self.is_drama.astype(int)
        df['is_horror'] = self.is_horror.astype(int)
        df['is_action'] = self.is_action.astype(int)
        df['is_comedy'] = self.is_comedy.astype(int)
        df['is_history'] = self.is_history.astype(int)
        df['is_western'] = self.is_western.astype(int)
        df['is_thriller'] = self.is_thriller.astype(int)
        df['is_crime'] = self.is_crime.astype(int)
        df['is_documentary'] = self.is_documentary.astype(int)
        df['is_sci_fi'] = self.is_sci_fi.astype(int)
        df['is_mystery'] = self.is_mystery.astype(int)
        df['is_music'] = self.is_music.astype(int)
        df['is_romance'] = self.is_romance.astype(int)
        df['is_family'] = self.is_family.astype(int)

        # Define release date parameters.
        #df['month'] = self.month

        df['is_jan'] = self.is_jan.astype(int)
        df['is_feb'] = self.is_feb.astype(int)
        df['is_mar'] = self.is_mar.astype(int)
        df['is_apr'] = self.is_apr.astype(int)
        df['is_may'] = self.is_may.astype(int)
        df['is_jun'] = self.is_jun.astype(int)
        df['is_jul'] = self.is_jul.astype(int)
        df['is_aug'] = self.is_aug.astype(int)
        df['is_sep'] = self.is_sep.astype(int)
        df['is_oct'] = self.is_oct.astype(int)
        df['is_nov'] = self.is_nov.astype(int)
        df['is_dec'] = self.is_dec.astype(int)

        # df['is_spring'] = self.is_spring.astype(int)
        # df['is_summer'] = self.is_summer.astype(int)
        # df['is_fall'] = self.is_fall.astype(int)
        # df['is_holiday'] = self.is_holiday.astype(int)

        # Define opening weekend parameters.
        df['opening_revenue'] = self.opening_revenue

        return df


    # Create Pandas dataframe of cleaned TMDB data.
    def get_clean_pd_df(self) -> pd.DataFrame:

        # Create base copy of Pandas dataframe.
        df = self.get_pd_df()

        # Clean opening revenue parameter in dataframe.
        cdf = df.copy(deep=True)
        cdf = cdf[cdf['opening_revenue'].gt(0)]

        return cdf
        
