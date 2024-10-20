import os
import ast
from typing import Tuple

import numpy as np
import pandas as pd


#%% Built-in TMDB utilities class
class utils():

    # Initialize utility parameters.
    def __init__(self):

        self.major_studio_id = np.array([33,                                # Universal Pictures
                                         3436,                              # Universal Studios Home Entertainment Family Productions
                                         5556,                              # Universal Animation Studios
                                         47046,                             # Universal Pictures do Brazil
                                         4,                                 # Paramount Pictures
                                         838,                               # Paramount Vantage
                                         21834,                             # Paramount Pictures Digital Entertainment
                                         220287,                            # Paramount Classics
                                         174,                               # Warner Bros. Pictures
                                         2785,                              # Warner Bros. Animation
                                         17466,                             # Warner Bros. Entertainment Germany
                                         12,                                # New Line Cinema
                                         2,                                 # Walt Disney Pictures
                                         6125,                              # Walt Disney Animation Studios
                                         178202,                            # Walt Disney Pictures China
                                         25,                                # 20th Century Fox
                                         3,                                 # Pixar
                                         34,                                # Sony Pictures
                                         58,                                # Sony Pictures Classics
                                         2251,                              # Sony Pictures Animation
                                         5388,                              # Sony Pictures Home Entertainment
                                         5,                                 # Columbia Pictures
                                         559],                              # TriStar Pictures
                                         dtype=np.int32)


    # Convert string to Boolean.
    def str_to_bool(self, string: str) -> bool:

        if (string.lower() == 'false'):

            return False
        
        else:

            return True
        
    
    # Convert string to list.
    def str_to_list(self, string: str) -> list:

        return ast.literal_eval(string)
    

    # Parse input TMDB CSV into Pandas dataframe.
    def parse_csv(self, fpath: str, fname: str) -> pd.DataFrame:

        return pd.read_csv(os.path.join(fpath, fname), header=0)
        
    
    # Calculate season from month.
    def get_season(self, month: int) -> str:

        if ((month == 3) or (month == 4)):

            return 'spring'
        
        elif ((month >= 5) and (month <= 8)):

            return 'summer'
        
        elif ((month == 9) or (month == 10)):

            return 'fall'
        
        elif ((month == 11) or (month == 12)):

            return 'holiday'
        
        else:

            return 'dump'


    # Calculate numerical identifier for season.
    def get_season_id(self, season: str) -> int:

        if (season.lower() == 'spring'):

            return 1
        
        elif (season.lower() == 'summer'):

            return 2
        
        elif (season.lower() == 'fall'):

            return 3
        
        elif (season.lower() == 'holiday'):

            return 4
        
        else:

            return 0
        
    
    # Check if production company is independent.
    def check_indie(self, production_co_id: np.array) -> bool:

        return (not np.any(np.isin(production_co_id, self.major_studio_id)))
    

    # Extract genre information from TMDB JSON list.
    def unpack_genre(self, df: pd.DataFrame, i: int) -> Tuple[np.array, np.array]:

        # Convert string to list of dicts.
        genre_eval = self.str_to_list(df.loc[i, 'genres'])

        # Extract all genres and corresponding numerical identifiers.
        n = len(genre_eval)

        genre = np.zeros((n,), dtype=object)
        genre_id = np.zeros((n,), dtype=np.int32)

        for k in range(n):

            genre[k] = genre_eval[k]['name'].lower()
            genre_id[k] = genre_eval[k]['id']
        
        return (genre, genre_id)


    # Extract countries of origin from TMDB JSON list.
    def unpack_origin_country(self, df: pd.DataFrame, i: int) -> np.array:

        # Convert string to list.
        origin_country_eval = self.str_to_list(df.loc[i, 'origin_country'])

        # Extract all countries of origin.
        n = len(origin_country_eval)

        origin_country = np.zeros((n,), dtype=object)

        for k in range(n):

            origin_country[k] = origin_country_eval[k].lower()

        return origin_country


    # Extract production company information from TMDB JSON list.
    def unpack_production_co(self, df: pd.DataFrame, i: int) -> Tuple[np.array, np.array, bool]:

        # Convert string to list of dicts.
        production_co_eval = self.str_to_list(df.loc[i, 'production_companies'])

        # Extract all genres and corresponding numerical identifiers.
        n = len(production_co_eval)

        production_co = np.zeros((n,), dtype=object)
        production_co_id = np.zeros((n,), dtype=np.int32)

        for k in range(n):

            production_co[k] = production_co_eval[k]['name'].lower()
            production_co_id[k] = production_co_eval[k]['id']
        
        return (production_co, production_co_id)
    

    # Extract release date information from TMDB JSON list.
    def unpack_release_date(self, df: pd.DataFrame, i: int) -> Tuple[int, int]:

        # Convert string to datetime object.
        release_date = pd.to_datetime(df.loc[i, 'release_date'])

        # Extract release month.
        month = release_date.month

        # Determine corresponding season.
        season = self.get_season(month)
        season_id = self.get_season_id(season)

        return season_id
    

    # Extract language information from TMDB JSON list.
    def unpack_lang(self, df: pd.DataFrame, i: int) -> Tuple[np.array, int]:

        # Convert string to list of dicts.
        lang_eval = self.str_to_list(df.loc[i, 'spoken_languages'])

        # Extract all spoken languages.
        n = len(lang_eval)

        spoken_lang = np.zeros((n,), dtype=object)

        for k in range(n):

            spoken_lang[k] = lang_eval[k]['iso_639_1'].lower()

        return (spoken_lang, n)
    

    # Extract collection information from TMDB JSON list.
    def unpack_collection(self, df: pd.DataFrame, i: int) -> Tuple[int, str]:

        # Determine if movie is part of a collection.
        if (not np.isnan(df.loc[i, 'belongs_to_collection.id'])):

            # Extract corresponding information.
            is_collection = True

            collection = str(df.loc[i, 'belongs_to_collection.name'])
            collection_id = int(df.loc[i, 'belongs_to_collection.id'])

        else:

            is_collection = False

            collection = ''
            collection_id = 0

        return (is_collection, collection, collection_id)
    
