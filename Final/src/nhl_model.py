from typing import Tuple

import numpy as np
import pandas as pd

from statsmodels.base.model import LikelihoodModel
from statsmodels.base.model import Results

from statsmodels.formula.api import ols
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from utils.tmdb_utils import utils


# TMDB model class & associated utilities
class tmdb_model(utils):

    # Initialize TMDB model parameters.
    def __init__(self):

        # Inherit parent utilities class.
        super().__init__()
    

    # Generate OLS model to fit Y from predictor(s) X.
    def get_ols_model(self, df: pd.DataFrame, Y: list, X: list, \
                      cov_transform: dict=None, cov_joint: list=None) -> Tuple[LikelihoodModel, Results]:

        # Define independent and dependent covariates.
        Y = df[Y]

        X = df[X]
        X = add_constant(X)     # Add column of ones for intercept

        # Calculate covariate transformations.
        if bool(cov_transform):

            for (key, val) in cov_transform.items():
                
                try:

                    c = np.array(X[key])
                    ct = self.get_cov_transform(c, shape=val)

                    X[str(val + '_' + key)] = ct

                except(KeyError):

                    c = np.array(Y[key])
                    ct = self.get_cov_transform(c, shape=val)

                    Y[str(val + '_' + key)] = ct

                    Y.drop(key, axis=1, inplace=True)

        # Calculate joint covariates.
        if bool(cov_joint):

            for val in cov_joint:

                n = len(val)

                if (n == 2):

                    ct = self.get_cov_joint(np.array(X[val[0]]), np.array(X[val[1]]))
                    X[str(val[0] + '*' + val[1])] = ct

                else:

                    ct = self.get_cov_joint(np.array(X[val[0]]), np.array(X[val[1]]), np.array(X[val[2]]))
                    X[str(val[0] + '*' + val[1] + '*' + val[2])] = ct

        # Generate and fit OLS model.
        model = OLS(Y, X)
        res = model.fit()

        return (model, res)
    

    # Generate OLS model to fit Y from predictor(s) X using R interface.
    def get_ols_model_r(self, df: pd.DataFrame, formula: str) -> Tuple[LikelihoodModel, Results]:

        # Generate and fit OLS model.
        model = ols(formula=formula, data=df)
        res = model.fit()

        return (model, res)
    

    # Transform continuous covariate X.
    def get_cov_transform(self, X: np.array, shape: str=None) -> np.array:

        # Perform square transformation.
        if ((shape.lower() == 'sq') or (shape.lower() == 'square')):

            return np.power(X, 2)
        
        # Perform cubic transformation.
        elif ((shape.lower() == 'cb') or (shape.lower() == 'cubic')):

            return np.power(X, 3)
        
        # Perform square root transformation.
        elif ((shape.lower() == 'sqrt') or (shape.lower() == 'square_root')):

            return np.sqrt(X)
        
        # Perform logarithmic transformation.
        elif ((shape.lower() == 'ln') or (shape.lower() == 'log') or (shape.lower() == 'logarithmic')):

            return np.log(X)
        
        # Perform exponential transformation.
        elif ((shape.lower() == 'exp') or (shape.lower() == 'exponential')):

            return np.exp(X)
        
        # Perform reciprocal transformation.
        elif ((shape.lower() == 'rec') or (shape.lower() == 'reciprocal')):

            return np.reciprocal(X)
        
        # Catch unsupported transformation shape.
        else:

            raise(TypeError('Unsupported transformation shape for continuous covariate.'))
    

    # Calculate joint covariate of X1, X2[, X3].
    def get_cov_joint(self, X1: np.array, X2: np.array, X3: np.array=None) -> np.array:

        if (np.any(X3)):

            return (X1.astype(float) * X2.astype(float) * X3.astype(float))
        
        else:

            return (X1.astype(float) * X2.astype(float))
    
