# custom_transformers.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class Numerical_Feature_Adder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.gdp_data = {
            'Mumbai': 310,
            'Delhi': 293.6,
            'Kolkata': 150.1,
            'Bangalore': 110,
            'Chennai': 78.6,
            'Hyderabad': 75.2,
            'Pune': 69,
            'Ahmedabad': 68
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        try:
            X['area'] = np.log1p(X['area'])
        except:
            pass
        X['city_wise_gdp'] = X['city'].map(self.gdp_data)
        return X.drop(columns=['city'])

class AvgPriceByLocalityAdder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.avg_price_per_locality_ = None
        self.overall_avg_price_ = None

    def fit(self, X, y):
        df = X.copy()
        df['target'] = y
        self.avg_price_per_locality_ = df.groupby('locality')['target'].mean().to_dict()
        self.overall_avg_price_ = y.mean()
        return self

    def transform(self, X):
        X = X.copy()
        X['avg_price_locality'] = X['locality'].map(self.avg_price_per_locality_)
        X['avg_price_locality'] = X['avg_price_locality'].fillna(self.overall_avg_price_)
        X['avg_price_locality'] = np.log1p(X['avg_price_locality'])
        return X[['avg_price_locality']]
