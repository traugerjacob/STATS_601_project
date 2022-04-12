from model import Model
from sklearn.ensemble import AdaBoostRegressor
import pandas as pd

class Adaboost(Model):
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.trained = False

    def fit(self, X, y):
        self.ada = AdaBoostRegressor(**self.kwargs).fit(X,y)
        self.trained = True

    def predict(self, X):
        if not self.trained:
            print("Model is not trained")
            return None
        else:
            return self.ada.predict(X)
