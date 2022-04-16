from model import Model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
import pandas as pd
from copy import deepcopy

class OLS_Tree(Model):
    def __init__(self, kwargs):
        args = deepcopy(kwargs)
        if isinstance(args, dict):
            self.v = args["v"]
            args.pop('v', None)
            self.kwargs_lr = {}
            self.kwargs_ada = args
        else:
            self.v = args[0]
            self.kwargs_lr = args[1]
            self.kwargs_ada = args[2]
            self.trained = False

    def fit(self, X, y):
        self.lr = LinearRegression(**self.kwargs_lr).fit(X,y)
        self.ada = AdaBoostRegressor(**self.kwargs_ada).fit(X,y)
        self.trained = True

    def predict(self, X):
        pred_lr = self.lr.predict(X)
        pred_ada = self.ada.predict(X)
        if not self.trained:
            print("Model is not trained")
            return None
        else:
            return self.v*pred_lr + (1 - self.v)*pred_ada
