from model import Model
from sklearn.linear_model import LinearRegression
import pandas as pd

class Lin_Reg(Model):
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.trained = False

    def fit(self, X, y):
        self.lr = LinearRegression(**self.kwargs).fit(X,y)
        self.trained = True

    def predict(self, X):
        if not self.trained:
            print("Model is not trained")
            return None
        else:
            return self.lr.predict(X)
