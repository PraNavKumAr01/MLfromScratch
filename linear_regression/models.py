import sys
sys.path.append('../')

from linear_regression.utilfunc import ols
from utils.tensors import Tensor

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        W, b = ols(X, y)
        self.weights = W
        self.bias = b
    
    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet. Call model.fit() to train the model")
        else:
            return Tensor([self.weights * sample + self.bias for sample in X]).data