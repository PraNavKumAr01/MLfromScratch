import sys
sys.path.append('../')

from linear_regression.utilfunc import ols
from utils.tensors import Tensor

class LinearRegression:
    def __init__(self):
        # Initializing the parameters of the model
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Calculating the paramters using Ordinary Least Squared algorithm
        W, b = ols(X, y)
        # Setting the paremeters, basically meaning the model has learned
        self.weights = W
        self.bias = b
    
    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet. Call model.fit() to train the model")
        else:
            # Making predictions
            return Tensor([self.weights * sample + self.bias for sample in X]).data