import sys
sys.path.append('../')

from utils.tensors import Tensor

# Implementing the Ordinary Least Squared algorithm
def ols(X: Tensor, y: Tensor):
    if len(X) != len(y):
        raise ValueError(f"Shape of features {X.shape} does not match the shape of targets {y.shape}")
    else:
        X_mean = X.mean()
        y_mean = y.mean()

        covar = 0.0
        var = 0.0

        for i in range(len(X)):
            covar += (X.data[i] - X_mean) * (y.data[i] - y_mean)
            var += ((X.data[i] - X_mean) ** 2)
            
        W = covar / var
        b = y_mean - W * X_mean
    
    return W, b