import numpy as np 

def meanSquareError(y_actual, y_pred, derivative = False) :
    if derivative :
        return (y_pred - y_actual) / y_actual.size
    else :
        return np.mean((y_actual - y_pred)**2)*0.5

