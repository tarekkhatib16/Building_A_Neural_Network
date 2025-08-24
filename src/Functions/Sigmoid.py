import numpy as np 

def sigmoid(x, derivative = False) :
    if derivative :
        return (np.exp(-x)) / ((np.exp(-x)+1)**2)
    else :
        return 1/(1+np.exp(-x))
