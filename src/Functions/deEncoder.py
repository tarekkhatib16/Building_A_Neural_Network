import numpy as np 

def deEncoder(x):
    return np.argmax(x, axis=1).tolist()