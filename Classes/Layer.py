import numpy as np

class Layer :
    def __init__(self, input_size, output_size, activation) :
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.activation = activation 

    # this method will return an output for a given input - uses dot to multiply matrices
    def forwardPropagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights)
        self.outputActivated = self.activation(self.output)
        return self.outputActivated
    
    # calculates dE/dW for an output error dE/dY and returns dE/dX
    def backPropagation(self, output_error, learning_rate) :

        input_error = np.dot((self.activation(self.output, derivative = True) * output_error), self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
