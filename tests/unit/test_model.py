"""

This file tests the NumpyNeuralNetwork class is functioning.

It is split into:
    - Initialisation and configuration
    - Core functionality testing (fit, predict, forward/backward propagation, model saving)

""" 

import unittest
import pandas as pd
import numpy as np
import os
import sys
import pickle
import tempfile 
import shutil 
from unittest.mock import patch

# Create mock Layer class and functions 
class MockLayer :
    """
    Mock Layer for test
    
    """
    def __init__(self, input_size, output_size, activation) :
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.biases = np.random.rand(1, output_size) - 0.5
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
        self.biases -= learning_rate * output_error
        return input_error
    
def mock_sigmoid(x, derivative = False) :
    if derivative :
        return (np.exp(-x)) / ((np.exp(-x)+1)**2)
    else :
        return 1/(1+np.exp(-x))

def mock_meanSquareError(y_actual, y_pred, derivative = False) :
    if derivative :
        return (y_pred - y_actual) / y_actual.size
    else :
        return np.mean((y_actual - y_pred)**2)*0.5
    
def mock_deEncoder(x):
    return np.argmax(x, axis=1).tolist()

def mock_toList(x) :
    y = []
    
    for i in range(len(x)) :
        y.append(x[i][0].tolist())

    return y

# Add the project root to sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the function to be tested
from src.model import NumpyNeuralNetwork

class TestNumpyNeuralNetwork(unittest.TestCase) :

    def setUp(self) :
        
        """
        Set up test fixtures before each test method.
        """

        self.network = NumpyNeuralNetwork()
        self.mock_layer1 = MockLayer(28*28,5,mock_sigmoid) # simulating MNIST 
        self.mock_layer2 = MockLayer(5,10,mock_sigmoid)
        
        # Sample training data
        np.random.seed(42)
        self.X_train = np.random.rand(10, 1, 784).astype('float32')
        self.y_train = np.random.rand(10, 10).astype('float32')
        self.X_test = [np.random.rand(2, 1, 784) for _ in range(3)]
        
        # Create temporary directory for file operations
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) :
        
        """
        Clean up after each test method.
        
        """

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialisation(self) : 
        """
        This function tests that NumpyNeuralNetwork intialises properly
        
        """

        network = NumpyNeuralNetwork()
        
        self.assertEqual(len(network.layers), 0)
        self.assertIsNone(network.loss)
        self.assertIsNone(network.lossPrime)
        self.assertIsNone(network.epochs)
        self.assertIsNone(network.learning_rate)
    
    def test_add_layer(self) : 
        """
        Tests the add method 
        
        """
        self.network.add(self.mock_layer1)
        self.network.add(self.mock_layer2)
        
        self.assertEqual(len(self.network.layers), 2)
        self.assertEqual(self.network.layers[0], self.mock_layer1)
        self.assertEqual(self.network.layers[1], self.mock_layer2)
    
    def test_loss_function(self) : 
        """
        Tests setLossFunction method 
        
        """

        self.network.setLossFunction(mock_meanSquareError)
        
        self.assertEqual(self.network.loss, mock_meanSquareError)
    
    @patch('builtins.print')
    def test_fit_method(self, mock_print) :
        """
        Tests the fit method 
        
        """

        # Set up the network
        self.network.add(self.mock_layer1)
        self.network.add(self.mock_layer2)
        self.network.setLossFunction(mock_meanSquareError)
        
        # Fit the model
        epochs = 2
        learning_rate = 0.01
        result = self.network.fit(self.X_train, self.y_train, epochs, learning_rate)
        
        # Check that parameters were set
        self.assertEqual(self.network.epochs, epochs)
        self.assertEqual(self.network.learning_rate, learning_rate)
        
        # Check that method returns self for chaining
        self.assertEqual(result, self.network)
        
        # Check that print was called (epoch messages)
        self.assertTrue(mock_print.called)

    def test_predict_method(self) : 
        """
        Tests predict method
        
        """

        # Set up network with layers
        self.network.add(self.mock_layer1)
        self.network.add(self.mock_layer2)
        
        # Make predictions
        predictions = self.network.predict(self.X_test)
        
        # Check that we get predictions for all test samples
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Each prediction should be a numpy array (result of forward propagation)
        for pred in predictions:
            self.assertIsInstance(pred, np.ndarray)

    @patch('builtins.print')
    def test_save_model(self, mock_print) : 
        """
        Tests save_model method
        
        """
        filepath = os.path.join(self.temp_dir, "test_model.pkl")
        
        # Add some configuration to make the test more meaningful
        self.network.add(self.mock_layer1)
        self.network.setLossFunction(mock_meanSquareError)
        self.network.epochs = 10
        self.network.learning_rate = 0.01
        
        # Save the model
        self.network.save_model(filepath)
        
        # Check that file was created
        self.assertTrue(os.path.exists(filepath))
        
        # Check that we can load it back
        with open(filepath, 'rb') as f:
            loaded_model = pickle.load(f)
        
        self.assertIsInstance(loaded_model, NumpyNeuralNetwork)
        self.assertEqual(len(loaded_model.layers), 1)
        self.assertEqual(loaded_model.epochs, 10)
        self.assertEqual(loaded_model.learning_rate, 0.01)
        
        # Check print message
        mock_print.assert_called_with(f"Model saved to {filepath}")
    
if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)