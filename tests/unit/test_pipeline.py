import unittest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, call
import sys
import pickle
import tempfile
import shutil

# Add the project root to sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import functions to be tested (and internal functions that will now run unmocked)
from src.pipeline import run_neural_network

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

class MockNumpyNeuralNetwork :
    """
    Numpy Neural Network for predicting the MNIST dataset
    """

    def __init__(self) :
        """
        Initialises the Neural Network
        """
        self.layers = []
        self.loss = None
        self.lossPrime = None
        self.epochs = None
        self.learning_rate = None
    
    def add(
        self, 
        layer: Layer
    ) -> None:
        """
        Adds a layer to the network
        
        Args :
            layer (Layer) : Layer to be added to network
            
        """
        self.layers.append(layer)
    
    def setLossFunction(
        self, 
        loss
    ) -> None:
        """
        Sets the network's loss function 
        
        Args :
            loss (Callable) : The loss function
        
        """
        self.loss = loss
    
    def fit(
        self, 
        X_train : pd.DataFrame, 
        y_train : pd.DataFrame, 
        epochs : int, 
        learning_rate : float
    ) -> 'NumpyNeuralNetwork':
        """
        Fit the model to the training data
        
        Args :
            X_train (pd.DataFrame) : Training data.
            y_train (pd.DataFrame) : Test data.
            epochs (int) : The number of passes on the entire dataset. 
            learning_rate (float) : The learning rate parameter.
             
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        samples = len(X_train)

        # training loop
        for i in range(epochs) :
            print("\nStart of epoch %d" % (i+1))

            err = 0
            for j in range(samples) :
                # carry out forward propagation
                output = X_train[j]
                for layer in self.layers :
                    output = layer.forwardPropagation(output)
                
                # calculate the loss
                err += self.loss(y_train[j], output)

                # carry out back-propagation
                error = self.loss(y_train[j], output, derivative = True)
                for layer in reversed(self.layers):
                    error = layer.backPropagation(error, learning_rate)
            
            # calculate average error on all samples
            err /= samples
            
            print("Error = %f" % err)
        
        return self
    
    def predict(
        self, 
        input_data : pd.DataFrame
    ) -> list :
        """
        Make predictions on input data.
        
        Args :
            input_data (pd.DataFrame) : Input data.
        
        Returns :
            result (list) : Predicted labels. 
        """

        samples = len(input_data)
        result = []

        # interate over all samples 
        for i in range(samples) :
            # carry out forward propagation
            output = input_data[i]
            for layer in self.layers :
                output = layer.forwardPropagation(output)
            result.append(output)
        
        return result
    
    def save_model(
        self, 
        filepath: str
    ) -> None :
        """
        Save the trained model to a pickle file
        
        Args :
            filepath (str) : Full path to save the model
            
        """

        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory) :
            os.makedirs(directory, exist_ok=True)

        with open(filepath, 'wb') as f :
            pickle.dump(obj = self, file = f)
        print(f"Model saved to {filepath}")
    
    def log_run(
        self,
        directory : str,
        metrics : dict,
        log_filename = str
    ) -> None :
        """
        Save model configuration and performance metrics to a JSON file.
        
        Args: 
            directory (str) : Directory where the JSON file will be stored.
            metrics (Dict) : Evaluation metrics to save.
            dataset_info (Dict) : Information about the dataset used.
            log_filename (str) : Name of hte log file.
        
        """

        if not os.path.exists(directory) :
            os.makedirs(directory, exist_ok=True)
        
        run_info = {
            "timestamp" : datetime.now().isoformat(),
            "model_class" : self.__class__.__name__,
            "dataset" : "MNIST",
            "parameters" : {
                "learning_rate" : self.learning_rate,
                "epochs" : self.epochs
            },
            "metrics" : metrics
        }

        log_file = os.path.join(directory, log_filename)

        # Load existing logs or create a new list
        if os.path.exists(log_file) :
            try :
                with open(log_file, "r") as f :
                    logs = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) :
                logs = []
        else :
            logs = []
        
        logs.append(run_info)

        # Save updated logs
        with open(log_file, "w") as f :
            json.dump(logs, f, indent=4, default=str)
        
        print(f"Run log saved to {log_file}")

def mock_data_loader(filepath: str) -> pd.DataFrame :
    """
    Loads the MNIST dataset from CSV file. 

    Args :
        filepath (str): Path to the CSV file.
    
    Returns :
        pd.DataFrame: The raw MNIST dataset. 

    """
    try :
        df = pd.read_csv(filepath)
        print(f"Dataset loaded from CSV: {filepath}")

        return df
    except Exception as e :
        raise RuntimeError(f"Failed to load dataset: {e}")
    
def mock_clean_dataset(
        train: pd.DataFrame,
        test: pd.DataFrame,
        target_column: str
) -> pd.DataFrame :
    """
    Cleans the dataset by splitting features and labels, reshaping dataset, and one-hot encoding labels.

    Args :
        train (pd.DataFrame) : Raw training data.
        test (pd.DataFrame) : Raw test data.
        target_column (str) : Target column name.
    
    Returns :
        Tuple: Split data - X_train, X_test, y_train_enc, y_test_enc
    
    """
    # Split features and labels
    X_train, y_train = train.drop(columns=[target_column]).values, train[target_column].values
    X_test, y_test = test.drop(columns=[target_column]).values, test[target_column].values

    # Reshape to (num_samples, 28, 28, 1) for CNNs (better than (1, 784))
    X_train = X_train.reshape(X_train.shape[0], 1, 28*28).astype(np.float32) / 255.0
    X_test = X_test.reshape(X_test.shape[0], 1, 28*28).astype(np.float32) / 255.0

    # One-hot encode labels using numpy.eye
    num_classes = len(np.unique(y_train))  # should be 10 for MNIST
    y_train_enc = np.eye(num_classes)[y_train]
    y_test_enc = np.eye(num_classes)[y_test]

    return X_train, X_test, y_train_enc, y_test_enc

def mock_compute_metrics(
        y_test : np.ndarray,
        y_pred : np.ndarray
    ) -> dict : 
        """
        Computes accuracy metrics for Neural Network
        
        Args :
            y_test (np.ndarray) : True labels.
            y_pred (np.ndarray) : Precicted labels.
        
        Returns :
            Dict : Dictionary containing all computed metrics 
        
        """

        y_pred_list = toList(np.array(y_pred))

        # Convert output into int - max value becomes 1 the rest is 0
        y_pred_int = [[1 if j == max_i else 0 for j in i] for i in y_pred_list for max_i in [max(i)]]

        # decode the output
        y_pred_denc = deEncoder(y_pred_int)

        y_test_list = np.argmax(y_test, axis=1).tolist()

        print(y_pred_denc[:10])
        print(y_test_list[:10])

        metrics = {
            "accuracy" : accuracy_score(y_test_list, y_pred_denc)
        }

        cm_display = confusion_matrix(y_test_list, y_pred_denc)

        cm_display = ConfusionMatrixDisplay(confusion_matrix = cm_display)

        cm_display.plot()
        plt.show()

        return metrics
    
class TestRunNeuralNetwork(unittest.TestCase) :
    
    def setUp(self) :
        """
        Set up test fixtures before each test method.
        
        """

        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.train_file = os.path.join(self.temp_dir, "train.csv")
        self.test_file = os.path.join(self.temp_dir, "test.csv")
        self.model_dir = os.path.join(self.temp_dir, "models")
        
        # Create dummy CSV files
        train_data = pd.DataFrame({
            'label': [0, 1, 2, 3, 4],
            **{f'pixel{i}': [100, 150, 200, 50, 75] for i in range(784)}
        })
        train_data.to_csv(self.train_file, index=False)
        
        test_data = pd.DataFrame({
            'label': [1, 2, 3],
            **{f'pixel{i}': [120, 180, 90] for i in range(784)}
        })
        test_data.to_csv(self.test_file, index=False)
        
        # Test parameters
        self.target_column = "label"
        self.model_filename = "test_model.pkl"
        self.log_filename = "test_log.json"
    
    
    def tearDown(self) :
        """
        Clean up after each test method.
        """

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('builtins.print')
    def test_successful_pipeline_execution(self, mock_print) :
        """
        Test that the complete pipeline executes successfully.
        
        """
        network, metrics = run_neural_network(
            self.train_file,
            self.test_file,
            self.target_column,
            self.model_dir,
            self.model_filename,
            self.log_filename
        )
        
        # Check return types
        self.assertIsInstance(network, MockNumpyNeuralNetwork)
        self.assertIsInstance(metrics, dict)
        
        # Check that metrics contain expected keys
        self.assertIn("accuracy", metrics)
        
        # Check that progress messages were printed
        expected_messages = [
            call("\n1. Loading datasets..."),
            call("\n2. Cleaning and preprocessing data..."),
            call("\n3. Creating and training model..."),
            call("\n4. Making predictions..."),
            call("\n5. Evaluating model"),
            call("\n6. Saving model and logs...")
        ]
        
        for expected_call in expected_messages:
            self.assertIn(expected_call, mock_print.call_args_list)

    @patch('builtins.print')
    def test_return_value_structure(self, mock_print) :
        """
        Test that the function returns the correct structure.
        """

        result = run_neural_network(
            self.train_file,
            self.test_file,
            self.target_column,
            self.model_dir,
            self.model_filename,
            self.log_filename
        )
        
        # Should return a tuple
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
        network, metrics = result
        
        # First element should be the network
        self.assertIsInstance(network, MockNumpyNeuralNetwork)
        
        # Second element should be metrics dictionary
        self.assertIsInstance(metrics, dict)
        self.assertIn("accuracy", metrics)

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)