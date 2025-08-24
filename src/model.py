# importing libraries, classes, and functions 

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

from src.Layer import Layer
from src.Functions.Sigmoid import sigmoid
from src.Functions.Loss import meanSquareError
from src.Functions.deEnoder import deEncoder
from src.Functions.toList import toList

class NumpyNeuralNetwork :
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

def compute_metrics(
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
    