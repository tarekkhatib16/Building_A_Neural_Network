import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from src.Layer import Layer
from src.Functions.Sigmoid import sigmoid
from src.Functions.Loss import meanSquareError
from src.Functions.deEnoder import deEncoder
from src.Functions.toList import toList

from src.data_loader import data_loader
from src.preprocessing import clean_dataset
from src.model import NumpyNeuralNetwork, compute_metrics

def run_neural_network(
    train_file_path : str,
    test_file_path : str,
    target_column : str,
    model_dir_path : str,
    model_filename : str, 
    log_filename : str
) -> tuple[NumpyNeuralNetwork, dict]: 
    """
    Runs the complete MNIST prediction pipeline
    
    Args : 
        train_file_path (str) : Path to the raw training CSV file.
        test_file_path (str) : Path to the raw test CSV file. 
        target_column (str) : Name of the target column. 
        model_dir_path (str) : Directory where the trained model and logs will be saved.
        model_filename (str) : Name of the file to save the trained model.
        log_filename (str) : Name of the file to save the run logs. 
    
    Returns : 
        Tuple[NumpyNeuralNetwork, Dict[str, Any]] : 
            - The trained NumpyNeuralNetwork.
            - A dictionary containing evaluation metrics.
    
    """

    print("\n1. Loading datasets...")
    traindf = data_loader(train_file_path)
    testdf = data_loader(test_file_path) 

    print("\n2. Cleaning and preprocessing data...")
    X_train, X_test, y_train, y_test = clean_dataset(traindf, testdf, target_column)

    print("\n3. Creating and training model...")
    network = NumpyNeuralNetwork()
    network.add(Layer(28*28, 100, sigmoid))
    network.add(Layer(100, 50, sigmoid))
    network.add(Layer(50, 10, sigmoid))

    # train on 1000 samples
    network.setLossFunction(meanSquareError)
    network.fit(X_train, y_train, epochs = 4, learning_rate = 0.1) 

    print("\n4. Making predictions...")
    y_pred = network.predict(X_test)

    print("\n5. Evaluating model")
    metrics = compute_metrics(y_test, y_pred)

    print("\n6. Saving model and logs...")
    model_file_path = os.path.join(model_dir_path, model_filename)
    network.save_model(filepath=model_file_path)
    network.log_run(model_dir_path, metrics, log_filename)

    print(f"Model saved to: {model_file_path}")
    print(f"Run log saved to: {os.path.join(model_dir_path, log_filename)}")

    return network, metrics