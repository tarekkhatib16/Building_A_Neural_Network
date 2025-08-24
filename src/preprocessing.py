import pandas as pd
import numpy as np

def clean_dataset(
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
    num_classes = 10  # should be 10 for MNIST
    y_train_enc = np.eye(num_classes)[y_train]
    y_test_enc = np.eye(num_classes)[y_test]

    return X_train, X_test, y_train_enc, y_test_enc