import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add the project root to sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the function to be tested
from src.preprocessing import clean_dataset

class TestCleanDataset(unittest.TestCase) : 

    def setUp(self) : 
        """
        This function sets up necessary data
        
        """
        
        # Create sample MNIST training data
        np.random.seed(42)
        self.train_data = {
            'label': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            **{f'pixel{i}': np.random.randint(0, 256, 10) for i in range(784)}
        }

        self.train_df = pd.DataFrame(self.train_data)

        # Create sample MNIST test data
        self.test_data = {
            'label': [1, 2, 3, 4, 5],
            **{f'pixel{i}': np.random.randint(0, 256, 5) for i in range(784)}
        }

        self.test_df = pd.DataFrame(self.test_data)

        self.target_column = 'label'
    
    def test_return_type_and_structure(self) : 
        
        """
        This function tests that clean_dataset returns correct structure tuple
        
        """

        result = clean_dataset(self.train_df, self.test_df, self.target_column)
        
        # Should return a tuple with 4 elements
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        
        X_train, X_test, y_train_enc, y_test_enc = result
        
        # Check types
        self.assertIsInstance(X_train, np.ndarray)
        self.assertIsInstance(X_test, np.ndarray)
        self.assertIsInstance(y_train_enc, np.ndarray)
        self.assertIsInstance(y_test_enc, np.ndarray)

    def test_feature_extraction_and_reshaping(self) :
        
        """
        This function tests that features are correctly extracted, reshaped and normalised
        
        """

        X_train, X_test, y_train_enc, y_test_enc = clean_dataset(
            self.train_df, self.test_df, self.target_column
        )
        
        # Check shapes
        self.assertEqual(X_train.shape, (10, 1, 784))  # 10 samples, reshaped
        self.assertEqual(X_test.shape, (5, 1, 784))    # 5 samples, reshaped
        
        # Check data type
        self.assertEqual(X_train.dtype, np.float32)
        self.assertEqual(X_test.dtype, np.float32)
        
        # Check normalization (values should be between 0 and 1)
        self.assertTrue(np.all(X_train >= 0))
        self.assertTrue(np.all(X_train <= 1))
        self.assertTrue(np.all(X_test >= 0))
        self.assertTrue(np.all(X_test <= 1))
    
    def test_one_hot_encoding(self) :
        
        """
        This function tests that one-hot encoding works
        
        """

        X_train, X_test, y_train_enc, y_test_enc = clean_dataset(
            self.train_df, self.test_df, self.target_column
        )
        
        # Check shapes (should be num_samples x num_classes)
        num_classes = len(np.unique(self.train_df[self.target_column]))
        self.assertEqual(y_train_enc.shape, (10, num_classes))
        self.assertEqual(y_test_enc.shape, (5, num_classes))
        
        # Check that each row sums to 1 (one-hot property)
        np.testing.assert_array_almost_equal(
            np.sum(y_train_enc, axis=1), 
            np.ones(10)
        )
        np.testing.assert_array_almost_equal(
            np.sum(y_test_enc, axis=1), 
            np.ones(5)
        )
        
        # Check that each row has exactly one 1 and the rest are 0s
        for i, label in enumerate(self.train_df[self.target_column]):
            expected_encoding = np.zeros(num_classes)
            expected_encoding[label] = 1
            np.testing.assert_array_equal(y_train_enc[i], expected_encoding)
    
    def test_empty_dataframes(self):
        
        """
        Test behavior with empty DataFrames.
        
        """

        empty_train = pd.DataFrame()
        empty_test = pd.DataFrame()
        
        with self.assertRaises((KeyError, ValueError)):
            clean_dataset(empty_train, empty_test, 'label')

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)