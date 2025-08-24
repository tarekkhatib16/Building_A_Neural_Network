import unittest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, call
from datetime import datetime
import sys
import pickle
import tempfile
import shutil

# Add the project root to sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import functions to be tested (and internal functions that will now run unmocked)
from src.pipeline import run_neural_network

# Import class to check instance type
from src.model import NumpyNeuralNetwork

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
            'label': np.array([0, 1, 2, 0, 1]),
            **{f'pixel{i}': np.random.randint(0, 255, 5) for i in range(784)}
        })
        train_data.to_csv(self.train_file, index=False)
        
        test_data = pd.DataFrame({
            'label': np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]),
            **{f'pixel{i}': np.random.randint(0, 255, 10) for i in range(784)}
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
        
        # Check that metrics contain expected keys
        self.assertIn("accuracy", metrics)
        
        # Check that progress messages were printed
        expected_messages = [
            call("\n1. Loading datasets..."),
            call("\n2. Cleaning and preprocessing data..."),
            call("\n3. Creating and training model..."),
            call("\n4. Making predictions..."),
            call("\n5. Evaluating model..."),
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
        self.assertIsInstance(network, NumpyNeuralNetwork)
        
        # Second element should be metrics dictionary
        self.assertIsInstance(metrics, dict)
        self.assertIn("accuracy", metrics)

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)