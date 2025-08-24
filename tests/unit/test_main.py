import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock, call
from pathlib import Path
from io import StringIO
import os
import tempfile
import shutil
import sys

# Add project root to sys.path for import resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from main import main

# Mock configuration constants
TARGET_COLUMN = 'label'
MODEL_FILENAME = 'numpy_neural_network_v1.pkl'
LOG_FILENAME = 'numpy_neural_network_log.json'
MODEL_STORE_DIR = 'model_store'
DATA_DIR_NAME = 'data'
RAW_DATA_DIR_NAME = 'raw'
TRAIN_FILENAME = 'mnist_train.csv'
TEST_FILENAME = 'mnist_test.csv'

def mock_run_neural_network_with_error(train_file_path, test_file_path, target_column,
                                      model_dir_path, model_filename, log_filename):
    """
    Mock implementation that raises an error
    """
    raise ValueError("Mock training error for testing")

class TestMainFunction(unittest.TestCase) :

    def setUp(self):
        """
        Set up test fixtures before each test method.
        """

        # Create temporary directory structure for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_base_path = Path(self.temp_dir)

        # Create expected directory structure
        self.data_dir = self.test_base_path / DATA_DIR_NAME / RAW_DATA_DIR_NAME
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dummy CSV files
        self.train_file = self.data_dir / TRAIN_FILENAME
        self.test_file = self.data_dir / TEST_FILENAME
        
        with open(self.train_file, 'w') as f:
            f.write("label," + ",".join([f"pixel{i}" for i in range(784)]) + "\n")
            f.write("0," + ",".join(["0"] * 784) + "\n")
            f.write("1," + ",".join(["255"] * 784) + "\n")
        
        with open(self.test_file, 'w') as f:
            f.write("label," + ",".join([f"pixel{i}" for i in range(784)]) + "\n")
            f.write("0," + ",".join(["0"] * 784) + "\n")
            f.write("1," + ",".join(["255"] * 784) + "\n")

    def tearDown(self):
        """
        Clean up after each test method.
        """

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('main.run_neural_network')
    def test_main_success(self, mock_run_pipeline):
        """
        Test that main correctly constructs paths and calls the pipeline
        when an output directory is provided.
        """
        dummy_model = MagicMock()
        dummy_metrics = {"accuracy": 0.92}
        mock_run_pipeline.return_value = (dummy_model, dummy_metrics)

        with patch('builtins.print'):
            main(output_base_dir=self.test_base_path)

        expected_training_data_path = self.test_base_path / DATA_DIR_NAME / RAW_DATA_DIR_NAME / TRAIN_FILENAME
        expected_test_data_path = self.test_base_path / DATA_DIR_NAME / RAW_DATA_DIR_NAME / TEST_FILENAME
        model_dir_path = self.test_base_path / MODEL_STORE_DIR

        mock_run_pipeline.assert_called_once_with(
            train_file_path=expected_training_data_path,
            test_file_path=expected_test_data_path,
            target_column=TARGET_COLUMN,
            model_dir_path=model_dir_path,
            model_filename=MODEL_FILENAME,
            log_filename=LOG_FILENAME
        )

    @patch('main.run_neural_network', side_effect=ValueError("Simulated pipeline error"))
    def test_main_pipeline_error_handling(self, mock_run_pipeline):
        """ Test that main logs errors when the pipeline fails. """
        with patch('sys.stderr', new_callable=StringIO) as mock_stderr:
            main(output_base_dir=self.test_base_path)
            stderr_output = mock_stderr.getvalue()

        self.assertIn("ERROR: Pipeline failed with exception:", stderr_output)
        self.assertIn("Simulated pipeline error", stderr_output)

if __name__ == '__main__':
    # Run all tests
    unittest.main()