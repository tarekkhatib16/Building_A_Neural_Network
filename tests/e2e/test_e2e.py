import unittest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
from io import StringIO
import os
import tempfile
import shutil
import sys
import json

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

# Mock run_neural_network function
def mock_run_neural_network(train_file_path, test_file_path, target_column, 
                            model_dir_path, model_filename, log_filename):
    """
    Mock implementation of run_neural_network
    """
    # Simulate successful training
    mock_model = MagicMock()
    mock_metrics = {"accuracy": 0.9234, "loss": 0.0456}
    return mock_model, mock_metrics

def mock_run_neural_network_with_error(train_file_path, test_file_path, target_column,
                                      model_dir_path, model_filename, log_filename):
    """
    Mock implementation that raises an error
    """
    raise ValueError("Mock training error for testing")

class TestMainIntegration(unittest.TestCase) :

    def setUp(self) : 
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
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('builtins.print')
    def test_main_pipeline_end_to_end(self, mock_print) :
        """
        End-to-end test of the main function with mocked pipeline.
        """
        print("\n--- Running End-to-End Pipeline Test ---")
        print(self.test_base_path)

        # Run the pipeline (main) 
        # Pass the temporary directory as the base output directory
        main(output_base_dir = self.test_base_path)

        # Expect output file paths
        model_path = self.test_base_path / MODEL_STORE_DIR / MODEL_FILENAME
        log_path = self.test_base_path / MODEL_STORE_DIR / LOG_FILENAME

        # Assert files were created
        self.assertTrue(model_path.exists(), f"Model file not found at {model_path}")
        self.assertTrue(log_path.exists(), f"Log file not found at {log_path}") 
        print("Model and log files confirmed to exist")

        # Load and verify log contents
        with open(log_path, 'r') as f :
            metrics_log = json.load(f)
            
            # The log file should contain accuracy and loss
            self.assertIsInstance(metrics_log, list)
            self.assertGreater(len(metrics_log), 0, "Log file should contain at least one run detail")
            
            first_run = metrics_log[0].get('metrics', {})
            self.assertIsNotNone(first_run, "First run metrics should contain 'metrics' key")

            # Assert specific metrics are present and have reasonable values
            self.assertIn('accuracy', first_run)
            self.assertIsInstance(first_run['accuracy'], float)
            self.assertGreaterEqual(first_run['accuracy'], 0.0)
            self.assertLessEqual(first_run['accuracy'], 1.0)
            print("Logged metrics verified (accuracy range).")

        print("--- End-to-End Pipeline Test Completed Successfully ---\n")

if __name__ == '__main__':
    unittest.main()
