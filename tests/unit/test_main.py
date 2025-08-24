import unittest
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
TARGET_COLUMN = "label"
MODEL_FILENAME = "mnist_model.pkl"
LOG_FILENAME = "training_log.json"
DATA_DIR_NAME = "data"
RAW_DATA_DIR_NAME = "raw"
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
MODEL_STORE_DIR = "models"

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
            f.write("label,pixel0,pixel1\n0,100,150\n1,200,250\n")
        
        with open(self.test_file, 'w') as f:
            f.write("label,pixel0,pixel1\n0,120,180\n1,220,280\n")

    def tearDown(self):
        """
        Clean up after each test method.
        """

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('builtins.print')
    def test_successful_execution_with_custom_base_dir(self, mock_print):
        """
        Test successful pipeline execution with custom base directory.
        """

        main(output_base_dir=self.test_base_path)
        
        # Check that all expected print statements were called
        expected_calls = [
            call("Starting the MNIST Prediction Pipeline"),
            call("="*60),
            call(f"\n{'='*60}"),
            call("Pipeline completed successfully!"),
            call("Final Model Accuracy: 0.9234")
        ]
        
        for expected_call in expected_calls:
            self.assertIn(expected_call, mock_print.call_args_list)

    @patch('sys.stderr', new_callable=StringIO)
    @patch('builtins.print')
    def test_pipeline_error_handling(self, mock_print, mock_stderr):
        """
        Test error handling when pipeline fails.
        """
        # Patch run_neural_network to raise an error
        with patch('__main__.mock_run_neural_network', mock_run_neural_network_with_error):
            main(output_base_dir=self.test_base_path)
        
        # Check that error was printed to stderr
        stderr_output = mock_stderr.getvalue()
        self.assertIn("ERROR: Pipeline failed with exception:", stderr_output)
        self.assertIn("Mock training error for testing", stderr_output)
        
        # Check that startup messages were still printed
        expected_startup_calls = [
            call("Starting the MNIST Prediction Pipeline"),
            call("="*60)
        ]
        
        for expected_call in expected_startup_calls:
            self.assertIn(expected_call, mock_print.call_args_list)
        
        # Success messages should NOT be printed
        success_calls = [
            call("Pipeline completed successfully!"),
            call(f"\n{'='*60}")
        ]
        
        for success_call in success_calls:
            self.assertNotIn(success_call, mock_print.call_args_list)