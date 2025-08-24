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

class TestMainIntegration(unittest.TestCase) :

    def setUp(self) : 
        self.temp_dir = tempfile.mkdtemp()
        self.test_base_path = Path(self.temp_dir)

        # Create directory strucutre 
        data_dir = self.test_base_path / DATA_DIR_NAME / RAW_DATA_DIR_NAME
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create MNIST-like CSV files
        train_data = "label," + ",".join([f"pixel{i}" for i in range(784)]) + "\n"
        for i in range(10):
            train_data += f"{i % 10}," + ",".join([str(j % 256) for j in range(784)]) + "\n"
        
        test_data = "label," + ",".join([f"pixel{i}" for i in range(784)]) + "\n"
        for i in range(5):
            test_data += f"{i % 10}," + ",".join([str(j % 128) for j in range(784)]) + "\n"
        
        with open(data_dir / TRAIN_FILENAME, 'w') as f:
            f.write(train_data)
        
        with open(data_dir / TEST_FILENAME, 'w') as f:
            f.write(test_data)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('builtins.print')
    def test_realistic_file_structure(self, mock_print):
        """
        Test with realistic MNIST file structure.
        """
        with patch('__main__.mock_run_neural_network') as mock_pipeline:
            mock_pipeline.return_value = (MagicMock(), {"accuracy": 0.9123})
            
            main(output_base_dir=self.test_base_path)
            
            # Verify the function completed without errors
            mock_pipeline.assert_called_once()
            
            # Check that the constructed paths point to existing files
            call_args = mock_pipeline.call_args.kwargs
            train_path = call_args['train_file_path']
            test_path = call_args['test_file_path']
            
            self.assertTrue(os.path.exists(train_path))
            self.assertTrue(os.path.exists(test_path))
