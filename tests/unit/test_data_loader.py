import unittest
import pandas as pd
import numpy as np 
import os
import sys
import tempfile

# Add the project root to sys.path to allow imports from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the function to be tested
from src.data_loader import data_loader

class TestDataLoader(unittest.TestCase) :
    def setUp(self) :
        
        """ 
        Sets up test scenarios before each test method 
        
        """
        
        # Create sample MNIST-like data for testing
        self.sample_mnist_data = {
            'label': [5, 0, 4, 1, 9, 2, 1, 3, 1, 4],
            **{f'pixel{i}': np.random.randint(0, 256, 10) for i in range(784)}
        }
        self.sample_df = pd.DataFrame(self.sample_mnist_data)
        
        # Create a temporary CSV file with sample data
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.sample_df.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()

    def tearDown(self) : 
        
        """ 
        This function cleans up the test data after each test mtehod 
        
        """
        
        if os.path.exists(self.temp_file.name) :
            os.unlink(self.temp_file.name)
    
    def test_successful_data_loading(self) : 
        
        """ This function tests that data_loader successfully loads a valid CSV file """
        
        with unittest.mock.patch('builtins.print') as mock_print :
            result = data_loader(self.temp_file.name)

            # Check that a DataFrame is returned 
            self.assertIsInstance(result, pd.DataFrame)

            # Check that the data matches expected structure 
            self.assertEqual(len(result), 10) # 10 samples
            self.assertEqual(len(result.columns), 785) # 1 label and 784 pixels
            self.assertIn('label', result.columns)

            # Check that labels are in expected range (0-9 for MNIST)
            self.assertTrue(all(result['label'].beteen(0,9)))

            # Check that pixel values are in expected range (0-255)
            pixel_cols = [col for col in result.columns if col.startswith('pixel')]
            for col in pixel_cols[:5] : # Test first 5 pixel columns 
                self.assertTrue(all(result[col].between(0,255)))
            
            # Check that success message was printed 
            mock_print.assert_called_once_with(f"Dataset loaded from CSV: {self.temp_file.name}")

    def test_file_not_found(self) : 
        
        """ 
        This function tests that data_loader raises RuntimeError
        
        """

        fake_file = "fake_file.csv"

        with self.assertRaises(RuntimeError) as context :
            data_loader(fake_file)

        self.assertIn("Failed to load dataset:", str(context.exception))
    
    def test_invalid_csv_format(self) : 
        """
        This function tests that data_loader handles invalid CSV formats

        """

        invalid_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        invalid_file.write('Not a valid CSV')
        invalid_file.close()

        try :
            result = data_loader(invalid_file.name)
            self.assertIsInstance(result, pd.DataFrame)
        finally : 
            os.unlink(invalid_file.name)
    
    def test_empty_file(self) : 
        """
        Test that data_loader handles empty CSV files 
        
        """

        empty_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        empty_file.close()

        try :
            result = data_loader(empty_file.name)
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(result.empty)
        finally :
            os.unlink(empty_file.name)

if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)