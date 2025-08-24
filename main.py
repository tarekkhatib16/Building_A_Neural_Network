import os 
import sys
from pathlib import Path
from src.pipeline import run_neural_network
from src.config import (
    TARGET_COLUMN, MODEL_FILENAME, LOG_FILENAME, DATA_DIR_NAME,
    RAW_DATA_DIR_NAME, TRAIN_FILENAME, TEST_FILENAME, MODEL_STORE_DIR)

def main(output_base_dir = None) -> None :
    """
    Main function to help orchestrate the entire NN pipeline.

    Args:
        output_base_dir (Path, optional) : The base directory where data and model 
                                            artifacts should be read from/saved to.
                                            If None, defaults to the script's directory.

    """
    print("Starting the MNIST Prediction Pipeline")
    print("="*60)

    try :
        # Construct necessary paths
        if output_base_dir is None : 
            # Default to script's directory for normal runs
            base_path = Path(os.path.dirname(os.path.abspath(__file__)))
        else : 
            # Use the provided base directory for testing 
            base_path = output_base_dir
        
        train_file_path = base_path / DATA_DIR_NAME / RAW_DATA_DIR_NAME / TRAIN_FILENAME
        test_file_path = base_path / DATA_DIR_NAME / RAW_DATA_DIR_NAME / TEST_FILENAME
        model_dir_path = base_path / MODEL_STORE_DIR

        # Run the entire training pipeline 
        trained_model, evaluation_metrics = run_neural_network(
            train_file_path = train_file_path,
            test_file_path = test_file_path,
            target_column = TARGET_COLUMN,
            model_dir_path = model_dir_path,
            model_filename = MODEL_FILENAME,
            log_filename = LOG_FILENAME
        ) 

        print(f"\n{"="*60}")
        print("Pipeline completed successfully!")
        print(f"Final Model Accuracy: {evaluation_metrics['accuracy']:.4f}")
    
    except Exception as e :
        print(f"ERROR: Pipeline failed with exception: {e}", file=sys.stderr)

if __name__ == "__main__" :
    main()