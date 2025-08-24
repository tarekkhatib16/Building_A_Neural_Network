import pandas as pd

def data_loader(filepath: str) -> pd.DataFrame :
    """
    Loads the MNIST dataset from CSV file. 

    Args :
        filepath (str): Path to the CSV file.
    
    Returns :
        pd.DataFrame: The raw MNIST dataset. 

    """
    try :
        df = pd.read_csv(filepath)
        print(f"Dataset loaded from CSV: {filepath}")

        return df
    except Exception as e :
        raise RuntimeError(f"Failed to load dataset: {e}")
