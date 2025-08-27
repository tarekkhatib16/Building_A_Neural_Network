from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from typing import List, Dict, Any, Union

from src.config import MODEL_FILENAME
from src.model import compute_metrics
from src.Functions.deEncoder import deEncoder
from src.Functions.toList import toList

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR_PATH = BASE_DIR / 'model_store'
MODEL_PATH = MODEL_DIR_PATH / MODEL_FILENAME

app = FastAPI(
    title="Neural Network API",
    description="API for training and evaluating a simple neural network on the MNIST dataset.",
    version="1.0.0",
)

model_pipeline = None

@app.on_event("startup")
async def load_model() :
    """
    Load the pre-trained model when FastAPI app starts up.
    
    """
    try :
        if not MODEL_PATH.exists() :
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")
        with open(MODEL_PATH, 'rb') as f :
            app.state.model_pipeline = pickle.load(f)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError as e : 
        raise RuntimeError(f"Failed to load model: {e}")
    except Exception as e :
        raise RuntimeError(f"An unexpected error occurred while loading the model: {e}")

class NumpyNNPredictRequest(BaseModel) : 
    # Flattened image as a list of 784 pixel values (0-255)
    pixels: List[Union[int, float]] = Field(
        ..., 
        description="Flattened 28x28 grayscale image as a list of 784 integers (0-255).",
        min_items=784,
        max_items=784,
    )

    @validator('pixels')
    def validate_pixels(cls, v):
        """
        Validates that each pixel value is an integer between 0 and 255.
        """
        if len(v) != 784 :
            raise ValueError("The 'pixels' list must contain exactly 784 values, got {len(v)}.")
        
        pixels_array = np.array(v)

        if not np.all((pixels_array >= 0) & (pixels_array <= 255)) :
            raise ValueError("All pixel values must be in the range 0-255.")
        
        normalized_pixels = (pixels_array / 255.0).astype(np.float32).tolist()

        return normalized_pixels

    class Config:
        extra = "forbid"  # Ensures only defined fields are accepted
        schema_extra = {
            "example": {
                "pixels": [0] * 784  # Example of a black 28x28 image
            }
        }

class PredictionResponse(BaseModel) :
    predicted_digit: int = Field(..., description="The predicted digit (0-9).")

@app.post("/predict", response_model=PredictionResponse)
async def predict_MNIST(request: NumpyNNPredictRequest) -> Dict[str, Any]:
    """
    Predicts the digit from the input image pixels using the pre-trained model.
    
    Parameters:
    - request: NumpyNNPredictRequest containing the flattened image pixels.
    
    Returns:
    - A dictionary with the predicted digit and confidence scores for each class.
    """
    if not hasattr(app.state, 'model_pipeline') or app.state.model_pipeline is None :
        raise HTTPException(
            status_code=500, 
            detail="Model is not loaded. Please ensure the model is trained and loaded correctly."
        )


    try:
        pixels_array = np.array(request.pixels, dtype='float32').reshape(1, 1, 784)
        print(f"Input shape: {pixels_array.shape}")
        print(f"Input data type: {pixels_array.dtype}")

    except KeyError as e : 
        raise HTTPException(status_code=400, detail=f"Missing required field: {e}")
    
    except Exception as e :
        raise HTTPException(status_code=400, detail=f"Error processing input data: {e}")
    
    try :
        # Make prediction
        prediction_raw = app.state.model_pipeline.predict(pixels_array)
        prediction = toList(np.array(prediction_raw))
        prediction_int = [[1 if j == max_i else 0 for j in i] for i in prediction for max_i in [max(i)]]
        prediction_denc = deEncoder(prediction_int)
        return PredictionResponse(predicted_digit=int(prediction_denc[0]))
    except Exception as e :
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")