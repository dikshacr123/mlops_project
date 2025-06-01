import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from helper_function import log_info, log_error

# Load environment variables from .env
load_dotenv()

# Define base paths dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'Data', 'data.csv')
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv('ARTIFACTS_DIR', 'Artifacts'))

# Ensure Artifacts directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Paths to save models/artifacts
PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "data_processing_pipeline.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")


def load_data():
    """
    Loads data from the data.csv file located in the Data directory.
    """
    try:
        data = pd.read_csv(DATA_PATH)
        log_info(f"Data loaded successfully from {DATA_PATH}")
        return data
    except Exception as e:
        log_error(f"Failed to load data: {str(e)}")
        return None


def create_data_pipeline(data):
    """
    Creates a preprocessing pipeline using OneHotEncoder for categorical features
    and MinMaxScaler for numerical features.
    """
    try:
        categorical_features = data.select_dtypes(include=['object']).columns.tolist()
        numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

        transformers = []
        if categorical_features:
            transformers.append(("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features))
        if numerical_features:
            transformers.append(("num", MinMaxScaler(), numerical_features))

        if not transformers:
            log_error("No categorical or numerical features found in the data.")
            return None

        pipeline = Pipeline(steps=[("preprocessor", ColumnTransformer(transformers))])
        log_info("Data processing pipeline created successfully.")
        return pipeline

    except Exception as e:
        log_error(f"Error in creating pipeline: {str(e)}")
        return None


def save_pipeline(pipeline):
    """
    Saves the preprocessing pipeline to a pickle file.
    """
    try:
        with open(PIPELINE_PATH, 'wb') as file:
            pickle.dump(pipeline, file)
        log_info(f"Pipeline saved at {PIPELINE_PATH}")
    except Exception as e:
        log_error(f"Failed to save pipeline: {str(e)}")


def encode_response_variable(y):
    """
    Encodes target variable using LabelEncoder and saves the encoder.
    """
    try:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        with open(LABEL_ENCODER_PATH, 'wb') as f:
            pickle.dump(label_encoder, f)

        log_info(f"Label encoder saved at {LABEL_ENCODER_PATH}")
        return y_encoded

    except Exception as e:
        log_error(f"Failed to encode response variable: {str(e)}")
        return None


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into train and test sets.
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        log_info(f"Data split into train and test sets: Train={len(X_train)}, Test={len(X_test)}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        log_error(f"Error in train-test split: {str(e)}")
        return None, None, None, None
