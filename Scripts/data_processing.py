import os
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from helper_function import log_info, log_error

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'Data', 'data.csv')
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv('ARTIFACTS_DIR', 'Artifacts'))

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

PIPELINE_PATH = os.path.join(ARTIFACTS_DIR, "data_processing_pipeline.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")

def create_data_pipeline(data):
    try:
        categorical = data.select_dtypes(include=['object']).columns.tolist()
        numerical = data.select_dtypes(include=['int64', 'float64', 'number']).columns.tolist()

        log_info(f"Categorical columns: {categorical}")
        log_info(f"Numerical columns: {numerical}")

        transformers = []
        if categorical:
            transformers.append(("cat", OneHotEncoder(drop='first', handle_unknown='ignore'), categorical))
        if numerical:
            transformers.append(("num", MinMaxScaler(), numerical))

        if not transformers:
            log_error("No categorical or numerical features found.")
            return None

        pipeline = Pipeline(steps=[("preprocessor", ColumnTransformer(transformers))])
        log_info("Data processing pipeline created.")
        return pipeline
    except Exception as e:
        log_error(f"Pipeline creation failed: {str(e)}")
        return None

def save_pipeline(pipeline):
    try:
        with open(PIPELINE_PATH, 'wb') as f:
            pickle.dump(pipeline, f)
        log_info(f"Pipeline saved at {PIPELINE_PATH}")
    except Exception as e:
        log_error(f"Saving pipeline failed: {str(e)}")

def encode_response_variable(y):
    try:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        with open(LABEL_ENCODER_PATH, 'wb') as f:
            pickle.dump(le, f)

        log_info(f"Label encoder saved at {LABEL_ENCODER_PATH}")
        return y_encoded
    except Exception as e:
        log_error(f"Encoding failed: {str(e)}")
        return None

def split_data(X, y, test_size=0.2, random_state=42):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        log_info(f"Split: Train={len(X_train)}, Test={len(X_test)}")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        log_error(f"Split failed: {str(e)}")
        return None, None, None, None