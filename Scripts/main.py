from data_processing import split_data, create_data_pipeline, save_pipeline, encode_response_variable
from ml_functions import train_model, evaluate_model
from helper_function import setup_logger, log_error, log_info
import pandas as pd

def load_data(filepath):
    try:
        # FIXED: Added sep=';' to support semicolon-delimited CSV
        df = pd.read_csv(filepath, sep=';')
        log_info(f"Data loaded successfully from {filepath}")
        return df
    except Exception as e:
        log_error(f"Failed to load data: {str(e)}")
        raise

def preprocess_data(X):
    pipeline = create_data_pipeline(X)
    if pipeline is None:
        raise ValueError("Pipeline creation failed.")
    
    pipeline.fit(X)
    save_pipeline(pipeline)

    return pipeline.transform(X)

def run_pipeline(filepath):
    try:
        setup_logger()

        df = load_data(filepath)
        X_raw = df.iloc[:, :-1]
        y_raw = df.iloc[:, -1]

        X_processed = preprocess_data(X_raw)
        y_encoded = encode_response_variable(y_raw)

        X_train, X_test, y_train, y_test = split_data(X_processed, y_encoded)
        model, accuracy = train_model(X_train, y_train)
        report = evaluate_model(model, X_test, y_test)

        return accuracy, report

    except Exception as e:
        log_error(f"Pipeline execution failed: {str(e)}")
        raise
