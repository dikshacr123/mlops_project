from helper_function import setup_logger, log_info, log_error
setup_logger()

from data_processing import create_data_pipeline, save_pipeline, encode_response_variable, split_data
from ml_functions import train_model, evaluate_model
import pandas as pd

def preprocess_data(X):
    pipeline = create_data_pipeline(X)
    if pipeline is None:
        raise ValueError("Pipeline creation failed.")
    pipeline.fit(X)
    save_pipeline(pipeline)
    return pipeline.transform(X)

def run_pipeline(filepath):
    try:
        df = pd.read_csv(filepath, sep=';')
        log_info("CSV read into DataFrame.")

        X_raw = df.iloc[:, :-1]
        y_raw = df.iloc[:, -1]

        X_processed = preprocess_data(X_raw)
        y_encoded = encode_response_variable(y_raw)

        X_train, X_test, y_train, y_test = split_data(X_processed, y_encoded)
        model, accuracy = train_model(X_train, y_train)
        report = evaluate_model(model, X_test, y_test)

        log_info(f"Pipeline completed. Accuracy: {accuracy}")
        return model,accuracy, report
    except Exception as e:
        log_error(f"Pipeline execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline("Data\data.csv")