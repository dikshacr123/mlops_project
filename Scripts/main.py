from data_processing import load_data, preprocess_data
from ml_functions import train_model, evaluate_model
from helper_function import setup_logger

def run_pipeline(filepath=r"C:/Users/crdik/OneDrive/Desktop/mlops_project/Data/data.csv"):
    setup_logger()

    df = load_data(filepath)

    # Assuming last column is the target
    X = preprocess_data(df.iloc[:, :-1])
    y = df.iloc[:, -1]

    model, accuracy = train_model(X, y)
    report = evaluate_model(model, X, y)

    return accuracy, report
