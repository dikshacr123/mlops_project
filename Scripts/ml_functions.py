import os
import pickle
import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
from helper_function import log_info, log_error

load_dotenv()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, os.getenv('ARTIFACTS_DIR', 'Artifacts'))
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "trained_model.pkl")

def train_model(X_train, y_train):
    log_info("Training model with GridSearchCV and MLflow")
    try:
        with mlflow.start_run():
            param_grid = {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            }

            grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy')
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            accuracy = grid.best_score_

            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("cv_accuracy", accuracy)
            mlflow.sklearn.log_model(best_model, "model")

            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(best_model, f)
            log_info(f"Model saved at {MODEL_PATH}")

            return best_model, accuracy
    except Exception as e:
        log_error(f"Training failed: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report_str = classification_report(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        log_info("Classification Report:\n" + report_str)
        log_info("Confusion Matrix:\n" + np.array2string(cm))

        return acc, report_dict
    except Exception as e:
        log_error(f"Evaluation failed: {str(e)}")
        return None, None