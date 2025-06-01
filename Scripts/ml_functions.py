import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import logging

logging.basicConfig(level=logging.INFO)

def preprocess_data(df):
    try:
        logging.info("Starting data preprocessing")

        # Drop rows with missing values
        df.dropna(inplace=True)

        # Assume the last column is the target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        # Create transformers
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Combine into a preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        # Apply transformation
        X_transformed = preprocessor.fit_transform(X)

        return X_transformed, y

    except Exception as e:
        logging.error(f"Error in preprocess_data: {e}")
        raise ValueError("Pipeline creation failed.")

def split_data(X, y):
    logging.info("Splitting data into train and test sets")
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    logging.info("Training model with GridSearchCV and tracking with MLflow")
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

        return best_model, accuracy

def evaluate_model(model, X_test, y_test):
    logging.info("Evaluating model")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, report
