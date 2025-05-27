from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import logging

def train_model(X, y):
    logging.info("Splitting data and training model")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    logging.info(f"Model Accuracy: {score:.2f}")
    return model, score

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    report = classification_report(y, predictions, output_dict=True)
    logging.info(f"Classification Report: {report}")
    return report
