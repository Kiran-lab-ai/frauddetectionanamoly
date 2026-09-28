from sklearn.ensemble import IsolationForest
import joblib

def train_model(X):
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )
    model.fit(X)
    return model

def score_transactions(model, X):
    predictions = model.predict(X)
    scores = model.decision_function(X)

    return predictions, scores