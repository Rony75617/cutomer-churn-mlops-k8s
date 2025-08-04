import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os
import json

def load_params():
    with open("params.yaml") as f:
        config = yaml.safe_load(f)

    train_cfg = config["train"]
    return train_cfg

# Load params
def train_model(train_cfg):
    
    data_path = train_cfg["data"]

# Load data
    df = pd.read_csv(data_path)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"]
    )

# MLflow tracking
    os.makedirs("/tmp/mlruns", exist_ok=True)
    mlflow.set_tracking_uri("/tmp/mlruns")
    
    with mlflow.start_run():
        clf = RandomForestClassifier(
            max_depth=train_cfg["max_depth"],
            n_estimators=train_cfg["n_estimators"],
            random_state=train_cfg["random_state"]
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        metrics = {
        "accuracy": acc,
        "auc": auc
        
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)   
        # Log metrics and model
        mlflow.log_param("max_depth", train_cfg["max_depth"])
        mlflow.log_param("n_estimators", train_cfg["n_estimators"])
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("auc", auc)

        mlflow.sklearn.log_model(clf, "model")

    return clf

def save_model(clf,train_cfg):
    # Save model locally
    os.makedirs(os.path.dirname(train_cfg["model_path"]), exist_ok=True)
    joblib.dump(clf, train_cfg["model_path"])

train_cgf=load_params()
mdl=train_model(train_cgf)
save_model(mdl,train_cgf)


