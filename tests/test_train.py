import os
import pytest
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.train import train_model, save_model,load_params

@pytest.fixture
def sample_data():
    df = pd.read_csv("processed/cleaned_telco.csv")
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y

def test_model_training(sample_data):
    X, y = sample_data
    train_cgf=load_params()
    model=train_model(train_cgf)
    
    assert hasattr(model, "fit"), "Model is not a scikit-learn compatible estimator."
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    assert acc > 0.5, f"Accuracy too low: {acc}"

def test_model_saving(tmp_path, sample_data):
    X, y = sample_data
    train_cgf=load_params()
    model=train_model(train_cgf)
    path = train_cgf["model_path"]
    
    save_model(model, train_cgf)
    assert os.path.exists(path), "Model file not saved"
    
    loaded = joblib.load(path)
    assert hasattr(loaded, "predict"), "Loaded model can't predict"