import pandas as pd
import pytest
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from notebooks.preprocessing import preprocess,clean_data,encode_data

@pytest.fixture
def raw_data():
    # Load a small sample from your actual dataset (or mock it)
    df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return df

def test_preprocess_shape_and_target(raw_data):
    df=clean_data(raw_data)
    df = encode_data(df)
    final_df = df
    
    # Ensure Churn column exists and is last
    assert "Churn" in final_df.columns, "Target column 'Churn' missing"
    assert final_df["Churn"].isnull().sum() == 0, "Target column has nulls"

    # Ensure no nulls in final features
    assert final_df.isnull().sum().sum() == 0, "Data contains null values after preprocessing"
    assert final_df.shape[1] > 10, "Too few features after encoding"

def test_feature_distribution(raw_data):
    df=clean_data(raw_data)
    df = encode_data(df)
    final_df = df
    feature_cols = final_df.drop(columns=["Churn"])

    # Check that all features are numeric (after encoding and scaling)
    assert all(pd.api.types.is_numeric_dtype(col) for col in feature_cols.dtypes), \
        "Not all features are numeric"

def test_scaling_range(raw_data):
    df=clean_data(raw_data)
    df = encode_data(df)
    final_df = df
    numeric_cols = [col for col in final_df.columns if "Tenure" in col or "Charges" in col]
    
    for col in numeric_cols:
        col_mean = final_df[col].mean()
        assert abs(col_mean) < 1.0, f"Mean for {col} is too large; check scaling"