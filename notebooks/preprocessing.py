import pandas as pd
import os
from sklearn.preprocessing import  StandardScaler,OneHotEncoder
import argparse

def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(df):
    # Drop customerID if exists
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # Convert total charges to numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    df = df.dropna(subset=['TotalCharges'])
   
    return df

def encode_data(df):

    df = df.rename(columns={
    'gender': 'Gender',
    'SeniorCitizen': 'Senior_Citizen',
    'tenure': 'Tenure',
    'PhoneService': 'Phone_Service',
    'MultipleLines': 'Multiple_Lines',
    'InternetService': 'Internet_Service',
    'OnlineSecurity': 'Online_Security',
    'OnlineBackup': 'Online_Backup',
    'DeviceProtection': 'Device_Protection',
    'TechSupport': 'Tech_Support',
    'StreamingTV': 'Streaming_TV',
    'StreamingMovies': 'Streaming_Movies',
    'PaperlessBilling': 'Paperless_Billing',
    'PaymentMethod': 'Payment_Method',
    'MonthlyCharges': 'Monthly_Charges',
    'TotalCharges': 'Total_Charges'
    })

    num_features = ['Tenure', 'Monthly_Charges', 'Total_Charges']

    cat_features = ['Gender','Senior_Citizen','Partner','Dependents','Phone_Service','Multiple_Lines',
                'Internet_Service','Online_Security','Online_Backup','Device_Protection','Tech_Support',
                'Streaming_TV', 'Streaming_Movies','Contract','Paperless_Billing','Payment_Method']


    #label_cols = df.select_dtypes(include='object').columns

    le = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    scaler = StandardScaler()
    onehot_encoded_array=le.fit_transform(df[cat_features])
    print(onehot_encoded_array.shape)
    feature_names = le.get_feature_names_out(cat_features)
    print(len(feature_names))
    one_hot_encoded_df = pd.DataFrame(onehot_encoded_array, columns=feature_names, index=df.index)
    target=df['Churn'].map({'Yes': 1, 'No': 0})
    df_non_categorical = scaler.fit_transform(df[num_features])
    #scalefeature_names = scaler.get_feature_names_out(num_features)
    scale_df = pd.DataFrame(df_non_categorical, columns=num_features, index=df.index)
    final_df = pd.concat([one_hot_encoded_df, scale_df,target.rename('Churn')], axis=1)

        
    
    return final_df



def preprocess(input_path, output_path):
    print("Loading data...")
    df = load_data(input_path)
    print(f"Original shape: {df.shape}")

    print("Cleaning data...")
    df = clean_data(df)

    print("Encoding data...")
    df = encode_data(df)

    

    print(f"Saving processed data to {output_path} ...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    parser.add_argument('--output', type=str, default='processed/cleaned_telco.csv')
    args = parser.parse_args()

    preprocess(args.input, args.output)