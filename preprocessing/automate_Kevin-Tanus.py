import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(input_path, output_path):
    # Load Data
    df = pd.read_csv(input_path)
    # Hapus Duplikat
    df_clean = df.drop_duplicates().copy()
    feature_cols = [col for col in df_clean.columns if col not in ['target', 'species']]
    # Encoding
    le = LabelEncoder()
    if 'species' in df_clean.columns:
        df_clean['target_encoded'] = le.fit_transform(df_clean['species'])
        y = df_clean['target_encoded']
    else:
        y = df_clean['target']

    # Scaling
    X = df_clean[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Final DataFrame
    df_preprocessed = pd.DataFrame(X_scaled, columns=feature_cols)
    df_preprocessed['target'] = y.values

    # Save result
    df_preprocessed.to_csv(output_path, index=False)
    print(f'Data processed and saved to {output_path}')

if __name__ == "__main__":
    preprocess_data('breast_cancer_raw.csv', 'preprocessing/breast_cancer_cleaned.csv')
