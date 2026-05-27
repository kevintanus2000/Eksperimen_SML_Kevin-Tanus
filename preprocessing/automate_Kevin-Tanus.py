import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(input_path, output_path):
    # Load
    df = pd.read_csv(input_path)
    
    # Preprocessing
    df_clean = df.drop(columns=['target'])
    le = LabelEncoder()
    df_clean['species_encoded'] = le.fit_transform(df_clean['species'])
    
    X = df_clean.drop(columns=['species', 'species_encoded'])
    y = df_clean['species_encoded']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    df_preprocessed = pd.DataFrame(X_scaled, columns=X.columns)
    df_preprocessed['target'] = y
    
    # Save result
    df_preprocessed.to_csv(output_path, index=False)
    print(f'Data processed and saved to {output_path}')

if __name__ == "__main__":
    preprocess_data('iris_raw.csv', 'preprocessing/iris_cleaned.csv')