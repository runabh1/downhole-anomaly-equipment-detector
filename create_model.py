import pandas as pd
import joblib
import json
from sklearn.ensemble import IsolationForest
from pathlib import Path

def create_and_save_model(data_path, features_path, model_output_path):
    """
    Trains an Isolation Forest model and saves it to a file.
    """
    print("Loading data and features...")
    # Load the feature list
    with open(features_path, 'r') as f:
        model_features = json.load(f)
    
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Ensure all required features are present
    if not all(feature in df.columns for feature in model_features):
        missing = [f for f in model_features if f not in df.columns]
        raise ValueError(f"Missing required features in data: {missing}")

    print("Training Isolation Forest model...")
    # Initialize and train the model
    # The contamination parameter is an estimate of the proportion of outliers in the data.
    # 'auto' is a good starting point, but this can be tuned (e.g., 0.05 for 5%).
    if_model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    if_model.fit(df[model_features])
    
    print(f"Saving model to {model_output_path}...")
    # Save the trained model
    joblib.dump(if_model, model_output_path)
    print("Model created and saved successfully.")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    create_and_save_model(
        data_path=BASE_DIR / 'sample_data.csv',
        features_path=BASE_DIR / 'model_features.json',
        model_output_path=BASE_DIR / 'isolation_forest_model.pkl'
    )