import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates

# --- Page Configuration ---
st.set_page_config(
    page_title="Downhole Equipment Anomaly Detector",
    page_icon="🤖",
    layout="wide"
)

# --- Constants ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "isolation_forest_model.pkl"
FEATURES_PATH = BASE_DIR / "model_features.json"
SAMPLE_DATA_PATH = BASE_DIR / "sample_data.csv"

# --- Asset Loading ---
@st.cache_resource(show_spinner="Loading AI model and features...")
def load_model_and_features():
    """
    Load the trained Isolation Forest model and the list of model features.
    Using st.cache_resource to load these assets only once.
    """
    try:
        model = joblib.load(MODEL_PATH)
        with open(FEATURES_PATH, 'r') as f:
            features = json.load(f)
        return model, features
    except FileNotFoundError:
        st.error(f"Model or features file not found. Ensure '{MODEL_PATH.name}' and '{FEATURES_PATH.name}' are in the script's directory.")
        return None, None

# --- Main Logic Functions ---
def process_uploaded_file(uploaded_file, model, model_features):
    """Processes the uploaded CSV file to find and display anomalies."""
    try:
        data = pd.read_csv(uploaded_file)
        
        # --- Validation ---
        required_cols = set(model_features + ['DateTime'])
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            st.error(f"The uploaded CSV is missing required columns: `{', '.join(missing_cols)}`")
            return

        # --- Data Preprocessing ---
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        data = data.set_index('DateTime').sort_index()

        # --- Anomaly Detection ---
        st.info("Running anomaly detection on the uploaded data...")
        X = data[model_features]
        data['Anomaly_Score'] = model.decision_function(X)
        data['Is_Anomaly'] = model.predict(X)  # -1 for anomalies, 1 for normal

        anomalies = data[data['Is_Anomaly'] == -1]
        num_anomalies = len(anomalies)
        total_points = len(data)
        anomaly_percentage = (num_anomalies / total_points) * 100 if total_points > 0 else 0

        st.success(f"Detection complete. Found {num_anomalies} potential anomalies.")

        # --- Display Results ---
        st.header("Anomaly Detection Results")

        # Key Metrics
        col1, col2 = st.columns(2)
        col1.metric("Total Anomalies Detected", f"{num_anomalies}")
        col2.metric("Anomaly Percentage", f"{anomaly_percentage:.2f}%")

        # Time-Series Plot
        if 'pressure' in data.columns:
            st.subheader("Pressure Time-Series with Anomalies")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(data.index, data['pressure'], color='cornflowerblue', label='Normal Pressure Reading', zorder=1)
            if not anomalies.empty:
                ax.scatter(anomalies.index, anomalies['pressure'], color='red', label='Detected Anomaly', zorder=2, s=50)
            
            ax.set_title('Pressure Sensor Readings Over Time', fontsize=16)
            ax.set_xlabel('DateTime', fontsize=12)
            ax.set_ylabel('Pressure', fontsize=12)
            ax.legend()
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            fig.autofmt_xdate()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            st.pyplot(fig)

        # Data Table of Anomalies
        if not anomalies.empty:
            st.subheader("Top 10 Most Significant Anomalies")
            st.markdown("These are the data points with the lowest anomaly scores, indicating the strongest deviations from normal behavior.")
            top_anomalies = anomalies.sort_values(by='Anomaly_Score', ascending=True).head(10)
            st.dataframe(top_anomalies)

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")

def main():
    """Main function to run the Streamlit application."""
    st.title("AI-Powered Downhole Equipment Anomaly Detector")
    st.markdown("""
    Upload a CSV file with sensor readings to detect anomalies in downhole equipment operations.
    The model will identify unusual patterns that could indicate potential equipment failure.
    """)

    model, model_features = load_model_and_features()

    if model and model_features:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file:
            process_uploaded_file(uploaded_file, model, model_features)
        else:
            st.info("Awaiting CSV file upload to begin analysis.")
            st.subheader("Expected Data Format")
            try:
                sample_df = pd.read_csv(SAMPLE_DATA_PATH, nrows=5)
                st.dataframe(sample_df)
            except FileNotFoundError:
                st.warning(f"`{SAMPLE_DATA_PATH.name}` not found. Cannot display sample format.")

if __name__ == "__main__":
    main()