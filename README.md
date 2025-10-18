# AI-Powered Downhole Equipment Anomaly Detector

A Streamlit web application that uses an unsupervised machine learning model (Isolation Forest) to detect anomalies in sensor data from downhole oil and gas equipment. It provides a user-friendly interface to upload time-series data and visualize potential operational issues in real-time.

  <!-- It's a good idea to add a screenshot of your running app here! -->

---

## The Potential: Unlocking Predictive Maintenance

This tool is more than just a data visualizer; it's a foundational component for a modern predictive maintenance strategy. By leveraging machine learning, it transforms raw sensor data into actionable intelligence.

### Key Business & Operational Value

*   **Proactive Failure Prevention**: Instead of reacting to equipment failures after they happen, this tool enables a proactive approach. By identifying subtle deviations from normal operating parameters (like unusual spikes in pressure or motor speed), engineers can investigate and perform maintenance *before* a critical failure occurs.

*   **Reduced Operational Downtime**: Unplanned downtime in drilling and extraction operations is incredibly costly. Predictive maintenance, powered by this anomaly detector, can significantly reduce these non-productive periods by allowing teams to schedule repairs during planned shutdowns, rather than being forced into emergency stops.

*   **Enhanced Safety**: Equipment failure can pose significant safety and environmental risks. Early detection of anomalies related to pressure, temperature, or vibration can help prevent catastrophic events, protecting personnel, the environment, and company assets.

*   **Optimized Resource Allocation**: Maintenance teams can focus their efforts on equipment that shows early signs of trouble, rather than adhering to rigid, time-based maintenance schedules that may be inefficient or unnecessary. This leads to better use of labor and spare parts inventory.

*   **Data-Driven Decision Making**: This tool empowers operations managers and engineers to make more informed decisions about equipment health and operational strategy. It provides clear, visual evidence of when and how a piece of equipment is behaving abnormally.

---

## Features

- **Interactive File Uploader**: Easily upload new CSV data for analysis.
- **Dashboard with Key Metrics**: At-a-glance view of the total anomalies detected and the overall anomaly percentage.
- **Dynamic Time-Series Visualization**: Plots the `pressure` sensor readings over time and highlights detected anomalies in red for immediate identification.
- **Detailed Anomaly Table**: Lists the top 10 most significant anomalous readings, sorted by their anomaly score, allowing for deeper investigation.

---

## Technology Stack

- **Backend & ML**: Python, Scikit-learn, Pandas, Joblib
- **Frontend**: Streamlit
- **Plotting**: Matplotlib

---

## Local Setup and Installation

Follow these steps to run the application on your local machine.

### 1. Prerequisites
- Python 3.8+
- `pip` (Python package installer)

### 2. Clone the Repository
```bash
git clone https://github.com/runabh1/downhole-anomaly-equipment-detector.git
cd downhole-anomaly-equipment-detector
```

### 3. Install Dependencies
Install all the required libraries from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

### 4. Generate the Model
Run this script **once** to train the Isolation Forest model and create the `isolation_forest_model.pkl` file.
```bash
python create_model.py
```

### 5. Run the Streamlit App
Launch the web application.
```bash
streamlit run app.py
```
Your web browser will automatically open a new tab with the application running.

---