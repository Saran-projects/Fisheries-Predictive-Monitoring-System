# Fisheries Predictive Monitoring MVP

This project contains a machine learning prototype for predicting fish biomass based on historical data. 

The system has been consolidated into a single, unified **Streamlit** application, which handles data processing, XGBoost model inference, and the interactive UI dashboard.

## Prerequisites

- Python 3.8+ installed

## How to Run

For complete details on how to set up and run the system, please refer to the [HOW_TO_RUN.md](HOW_TO_RUN.md) document.

### Quick Start:

```powershell
# Navigate to the project root
cd "C:\Users\saran\ml aat"

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies (if you haven't already)
pip install -r requirements.txt

# Start the Streamlit application
streamlit run app.py
```

Once running, the dashboard will be available at **[http://localhost:8501](http://localhost:8501)**.

---

## Retraining the Model

If you modify the Excel dataset and want to retrain the model:

```powershell
cd "C:\Users\saran\ml aat"
.\venv\Scripts\Activate.ps1
python ml_service\train_model.py
```
