# How to Run the Global Fisheries Predictive Monitoring System

This project is a single, unified **Streamlit** application that predicts global fisheries production using an XGBoost model.

## Prerequisites

- **Python 3.8+** installed on your system.

## Step-by-Step Instructions

### 1. Open your terminal
Open PowerShell or Command Prompt and navigate to the project directory:

```powershell
cd "C:\Users\saran\ml aat"
```

### 2. Install Dependencies
Ensure you have all the required Python packages installed by running:

```powershell
py -m pip install -r requirements.txt
```

### 3. Run the Application
Start the Streamlit server using the Python module launcher (this avoids Windows PATH issues with the `streamlit` command):

```powershell
py -m streamlit run app.py
```

### 4. Access the Dashboard
After running the command, Streamlit should automatically open your default web browser and load the application. 

If it doesn't open automatically, you can manually open your browser and go to the local URL provided in the terminal, which is usually:
**[http://localhost:8501](http://localhost:8501)**

---

## Retraining the Model and Re-Processing Data (Optional)

If you have downloaded an updated global dataset (`Global_production_quantity.csv`), you will need to re-run the data processing and model training scripts:

1. **Preprocess the Data**:
   This parses the CSVs into a cleaned `global_data.csv`.
   ```powershell
   py preprocess_global_data.py
   ```

2. **Train the Model**:
   This trains the XGBoost model on the newly cleaned data and saves it to `models/global_model.pkl`.
   ```powershell
   py train_global_model.py
   ```

3. Restart your Streamlit app using Step 3 above.
