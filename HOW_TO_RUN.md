# How to Run the Fisheries Predictive Monitoring System

This project has been consolidated into a single, unified **Streamlit** application. All data processing, machine learning predictions, and the interactive dashboard are now handled by the `app.py` script.

## Prerequisites

- **Python 3.8+** installed on your system.

## Step-by-Step Instructions

### 1. Open your terminal
Open PowerShell or Command Prompt and navigate to the project directory:

```powershell
cd "C:\Users\saran\ml aat"
```

### 2. Activate the Virtual Environment
Before running the application or installing dependencies, you need to activate the existing virtual environment.

**For PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**For Command Prompt:**
```cmd
.\venv\Scripts\activate.bat
```

*(You should see `(venv)` appear at the beginning of your terminal prompt once activated.)*

### 3. Install Dependencies
Ensure you have all the required Python packages installed by running:

```powershell
pip install -r requirements.txt
```

### 4. Run the Application
Start the Streamlit server using the following command:

```powershell
streamlit run app.py
```

### 5. Access the Dashboard
After running the command, Streamlit should automatically open your default web browser and load the application. 

If it doesn't open automatically, you can manually open your browser and go to the local URL provided in the terminal, which is usually:
**[http://localhost:8501](http://localhost:8501)**

---

## Retraining the Model (Optional)

If you have updated the underlying Excel dataset and want to retrain the XGBoost model:

1. Make sure your virtual environment is activated.
2. Run the training script:
```powershell
python ml_service\train_model.py
```
*(Note: Ensure the `models` folder exists and the paths inside the training script are correct.)*
