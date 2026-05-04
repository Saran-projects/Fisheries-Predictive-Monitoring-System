import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.preprocessing import LabelEncoder
import os

def main():
    print("Loading global_data.csv...")
    df = pd.read_csv("global_data.csv")
    
    # Drop rows with NaN in critical features
    df = df.dropna(subset=['Country', 'Species', 'Area', 'Production_Quantity'])
    
    # For Streamlit app, we will use LabelEncoder for categorical features
    le_country = LabelEncoder()
    le_species = LabelEncoder()
    le_area = LabelEncoder()
    
    print("Encoding features...")
    df['Country_Enc'] = le_country.fit_transform(df['Country'])
    df['Species_Enc'] = le_species.fit_transform(df['Species'])
    df['Area_Enc'] = le_area.fit_transform(df['Area'])
    
    # We will use Country_Enc, Species_Enc, Area_Enc, and Year to predict Production_Quantity
    X = df[['Country_Enc', 'Species_Enc', 'Area_Enc', 'Year']]
    y = df['Production_Quantity']
    
    print("Training XGBoost Regressor...")
    model = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Save the model and encoders
    os.makedirs('models', exist_ok=True)
    
    model_data = {
        'model': model,
        'le_country': le_country,
        'le_species': le_species,
        'le_area': le_area,
        'country_classes': list(le_country.classes_),
        'species_classes': list(le_species.classes_),
        'area_classes': list(le_area.classes_)
    }
    
    print("Saving model to models/global_model.pkl...")
    with open('models/global_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
        
    print("Model training complete!")

if __name__ == "__main__":
    main()
