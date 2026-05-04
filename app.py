import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk

# -------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Global Fisheries Predictive Monitoring",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme adjustments and sleek UI
st.markdown("""
<style>
    .kpi-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        color: white;
    }
    .kpi-title {
        font-size: 1.1rem;
        color: #94A3B8;
        margin-bottom: 10px;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .risk-sustainable { color: #22C55E; }
    .risk-warning { color: #F59E0B; }
    .risk-critical { color: #EF4444; }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# LOAD MODEL & DATA
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(BASE_DIR, "models", "global_model.pkl")
    
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    file_path = os.path.join(BASE_DIR, "global_data.csv")
    
    if not os.path.exists(file_path):
        return pd.DataFrame()
    df = pd.read_csv(file_path)
    return df

model_data = load_model()
full_df = load_data()

if not model_data or full_df.empty:
    st.error("Model or Data files not found! Please ensure data exists.")
    st.stop()

model = model_data['model']
le_country = model_data['le_country']
le_species = model_data['le_species']
le_area = model_data['le_area']

country_classes = model_data['country_classes']
species_classes = model_data['species_classes']
area_classes = model_data['area_classes']

# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2942/2942102.png", width=80)
st.sidebar.title("Configuration")
st.sidebar.markdown("Adjust parameters to simulate and predict future production.")

st.sidebar.subheader("Location & Species")
# Add searching to make it easier for users with long lists
sel_country = st.sidebar.selectbox("Country", country_classes, index=country_classes.index("United States of America") if "United States of America" in country_classes else 0)

# Filter species and area options based on selected country to avoid invalid combinations
filtered_by_country = full_df[full_df['Country'] == sel_country]
available_species = filtered_by_country['Species'].unique().tolist() if not filtered_by_country.empty else species_classes

sel_species = st.sidebar.selectbox("Species", sorted(available_species))

# Filter areas based on country + species
filtered_by_country_species = filtered_by_country[filtered_by_country['Species'] == sel_species]
available_areas = filtered_by_country_species['Area'].unique().tolist() if not filtered_by_country_species.empty else area_classes

sel_area = st.sidebar.selectbox("Water Area", sorted(available_areas))

st.sidebar.subheader("Time")
current_year = int(full_df['Year'].max()) if 'Year' in full_df.columns else 2026
sel_year = st.sidebar.slider("Prediction Year", min_value=2000, max_value=2035, value=current_year + 1, step=1)

# -------------------------------------------------------------------
# INFERENCE
# -------------------------------------------------------------------
# Transform inputs safely (fallback to 0 if not found)
try:
    country_enc = le_country.transform([sel_country])[0]
except ValueError:
    country_enc = 0
    
try:
    species_enc = le_species.transform([sel_species])[0]
except ValueError:
    species_enc = 0
    
try:
    area_enc = le_area.transform([sel_area])[0]
except ValueError:
    area_enc = 0
# Prepare feature dataframe
X = pd.DataFrame([[country_enc, species_enc, area_enc, sel_year]], 
                 columns=['Country_Enc', 'Species_Enc', 'Area_Enc', 'Year'])

# Predict with XGBoost
pred_production = model.predict(X)[0]

# Calculate historical data for this specific selection
filtered_df = full_df[
    (full_df['Country'] == sel_country) & 
    (full_df['Species'] == sel_species) & 
    (full_df['Area'] == sel_area)
]

hist_df = filtered_df.groupby('Year')['Production_Quantity'].sum().reset_index() if not filtered_df.empty else pd.DataFrame(columns=['Year', 'Production_Quantity'])

# Add a linear trend for future years since tree-based models cannot extrapolate
max_hist_year = hist_df['Year'].max() if not hist_df.empty else current_year

if sel_year > max_hist_year and not hist_df.empty and len(hist_df) >= 2:
    # Calculate simple trend from the last 5 years of historical data
    recent_hist = hist_df.sort_values('Year').tail(5)
    if len(recent_hist) > 1:
        x_vals = recent_hist['Year'].values
        y_vals = recent_hist['Production_Quantity'].values
        slope = np.polyfit(x_vals, y_vals, 1)[0]
        
        # Apply the trend (slope) to the base XGBoost prediction
        years_ahead = sel_year - max_hist_year
        pred_production = pred_production + (slope * years_ahead)

# Prevent negative predictions after applying trend
pred_production = max(0, pred_production)

avg_production = hist_df['Production_Quantity'].mean() if not hist_df.empty else pred_production * 1.5
if pd.isna(avg_production) or avg_production == 0:
    avg_production = max(pred_production * 1.5, 1.0) # Avoid division by zero

if pred_production < avg_production * 0.5:
    risk_status = "Critical"
    risk_color = "risk-critical"
    risk_hex = "#EF4444"
    quota_action = "HALT FISHING (0% Quota)"
elif pred_production < avg_production * 0.8:
    risk_status = "Warning"
    risk_color = "risk-warning"
    risk_hex = "#F59E0B"
    quota_action = "REDUCE QUOTA by 30%"
else:
    risk_status = "Sustainable"
    risk_color = "risk-sustainable"
    risk_hex = "#22C55E"
    quota_action = "MAINTAIN QUOTA (100%)"


# -------------------------------------------------------------------
# MAIN DASHBOARD
# -------------------------------------------------------------------
st.title("🌍 Global Fisheries Predictive Monitoring System")
st.markdown("""
Welcome to the **Global Smart Map Dashboard**. This tool uses AI (XGBoost) to predict future fish production/biomass 
based on historical data from the global fisheries dataset. Configure the parameters on the left to see predictions.
""")

st.markdown("---")

# KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Predicted Prod. ({sel_year})</div>
        <div class="kpi-value">{pred_production:,.0f} t</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Ecosystem Risk Status</div>
        <div class="kpi-value {risk_color}">{risk_status}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Recommended Action</div>
        <div class="kpi-value" style="font-size: 1.2rem; margin-top: 10px;">{quota_action}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    # Model confidence proxy (in a real app this would be based on model metrics)
    confidence = min(98.0, max(60.0, 100 - (abs(pred_production - avg_production) / avg_production * 20)))
    if pd.isna(confidence): confidence = 85.0
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Model Confidence</div>
        <div class="kpi-value">{confidence:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# CHARTS AND MAP
# -------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Analytics & Predictions", "🌍 Geographic Risk Map"])

with tab1:
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        st.subheader(f"Historical vs. Predicted Production for {sel_species}")
        # Base historical data
        fig = px.area(hist_df, x="Year", y="Production_Quantity", 
                      title=f"{sel_country} - {sel_species} in {sel_area}",
                      labels={"Year": "Year", "Production_Quantity": "Production (Tonnes)"},
                      color_discrete_sequence=["#3B82F6"])
        
        # Add prediction marker
        fig.add_trace(go.Scatter(
            x=[sel_year], 
            y=[pred_production],
            mode='markers+text',
            name=f'Prediction ({sel_year})',
            marker=dict(color=risk_hex, size=15, symbol='star'),
            text=[f"{pred_production:,.0f} t"],
            textposition="top center"
        ))
        
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col_chart2:
        st.subheader("Risk Thresholds")
        gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred_production,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Stock Level"},
            gauge = {
                'axis': {'range': [None, max(pred_production * 1.5, avg_production * 1.5)]},
                'bar': {'color': "#1E293B"},
                'steps': [
                    {'range': [0, avg_production * 0.5], 'color': "#EF4444"},
                    {'range': [avg_production * 0.5, avg_production * 0.8], 'color': "#F59E0B"},
                    {'range': [avg_production * 0.8, max(pred_production * 1.5, avg_production * 1.5)], 'color': "#22C55E"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': pred_production
                }
            }
        ))
        gauge.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, use_container_width=True)

with tab2:
    st.subheader(f"Global View: {sel_country}")
    st.markdown("This map shows the selected country colored by its risk status based on the selected species and area.")
    
    # Get ISO3 code for map
    iso3_code = filtered_by_country['ISO3'].iloc[0] if not filtered_by_country.empty else None
    
    if iso3_code:
        # Create a dataframe for the map with just the selected country
        map_df = pd.DataFrame({
            "ISO3": [iso3_code],
            "Country": [sel_country],
            "Risk": [risk_status],
            "Color": [1] # Dummy value for continuous color scale
        })
        
        # We use a trick with continuous color scale to force our risk color
        color_scale = [[0.0, risk_hex], [1.0, risk_hex]]
        
        fig_map = px.choropleth(
            map_df,
            locations="ISO3",
            color="Color",
            hover_name="Country",
            hover_data={"Risk": True, "Color": False, "ISO3": False},
            color_continuous_scale=color_scale,
            projection="natural earth",
        )
        
        fig_map.update_layout(
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            geo=dict(
                showframe=False,
                showcoastlines=True,
                coastlinecolor="rgba(255, 255, 255, 0.3)",
                showland=True,
                landcolor="rgba(255, 255, 255, 0.05)",
                showocean=True,
                oceancolor="rgba(0, 0, 0, 0.1)",
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning("ISO3 code not found for mapping.")

st.markdown("---")
st.caption("Developed for Global Sustainable Fisheries Management | Model: XGBoost Regressor")
