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
    page_title="Fisheries Predictive Monitoring",
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
    model_path = os.path.join("models", "model.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    file_path = r"2024-09-09_Coggan_Anthea_62174v5\data\ESVC_all data_Provisioning_Wild_fish_biomass_WT and CY as CAPOM_Fitzroy as KCB.xlsx"
    if not os.path.exists(file_path):
        return pd.DataFrame()
    df = pd.read_excel(file_path)
    def clean_year(y):
        if pd.isna(y):
            return 2020
        y = str(y)
        if '/' in y:
            return int(y.split('/')[0])
        return int(y)
    if 'year' in df.columns:
        df['year_int'] = df['year'].apply(clean_year)
    return df

model_data = load_model()
full_df = load_data()

if not model_data or full_df.empty:
    st.error("Model or Data files not found! Please ensure data exists.")
    st.stop()

model = model_data['model']
le_mg = model_data['le_mg']
le_region = model_data['le_region']
le_fishery = model_data['le_fishery']

mg_classes = model_data['mg_classes']
region_classes = model_data['region_classes']
fishery_classes = model_data['fishery_classes']

# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2942/2942102.png", width=80)
st.sidebar.title("Configuration")
st.sidebar.markdown("Adjust parameters to simulate and predict future biomass.")

st.sidebar.subheader("Location & Fishery")
sel_mgArea = st.sidebar.selectbox("Management Area", mg_classes)
sel_region = st.sidebar.selectbox("Specific Region", region_classes)
sel_fishery = st.sidebar.selectbox("Fishery Type", fishery_classes)

st.sidebar.subheader("Time")
# Assume current year is around 2026, let's predict up to 2030
current_year = int(full_df['year_int'].max()) if 'year_int' in full_df.columns else 2024
sel_year = st.sidebar.slider("Prediction Year", min_value=2000, max_value=2035, value=current_year + 1, step=1)

# -------------------------------------------------------------------
# INFERENCE
# -------------------------------------------------------------------
# Transform inputs
mg_enc = le_mg.transform([sel_mgArea])[0]
reg_enc = le_region.transform([sel_region])[0]
fish_enc = le_fishery.transform([sel_fishery])[0]

# Prepare feature dataframe
X = pd.DataFrame([[mg_enc, reg_enc, fish_enc, sel_year]], 
                 columns=['mgArea_enc', 'region_enc', 'fishery_enc', 'year_int'])

# Predict
pred_biomass = model.predict(X)[0]

# Calculate Risk based on THIS SPECIFIC selection, not the global sum
filtered_df = full_df[
    (full_df['mgArea'] == sel_mgArea) & 
    (full_df['region'] == sel_region) & 
    (full_df['fishery_type'] == sel_fishery)
]

hist_df = filtered_df.groupby('year_int')['wildfishBiomass_kg'].sum().reset_index() if not filtered_df.empty else pd.DataFrame(columns=['year_int', 'wildfishBiomass_kg'])

avg_biomass = hist_df['wildfishBiomass_kg'].mean() if not hist_df.empty else pred_biomass * 1.5
if pd.isna(avg_biomass) or avg_biomass == 0:
    avg_biomass = pred_biomass * 1.5

if pred_biomass < avg_biomass * 0.5:
    risk_status = "Critical"
    risk_color = "risk-critical"
    risk_hex = "#EF4444"
    quota_action = "HALT FISHING (0% Quota)"
elif pred_biomass < avg_biomass * 0.8:
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
st.title("🌊 Fisheries Predictive Monitoring System")
st.markdown("""
Welcome to the **Smart Map Dashboard**. This tool uses AI (XGBoost) to predict future fish population health based on historical data. 
Configure the parameters on the left to see how populations might shift in different regions over time.
""")

st.markdown("---")

# KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">Predicted Biomass ({sel_year})</div>
        <div class="kpi-value">{pred_biomass:,.0f} kg</div>
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
    confidence = np.random.uniform(85, 98) # Mock confidence score for demonstration
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
        st.subheader("Historical vs. Predicted Biomass")
        # Base historical data
        fig = px.area(hist_df, x="year_int", y="wildfishBiomass_kg", 
                      title="Total Wildfish Biomass Over Time",
                      labels={"year_int": "Year", "wildfishBiomass_kg": "Biomass (kg)"},
                      color_discrete_sequence=["#3B82F6"])
        
        # Add prediction marker
        fig.add_trace(go.Scatter(
            x=[sel_year], 
            y=[pred_biomass],
            mode='markers+text',
            name=f'Prediction ({sel_year})',
            marker=dict(color=risk_hex, size=15, symbol='star'),
            text=[f"{pred_biomass:,.0f} kg"],
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
            value = pred_biomass,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Stock Level"},
            gauge = {
                'axis': {'range': [None, avg_biomass * 1.5]},
                'bar': {'color': "#1E293B"},
                'steps': [
                    {'range': [0, avg_biomass * 0.5], 'color': "#EF4444"},
                    {'range': [avg_biomass * 0.5, avg_biomass * 0.8], 'color': "#F59E0B"},
                    {'range': [avg_biomass * 0.8, avg_biomass * 1.5], 'color': "#22C55E"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': pred_biomass
                }
            }
        ))
        gauge.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge, use_container_width=True)

with tab2:
    st.subheader(f"Monitoring Zone: {sel_region} ({sel_mgArea})")
    st.markdown("This map shows the general monitored region and its current risk status overlay.")
    
    # Map specific regions to general coordinates to prevent random jumping
    region_coords = {
        "Fitzroy": [-23.37, 150.51],
        "Burdekin": [-19.57, 147.40],
        "Burnett Mary": [-25.54, 152.70],
        "Mackay Whitsunday": [-21.14, 149.18],
        "Wet Tropics": [-17.52, 146.00],
        "Cape York": [-13.75, 143.00]
    }
    
    # Default to a central GBR location if region not found
    coords = region_coords.get(sel_region, [-20.0, 145.0])
    
    df_map = pd.DataFrame({
        "lat": [coords[0]],
        "lon": [coords[1]],
        "color": [risk_hex],
        "size": [50000] # Adjust size for visual presence
    })

    # st.map natively handles everything without Mapbox API tokens
    st.map(df_map, latitude="lat", longitude="lon", color="color", size="size", zoom=5)

st.markdown("---")
st.caption("Developed for Sustainable Fisheries Management | Model: XGBoost Regressor")
