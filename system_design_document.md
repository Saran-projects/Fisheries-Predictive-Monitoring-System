# End-to-End Predictive Monitoring System for Sustainable Fisheries Management

## 1. Problem Formulation

**Machine Learning Task Type:** Regression (Continuous prediction of biomass).
**Prediction Objective:** To predict the future fish biomass (stock level) in designated geographic fishing zones to prevent overfishing and maintain ecological balance.

**Input Feature Categories:**
- **Sonar Density:** Acoustic backscatter values estimating fish school volume and density.
- **Salinity:** Water salinity measurements which influence fish habitat preference and migration.
- **Catch Logs:** Historical and current commercial fishing extraction volumes.
- **Time Variables:** Seasonality, time of day, lunar cycles affecting fish behavior.
- **Location Variables:** GPS coordinates, depth, and specific bathymetric zones.

**Output Variable:** Future Fish Biomass Estimate (measured in metric tons per square kilometer).

**Prediction Horizon:** Weekly forecast. A weekly horizon balances the operational responsiveness needed by fishery authorities with the statistical smoothing required for noisy environmental data.

**Mathematical Objective Function:**
The primary objective is to minimize the Mean Squared Error (MSE) between the predicted biomass $\hat{y}_i$ and the actual biomass $y_i$, with an asymmetric penalty weighting to heavily penalize overestimation.

$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

**Operational Trade-offs:**
- **Overestimation → Overfishing Risk:** If the model predicts more fish than exist, quotas will be set too high, leading to catastrophic depletion of the stock.
- **Underestimation → Economic Restriction Risk:** If the model predicts fewer fish, quotas will be overly strict, unnecessarily harming the livelihoods of local fishing communities.

**Minimizing Overestimation Error:**
Minimizing overestimation error is critical because biological recovery from overfishing takes decades and can lead to permanent ecosystem collapse. Economic restrictions, while painful, are reversible in subsequent seasons once stock levels are re-verified.

---

## 2. Dataset Strategy

**Dataset Schema Integration:**
- **Sonar Sensor Data:** `timestamp`, `zone_id`, `depth_m`, `acoustic_density_db`.
- **Salinity Sensor Readings:** `timestamp`, `sensor_id`, `zone_id`, `salinity_ppt`, `temp_c`.
- **Commercial Catch Records:** `date`, `vessel_id`, `zone_id`, `species`, `catch_weight_kg`.
- **Temporal Environmental Variables:** `date`, `season`, `lunar_phase`.

**Handling Data Anomalies:**
- **Missing Sonar Signals:** Use spatial interpolation (Kriging) from nearby zones or temporal forward-filling for short gaps (< 24 hours).
- **Sensor Noise:** Apply moving average filters and isolation forests to remove anomalous acoustic spikes caused by passing ships or biological noise (e.g., plankton blooms).
- **Inconsistent Timestamps:** Discretize all continuous time-series data into standardized 6-hour and 24-hour tumbling windows to force temporal alignment.
- **Multi-Source Data Integration:** Establish `(zone_id, date)` as the primary composite key for relational joins across all data sources.

**Preprocessing Pipeline:**
1. **Cleaning:** Remove duplicate records and drop out-of-bounds sensor readings (e.g., negative salinity).
2. **Normalization:** Min-Max scaling for catch logs; Z-score standardization for sensor readings (salinity, sonar).
3. **Aggregation:** Roll up hourly sensor data into daily median values to match the granularity of daily catch logs.
4. **Alignment:** Left-join sensor data onto the spatial grid master table using the composite key.
5. **Feature Merging:** Construct the final tabular dataset ensuring no forward-looking data leakage exists.

---

## 3. Feature Engineering Strategy

**Engineered Features:**
- **Lag Features (Previous stock estimates):** `biomass_t_minus_1`, `biomass_t_minus_7`. Improves robustness by grounding current predictions in recent reality, acting as an autoregressive baseline.
- **Rolling Averages:** `7_day_avg_salinity`, `30_day_avg_catch`. Smooths out daily volatility and captures broader environmental trends.
- **Seasonal Indicators:** Sine/Cosine transformations of the day of the year. Captures cyclical biological behaviors like spawning without artificial hard-boundaries at year-end.
- **Migration-Cycle Indicators:** Boolean flags for known regional migration windows, providing the model with hard-coded domain expertise.
- **Catch Pressure Ratio:** `(catch_last_7_days) / (estimated_biomass_last_7_days)`. A critical stress indicator; high pressure precedes stock crashes.
- **Salinity-Temperature Interaction Terms:** `salinity * temperature`. Water density affects acoustic sonar propagation and fish habitat selection; the interaction term captures this nonlinear relationship.

---

## 4. Model Design Strategy

**Model Suitability Comparison:**
- **Random Forest:** Highly robust to outliers, easily handles non-linear interactions, but struggles to extrapolate trends outside the training data.
- **Gradient Boosting (XGBoost/LightGBM):** Exceptional performance on tabular data, handles missing values natively, but prone to overfitting on small, noisy datasets.
- **LSTM Time-Series Models:** Excellent for sequential patterns, but requires massive datasets and is highly opaque (low interpretability).

**Recommendation:**
**Gradient Boosting (XGBoost)** is recommended. It dominates structured environmental tabular datasets, natively handles the missing values common in sensor networks, and provides feature importance scores crucial for regulatory transparency.

**Pipeline & Methodology:**
- **Training Pipeline:** Raw Data $\rightarrow$ Imputer $\rightarrow$ Scaler $\rightarrow$ Polynomial Features $\rightarrow$ XGBoost Regressor.
- **Cross-Validation Approach:** Time-Series Split (Walk-Forward Validation). Standard K-Fold is strictly prohibited as it causes future data leakage into past training sets.
- **Hyperparameter Tuning:** Bayesian Optimization using Tree-structured Parzen Estimator (TPE) to efficiently find optimal `learning_rate`, `max_depth`, and `subsample` without exhaustive grid search.
- **Handling Noisy Inputs:** L1/L2 regularization terms in the XGBoost objective function to penalize overly complex trees that fit to sensor noise.
- **Small Dataset Scenarios:** Utilize transfer learning from data-rich adjacent geographic zones or employ SMOTE for synthetic minority over-sampling of rare "low-stock" events.

**Trade-offs:**
- **Accuracy vs. Interpretability:** XGBoost offers high accuracy but requires SHAP (SHapley Additive exPlanations) values to explain decisions to fishery stakeholders, increasing computational overhead.
- **Deployment Complexity:** Tree-based models are lightweight to deploy compared to LSTMs, requiring only standard CPU inference environments.

---

## 5. Evaluation Strategy

**Evaluation Metrics:**
- **MAE (Mean Absolute Error):** Provides a linear penalty, highly interpretable in terms of physical metric tons of fish.
- **RMSE (Root Mean Squared Error):** Heavily penalizes large prediction errors, aligning with the need to avoid catastrophic overestimations.
- **R² Score:** Measures the proportion of variance explained; useful for communicating baseline effectiveness to non-technical policymakers.

**Recall in Classification Context:**
If converting the output to a classification task (e.g., "Normal" vs. "Critically Low Stock"), **Recall** becomes the paramount metric. Missing a "Critically Low" alert (False Negative) results in overfishing a depleted stock, whereas a False Positive merely pauses fishing unnecessarily.

**Time-Series Validation Strategy:**
Use a Walk-Forward Validation approach. Train on Year 1, validate on Year 2 (Q1). Train on Year 1 + Year 2 (Q1), validate on Year 2 (Q2). This strictly prevents the model from peeking into the future.

**Acceptable Performance Thresholds:**
- MAPE (Mean Absolute Percentage Error) < 15%.
- Zero instances of overestimation exceeding 25% of actual biomass in any given validation window.

---

## 6. Deployment Architecture

**ML Deployment Pipeline:**
`Sensors` $\rightarrow$ `MQTT Broker (IoT Data Ingestion)` $\rightarrow$ `Apache Kafka` $\rightarrow$ `Python Preprocessing Worker` $\rightarrow$ `Feature Store (Redis)` $\rightarrow$ `FastAPI Model Inference Server` $\rightarrow$ `Node.js API` $\rightarrow$ `React/Dashboard`

**Architecture Considerations:**
- **Latency Constraints:** Real-time latency is not critical for weekly forecasts; hourly batch processing is sufficient, reducing infrastructure costs.
- **Scalability Across Fishing Zones:** Containerized inference endpoints (Docker/Kubernetes) allow horizontal scaling as more geographic zones and sensors are added to the network.
- **Fault Tolerance:** If a sensor node goes offline, the system automatically falls back to the last known spatial average or uses the `timestamp_lag` features to impute missing arrays without failing the entire inference pipeline.

---

## 7. Node.js Dashboard Frontend Architecture

**Frontend System:**
- **Runtime:** Node.js for server-side rendering and API aggregation.
- **Framework:** Express.js for routing and middleware.
- **Visualization:** Chart.js for time-series analytics and Leaflet.js for geographic mapping.

**Routing Structure:**
- `/dashboard`: Main aggregated overview.
- `/predictions`: Detailed predictive modeling outputs and confidence intervals.
- `/zones`: Granular data filtered by specific geographic fishing regions.
- `/alerts`: Log of automated system warnings regarding stock depletion.
- `/analytics`: Historical data exploration and reporting.

**Middleware:**
- **Authentication:** JWT (JSON Web Tokens) for securing endpoints.
- **Logging:** `morgan` or `winston` for comprehensive request and error logging.
- **Role-Based Access (RBAC):** Middleware to ensure only "Regulator" roles can view exact quota recommendations, while "Fishery" roles only see public zonal health statuses.

---

## 8. Dashboard UI Components

**Homepage Dashboard:**
- **Current Fish Stock Estimate:** Prominent KPI card displaying total estimated biomass.
- **Prediction Confidence Score:** Percentage metric indicating model certainty (derived from ensemble variance).
- **Stock Health Category:** Traffic light indicator (Green: Sustainable, Yellow: Warning, Red: Critical).
- **Recommended Fishing Quota:** Actionable metric for regulators.
- **Active Sustainability Alerts:** Scrolling ticker or notification bell for urgent anomalies.

**Analytical Charts (Chart.js):**
- **Historical Stock Trends:** Multi-line chart overlaying historical biomass.
- **Salinity Variation:** Dual-axis chart correlating salinity dips with biomass changes.
- **Sonar Density Trends:** Heat-mapped bar charts of acoustic returns over time.
- **Catch Volume Trends:** Stacked area chart showing extraction rates.
- **Predicted vs Actual Stock Comparison:** Line chart with shaded confidence interval bands evaluating model drift.

---

## 9. Geographic Monitoring Interface

**Leaflet.js Map Visualization:**
- **Fishing Zones:** Polygon overlays delineating regulatory boundaries.
- **Stock-Density Heatmaps:** Gradient overlays (blue to deep red) representing predicted biomass density per square kilometer.
- **Sensor Locations:** Interactive markers showing active/inactive sonar and salinity buoys.
- **High-Risk Overfishing Regions:** Pulsing red polygon borders highlighting zones approaching critical depletion thresholds.

**GeoJSON Integration:**
The Node.js backend serves spatial data via endpoints (e.g., `GET /api/geojson/zones`). The frontend fetches this JSON and directly injects it into Leaflet's `L.geoJSON()` layer, binding popup data dynamically to the polygons based on the latest ML inference results.

---

## 10. Backend API Integration

**REST Endpoints:**
- `GET /api/predict_stock?zone=A`: Returns current and forecasted biomass with confidence bounds.
- `GET /api/zone_risk`: Returns a list of all zones sorted by depletion risk factor.
- `GET /api/salinity_trend?days=30`: Returns aggregated time-series salinity data.
- `GET /api/catch_logs?vessel=all`: Returns paginated extraction data.
- `GET /api/alerts`: Returns active system notifications.

**Response Format & Rendering:**
Standardized JSON responses containing `status`, `data`, and `metadata` (e.g., last update timestamp). The Express.js backend aggregates data from the Python ML service and a PostgreSQL database, formats it, and passes it to the frontend where client-side JavaScript updates the DOM asynchronously.

---

## 11. Monitoring and Maintenance Strategy

**Detection Mechanisms:**
- **Data Drift:** Monitor statistical distributions of incoming salinity and sonar data using Kolmogorov-Smirnov tests to detect physical sensor degradation.
- **Concept Drift:** Track when the relationship between features and biomass changes (e.g., due to climate-change-induced migration pattern shifts).
- **Sensor Reliability Failures:** Alert when null values from a specific `sensor_id` exceed 10% in a 24-hour window.
- **Prediction Accuracy Degradation:** Track moving average of RMSE over the last 4 weeks.

**Retraining Triggers:**
- **Season Change:** Scheduled retraining at the onset of major oceanic seasons (e.g., El Niño transitions).
- **Performance Drop Threshold:** Automated retraining triggered if weekly MAPE exceeds 15%.
- **Distribution Shift Detection:** Triggered when the feature store detects a >2 standard deviation shift in core variables like salinity over a 30-day period.

**Fallback Strategy:**
If the ML service crashes or data drift is unmanageable, the system automatically falls back to a simplistic moving-average heuristic model, flagging all outputs with a "Degraded System State" warning on the dashboard to prevent false confidence.

---

## 12. Ethical and Risk Considerations

**Systemic Risks:**
- **Ecosystem Imbalance:** Incorrect quotas could collapse a target species, leading to trophic cascades (starvation of predators, overpopulation of prey).
- **Biased Sonar Coverage:** Sensor placement heavily weighted toward wealthy coastal regions may leave remote zones unmonitored, skewing total biomass estimates.
- **Underreported Catch Logs:** Illegal, Unreported, and Unregulated (IUU) fishing creates invisible data gaps, making the model overly optimistic.
- **Policy Misuse:** Regulators might use the "predicted" data to enforce punitive measures without field verification.

**Proposed Mitigations:**
- **Confidence Intervals:** Never present a single point estimate; always display the 95% confidence bound to highlight uncertainty.
- **Transparent Reporting:** Make anonymized aggregate data publicly accessible to NGOs and researchers for independent validation.
- **Human-in-the-Loop Validation:** ML outputs must be classified strictly as *Decision Support Systems (DSS)*. Final quota adjustments require physical marine biologist sign-off.
- **Multi-Source Verification:** Cross-reference sonar predictions with satellite AIS (Automatic Identification System) vessel density data.

---

## 13. SDG Alignment (Mandatory)

**Alignment with Sustainable Development Goals:**

- **SDG 14: Life Below Water:** This system directly addresses Target 14.4 (regulate harvesting and end overfishing, IUU fishing, and destructive fishing practices). By providing accurate, predictive insights into fish biomass, it enables science-based management plans that restore fish stocks in the shortest time feasible.
- **SDG 12: Responsible Consumption and Production:** By optimizing the extraction of marine resources, the framework ensures that commercial fishing yields meet current human demands without compromising the ability of the marine ecosystem to support future generations.

---

## 14. System Architecture Diagram Description

**Architecture Flow for Academic Diagram:**

1. **Sensors (Edge Layer):**
   - Sonar Buoys (Acoustic Data)
   - Salinity/Temperature Probes
   - Satellite AIS (Vessel Tracking)
   *$\downarrow$ via IoT MQTT / Sat-link*
2. **Data Ingestion Service (Backend):**
   - Apache Kafka (Message Queue)
   - Data Validator (Drops malformed packets)
   *$\downarrow$ via internal network*
3. **Feature Engineering Pipeline (Data Layer):**
   - Apache Spark / Python Pandas Workers (Temporal alignment, Aggregation)
   - PostgreSQL (Relational storage for catch logs & zones)
   - Redis Feature Store (Low-latency engineered features)
   *$\downarrow$ via RPC / REST*
4. **ML Prediction Service (Inference Layer):**
   - FastAPI Server hosting XGBoost model
   - Drift Monitoring Agent
   *$\downarrow$ via HTTP API*
5. **Node.js API Server (Application Layer):**
   - Express.js Application (Authentication, Rate Limiting, API Aggregation)
   *$\downarrow$ via JSON / WebSocket*
6. **Dashboard Interface (Presentation Layer):**
   - React/HTML5 DOM (UI Components, Alerts)
   - Chart.js (Statistical Visualizations)
   - Leaflet.js (Geospatial Mapping with GeoJSON)
