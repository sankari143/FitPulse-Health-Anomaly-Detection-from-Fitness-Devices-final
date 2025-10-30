# fitpulse_full_pipeline_with_m3.py
# ================================================
# FitPulse - Full Unified Pipeline (M1 + M2) then M3 (Anomaly Detection module)
# Single-file Streamlit app
# ================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import warnings
from datetime import datetime, timedelta
import time
warnings.filterwarnings('ignore')

# Visualization
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

# TSFresh (optional)
try:
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute
    from tsfresh.feature_extraction import MinimalFCParameters
except Exception:
    extract_features = None
    impute = None
    MinimalFCParameters = None

# Forecasting (optional)
try:
    from prophet import Prophet
except Exception:
    Prophet = None

# Clustering / Scalers / Metrics
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch, OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, davies_bouldin_score

# For cluster-based anomaly detection alternative
from sklearn.neighbors import LocalOutlierFactor

# -------------------- Page Config & CSS --------------------
# -------------------- Page Config & CSS --------------------
import streamlit as st
import base64

st.set_page_config(
    page_title="FitPulse - Full Pipeline + M3",
    layout="wide",
    page_icon="💓"
)

# --- Add background image ---
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        /* 🩺 Healthcare Background with Gradient Overlay */
        .stApp {{
            background: linear-gradient(rgba(255,255,255,0.75), rgba(255,255,255,0.85)),
                        url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: #000000 !important;
        }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background: rgba(255,255,255,0.9);
            color: #000;
            border-right: 3px solid #b2ebf2;
        }}

        /* Title Banner */
        .title-banner {{
            text-align: center;
            font-size: 40px;
            color: #004d99;
            background: linear-gradient(90deg, #b2fefa, #0ed2f7);
            border-radius: 12px;
            padding: 15px;
            box-shadow: 3px 3px 12px rgba(0,0,0,0.2);
            margin-bottom: 25px;
            font-weight: bold;
        }}

        /* Buttons */
        .stButton>button, .stDownloadButton>button {{
            border-radius: 12px;
            padding: 0.6em 1.2em;
            font-weight: bold;
            font-size: 16px;
            border: none;
            box-shadow: 0 3px 8px rgba(0,0,0,0.2);
            transition: 0.3s ease;
        }}

        .stButton>button {{
            background-color: #00bfa6;
            color: white;
        }}
        .stButton>button:hover {{
            background-color: #009688;
            transform: scale(1.05);
        }}

        .stDownloadButton>button {{
            background-color: #ff7043;
            color: white;
        }}
        .stDownloadButton>button:hover {{
            background-color: #f4511e;
            transform: scale(1.05);
        }}

        /* Metric Cards */
        .metric-card {{
            background: rgba(255, 255, 255, 0.85);
            border-radius: 20px;
            padding: 20px;
            box-shadow: 4px 4px 15px rgba(0,0,0,0.15);
            margin-bottom: 25px;
        }}

        h1, h2, h3 {{
            color: #004d99;
            font-weight: 700;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }}

        .stMarkdown, .stPlotlyChart, .stDataFrame {{
            background: rgba(255,255,255,0.85);
            border-radius: 15px;
            padding: 15px;
            box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply the background
add_bg_from_local("5.jpg")

# Title
st.markdown("<div class='title-banner'>💓 FitPulse - Smart Health Anomaly Detection Dashboard</div>", unsafe_allow_html=True)


# =====================================================
# Utility / Preprocessing
# =====================================================
def preprocess_df(df):
    # Ensure timestamp is properly formatted
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values(by='timestamp')

    # Handle numeric columns safely
    for col in ['heart_rate','steps','resting_heart','step','sleep']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)

    # Reset index after cleaning
    df = df.reset_index(drop=True)
    return df

# Z-score helper
def zscore(series):
    return (series - series.mean()) / (series.std(ddof=0) + 1e-9)

# Basic z-score anomaly detection (exists in M1/M2)
def detect_anomalies_zscore(df, hr_thresh=3.0, steps_thresh=3.0, rest_thresh=2.5):
    df = df.copy()
    if 'heart_rate' in df.columns:
        df['hr_zscore'] = zscore(df['heart_rate'])
        df['hr_anomaly'] = (df['hr_zscore'].abs() > hr_thresh).astype(int)
    else:
        df['hr_anomaly'] = 0

    if 'steps' in df.columns:
        df['steps_zscore'] = zscore(df['steps'])
        df['steps_anomaly'] = (df['steps_zscore'].abs() > steps_thresh).astype(int)
    else:
        df['steps_anomaly'] = 0

    if 'resting_heart' in df.columns:
        df['rest_zscore'] = zscore(df['resting_heart'])
        df['rest_anomaly'] = (df['rest_zscore'].abs() > rest_thresh).astype(int)
    else:
        df['rest_anomaly'] = 0

    return df

# =====================================================
# Streamlit Sidebar: Inputs & Config
# =====================================================
st.sidebar.markdown('<div class="sidebar-title">💓 FitPulse Pipeline</div>', unsafe_allow_html=True)

# Data selection
file = st.sidebar.file_uploader("Upload CSV/JSON (timestamp, heart_rate, steps, resting_heart)", type=['csv', 'json'])
use_sample = st.sidebar.checkbox("Use built-in sample data")

# Pipeline options (M1+M2)
run_anomaly = st.sidebar.checkbox("Run Anomaly Detection (Z-score)", value=True)
run_tsfresh = st.sidebar.checkbox("Run TSFresh Feature Extraction", value=True)
window_size = st.sidebar.slider("TSFresh Window Size (minutes)", 10, 240, 60)
forecast_periods = st.sidebar.slider("Prophet Forecast Horizon (days)", 1, 30, 7)
clustering_method = st.sidebar.selectbox("Clustering Method", ["KMeans", "DBSCAN", "Agglomerative", "Birch", "OPTICS"]) 
n_clusters = st.sidebar.slider("K (for KMeans / Agglomerative / Birch)", 2, 8, 3)

# Thresholds for anomalies (M1 z-score)
hr_thresh = st.sidebar.number_input("HR z-score threshold", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
steps_thresh = st.sidebar.number_input("Steps z-score threshold", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
rest_thresh = st.sidebar.number_input("Resting HR z-score threshold", min_value=0.5, max_value=10.0, value=2.5, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("Developed: FitPulse Full Pipeline")

# =====================================================
# Load or create sample data (M1+M2)
# =====================================================
if use_sample and file is None:
    rng = pd.date_range("2025-01-01 08:00", periods=8*60, freq='T')  # 8 hours minute-level
    df = pd.DataFrame({
        "timestamp": rng,
        "heart_rate": np.clip((60 + 10*np.sin(np.linspace(0,20,len(rng))) + np.random.randn(len(rng))*3).astype(int), 45, 180),
        "steps": np.random.poisson(2, len(rng)),
        "resting_heart": np.clip((55 + 3*np.sin(np.linspace(0,6,len(rng))) + np.random.randn(len(rng))*1).astype(int), 40, 90)
    })
    st.sidebar.success("Sample data loaded")
elif file is not None:
    try:
        if hasattr(file, 'name') and file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_json(file)
        st.sidebar.success("Uploaded file loaded")
    except Exception as e:
        st.sidebar.error(f"Failed to read file: {e}")
        st.stop()
else:
    st.sidebar.info("Upload a file or choose sample data to proceed")
    st.stop()

# Show raw preview
st.markdown("# 💓 FitPulse — Full Pipeline (M1 + M2 then M3)")
st.markdown("### 📊 Raw Data Preview")
st.dataframe(df.head(200), use_container_width=True)

# =====================================================
# Preprocess (M1)
# =====================================================
df = preprocess_df(df)
st.markdown("### ⚙️ Processed Data Preview")
st.dataframe(df.head(200), use_container_width=True)

# =====================================================
# Anomaly Detection (Z-score) - M1 (unchanged)
# =====================================================
if run_anomaly:
    st.markdown("## 🚨 M1 — Anomaly Detection (Z-score)")
    df = detect_anomalies_zscore(df, hr_thresh=hr_thresh, steps_thresh=steps_thresh, rest_thresh=rest_thresh)

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### ❤️ Heart Rate Anomalies")
        st.metric("Count", int(df.get('hr_anomaly', pd.Series([])).sum()))
        st.metric("Mean HR", round(df['heart_rate'].mean() if 'heart_rate' in df.columns else 0, 2))
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 👟 Steps Anomalies")
        st.metric("Count", int(df.get('steps_anomaly', pd.Series([])).sum()))
        st.metric("Mean Steps", round(df['steps'].mean() if 'steps' in df.columns else 0, 2))
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🛌 Sleep Anomalies")
        st.metric("Count", int(df.get('rest_anomaly', pd.Series([])).sum()))
        st.metric("Mean Resting HR", round(df['resting_heart'].mean() if 'resting_heart' in df.columns else 0, 2))
        st.markdown('</div>', unsafe_allow_html=True)

    # Visualizations
    if 'heart_rate' in df.columns:
        st.markdown("### ❤️ Heart Rate Over Time (Z-score anomalies marked)")
        fig_hr = px.line(df, x='timestamp', y='heart_rate', title='Heart Rate')
        if 'hr_anomaly' in df.columns:
            anomalies_hr = df[df['hr_anomaly'] == 1]
            fig_hr.add_scatter(x=anomalies_hr['timestamp'], y=anomalies_hr['heart_rate'],
                               mode='markers', name='HR Anomaly', marker=dict(size=8, color='black', symbol='x'))
        st.plotly_chart(fig_hr, use_container_width=True)

    if 'steps' in df.columns:
        st.markdown("### 👟 Steps Over Time (Z-score anomalies marked)")
        fig_steps = px.line(df, x='timestamp', y='steps', title='Steps')
        if 'steps_anomaly' in df.columns:
            anomalies_st = df[df['steps_anomaly'] == 1]
            fig_steps.add_scatter(x=anomalies_st['timestamp'], y=anomalies_st['steps'],
                                  mode='markers', name='Steps Anomaly', marker=dict(size=8, color='red', symbol='x'))
        st.plotly_chart(fig_steps, use_container_width=True)

    if 'resting_heart' in df.columns:
        st.markdown("### 🛌 Resting Heart Over Time (Z-score anomalies marked)")
        fig_rest = px.line(df, x='timestamp', y='resting_heart', title='Resting Heart')
        if 'rest_anomaly' in df.columns:
            anomalies_rest = df[df['rest_anomaly'] == 1]
            fig_rest.add_scatter(x=anomalies_rest['timestamp'], y=anomalies_rest['resting_heart'],
                                 mode='markers', name='Rest Anomaly', marker=dict(size=8, color='orange', symbol='x'))
        st.plotly_chart(fig_rest, use_container_width=True)

# =====================================================
# TSFresh feature extraction (windowed) - M2
# =====================================================
features = pd.DataFrame()
if run_tsfresh and extract_features is not None:
    st.markdown("## 🧠 M2 — TSFresh Feature Extraction")

    # Choose metric based on available columns
    metric_candidates = [c for c in ['heart_rate', 'steps', 'resting_heart'] if c in df.columns]
    metric_col = st.selectbox("TSFresh Metric", metric_candidates, index=0)

    step_size = max(1, window_size // 2)
    prepared = []
    window_id = 0
    for i in range(0, len(df) - window_size + 1, step_size):
        win = df.iloc[i:i + window_size].copy()
        win['window_id'] = window_id
        win = win[['window_id', 'timestamp', metric_col]].rename(columns={metric_col: 'value'})
        prepared.append(win)
        window_id += 1

    if not prepared:
        st.error("⚠️ Not enough data for TSFresh windows. Reduce window size or use more data.")
    else:
        df_tsfresh = pd.concat(prepared, ignore_index=True)

        fc_parameters = MinimalFCParameters()
        with st.spinner('Extracting features using TSFresh...'):
            try:
                features = extract_features(
                    df_tsfresh,
                    column_id='window_id',
                    column_sort='timestamp',
                    default_fc_parameters=fc_parameters,
                    n_jobs=1
                )
                features = impute(features)
                features = features.loc[:, features.std() > 0]  # remove constant cols
                st.success(f"✅ Extracted {features.shape[1]} features from {features.shape[0]} windows")
            except Exception as e:
                st.error(f"TSFresh extraction failed: {e}")
                features = pd.DataFrame()

        if not features.empty:
            # Top variable features and charts
            st.markdown("### 🔝 Top 10 Most Variable Features")
            top_feats = features.var().sort_values(ascending=False).head(10).index.tolist()
            st.dataframe(features[top_feats].head())

            st.markdown("### 📊 Feature Variance Overview")
            var_df = features.var().sort_values(ascending=False).reset_index()
            var_df.columns = ['Feature', 'Variance']
            fig_var = px.bar(var_df.head(20), x='Feature', y='Variance', title="Top 20 Features by Variance", text_auto='.2f', color='Variance')
            fig_var.update_layout(xaxis_tickangle=-45, height=450)
            st.plotly_chart(fig_var, use_container_width=True)

            st.markdown("### 🔥 Feature Correlation Heatmap")
            corr = features[top_feats].corr()
            fig_corr = px.imshow(corr, text_auto=True, title="Correlation Between Top Features")
            st.plotly_chart(fig_corr, use_container_width=True)

            st.markdown("### 📈 Feature Trends Across Windows")
            selected_feat = st.selectbox("Select feature to visualize trend:", top_feats)
            trend_df = features[[selected_feat]].reset_index().rename(columns={'index': 'Window ID'})
            fig_trend = px.line(trend_df, x='Window ID', y=selected_feat, title=f"Trend of {selected_feat} Across Windows", markers=True)
            st.plotly_chart(fig_trend, use_container_width=True)
else:
    if run_tsfresh:
        st.warning("TSFresh not installed — skipping feature extraction. Install tsfresh to enable this feature.")

# =====================================================
# Prophet Forecasting (optional) - M2
# =====================================================
prophet_df = pd.DataFrame()
mae = rmse = mape = None
if Prophet is None:
    st.warning("Prophet library not available. Forecasting disabled. Install prophet to enable forecasting.")
else:
    st.markdown("## 📈 Prophet Forecasting (M2)")
    forecastable_cols = [c for c in ['heart_rate','steps','resting_heart'] if c in df.columns and df[c].notnull().all()]
    if forecastable_cols:
        forecast_metric = st.selectbox("Forecast Metric", forecastable_cols, index=0)
        prophet_df = df[['timestamp', forecast_metric]].rename(columns={'timestamp':'ds', forecast_metric:'y'}).dropna()
        if len(prophet_df) < 10:
            st.info('Not enough rows for Prophet modeling (>=10 required)')
        else:
            try:
                model = Prophet(daily_seasonality=True, weekly_seasonality=True)
                with st.spinner('Training Prophet model ...'):
                    model.fit(prophet_df)
                future = model.make_future_dataframe(periods=forecast_periods)
                forecast = model.predict(future)

                merged = prophet_df.merge(forecast[['ds','yhat']], on='ds', how='left')
                merged['residual'] = merged['y'] - merged['yhat']
                threshold = 3 * merged['residual'].std()
                merged['anomaly'] = merged['residual'].abs() > threshold

                figf = go.Figure()
                figf.add_trace(go.Scatter(x=prophet_df['ds'], y=prophet_df['y'], mode='markers', name='Actual'))
                figf.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
                figf.add_trace(go.Scatter(x=merged[merged['anomaly']]['ds'], y=merged[merged['anomaly']]['y'], mode='markers', name='Anomalies', marker=dict(color='red', size=8)))
                figf.update_layout(title=f"{forecast_metric} Forecast ({len(prophet_df)} history + {forecast_periods} days future)", xaxis_title='Date', yaxis_title=forecast_metric)
                st.plotly_chart(figf, use_container_width=True)

                mae = mean_absolute_error(prophet_df['y'], merged['yhat'].fillna(method='ffill'))
                rmse = np.sqrt(mean_squared_error(prophet_df['y'], merged['yhat'].fillna(method='ffill')))
                mape = np.mean(np.abs((prophet_df['y'] - merged['yhat'].fillna(method='ffill')) / (prophet_df['y'].replace(0, np.nan)))) * 100
                st.markdown(f"**Forecast Metrics:** MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%")
            except Exception as e:
                st.error(f"Prophet modeling failed: {e}")
    else:
        st.info("No forecastable columns available for Prophet.")

# =====================================================
# Clustering on TSFresh features + PCA visualization - M2
# =====================================================
labels = None
if not features.empty:
    st.markdown("## 🧩 Behavioral Clustering (on extracted TSFresh features) - M2")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Sidebar clustering view choice
    cluster_view = st.sidebar.radio("🔍 Cluster Visualization Type", ["2D", "3D"], index=0)

    # Select clustering model
    if clustering_method == 'KMeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif clustering_method == 'DBSCAN':
        model = DBSCAN(eps=1.5, min_samples=3)
    elif clustering_method == 'Agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif clustering_method == 'Birch':
        model = Birch(n_clusters=n_clusters)
    elif clustering_method == 'OPTICS':
        model = OPTICS(min_samples=3)
    else:
        model = KMeans(n_clusters=n_clusters)

    # Fit model
    try:
        labels = model.fit_predict(features_scaled)
        features['Cluster'] = labels
    except Exception as e:
        st.error(f"Clustering failed: {e}")
        labels = None

    # PCA Reduction for visualization
    try:
        pca_2d = PCA(n_components=2, random_state=42)
        reduced_2d = pca_2d.fit_transform(features_scaled)

        pca_3d = PCA(n_components=3, random_state=42)
        reduced_3d = pca_3d.fit_transform(features_scaled)
    except Exception:
        reduced_2d = np.zeros((features_scaled.shape[0], 2))
        reduced_3d = np.zeros((features_scaled.shape[0], 3))

    # Visualizations
    cluster_view_choice = st.sidebar.radio("Cluster plot choice (M2)", ["2D", "3D"], index=0)
    if cluster_view_choice == "2D":
        fig_pca = px.scatter(x=reduced_2d[:, 0], y=reduced_2d[:, 1], color=labels.astype(str) if labels is not None else None,
                             title=f"2D Cluster Visualization ({clustering_method})", labels={"x": "PCA 1", "y": "PCA 2"})
        st.plotly_chart(fig_pca, use_container_width=True)
    else:
        fig_3d = px.scatter_3d(x=reduced_3d[:, 0], y=reduced_3d[:, 1], z=reduced_3d[:, 2],
                               color=labels.astype(str) if labels is not None else None,
                               title=f"3D Cluster Visualization ({clustering_method})", opacity=0.8)
        fig_3d.update_traces(marker=dict(size=5))
        st.plotly_chart(fig_3d, use_container_width=True)

    # Cluster count bar chart
    if labels is not None:
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        fig_bar = px.bar(x=cluster_counts.index.astype(str), y=cluster_counts.values, labels={'x': 'Cluster ID', 'y': 'Size'}, title='Cluster Sizes')
        st.plotly_chart(fig_bar, use_container_width=True)

        if len(set(labels)) > 1 and -1 not in set(labels):
            try:
                sil = silhouette_score(features_scaled, labels)
                db = davies_bouldin_score(features_scaled, labels)
                st.markdown(f"**Clustering Metrics:** Silhouette={sil:.3f}, Davies-Bouldin={db:.3f}")
            except Exception:
                st.info('Silhouette/Davies-Bouldin metrics not computable for this clustering.')
    else:
        st.info('Clustering labels not produced.')

# =====================================================
# -------------------- M3 START --------------------
# M3: Threshold-based, Prophet residual-based, and cluster-based anomaly detection
# Runs AFTER M1 + M2 completed above (as requested)
# =====================================================
st.markdown("## 🔁 M3 — Extended Anomaly Detection & Visualization (Runs after M1 & M2)")

# M3 Config
st.sidebar.markdown("### M3 Settings")
m3_threshold_hr = st.sidebar.number_input("M3: Absolute HR threshold (bpm) - alerts", value=180, step=1)
m3_recent_minutes = st.sidebar.slider("M3: Recent window minutes for 'real-time' summary", 1, 60, 10)
m3_prophet_sigma = st.sidebar.number_input("M3: Prophet residual sigma multiplier", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
m3_dbscan_eps = st.sidebar.number_input("M3: DBSCAN eps (cluster-based anomaly)", min_value=0.1, max_value=10.0, value=1.5, step=0.1)
m3_dbscan_min = st.sidebar.number_input("M3: DBSCAN min_samples", min_value=1, max_value=10, value=3, step=1)

# Ensure we have data
if df is None or df.empty:
    st.error("No data available for M3.")
else:
    # 1) Threshold-based anomalies (absolute thresholds + simple domain checks)
    df_m3 = df.copy()
    df_m3['m3_hr_threshold_anomaly'] = 0
    if 'heart_rate' in df_m3.columns:
        df_m3.loc[df_m3['heart_rate'] > m3_threshold_hr, 'm3_hr_threshold_anomaly'] = 1

    # 2) Prophet-residual based anomaly detection (if prophet_df exists)
    df_m3['m3_prophet_anomaly'] = 0
    if Prophet is not None and not prophet_df.empty:
        try:
            # Use model already trained earlier if available; otherwise train quickly on the forecast_metric
            # We will retrain a lightweight prophet if needed to compute residuals on training window
            pf_metric = prophet_df['y'].name if 'y' in prophet_df.columns else None
            prophet_local = Prophet(daily_seasonality=True, weekly_seasonality=True)
            with st.spinner('(M3) Training lightweight Prophet for residual-based anomalies...'):
                prophet_local.fit(prophet_df)
            fut = prophet_local.make_future_dataframe(periods=0)  # only in-sample
            pred = prophet_local.predict(fut)
            merged_m3 = prophet_df.merge(pred[['ds','yhat']], on='ds', how='left')
            merged_m3['residual'] = merged_m3['y'] - merged_m3['yhat']
            sigma = merged_m3['residual'].std()
            merged_m3['m3_prophet_anomaly'] = merged_m3['residual'].abs() > (m3_prophet_sigma * sigma)
            # Map anomalies back to df_m3 by timestamp
            df_m3 = df_m3.merge(merged_m3[['ds','m3_prophet_anomaly']], left_on='timestamp', right_on='ds', how='left')
            df_m3['m3_prophet_anomaly'] = df_m3['m3_prophet_anomaly'].fillna(False).astype(int)
            df_m3.drop(columns=['ds'], inplace=True)
        except Exception as e:
            st.warning(f"(M3) Prophet residual anomaly detection failed: {e}")
            df_m3['m3_prophet_anomaly'] = 0
    else:
        df_m3['m3_prophet_anomaly'] = 0

    # 3) Cluster-based anomaly detection (on windows / features)
    # Strategy: prefer DBSCAN on TSFresh windows (if available); otherwise use sliding-window features (simple aggregates).
    df_m3['m3_cluster_anomaly'] = 0
    cluster_anomaly_indices = set()

    if not features.empty:
        try:
            # Use PCA reduction + DBSCAN on TSFresh features
            scaler_m3 = StandardScaler()
            feats_scaled = scaler_m3.fit_transform(features)
            pca = PCA(n_components=min(10, feats_scaled.shape[1]))
            feats_pca = pca.fit_transform(feats_scaled)
            db = DBSCAN(eps=float(m3_dbscan_eps), min_samples=int(m3_dbscan_min))
            lab = db.fit_predict(feats_pca)
            # Mark windows labeled -1 as anomalies, then map back to original df timestamps using window ids
            if 'window_id' in locals() and 'df_tsfresh' in locals():
                # features.index holds window ids (if produced), else use position mapping
                try:
                    window_index = features.index.to_list()
                except Exception:
                    window_index = list(range(features.shape[0]))
                anomalous_windows = [window_index[i] for i, l in enumerate(lab) if l == -1]
                # Map window_id -> row ranges in original df_tsfresh
                if 'df_tsfresh' in locals():
                    anomalous_rows = df_tsfresh[df_tsfresh['window_id'].isin(anomalous_windows)]
                    ts_anom_timestamps = anomalous_rows['timestamp'].unique().tolist()
                    cluster_anomaly_indices.update(ts_anom_timestamps)
            else:
                # fallback: mark no cluster anomalies
                pass
        except Exception as e:
            st.warning(f"(M3) Cluster-based anomaly detection on TSFresh failed: {e}")
            # fallback later to LOF on simple aggregates
            pass

    # fallback cluster-based approach if no TSFresh features available
    if not cluster_anomaly_indices:
        try:
            # Create sliding-window aggregates (mean,std) and run LocalOutlierFactor
            win = max(5, min(60, window_size))
            agg_list = []
            idx_map = []
            for i in range(0, len(df_m3) - win + 1, max(1, win // 2)):
                w = df_m3.iloc[i:i+win]
                agg = {
                    'mean_hr': w['heart_rate'].mean() if 'heart_rate' in w else 0,
                    'std_hr': w['heart_rate'].std() if 'heart_rate' in w else 0,
                    'mean_steps': w['steps'].mean() if 'steps' in w else 0,
                    'std_steps': w['steps'].std() if 'steps' in w else 0
                }
                agg_list.append(agg)
                idx_map.append((i, i+win-1))
            if agg_list:
                agg_df = pd.DataFrame(agg_list).fillna(0)
                lof = LocalOutlierFactor(n_neighbors=20 if len(agg_df) > 20 else max(2, len(agg_df)-1), contamination=0.05)
                lof_labels = lof.fit_predict(agg_df)
                outlier_windows = [i for i, l in enumerate(lof_labels) if l == -1]
                for ow in outlier_windows:
                    start, end = idx_map[ow]
                    times = df_m3.iloc[start:end+1]['timestamp'].unique().tolist()
                    cluster_anomaly_indices.update(times)
        except Exception as e:
            st.warning(f"(M3) fallback cluster-based anomaly detection failed: {e}")

    # Assign cluster anomalies to main df_m3
    if cluster_anomaly_indices:
        df_m3['m3_cluster_anomaly'] = df_m3['timestamp'].isin(cluster_anomaly_indices).astype(int)
    else:
        df_m3['m3_cluster_anomaly'] = 0

    # Combined M3 anomaly flags (any of the methods)
    df_m3['m3_any_anomaly'] = ((df_m3.get('m3_hr_threshold_anomaly',0) == 1) |
                               (df_m3.get('m3_prophet_anomaly',0) == 1) |
                               (df_m3.get('m3_cluster_anomaly',0) == 1)).astype(int)

    # ---------- M3 Visualizations & Real-time summary ----------
    st.markdown("### M3 — Anomaly Summary & Visualizations")
    # Counts
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Threshold anomalies (HR)", int(df_m3['m3_hr_threshold_anomaly'].sum()))
    with col_b:
        st.metric("Prophet-residual anomalies", int(df_m3['m3_prophet_anomaly'].sum()))
    with col_c:
        st.metric("Cluster-based anomalies", int(df_m3['m3_cluster_anomaly'].sum()))
    with col_d:
        st.metric("Any M3 anomaly", int(df_m3['m3_any_anomaly'].sum()))

    # Real-time / recent anomalies within the last m3_recent_minutes
    recent_cut = df_m3['timestamp'].max() - pd.Timedelta(minutes=int(m3_recent_minutes))
    recent_slice = df_m3[df_m3['timestamp'] >= recent_cut]
    recent_any = recent_slice[recent_slice['m3_any_anomaly'] == 1]

    st.markdown(f"#### Real-time summary (last {m3_recent_minutes} minutes)")
    if recent_any.empty:
        st.success(f"No M3 anomalies detected in the last {m3_recent_minutes} minutes.")
    else:
        st.warning(f"{len(recent_any)} anomaly rows detected in the last {m3_recent_minutes} minutes.")
        st.dataframe(recent_any[['timestamp','heart_rate','steps','resting_heart','m3_hr_threshold_anomaly','m3_prophet_anomaly','m3_cluster_anomaly']].head(50), use_container_width=True)

    # Interactive charts
    st.markdown("### M3 — Interactive Charts (combined)")
    # Combined heart rate view with markers for each anomaly type
    if 'heart_rate' in df_m3.columns:
        fig_m3_hr = px.line(df_m3, x='timestamp', y='heart_rate', title='M3 — Heart Rate with multiple anomaly types')
        if df_m3['m3_hr_threshold_anomaly'].sum() > 0:
            t = df_m3[df_m3['m3_hr_threshold_anomaly'] == 1]
            fig_m3_hr.add_scatter(x=t['timestamp'], y=t['heart_rate'], mode='markers', name='Threshold HR', marker=dict(size=9, symbol='triangle-up', color='purple'))
        if df_m3['m3_prophet_anomaly'].sum() > 0:
            t = df_m3[df_m3['m3_prophet_anomaly'] == 1]
            fig_m3_hr.add_scatter(x=t['timestamp'], y=t['heart_rate'], mode='markers', name='Prophet Residual', marker=dict(size=8, symbol='x', color='red'))
        if df_m3['m3_cluster_anomaly'].sum() > 0:
            t = df_m3[df_m3['m3_cluster_anomaly'] == 1]
            fig_m3_hr.add_scatter(x=t['timestamp'], y=t['heart_rate'], mode='markers', name='Cluster-based', marker=dict(size=7, symbol='circle-open', color='orange'))
        st.plotly_chart(fig_m3_hr, use_container_width=True)

    # Steps chart
    if 'steps' in df_m3.columns:
        fig_m3_steps = px.line(df_m3, x='timestamp', y='steps', title='M3 — Steps with anomaly markers')
        if df_m3['m3_any_anomaly'].sum() > 0:
            t = df_m3[df_m3['m3_any_anomaly'] == 1]
            fig_m3_steps.add_scatter(x=t['timestamp'], y=t['steps'], mode='markers', name='Any M3 anomaly', marker=dict(size=7, symbol='x', color='black'))
        st.plotly_chart(fig_m3_steps, use_container_width=True)

    # Sleep / resting heart chart
    if 'resting_heart' in df_m3.columns:
        fig_m3_rest = px.line(df_m3, x='timestamp', y='resting_heart', title='M3 — Resting Heart with anomaly markers')
        if df_m3['m3_any_anomaly'].sum() > 0:
            t = df_m3[df_m3['m3_any_anomaly'] == 1]
            fig_m3_rest.add_scatter(x=t['timestamp'], y=t['resting_heart'], mode='markers', name='Any M3 anomaly', marker=dict(size=7, symbol='diamond', color='magenta'))
        st.plotly_chart(fig_m3_rest, use_container_width=True)

    # Anomaly distribution pie
    anomaly_counts = {
        "Threshold HR": int(df_m3['m3_hr_threshold_anomaly'].sum()),
        "Prophet Residual": int(df_m3['m3_prophet_anomaly'].sum()),
        "Cluster-based": int(df_m3['m3_cluster_anomaly'].sum())
    }
    fig_pie_m3 = px.pie(values=list(anomaly_counts.values()), names=list(anomaly_counts.keys()), title='M3 Anomaly Distribution')
    st.plotly_chart(fig_pie_m3, use_container_width=True)

    # Alerts
    if df_m3['m3_hr_threshold_anomaly'].sum() > 0:
        st.error(f"⚠️ Threshold alert: {int(df_m3['m3_hr_threshold_anomaly'].sum())} HR readings exceed {m3_threshold_hr} bpm.")
    if df_m3['m3_prophet_anomaly'].sum() > 0:
        st.warning(f"⚠️ Prophet residual anomalies: {int(df_m3['m3_prophet_anomaly'].sum())} points flagged.")
    if df_m3['m3_cluster_anomaly'].sum() > 0:
        st.info(f"ℹ️ Cluster-based anomalies detected: {int(df_m3['m3_cluster_anomaly'].sum())} points/windows.")

# =====================================================
# Export processed data and features (unchanged)
# =====================================================
buffer = io.StringIO()
export_df = df.copy()
if 'df_m3' in locals() and not df_m3.empty:
    export_df = df_m3  # include M3 flags if available
export_df.to_csv(buffer, index=False)
st.download_button('Download Processed CSV (with M3 flags if present)', data=buffer.getvalue(), file_name='fitpulse_processed_with_m3.csv', mime='text/csv')

if not features.empty:
    buf_feat = io.StringIO()
    features.to_csv(buf_feat)
    st.download_button('Download Extracted Features (CSV)', data=buf_feat.getvalue(), file_name='fitpulse_tsfresh_features.csv', mime='text/csv')

st.sidebar.markdown('---')
st.sidebar.info('Merged full pipeline: Anomaly detection + TSFresh + Prophet + Clustering + M3')

# =====================================================
# Step 4: Detailed Report Cards + Clustering 3D Plot (updated names safe)
# =====================================================
# Build safe report metrics (use previously computed vars or defaults)
data_choice = "Activity"  # Default label

feature_report = {
    "data_type": data_choice.lower(),
    "original_rows": len(df),
    "window_size": window_size,
    "features_extracted": features.shape[1] if features is not None and not features.empty else 0,
    "feature_windows": features.shape[0] if features is not None and not features.empty else 0,
    "success": True,
    "extraction_time": round(time.time() % 1, 6)
}

trend_report = {
    "data_type": data_choice.lower(),
    "training_rows": len(prophet_df) if 'prophet_df' in locals() and not prophet_df.empty else 0,
    "forecast_periods": forecast_periods,
    "mae": round(mae, 6) if mae is not None else None,
    "rmse": round(rmse, 6) if rmse is not None else None,
    "mape": round(mape, 6) if mape is not None else None,
    "success": True if mae is not None else False
}

# -----------------------------
# Safe metric extraction
# -----------------------------
sil_score_val = None
db_score_val = None
if 'sil' in locals() and isinstance(sil, (float, int)):
    sil_score_val = round(sil, 6)
if 'db' in locals() and isinstance(db, (float, int)):
    db_score_val = round(db, 6)

cluster_report = {
    "data_type": data_choice.lower(),
    "method": clustering_method.lower(),
    "n_samples": features.shape[0],
    "n_features": features.shape[1] - 1 if 'Cluster' in features.columns else features.shape[1],
    "n_clusters": n_clusters if clustering_method == "KMeans" else (len(set(labels)) - (1 if -1 in labels else 0)),
    "silhouette_score": sil_score_val,
    "davies_bouldin_score": db_score_val,
    "success": True
}


feature_card = f"""
<div style='background:#E0F7FA; padding:15px; border-radius:10px; width:250px; margin-bottom:10px'>
<b>Feature Extraction</b><br>
Data Type: {feature_report['data_type']}<br>
Original Rows: {feature_report['original_rows']}<br>
Window Size: {feature_report['window_size']}<br>
Features Extracted: {feature_report['features_extracted']}<br>
Feature Windows: {feature_report['feature_windows']}<br>
Success: {feature_report['success']}
</div>
"""

trend_card = f"""
<div style='background:#FFF3E0; padding:15px; border-radius:10px; width:250px; margin-bottom:10px'>
<b>Trend Modeling</b><br>
Training Rows: {trend_report['training_rows']}<br>
Forecast Periods: {trend_report['forecast_periods']}<br>
MAE: {trend_report['mae']}<br>
RMSE: {trend_report['rmse']}<br>
MAPE: {trend_report['mape']}<br>
Success: {trend_report['success']}
</div>
"""

cluster_card = f"""
<div style='background:#E8F5E9; padding:15px; border-radius:10px; width:250px; margin-bottom:10px'>
<b>Clustering</b><br>
Method: {cluster_report['method']}<br>
N Samples: {cluster_report['n_samples']}<br>
N Features: {cluster_report['n_features']}<br>
N Clusters: {cluster_report['n_clusters']}<br>
Silhouette Score: {cluster_report.get('silhouette_score','-')}<br>
Davies-Bouldin: {cluster_report.get('davies_bouldin_score','-')}<br>
Success: {cluster_report['success']}
</div>
"""

st.subheader("📋 Detailed Reports (Card Style)")
st.markdown(f"""
<div style='display:flex; gap:20px; flex-wrap:wrap'>
{feature_card}
{trend_card}
{cluster_card}
</div>
""", unsafe_allow_html=True)

# End of file
