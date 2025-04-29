import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
import numpy as np

# Helper Functions
def extract_zip(zip_path, extract_dir="extracted_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    except zipfile.BadZipFile:
        st.error("The uploaded file is not a valid ZIP file.")
        st.stop()
    csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
    if not csv_files:
        st.error("No CSV files found in the ZIP file.")
        st.stop()
    return [os.path.join(extract_dir, f) for f in csv_files], extract_dir

def segment_beads(df, column, threshold):
    start_indices, end_indices = [], []
    signal = df[column].to_numpy()
    i = 0
    while i < len(signal):
        if signal[i] > threshold:
            start = i
            while i < len(signal) and signal[i] > threshold:
                i += 1
            end = i - 1
            start_indices.append(start)
            end_indices.append(end)
        else:
            i += 1
    return list(zip(start_indices, end_indices))

def normalize_signal_with_scaler(signal):
    scaler = RobustScaler()
    signal_reshaped = signal.reshape(-1, 1)
    normalized_signal = scaler.fit_transform(signal_reshaped).flatten()
    return normalized_signal

def extract_advanced_features(signal):
    n = len(signal)
    if n == 0:
        return [0] * 17
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    median_val = np.median(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)
    peak_to_peak = max_val - min_val
    energy = np.sum(signal**2)
    cv = std_val / mean_val if mean_val != 0 else 0
    signal_fft = fft(signal)
    psd = np.abs(signal_fft)**2
    freqs = fftfreq(n, 1)
    positive_freqs = freqs[:n // 2]
    positive_psd = psd[:n // 2]
    psd_normalized = positive_psd / np.sum(positive_psd) if np.sum(positive_psd) > 0 else np.zeros_like(positive_psd)
    spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized + 1e-12))
    autocorrelation = np.corrcoef(signal[:-1], signal[1:])[0, 1] if n > 1 else 0
    rms = np.sqrt(np.mean(signal**2))
    slope, _ = np.polyfit(np.arange(n), signal, 1)
    return [mean_val, std_val, min_val, max_val, median_val, skewness, kurt, peak_to_peak, energy, cv, 
            spectral_entropy, autocorrelation, rms, slope]

# Streamlit UI
st.set_page_config(layout="wide")
st.title("Anomaly Detection and Clustering for Processed Data")

uploaded_file = st.sidebar.file_uploader("Upload a ZIP file containing CSV files", type=["zip"])

if uploaded_file:
    with open("temp.zip", "wb") as f:
        f.write(uploaded_file.getbuffer())
    csv_files, extract_dir = extract_zip("temp.zip")
    st.sidebar.success(f"Extracted {len(csv_files)} CSV files")
    
    df_sample = pd.read_csv(csv_files[0])
    columns = df_sample.columns.tolist()
    
    analysis_column = st.sidebar.selectbox("Select column for signal analysis", columns)
    filter_column = st.sidebar.selectbox("Select column for filtering", columns)
    threshold = st.sidebar.number_input("Enter filtering threshold", value=0.0)
    num_clusters = st.sidebar.slider("Select Number of Clusters (for anomalies)", min_value=2, max_value=20, value=3)
    
    feature_names = ["Mean Value", "STD Value", "Min Value", "Max Value", "Median Value", "Skewness", "Kurtosis", "Peak-to-Peak",
                     "Energy", "Coefficient of Variation (CV)", "Spectral Entropy", "Autocorrelation", "Root Mean Square (RMS)", "Slope"]
    selected_features = st.sidebar.multiselect("Select Features for Analysis", feature_names, default=feature_names)
    
    if st.sidebar.button("Run Analysis"):
        with st.spinner("Processing data..."):
            bead_segments = {}
            metadata = []
            for file in csv_files:
                df = pd.read_csv(file)
                segments = segment_beads(df, filter_column, threshold)
                if segments:
                    bead_segments[file] = segments
                    for bead_num, (start, end) in enumerate(segments, start=1):
                        metadata.append({"file": file, "bead_number": bead_num, "start_index": start, "end_index": end})

            features_global = []
            file_bead_info = []
            for entry in metadata:
                df = pd.read_csv(entry["file"])
                bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1]
                signal = bead_segment[analysis_column].values
                signal = normalize_signal_with_scaler(signal)
                full_features = extract_advanced_features(signal)
                feature_indices = [feature_names.index(f) for f in selected_features]
                selected_feature_values = [full_features[i] for i in feature_indices]
                features_global.append(selected_feature_values)
                file_bead_info.append({
                    "file_name": entry["file"],
                    "bead_number": entry["bead_number"]
                })
            
            # Scale features and perform anomaly detection
            scaler = RobustScaler()
            scaled_features = scaler.fit_transform(features_global)
            isolation_forest = IsolationForest(random_state=42)
            anomaly_labels = isolation_forest.fit_predict(scaled_features)
            
            # Separate anomalies and normal data
            reduced_features = PCA(n_components=2).fit_transform(scaled_features)
            cluster_df = pd.DataFrame({
                "PCA1": reduced_features[:, 0],
                "PCA2": reduced_features[:, 1],
                "Anomaly": ["Anomaly" if label == -1 else "Normal" for label in anomaly_labels],
                "File Name": [info["file_name"] for info in file_bead_info],
                "Bead Number": [info["bead_number"] for info in file_bead_info]
            })
            
            # Plot anomaly detection result
            fig_anomaly = px.scatter(
                cluster_df,
                x="PCA1",
                y="PCA2",
                color="Anomaly",
                hover_data=["File Name", "Bead Number"],
                title="Anomaly Detection Results"
            )
            # Add annotations
            for i in range(len(cluster_df)):
                fig_anomaly.add_annotation(
                    x=cluster_df.loc[i, "PCA1"],
                    y=cluster_df.loc[i, "PCA2"],
                    text=str(cluster_df.loc[i, "Bead Number"]),
                    showarrow=False,
                    font=dict(size=10, color="black")
                )
            st.plotly_chart(fig_anomaly)
            
            # Cluster anomalies
            anomalies = scaled_features[anomaly_labels == -1]
            anomaly_pca = reduced_features[anomaly_labels == -1]
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            anomaly_clusters = kmeans.fit_predict(anomalies)
            
            # Plot clustering results for anomalies
            cluster_df_anomalies = pd.DataFrame({
                "PCA1": anomaly_pca[:, 0],
                "PCA2": anomaly_pca[:, 1],
                "Cluster": anomaly_clusters
            })
            fig_clusters = px.scatter(
                cluster_df_anomalies,
                x="PCA1",
                y="PCA2",
                color=cluster_df_anomalies["Cluster"].astype(str),
                title="Clustering Results for Anomalies"
            )
            st.plotly_chart(fig_clusters)
