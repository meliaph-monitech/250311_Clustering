import streamlit as st
import zipfile
import os
import shutil
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
    # Clear the extraction directory
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            full_path = os.path.join(extract_dir, file)
            if os.path.isfile(full_path):
                os.remove(full_path)
            elif os.path.isdir(full_path):
                shutil.rmtree(full_path)
    else:
        os.makedirs(extract_dir)

    # Try to extract the ZIP
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    except zipfile.BadZipFile:
        st.error("The uploaded file is not a valid ZIP file.")
        st.stop()

    # Gather CSV files
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

    # Allow user to select analysis column and filtering options
    st.sidebar.subheader("Signal Analysis Options")
    analysis_column = st.sidebar.selectbox("Select column for signal analysis", columns)
    filter_column = st.sidebar.selectbox("Select column for filtering", columns)
    threshold = st.sidebar.number_input("Enter filtering threshold", value=0.0)

    # Add option to analyze all bead numbers or specific bead numbers
    st.sidebar.subheader("Bead Selection")
    bead_selection_mode = st.sidebar.radio(
        "Select Bead Numbers for Analysis",
        options=["Analyze All Beads", "Specify Bead Numbers"]
    )

    specific_beads = []
    if bead_selection_mode == "Specify Bead Numbers":
        bead_input = st.sidebar.text_input(
            "Enter bead numbers (comma-separated)",
            value=""
        )
        if bead_input.strip():
            try:
                specific_beads = [int(bead.strip()) for bead in bead_input.split(",")]
            except ValueError:
                st.sidebar.error("Invalid input. Please enter integers separated by commas.")

    # Feature selection (ensure this is visible in the sidebar)
    st.sidebar.subheader("Feature Selection")
    feature_names = [
        "Mean Value", "STD Value", "Min Value", "Max Value", "Median Value",
        "Skewness", "Kurtosis", "Peak-to-Peak", "Energy",
        "Coefficient of Variation (CV)", "Spectral Entropy", "Autocorrelation",
        "Root Mean Square (RMS)", "Slope"
    ]
    selected_features = st.sidebar.multiselect(
        "Select Features for Analysis",
        feature_names,
        default=feature_names
    )

    # Add slider for number of clusters
    st.sidebar.subheader("Clustering Options")
    num_clusters = st.sidebar.slider("Select Number of Clusters (for anomalies)", min_value=2, max_value=20, value=3)

    # Add "Filter Data" button for memory efficiency
    if st.sidebar.button("Run Analysis"):
        with st.spinner("Processing data..."):
            metadata = []
            features_global = []
            file_bead_info = []

            for file in csv_files:
                df = pd.read_csv(file)
                segments = segment_beads(df, filter_column, threshold)
                for bead_num, (start, end) in enumerate(segments, start=1):
                    # If specific beads are selected, skip others
                    if specific_beads and bead_num not in specific_beads:
                        continue

                    bead_segment = df.iloc[start:end + 1]
                    signal = bead_segment[analysis_column].values
                    signal = normalize_signal_with_scaler(signal)
                    full_features = extract_advanced_features(signal)

                    # Extract only selected features
                    feature_indices = [feature_names.index(f) for f in selected_features]
                    selected_feature_values = [full_features[i] for i in feature_indices]
                    features_global.append(selected_feature_values)
                    metadata.append({"file": file, "bead_number": bead_num})

            # Scale features and perform anomaly detection
            scaler = RobustScaler()
            scaled_features = scaler.fit_transform(features_global)
            isolation_forest = IsolationForest(random_state=42)
            anomaly_labels = isolation_forest.fit_predict(scaled_features)

            # PCA for visualization
            pca = PCA(n_components=2)
            reduced_features = pca.fit_transform(scaled_features)

            # Annotate data
            cluster_df = pd.DataFrame({
                "PCA1": reduced_features[:, 0],
                "PCA2": reduced_features[:, 1],
                "Anomaly": anomaly_labels,
                "File Name": [m["file"] for m in metadata],
                "Bead Number": [m["bead_number"] for m in metadata]
            })
            cluster_df["Annotation"] = cluster_df["File Name"].apply(lambda x: x.split("_")[-1].split(".csv")[0])

            # Plot anomaly detection result
            cluster_df["Color"] = cluster_df["Anomaly"].apply(lambda x: "red" if x == -1 else "black")
            fig_anomaly = px.scatter(
                cluster_df,
                x="PCA1",
                y="PCA2",
                color=cluster_df["Anomaly"].map({-1: "Anomaly", 1: "Normal"}),
                title="Anomaly Detection Results",
                hover_data=["File Name", "Bead Number"]
            )
            for i in range(len(cluster_df)):
                fig_anomaly.add_annotation(
                    x=cluster_df.loc[i, "PCA1"],
                    y=cluster_df.loc[i, "PCA2"],
                    text=cluster_df.loc[i, "Annotation"],
                    showarrow=False,
                    font=dict(size=10, color="black")
                )
            st.plotly_chart(fig_anomaly)

            # Cluster anomalies
            anomalies = scaled_features[anomaly_labels == -1]
            anomaly_pca = reduced_features[anomaly_labels == -1]
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            anomaly_clusters = kmeans.fit_predict(anomalies)

            # Plot clustering results
            cluster_df_anomalies = cluster_df[cluster_df["Anomaly"] == -1].copy()
            cluster_df_anomalies["Cluster"] = anomaly_clusters
            cluster_df["Cluster"] = cluster_df["Anomaly"].apply(lambda x: -1 if x == 1 else None)
            cluster_df.loc[cluster_df["Anomaly"] == -1, "Cluster"] = anomaly_clusters

            fig_clusters = px.scatter(
                cluster_df,
                x="PCA1",
                y="PCA2",
                color=cluster_df["Cluster"].astype(str),
                title="Clustering Results for Anomalies",
                hover_data=["File Name", "Bead Number"]
            )
            for i in range(len(cluster_df)):
                fig_clusters.add_annotation(
                    x=cluster_df.loc[i, "PCA1"],
                    y=cluster_df.loc[i, "PCA2"],
                    text=cluster_df.loc[i, "Annotation"],
                    showarrow=False,
                    font=dict(size=10, color="black")
                )
            st.plotly_chart(fig_clusters)

# ==================== Summary Table ====================

# Create a new DataFrame combining annotations and cluster info
summary_df = cluster_df.copy()
summary_df["Cluster"] = summary_df["Cluster"].astype("Int64")  # Allows NA

# Pivot table: counts grouped by expert annotation and predicted label
summary_table = pd.pivot_table(
    summary_df,
    index="Annotation", 
    columns="Cluster",
    values="Bead Number",
    aggfunc="count",
    fill_value=0,
    margins=True,
    margins_name="Total"
)

# Rename the cluster columns: Cluster -1 = Normal, Cluster 0/1/... = Anomaly Clusters
cluster_col_map = {-1: "Normal"}
for col in summary_table.columns:
    if col != -1 and col != "Total":
        cluster_col_map[col] = f"Anomaly Cluster {col}"
summary_table.rename(columns=cluster_col_map, inplace=True)

# Display
st.subheader("Summary of Expert Annotations vs. ML Classification")
st.dataframe(summary_table)
