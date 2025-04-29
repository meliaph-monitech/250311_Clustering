import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
import numpy as np

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
    start_indices = []
    end_indices = []
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

def extract_advanced_features(signal, sampling_rate=240):
    if len(signal) == 0:
        return [0] * 17  # Updated to accommodate new features
    
    signal = normalize_signal_with_scaler(signal)
    
    # Basic Statistical Features
    mean_value = np.mean(signal)
    median_value = np.median(signal)
    max_value = np.max(signal)
    min_value = np.min(signal)
    std_dev = np.std(signal)
    variance_value = np.var(signal)
    z_scores = (signal - mean_value) / (std_dev + 1e-12)  # Avoid division by zero
    mean_z_score = np.mean(z_scores)  # Mean of Z-scores
    
    # FFT-Based Features
    N = len(signal)
    fft_values = fft(signal)
    fft_magnitudes = np.abs(fft_values[:N // 2])
    freqs = fftfreq(N, 1 / sampling_rate)[:N // 2]
    
    total_power = np.sum(fft_magnitudes**2)
    mean_freq = np.sum(freqs * fft_magnitudes) / np.sum(fft_magnitudes)
    peak_freq = freqs[np.argmax(fft_magnitudes)]
    bandwidth = np.sqrt(np.sum((freqs - mean_freq)**2 * fft_magnitudes) / np.sum(fft_magnitudes))
    spectral_entropy = -np.sum((fft_magnitudes / np.sum(fft_magnitudes)) * 
                               np.log2(fft_magnitudes / np.sum(fft_magnitudes) + 1e-12))
    skewness = skew(fft_magnitudes)
    kurt = kurtosis(fft_magnitudes)
    
    band_mask = (freqs >= 0) & (freqs <= sampling_rate / 2)
    band_power = np.sum(fft_magnitudes[band_mask]**2)
    
    return [
        mean_value, median_value, max_value, min_value, std_dev, variance_value, mean_z_score,
        total_power, mean_freq, peak_freq, bandwidth, spectral_entropy, skewness, kurt, 
        band_power, np.max(fft_magnitudes), np.sum(fft_magnitudes)
    ]

st.set_page_config(layout="wide")
st.title("Global K-Means Clustering Across Selected Bead Numbers")

uploaded_file = st.sidebar.file_uploader("Upload a ZIP file containing CSV files", type=["zip"])

# Allow the user to choose which column to use for analysis
if uploaded_file:
    with open("temp.zip", "wb") as f:
        f.write(uploaded_file.getbuffer())
    csv_files, extract_dir = extract_zip("temp.zip")
    st.sidebar.success(f"Extracted {len(csv_files)} CSV files")
    
    # Read the first CSV file to get column names
    df_sample = pd.read_csv(csv_files[0])
    columns = df_sample.columns.tolist()
    
    # Sidebar option to select the column to use for signal analysis
    analysis_column = st.sidebar.selectbox("Select column for signal analysis", columns)
    
    # Sidebar option to select column for filtering
    filter_column = st.sidebar.selectbox("Select column for filtering", columns)
    threshold = st.sidebar.number_input("Enter filtering threshold", value=0.0)
    num_clusters = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=20, value=3)
    
    if st.sidebar.button("Segment Beads"):
        with st.spinner("Segmenting beads..."):
            bead_segments = {}
            metadata = []
            for file in csv_files:
                df = pd.read_csv(file)
                segments = segment_beads(df, filter_column, threshold)
                if segments:
                    bead_segments[file] = segments
                    for bead_num, (start, end) in enumerate(segments, start=1):
                        metadata.append({"file": file, "bead_number": bead_num, "start_index": start, "end_index": end})
            st.success("Bead segmentation complete")
            st.session_state["metadata"] = metadata
    
    if "metadata" in st.session_state:
        bead_numbers = sorted(set(entry["bead_number"] for entry in st.session_state["metadata"]))
        selected_bead_numbers = st.sidebar.multiselect("Select Bead Numbers for Clustering", bead_numbers)
    
    feature_names = [
        "Mean", "Median", "Max", "Min", "Standard Deviation", "Variance", "Mean Z-Score",
        "Total Power", "Mean Frequency", "Peak Frequency", "Bandwidth", 
        "Spectral Entropy", "Skewness", "Kurtosis", "Band Power", 
        "Max Amplitude", "Sum of Amplitudes"
    ]
    selected_features = st.sidebar.multiselect("Select Features for Clustering", feature_names, default=feature_names)
    
    if st.sidebar.button("Run K-Means Clustering") and "metadata" in st.session_state:
        with st.spinner("Running K-Means Clustering..."):
            features_global = []
            file_bead_info = []  # Store file and bead info for annotation
            
            for entry in st.session_state["metadata"]:
                if entry["bead_number"] in selected_bead_numbers:
                    df = pd.read_csv(entry["file"])
                    bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1]
                    
                    # Use the selected column for analysis
                    signal = bead_segment[analysis_column].values
                    
                    normalized_signal = normalize_signal_with_scaler(signal)
                    
                    full_features = extract_advanced_features(normalized_signal)
                    
                    feature_indices = [feature_names.index(f) for f in selected_features]
                    features = [full_features[i] for i in feature_indices]
                    
                    features_global.append(features)
                    file_bead_info.append({
                        "file_name": entry["file"],
                        "bead_number": entry["bead_number"]
                    })
            
            scaler = RobustScaler()
            scaled_features = scaler.fit_transform(features_global)
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            
            pca_2d = PCA(n_components=2)
            reduced_features_2d = pca_2d.fit_transform(scaled_features)
            cluster_df_2d = pd.DataFrame({
                "PCA1": reduced_features_2d[:, 0],
                "PCA2": reduced_features_2d[:, 1],
                "Cluster": clusters,
                "File Name": [info["file_name"] for info in file_bead_info],
                "Bead Number": [info["bead_number"] for info in file_bead_info]
            })
            
            # cluster_df_2d["Annotation"] = cluster_df_2d.apply(
            #     lambda row: row["File Name"].split("_")[2] + f" - {row['Bead Number']}" if len(row["File Name"].split("_")) > 2 else "Invalid File Name",
            #     axis=1
            # )

            cluster_df_2d["Annotation"] = cluster_df_2d["File Name"].apply(
                lambda x: x.split("_")[-1].split(".csv")[0]
            )
            
            pca2_range = cluster_df_2d["PCA2"].max() - cluster_df_2d["PCA2"].min()
            offset = pca2_range * 0.05  # Adjust annotation position
            
            fig_2d = px.scatter(
                cluster_df_2d,
                x="PCA1",
                y="PCA2",
                color=cluster_df_2d["Cluster"].astype(str),
                hover_data=["File Name", "Bead Number", "Cluster"],
                title="2D PCA Visualization (Global K-Means Clustering)"
            )
            
            for i in range(len(cluster_df_2d)):
                fig_2d.add_annotation(
                    x=cluster_df_2d.loc[i, "PCA1"],
                    y=cluster_df_2d.loc[i, "PCA2"] + offset,
                    text=cluster_df_2d.loc[i, "Annotation"],
                    showarrow=False,
                    font=dict(size=10, color="black")
                )
            
            st.write("### 2D PCA Visualization")
            st.plotly_chart(fig_2d, key="2d_chart")
