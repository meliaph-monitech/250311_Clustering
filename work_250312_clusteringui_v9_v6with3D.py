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

def extract_advanced_features(signal, sampling_rate=240):
    if len(signal) == 0:
        return [0] * 10
    
    # Perform FFT
    N = len(signal)
    fft_values = fft(signal)
    fft_magnitudes = np.abs(fft_values[:N // 2])  # Take magnitudes of positive frequencies
    freqs = fftfreq(N, 1 / sampling_rate)[:N // 2]  # Positive frequency components
    
    # Frequency-domain features
    total_power = np.sum(fft_magnitudes**2)  # Total power of the spectrum
    mean_freq = np.sum(freqs * fft_magnitudes) / np.sum(fft_magnitudes)  # Mean frequency (centroid)
    peak_freq = freqs[np.argmax(fft_magnitudes)]  # Frequency with maximum amplitude
    bandwidth = np.sqrt(np.sum((freqs - mean_freq)**2 * fft_magnitudes) / np.sum(fft_magnitudes))  # Spectral bandwidth
    spectral_entropy = -np.sum((fft_magnitudes / np.sum(fft_magnitudes)) * 
                               np.log2(fft_magnitudes / np.sum(fft_magnitudes) + 1e-12))  # Spectral entropy
    skewness = skew(fft_magnitudes)  # Skewness of the spectrum
    kurt = kurtosis(fft_magnitudes)  # Kurtosis of the spectrum
    
    band_mask = (freqs >= 0) & (freqs <= sampling_rate / 2)
    band_power = np.sum(fft_magnitudes[band_mask]**2)  # Power within the band
    
    return [
        total_power, mean_freq, peak_freq, bandwidth, 
        spectral_entropy, skewness, kurt, band_power, 
        np.max(fft_magnitudes), np.sum(fft_magnitudes)
    ]

st.set_page_config(layout="wide")
st.title("Laser Welding K-Means Clustering V6 Global Analysis with Frequency-domain Features")

uploaded_file = st.sidebar.file_uploader("Upload a ZIP file containing CSV files", type=["zip"])

if uploaded_file:
    with open("temp.zip", "wb") as f:
        f.write(uploaded_file.getbuffer())
    csv_files, extract_dir = extract_zip("temp.zip")
    st.sidebar.success(f"Extracted {len(csv_files)} CSV files")
    df_sample = pd.read_csv(csv_files[0])
    columns = df_sample.columns.tolist()
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
        selected_bead_number = st.sidebar.selectbox("Select Bead Number for Clustering", bead_numbers)
    
    feature_names = [
        "Total Power", "Mean Frequency", "Peak Frequency", "Bandwidth", 
        "Spectral Entropy", "Skewness", "Kurtosis", "Band Power", 
        "Max Amplitude", "Sum of Amplitudes"
    ]
    selected_features = st.sidebar.multiselect("Select Features for Clustering", feature_names, default=feature_names)
    
    if st.sidebar.button("Run K-Means Clustering") and "metadata" in st.session_state:
        with st.spinner("Running K-Means Clustering..."):
            features_by_bead = []
            file_names = []
            for entry in st.session_state["metadata"]:
                if entry["bead_number"] == selected_bead_number:
                    df = pd.read_csv(entry["file"])
                    bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1]
                    full_features = extract_advanced_features(bead_segment.iloc[:, 0].values)
                    
                    feature_indices = [feature_names.index(f) for f in selected_features]
                    features = [full_features[i] for i in feature_indices]
                    
                    features_by_bead.append(features)
                    file_names.append(entry["file"])
            
            scaler = RobustScaler()
            scaled_features = scaler.fit_transform(features_by_bead)
            
            st.session_state["scaled_features"] = scaled_features
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            
            st.session_state["clusters"] = clusters
            st.session_state["file_names"] = file_names
            
            pca = PCA(n_components=2)
            reduced_features = pca.fit_transform(scaled_features)
            cluster_df = pd.DataFrame({
                "PCA1": reduced_features[:, 0],
                "PCA2": reduced_features[:, 1],
                "Cluster": clusters,
                "File Name": file_names,
                "Bead Number": [selected_bead_number] * len(file_names)
            })
            
            st.session_state["clustering_results"] = cluster_df
            
            fig = px.scatter(
                cluster_df,
                x="PCA1",
                y="PCA2",
                color=cluster_df["Cluster"].astype(str),
                hover_data=["File Name", "Bead Number", "Cluster"],
                title="K-Means Clustering Visualization (PCA Reduced)"
            )
            st.session_state["fig_2d"] = fig
            st.plotly_chart(fig)

if "clustering_results" in st.session_state:
    st.write("### 2D PCA Visualization")
    st.plotly_chart(st.session_state["fig_2d"])

    if st.button("Show 3D PCA"):
        if "scaled_features" in st.session_state and "clusters" in st.session_state:
            scaled_features = st.session_state["scaled_features"]
            clusters = st.session_state["clusters"]
            file_names = st.session_state["file_names"]

            pca_3d = PCA(n_components=3)
            reduced_features_3d = pca_3d.fit_transform(scaled_features)
            cluster_df_3d = pd.DataFrame({
                "PCA1": reduced_features_3d[:, 0],
                "PCA2": reduced_features_3d[:, 1],
                "PCA3": reduced_features_3d[:, 2],
                "Cluster": clusters,
                "File Name": file_names,
                "Bead Number": [selected_bead_number] * len(file_names)
            })

            fig_3d = go.Figure()
            unique_clusters = cluster_df_3d["Cluster"].unique()
            for cluster in unique_clusters:
                cluster_data = cluster_df_3d[cluster_df_3d["Cluster"] == cluster]
                fig_3d.add_trace(go.Scatter3d(
                    x=cluster_data["PCA1"],
                    y=cluster_data["PCA2"],
                    z=cluster_data["PCA3"],
                    mode="markers",
                    marker=dict(size=6),
                    name=f"Cluster {cluster}",
                    hovertext=cluster_data["File Name"]
                ))

            fig_3d.update_layout(
                title="3D PCA Visualization of K-Means Clusters",
                scene=dict(
                    xaxis_title="PCA1",
                    yaxis_title="PCA2",
                    zaxis_title="PCA3",
                    aspectmode="manual",
                    aspectratio=dict(x=2, y=1, z=0.5)
                ),
                height=700
            )
            st.plotly_chart(fig_3d)
