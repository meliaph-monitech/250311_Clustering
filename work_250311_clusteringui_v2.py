import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
import numpy as np
from collections import defaultdict

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
    rms = np.sqrt(np.mean(signal**2))
    slope, _ = np.polyfit(np.arange(n), signal, 1)
    return [mean_val, std_val, min_val, max_val, median_val, skewness, kurt, peak_to_peak, energy, cv, spectral_entropy, rms, slope]

st.set_page_config(layout="wide")
st.title("Laser Welding K-Means Clustering - Global Analysis with Feature Selection")

with st.sidebar:
    uploaded_file = st.file_uploader("Upload a ZIP file containing CSV files", type=["zip"])
    if uploaded_file:
        with open("temp.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())
        csv_files, extract_dir = extract_zip("temp.zip")
        st.success(f"Extracted {len(csv_files)} CSV files")
        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()
        filter_column = st.selectbox("Select column for filtering", columns)
        threshold = st.number_input("Enter filtering threshold", value=0.0)
        if st.button("Segment Beads"):
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
    num_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
    if st.button("Run K-Means Clustering") and "metadata" in st.session_state:
        with st.spinner("Running K-Means Clustering..."):
            features_by_bead = defaultdict(list)
            files_by_bead = defaultdict(list)
            for entry in st.session_state["metadata"]:
                df = pd.read_csv(entry["file"])
                bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1]
                features = extract_advanced_features(bead_segment.iloc[:, 0].values)
                bead_number = entry["bead_number"]
                features_by_bead[bead_number].append(features)
                files_by_bead[bead_number].append((entry["file"], bead_number))
            all_scaled_features = []
            all_file_names = []
            for bead_number, feature_matrix in features_by_bead.items():
                scaler = RobustScaler()
                scaled_features = scaler.fit_transform(feature_matrix)
                all_scaled_features.extend(scaled_features)
                all_file_names.extend(files_by_bead[bead_number])
            all_scaled_features = np.array(all_scaled_features)
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(all_scaled_features)
            st.session_state["clustering_results"] = {fn: cluster for fn, cluster in zip(all_file_names, clusters)}

if "clustering_results" in st.session_state:
    bead_numbers = sorted(set(num for _, num in st.session_state["clustering_results"].keys()))
    selected_bead = st.selectbox("Select Bead Number to Display", bead_numbers)
    if st.button("Generate Result"):
        results_df = pd.DataFrame([
            {"File Name": file_name, "Bead Number": bead_num, "Cluster": cluster}
            for (file_name, bead_num), cluster in st.session_state["clustering_results"].items()
        ])
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv_data, file_name="clustering_results.csv", mime="text/csv")
    st.write("## Visualization")
    if selected_bead:
        filtered_results = [(fn, bead, cluster) for (fn, bead), cluster in st.session_state["clustering_results"].items() if bead == selected_bead]
        file_names, bead_numbers, cluster_labels = zip(*filtered_results)
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(np.array([features_by_bead[selected_bead][i] for i in range(len(filtered_results))]))
        fig = px.scatter(x=reduced_features[:, 0], y=reduced_features[:, 1], color=cluster_labels,
                         hover_data={'File Name': file_names, 'Bead Number': bead_numbers, 'Cluster': cluster_labels})
        fig.update_layout(title=f"Bead Number {selected_bead}: K-Means Clustering Results", xaxis_title="PCA Component 1", yaxis_title="PCA Component 2")
        st.plotly_chart(fig)
