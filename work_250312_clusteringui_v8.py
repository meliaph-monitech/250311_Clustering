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
import shutil  # Required for removing directories

def extract_zip(main_zip_path, extract_dir="extracted_zip_contents"):
    # Clear the extraction directory if it exists
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            file_path = os.path.join(extract_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Remove files
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directories
    else:
        os.makedirs(extract_dir)
    
    # Extract the main ZIP file
    try:
        with zipfile.ZipFile(main_zip_path, 'r') as main_zip:
            main_zip.extractall(extract_dir)
    except zipfile.BadZipFile:
        st.error("The uploaded file is not a valid ZIP file.")
        st.stop()
    
    # Find all ZIP files inside the extracted folder
    inner_zip_files = [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.zip')]
    if not inner_zip_files:
        st.error("No ZIP files found inside the uploaded ZIP file.")
        st.stop()

    # Create a mapping of file names to full paths and sort alphabetically
    zip_file_names = sorted([os.path.basename(f) for f in inner_zip_files])  # Extract only the file names and sort

    # Let the user select an inner ZIP file (only show the file names in the dropdown)
    selected_file_name = st.sidebar.selectbox("Select a ZIP file", zip_file_names)
    selected_inner_zip = inner_zip_files[zip_file_names.index(selected_file_name)]  # Map back to full path

    # Create a subdirectory to extract the selected inner ZIP file
    inner_extract_dir = os.path.join(extract_dir, "inner_extracted_csvs")
    if os.path.exists(inner_extract_dir):
        for file in os.listdir(inner_extract_dir):
            file_path = os.path.join(inner_extract_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        os.makedirs(inner_extract_dir)
    
    # Extract the selected inner ZIP file
    try:
        with zipfile.ZipFile(selected_inner_zip, 'r') as inner_zip:
            inner_zip.extractall(inner_extract_dir)
    except zipfile.BadZipFile:
        st.error("The selected file is not a valid ZIP file.")
        st.stop()
    
    # Find all CSV files in the extracted inner ZIP folder
    csv_files = [os.path.join(inner_extract_dir, f) for f in os.listdir(inner_extract_dir) if f.endswith('.csv')]
    if not csv_files:
        st.error("No CSV files found in the selected ZIP file.")
        st.stop()
    
    return csv_files, inner_extract_dir, selected_file_name

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
    total_power = np.sum(fft_magnitudes**2)
    mean_freq = np.sum(freqs * fft_magnitudes) / np.sum(fft_magnitudes)
    peak_freq = freqs[np.argmax(fft_magnitudes)]
    bandwidth = np.sqrt(np.sum((freqs - mean_freq)**2 * fft_magnitudes) / np.sum(fft_magnitudes))
    spectral_entropy = -np.sum((fft_magnitudes / np.sum(fft_magnitudes)) * 
                               np.log2(fft_magnitudes / np.sum(fft_magnitudes) + 1e-12))
    skewness = skew(fft_magnitudes)
    kurt = kurtosis(fft_magnitudes)
    band_power = np.sum(fft_magnitudes**2)
    
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
    csv_files, extract_dir, selected_zip_name = extract_zip("temp.zip")
    st.sidebar.success(f"Extracted {len(csv_files)} CSV files")
    df_sample = pd.read_csv(csv_files[0])
    columns = df_sample.columns.tolist()
    filter_column = st.sidebar.selectbox("Select column for filtering", columns)
    threshold = st.sidebar.number_input("Enter filtering threshold", value=0.0)
    
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
    
    # Cluster adjustment comes AFTER feature selection
    num_clusters = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=20, value=3)
    
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
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            
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
            
            cluster_df["Annotation"] = cluster_df["File Name"].apply(
                lambda x: x.split("_")[-1].split(".csv")[0]
            )
            
            pca2_range = cluster_df["PCA2"].max() - cluster_df["PCA2"].min()
            offset = pca2_range * 0.05
            
            fig = px.scatter(
                cluster_df,
                x="PCA1",
                y="PCA2",
                color=cluster_df["Cluster"].astype(str),
                hover_data=["File Name", "Bead Number", "Cluster"],
                title=f"K-Means Clustering Visualization ({selected_zip_name})"
            )
            
            # Add annotations with dynamic text color
            for i in range(len(cluster_df)):
                fig.add_annotation(
                    x=cluster_df.loc[i, "PCA1"],
                    y=cluster_df.loc[i, "PCA2"] + offset,
                    text=cluster_df.loc[i, "Annotation"],
                    showarrow=False,
                    font=dict(size=11, color="black"),  # Use dynamic text color
                    align="center"
                )
            
            # Display the plot
            st.plotly_chart(fig)
            
if "clustering_results" in st.session_state:
    if st.button("Download Results"):
        csv_data = st.session_state["clustering_results"].to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_data, file_name="clustering_results.csv", mime="text/csv")
