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
    
    return csv_files, inner_extract_dir
    
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
    """
    Extracts frequency-domain features from the signal using FFT.
    Assumes the signal is sampled at the given sampling rate.

    Parameters:
    - signal: 1D array-like signal data
    - sampling_rate: Sampling frequency in Hz (default is 240 Hz)

    Returns:
    - A list of frequency-domain features
    """
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
    
    # Select features in the 240 Hz band if needed (optional, based on requirements)
    band_mask = (freqs >= 0) & (freqs <= sampling_rate / 2)
    band_power = np.sum(fft_magnitudes[band_mask]**2)  # Power within the band
    
    # Return frequency-domain features
    return [
        total_power, # Sum of squared magnitudes of the FFT, representing the signal's energy in the frequency domain.
        mean_freq, # Weighted average of frequencies, often referred to as the spectral centroid.
        peak_freq, # Frequency with the highest magnitude (dominant frequency).
        bandwidth, # Measure of the spread of the spectrum around the mean frequency.
        spectral_entropy, # A measure of the signal's spectral complexity or randomness.
        skewness, # Skewness of the FFT magnitudes, indicating asymmetry in the spectrum.
        kurt, # Kurtosis of the FFT magnitudes, indicating how peaked the spectrum is.
        band_power, # Total power within a specific frequency band (0–240 Hz in this case).
        np.max(fft_magnitudes),  # aximum magnitude in the frequency spectrum.
        np.sum(fft_magnitudes)   # Sum of all FFT magnitudes, representing the total spectral amplitude.
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
    
    # Feature selection multi-select
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
                    
                    # Map selected features to indices
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
            
            # Extract the annotation string from the file name
            cluster_df["Annotation"] = cluster_df["File Name"].apply(
                lambda x: x.split("_")[-1].split(".csv")[0]
            )
            
            # Calculate a suitable offset based on the PCA2 range
            pca2_range = cluster_df["PCA2"].max() - cluster_df["PCA2"].min()
            offset = pca2_range * 0.05  # 5% of the PCA2 range as the vertical offset
            
            # Create the scatter plot
            fig = px.scatter(
                cluster_df,
                x="PCA1",
                y="PCA2",
                color=cluster_df["Cluster"].astype(str),
                hover_data=["File Name", "Bead Number", "Cluster"],
                title="K-Means Clustering Visualization (PCA Reduced)"
            )
            
            # Add annotations for each point (text slightly above the dots)
            for i in range(len(cluster_df)):
                fig.add_annotation(
                    x=cluster_df.loc[i, "PCA1"],
                    y=cluster_df.loc[i, "PCA2"] + offset,  # Offset to place the text above the dot
                    text=cluster_df.loc[i, "Annotation"],
                    showarrow=False,  # No arrow
                    font=dict(size=10, color="black"),
                    align="center"
                )
            
            # Display the plot
            st.plotly_chart(fig)
            
if "clustering_results" in st.session_state:
    if st.button("Download Results"):
        csv_data = st.session_state["clustering_results"].to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_data, file_name="clustering_results.csv", mime="text/csv")
