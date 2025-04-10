import streamlit as st
import zipfile
import os
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import numpy as np
import scipy.signal as signal

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

st.set_page_config(layout="wide")
st.title("Laser Welding K-Means Clustering for Signal Intensities Over Time")

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
    fs = st.sidebar.number_input("Sampling Frequency (fs)", min_value=1000, max_value=50000, value=10000)
    nperseg = st.sidebar.number_input("Spectrogram Window Size (nperseg)", min_value=256, max_value=4096, value=1024)
    overlap_ratio = st.sidebar.number_input("Overlap Ratio", min_value=0.0, max_value=1.0, value=0.5)
    db_scale = st.sidebar.number_input("dB Scale", min_value=50, max_value=150, value=110)
    selected_frequencies = st.sidebar.text_area("Enter Frequencies (Hz, comma-separated)", "240, 500").split(",")
    
    try:
        selected_frequencies = [float(freq.strip()) for freq in selected_frequencies if freq.strip()]
    except ValueError:
        selected_frequencies = []
        st.error("Invalid frequency input. Please enter numeric values separated by commas.")
    
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
    
    if st.sidebar.button("Run K-Means Clustering") and "metadata" in st.session_state:
        with st.spinner("Running K-Means Clustering..."):
            intensities_by_bead = []
            file_names = []
            for entry in st.session_state["metadata"]:
                if entry["bead_number"] == selected_bead_number:
                    df = pd.read_csv(entry["file"])
                    bead_segment = df.iloc[entry["start_index"]:entry["end_index"] + 1].to_numpy()

                    # Compute spectrogram
                    noverlap = int(overlap_ratio * nperseg)
                    f, t, Sxx = signal.spectrogram(bead_segment[:, 0], fs, nperseg=nperseg, noverlap=noverlap)
                    Sxx_dB = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)
                    min_disp_dB = np.max(Sxx_dB) - db_scale
                    Sxx_dB[Sxx_dB < min_disp_dB] = min_disp_dB

                    intensities = []
                    for freq in selected_frequencies:
                        freq_indices = np.where((f >= freq - 5) & (f <= freq + 5))[0]
                        if len(freq_indices) > 0:
                            intensity_over_time = np.mean(Sxx_dB[freq_indices, :] - min_disp_dB, axis=0)
                            intensities.extend(intensity_over_time)
                    
                    if intensities:
                        intensities_by_bead.append(intensities)
                        file_names.append(entry["file"])
            
            if not intensities_by_bead:
                st.error("No signal intensities found for the selected frequencies.")
                st.stop()
            
            scaler = RobustScaler()
            scaled_intensities = scaler.fit_transform(intensities_by_bead)
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_intensities)
            
            pca = PCA(n_components=2)
            reduced_features = pca.fit_transform(scaled_intensities)
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
