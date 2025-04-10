import streamlit as st
import numpy as np
import pandas as pd
import scipy.signal as signal
import zipfile
import os
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Spectrogram VIZ with Line Plot Comparison")

# Function to extract CSV files from ZIP
def extract_zip(zip_path, extract_dir="extracted_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    csv_files = [f for f in os.listdir(extract_dir) if f.endswith('.csv')]
    return [os.path.join(extract_dir, f) for f in csv_files], extract_dir

# Function to segment beads
def segment_beads(df, column, threshold):
    start_indices, end_indices = [], []
    signal_values = df[column].to_numpy()
    i = 0
    while i < len(signal_values):
        if signal_values[i] > threshold:
            start = i
            while i < len(signal_values) and signal_values[i] > threshold:
                i += 1
            end = i - 1
            start_indices.append(start)
            end_indices.append(end)
        else:
            i += 1
    return list(zip(start_indices, end_indices))

# Sidebar for file upload and settings
with st.sidebar:
    uploaded_file = st.file_uploader("Upload ZIP file containing CSV files", type=["zip"])
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
                metadata = {}
                for file in csv_files:
                    df = pd.read_csv(file)
                    segments = segment_beads(df, filter_column, threshold)
                    metadata[file] = segments
                st.success("Bead segmentation complete")
                st.session_state["metadata"] = metadata

# Spectrogram parameters
fs = st.sidebar.number_input("Sampling Frequency (fs)", min_value=1000, max_value=50000, value=10000)
a = st.sidebar.number_input("nperseg parameter (a)", min_value=256, max_value=4096, value=1024)
b = st.sidebar.number_input("Division Factor (b)", min_value=1, max_value=10, value=4)
c = st.sidebar.number_input("Overlap Ratio (c)", min_value=0.0, max_value=1.0, value=0.99)
d = st.sidebar.number_input("nfft parameter (d)", min_value=512, max_value=8192, value=2048)
db_scale = st.sidebar.number_input("dB Scale", min_value=50, max_value=150, value=110)
ylimit = st.sidebar.number_input("Y-Axis Limit", min_value=100, max_value=int(fs / 2), value=500)

if "metadata" in st.session_state and isinstance(st.session_state["metadata"], dict):
    # Select files and beads
    selected_files = st.sidebar.multiselect("Select CSV files", list(st.session_state["metadata"].keys()))
    selected_frequencies = st.sidebar.text_area("Enter Frequencies (Hz, comma-separated)", "240, 500").split(",")
    try:
        selected_frequencies = [float(freq.strip()) for freq in selected_frequencies if freq.strip()]
    except ValueError:
        selected_frequencies = []
        st.error("Invalid frequency input. Please enter numeric values separated by commas.")
    
    if selected_files and selected_frequencies:
        # Initialize Plotly figure
        fig = go.Figure()
        
        for file in selected_files:
            df = pd.read_csv(file)
            bead_options = list(range(1, len(st.session_state["metadata"][file]) + 1))
            selected_bead = st.sidebar.selectbox(f"Select Bead Number for {os.path.basename(file)}", bead_options)
            
            start, end = st.session_state["metadata"][file][selected_bead - 1]
            sample_data = df.iloc[start:end].to_numpy()

            # Spectrogram computation
            nperseg = min(a, len(sample_data) // b)
            noverlap = int(c * nperseg)
            nfft = min(d, b ** int(np.ceil(np.log2(nperseg * 2))))
            
            f, t, Sxx = signal.spectrogram(sample_data[:, 0], fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
            Sxx_dB = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)
            min_disp_dB = np.max(Sxx_dB) - db_scale
            Sxx_dB[Sxx_dB < min_disp_dB] = min_disp_dB
            
            # Extract intensities for selected frequencies
            for freq in selected_frequencies:
                freq_indices = np.where((f >= freq - 5) & (f <= freq + 5))[0]
                if len(freq_indices) > 0:
                    intensity_over_time = np.mean(Sxx_dB[freq_indices, :] - min_disp_dB, axis=0)
                    fig.add_trace(go.Scatter(
                        x=t,
                        y=intensity_over_time,
                        mode='lines',
                        name=f"{os.path.basename(file)} - {freq} Hz"
                    ))
        
        # Update layout of the Plotly figure
        fig.update_layout(
            title="Frequency Intensity Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Signal Intensity (dB)",
            legend_title="File and Frequency",
            height=600,
            width=1000
        )
        
        # Display the Plotly figure
        st.plotly_chart(fig)
