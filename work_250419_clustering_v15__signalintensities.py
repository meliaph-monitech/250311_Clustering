import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from scipy.signal import spectrogram

# === Setup and Helper Functions ===

st.set_page_config(layout="wide")
st.title("Laser Welding Clustering Visualization - V15 Signal Intensities")

def extract_zip(zip_path, extract_dir="extracted_csvs"):
    if os.path.exists(extract_dir):
        for file in os.listdir(extract_dir):
            os.remove(os.path.join(extract_dir, file))
    else:
        os.makedirs(extract_dir)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return [os.path.join(extract_dir, f) for f in os.listdir(extract_dir) if f.endswith('.csv')]

def segment_beads(df, column, threshold):
    starts, ends = [], []
    signal = df[column].to_numpy()
    i = 0
    while i < len(signal):
        if signal[i] > threshold:
            start = i
            while i < len(signal) and signal[i] > threshold:
                i += 1
            ends.append(i - 1)
            starts.append(start)
        i += 1
    return list(zip(starts, ends))

def compute_spectrogram_intensity(signal, fs, fmin, fmax, nperseg, noverlap, nfft):
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    Sxx_dB = 20 * np.log10(np.abs(Sxx) + np.finfo(float).eps)
    band_indices = np.where((f >= fmin) & (f <= fmax))[0]
    if len(band_indices) == 0:
        return np.zeros_like(t)
    return np.mean(Sxx_dB[band_indices, :], axis=0)

# === Sidebar Controls ===

with st.sidebar:
    uploaded_file = st.file_uploader("Upload ZIP of CSVs", type=["zip"])
    clustering_mode = st.radio("Clustering Mode", ["Time Domain", "Frequency Domain"])

    if uploaded_file:
        with open("temp.zip", "wb") as f:
            f.write(uploaded_file.getbuffer())
        csv_files = extract_zip("temp.zip")
        st.success(f"Extracted {len(csv_files)} CSV files")

        df_sample = pd.read_csv(csv_files[0])
        columns = df_sample.columns.tolist()
        filter_column = st.selectbox("Select filter column", columns)
        threshold = st.number_input("Bead detection threshold", value=0.0)

        if clustering_mode == "Frequency Domain":
            fs = st.number_input("Sampling frequency (Hz)", min_value=1000, value=10000)
            fmin = st.number_input("Min frequency (Hz)", min_value=1, value=15000)
            fmax = st.number_input("Max frequency (Hz)", min_value=1, value=20000)
            nperseg = st.number_input("nperseg", min_value=128, max_value=8192, value=1024)
            noverlap_ratio = st.slider("Overlap ratio", 0.0, 0.99, value=0.9)
            noverlap = int(nperseg * noverlap_ratio)
            nfft = st.number_input("nfft", min_value=256, value=2048)

        if st.button("Segment Beads"):
            metadata = []
            for file in csv_files:
                df = pd.read_csv(file)
                segments = segment_beads(df, filter_column, threshold)
                for bead_number, (start, end) in enumerate(segments, start=1):
                    metadata.append({
                        "file": file,
                        "bead_number": bead_number,
                        "start": start,
                        "end": end
                    })
            st.session_state["metadata"] = metadata
            st.success("Beads segmented.")

        bead_input = st.text_input("Bead numbers (comma-separated)")
        if st.button("Select Beads") and "metadata" in st.session_state:
            selected_beads = [int(b.strip()) for b in bead_input.split(",") if b.strip().isdigit()]
            chosen = []
            for m in st.session_state["metadata"]:
                if m["bead_number"] in selected_beads:
                    df = pd.read_csv(m["file"])
                    seg = df.iloc[m["start"]:m["end"] + 1]
                    chosen.append({**m, "df": seg})
            st.session_state["chosen"] = chosen
            st.success("Beads loaded.")

        num_clusters = st.slider("Number of clusters", 2, 10, 4)

        if st.button("Run Clustering") and "chosen" in st.session_state:
            chosen = st.session_state["chosen"]
            results = {}

            # Perform clustering per bead number
            bead_groups = {}
            for bead in chosen:
                bead_num = bead["bead_number"]
                if bead_num not in bead_groups:
                    bead_groups[bead_num] = []
                bead_groups[bead_num].append(bead)

            for bead_num, beads in bead_groups.items():
                features = []
                signals_for_plot = []
                for bead in beads:
                    signal = bead["df"].iloc[:, 0].values
                    if clustering_mode == "Time Domain":
                        scaler = RobustScaler()
                        norm = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
                        features.append(norm[:1000] if len(norm) > 1000 else np.pad(norm, (0, 1000 - len(norm))))
                        signals_for_plot.append(norm)
                    else:
                        intensity = compute_spectrogram_intensity(signal, fs, fmin, fmax, nperseg, noverlap, nfft)
                        features.append(intensity[:1000] if len(intensity) > 1000 else np.pad(intensity, (0, 1000 - len(intensity))))
                        signals_for_plot.append(intensity)

                kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
                labels = kmeans.fit_predict(features)
                results[bead_num] = {
                    "labels": labels,
                    "signals": signals_for_plot,
                    "files": [b["file"] for b in beads],
                    "beads": [b["bead_number"] for b in beads]
                }
            st.session_state["results"] = results
            st.success("Clustering complete!")

# === Visualization ===

if "results" in st.session_state:
    results = st.session_state["results"]
    colors = ["red", "blue", "green", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"]

    for bead_num, result in results.items():
        labels = result["labels"]
        signals = result["signals"]
        files = result["files"]
        beads = result["beads"]

        fig = go.Figure()
        for i, b in enumerate(beads):
            label = labels[i]
            sig = signals[i]
            fig.add_trace(go.Scatter(
                y=sig,
                mode="lines",
                name=f"Cluster {label}: {files[i]}",
                line=dict(color=colors[label % len(colors)], width=1),
                hoverinfo="text",
                text=f"File: {files[i]}<br>Cluster: {label}"
            ))
        fig.update_layout(
            title=f"Bead {bead_num} - Clustered ({clustering_mode})",
            xaxis_title="Time Index" if clustering_mode == "Time Domain" else "Time Frame",
            yaxis_title="Signal" if clustering_mode == "Time Domain" else "Intensity (dB)",
            showlegend=True
        )
        st.plotly_chart(fig)
