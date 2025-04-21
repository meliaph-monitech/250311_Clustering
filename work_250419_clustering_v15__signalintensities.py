import streamlit as st
import zipfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import spectrogram
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

def extract_zip(zip_file, extract_to="extracted"):
    if os.path.exists(extract_to):
        for f in os.listdir(extract_to):
            os.remove(os.path.join(extract_to, f))
    else:
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return [os.path.join(extract_to, f) for f in os.listdir(extract_to) if f.endswith(".csv")]

def segment_beads(df, column, threshold):
    signal = df[column].values
    start, end = [], []
    i = 0
    while i < len(signal):
        if signal[i] > threshold:
            s = i
            while i < len(signal) and signal[i] > threshold:
                i += 1
            e = i - 1
            start.append(s)
            end.append(e)
        else:
            i += 1
    return list(zip(start, end))

def compute_spectrogram_intensity(signal, fs, fmin, fmax, nperseg, noverlap, nfft):
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    Sxx = 10 * np.log10(Sxx + 1e-12)  # Convert to dB
    freq_mask = (f >= fmin) & (f <= fmax)
    if np.sum(freq_mask) == 0:
        return np.zeros_like(t)
    selected = Sxx[freq_mask, :]
    return np.mean(selected, axis=0)

st.set_page_config(layout="wide")
st.title("Laser Welding Clustering Visualization - V15 Signal Intensities")

with st.sidebar:
    uploaded = st.file_uploader("Upload ZIP of CSVs", type="zip")
    if uploaded:
        with open("temp.zip", "wb") as f:
            f.write(uploaded.getbuffer())
        csv_files = extract_zip("temp.zip")
        df_sample = pd.read_csv(csv_files[0])
        cols = df_sample.columns.tolist()
        filter_col = st.selectbox("Filter Column", cols)
        threshold = st.number_input("Threshold", value=0.0)
        
        if st.button("Segment Beads"):
            metadata = []
            for file in csv_files:
                df = pd.read_csv(file)
                segments = segment_beads(df, filter_col, threshold)
                for i, (start, end) in enumerate(segments):
                    metadata.append({
                        "file": file,
                        "bead_number": i + 1,
                        "start": start,
                        "end": end
                    })
            st.session_state["meta"] = metadata
            st.success(f"Segmented {len(metadata)} beads!")

        bead_str = st.text_input("Select Bead Numbers (comma-separated)")
        if st.button("Load Beads") and "meta" in st.session_state:
            bead_nums = [int(b.strip()) for b in bead_str.split(",") if b.strip().isdigit()]
            chosen = []
            for entry in st.session_state["meta"]:
                if entry["bead_number"] in bead_nums:
                    df = pd.read_csv(entry["file"])
                    seg = df.iloc[entry["start"]:entry["end"]+1]
                    chosen.append({
                        "bead_number": entry["bead_number"],
                        "file": entry["file"],
                        "df": seg
                    })
            st.session_state["chosen"] = chosen
            st.success("Loaded selected beads!")

        clustering_mode = st.radio("Clustering Mode", ["Time Domain", "Frequency Domain"])
        num_clusters = st.slider("Number of Clusters", 2, 10, 4)

        if clustering_mode == "Frequency Domain":
            fmin = st.number_input("Min Frequency (Hz)", value=1000)
            fmax = st.number_input("Max Frequency (Hz)", value=1000)
            fs = st.number_input("Sampling Rate (Hz)", value=44100)
            nperseg = st.number_input("nperseg", value=256)
            noverlap = st.number_input("noverlap", value=128)
            nfft = st.number_input("nfft", value=512)

if st.sidebar.button("Run Clustering") and "chosen" in st.session_state:
    chosen = st.session_state["chosen"]
    per_bead = {}
    for bead in chosen:
        bead_num = bead["bead_number"]
        signal = bead["df"].iloc[:, 0].values

        if clustering_mode == "Time Domain":
            scaler = RobustScaler()
            norm = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
            feat = norm[:1000] if len(norm) > 1000 else np.pad(norm, (0, 1000 - len(norm)))
        else:
            intensity = compute_spectrogram_intensity(signal, fs, fmin, fmax, nperseg, noverlap, nfft)
            feat = intensity[:1000] if len(intensity) > 1000 else np.pad(intensity, (0, 1000 - len(intensity)))

        if bead_num not in per_bead:
            per_bead[bead_num] = {"features": [], "signals": [], "files": []}

        per_bead[bead_num]["features"].append(feat)
        per_bead[bead_num]["signals"].append(norm if clustering_mode == "Time Domain" else intensity)
        per_bead[bead_num]["files"].append(bead["file"])

    clustering_results = {}

    for bead_num, data in per_bead.items():
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(data["features"])
        clustering_results[bead_num] = {
            "labels": labels,
            "signals": data["signals"],
            "files": data["files"]
        }

    st.session_state["results"] = clustering_results
    st.success("Clustering per bead number complete!")

if "results" in st.session_state:
    st.write("## Visualization Results")
    results = st.session_state["results"]
    colors = ["red", "blue", "green", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"]

    for bead_num, data in results.items():
        fig = go.Figure()
        for i, signal in enumerate(data["signals"]):
            label = data["labels"][i]
            file = data["files"][i]
            fig.add_trace(go.Scatter(
                y=signal,
                mode="lines",
                name=f"Cluster {label}: {file}",
                line=dict(color=colors[label % len(colors)], width=1),
                hoverinfo="text",
                text=f"File: {file}<br>Cluster: {label}"
            ))
        fig.update_layout(
            title=f"Bead {bead_num} - Clustered ({clustering_mode})",
            xaxis_title="Time Index" if clustering_mode == "Time Domain" else "Time Frame",
            yaxis_title="Signal" if clustering_mode == "Time Domain" else "Intensity (dB)",
            showlegend=True
        )
        st.plotly_chart(fig)
