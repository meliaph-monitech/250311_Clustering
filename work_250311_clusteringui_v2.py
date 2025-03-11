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

def extract_advanced_features(signal):
    n = len(signal)
    if n == 0:
        return [0] * 17
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    skewness = skew(signal)
    kurt = kurtosis(signal)
    peak_to_peak = max_val - min_val
    energy = np.sum(signal**2)
    rms = np.sqrt(np.mean(signal**2))
    return [mean_val, std_val, min_val, max_val, skewness, kurt, peak_to_peak, energy, rms]

st.set_page_config(layout="wide")
st.title("Laser Welding K-Means Clustering")

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
    num_clusters = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
    if st.sidebar.button("Run K-Means Clustering"):
        with st.spinner("Running K-Means Clustering..."):
            features_by_file = []
            file_names = []
            for file in csv_files:
                df = pd.read_csv(file)
                features = extract_advanced_features(df[filter_column].values)
                features_by_file.append(features)
                file_names.append(file)
            
            scaler = RobustScaler()
            scaled_features = scaler.fit_transform(features_by_file)
            
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            
            results_df = pd.DataFrame({
                "File Name": file_names,
                "Cluster": clusters
            })
            st.session_state["clustering_results"] = results_df
            
            pca = PCA(n_components=2)
            reduced_features = pca.fit_transform(scaled_features)
            cluster_df = pd.DataFrame({
                "PC1": reduced_features[:, 0],
                "PC2": reduced_features[:, 1],
                "Cluster": clusters
            })
            
            st.subheader("K-Means Clustering Visualization")
            fig = px.scatter(cluster_df, x="PC1", y="PC2", color=cluster_df["Cluster"].astype(str),
                             title="K-Means Clustering Visualization", labels={"color": "Cluster"})
            st.plotly_chart(fig)

if "clustering_results" in st.session_state:
    csv_data = st.session_state["clustering_results"].to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv_data, file_name="clustering_results.csv", mime="text/csv")
