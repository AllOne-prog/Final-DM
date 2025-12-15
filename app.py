import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

st.set_page_config(
    page_title="Ensemble & Meta-Ensemble Clustering",
    layout="wide"
)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("hasil_clustering.csv")

st.title("🚗 Ensemble & Meta-Ensemble Clustering Dataset Mobil")

# =========================
# PREVIEW DATASET
# =========================
st.subheader("Preview Dataset")
st.dataframe(df.head())

# =========================
# SEARCH BY USER ID
# =========================
st.subheader("🔍 Cari Hasil Clustering Berdasarkan User ID")

user_id_input = st.number_input(
    "Masukkan User ID",
    min_value=int(df["User ID"].min()),
    max_value=int(df["User ID"].max()),
    step=1
)

if st.button("Cari User"):
    result = df[df["User ID"] == user_id_input]
    if len(result) > 0:
        st.success("Data ditemukan")
        st.dataframe(result)
    else:
        st.warning("User ID tidak ditemukan")

# =========================
# INPUT MANUAL (SIMULASI REGRESI + CLUSTER)
# =========================
st.subheader("🧠 Input Manual (Prediksi Cluster Terdekat)")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=30)

with col2:
    salary = st.number_input("Annual Salary", min_value=0, value=50000)

with col3:
    purchased = st.selectbox("Purchased", [0, 1])

if st.button("Prediksi Cluster"):
    # gunakan PCA space untuk cari data terdekat
    features = df[["Age", "AnnualSalary", "Purchased"]]

    input_vector = np.array([[age, salary, purchased]])
    distances = euclidean_distances(features, input_vector)

    nearest_index = distances.argmin()
    nearest_data = df.iloc[nearest_index]

    st.success("Prediksi berdasarkan data terdekat")
    st.write("Cluster (Ensemble):", nearest_data["Cluster"])
    st.write("Cluster Final (Logistic Regression):", nearest_data["Cluster_LogReg"])

    st.info("Data referensi terdekat:")
    st.dataframe(nearest_data.to_frame().T)

# =========================
# VISUALISASI PCA
# =========================
st.subheader("📊 Visualisasi Cluster (PCA)")

fig, ax = plt.subplots(figsize=(7,5))
scatter = ax.scatter(
    df["PCA1"],
    df["PCA2"],
    c=df["Cluster_LogReg"],
    cmap="viridis",
    alpha=0.8
)
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_title("PCA Visualization - Meta Ensemble Clustering")
st.pyplot(fig)

# =========================
# DISTRIBUSI CLUSTER
# =========================
st.subheader("📈 Jumlah Data per Cluster")
st.bar_chart(df["Cluster_LogReg"].value_counts())
