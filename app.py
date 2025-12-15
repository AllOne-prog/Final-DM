import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# =================================
# PAGE CONFIG
# =================================
st.set_page_config(
    page_title="Ensemble & Meta-Ensemble Clustering",
    layout="wide"
)

# =================================
# LOAD DATA
# =================================
df = pd.read_csv("hasil_clustering.csv")

# =================================
# TITLE
# =================================
st.title("🚗 Ensemble & Meta-Ensemble Clustering Dataset Mobil")
st.markdown(
    """
    Aplikasi ini menampilkan hasil **clustering**, **meta-ensemble Logistic Regression**,
    serta **estimasi Spending Score** berdasarkan karakteristik data.
    """
)

# =================================
# PREVIEW DATASET
# =================================
st.subheader("📄 Preview Dataset")
st.dataframe(df.head())

# =================================
# SEARCH BY USER ID
# =================================
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

# =================================
# INPUT MANUAL + SPENDING SCORE
# =================================
# =================================
# INPUT MANUAL + INTERPRETASI CLUSTER
# =================================
st.subheader("🧠 Input Manual & Interpretasi Segment Pembeli")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=30)

with col2:
    salary = st.number_input("Annual Salary", min_value=0, value=50000)

with col3:
    purchased = st.selectbox("Purchased", [0, 1])

if st.button("Analisis Segment"):
    # Cari data terdekat
    features = df[["Age", "AnnualSalary", "Purchased"]]
    input_vector = np.array([[age, salary, purchased]])

    distances = euclidean_distances(features, input_vector)
    nearest_index = distances.argmin()
    nearest_data = df.iloc[nearest_index]

    cluster_final = nearest_data["Cluster_LogReg"]

    st.success("Hasil Segmentasi Pengguna")

    st.write("🔹 Cluster Final:", cluster_final)

    # ===============================
    # KESIMPULAN BERDASARKAN CLUSTER
    # ===============================
    if cluster_final == 0:
        conclusion = (
            "Pengguna termasuk dalam segmen pembeli dengan pendapatan rendah hingga menengah "
            "dan kecenderungan pembelian yang relatif rendah. Segmen ini cocok untuk ditargetkan "
            "mobil ekonomis atau city car dengan fokus pada harga terjangkau dan efisiensi."
        )
    elif cluster_final == 1:
        conclusion = (
            "Pengguna termasuk dalam segmen pembeli dengan pendapatan menengah dan usia produktif. "
            "Segmen ini memiliki potensi pembelian yang cukup baik dan cocok untuk ditargetkan "
            "mobil keluarga atau kendaraan kelas menengah dengan fokus pada kenyamanan dan fitur."
        )
    else:
        conclusion = (
            "Pengguna termasuk dalam segmen pembeli dengan pendapatan tinggi dan potensi pembelian besar. "
            "Segmen ini cocok untuk ditargetkan kendaraan SUV atau mobil premium dengan fokus pada "
            "performa, kenyamanan, dan prestise."
        )

    st.info("📌 Kesimpulan:")
    st.write(conclusion)

    st.subheader("📍 Data Referensi Terdekat")
    st.dataframe(nearest_data.to_frame().T)

    # ---------------------------------
    # CARI DATA TERDEKAT
    # ---------------------------------
    features = df[["Age", "AnnualSalary", "Purchased"]]
    input_vector = np.array([[age, salary, purchased]])

    distances = euclidean_distances(features, input_vector)
    nearest_index = distances.argmin()
    nearest_data = df.iloc[nearest_index]

    cluster_ensemble = nearest_data["Cluster"]
    cluster_final = nearest_data["Cluster_LogReg"]

    # ---------------------------------
    # HITUNG SPENDING SCORE (SINTETIS)
    # ---------------------------------
    salary_norm = min(salary / df["AnnualSalary"].max(), 1)

    cluster_weight = {
        0: 0.3,  # low spender
        1: 0.6,  # medium spender
        2: 0.9   # high spender
    }.get(cluster_final, 0.5)

    spending_score = (
        salary_norm * 0.6 +
        purchased * 0.3 +
        cluster_weight * 0.1
    ) * 100

    spending_score = round(spending_score, 2)

    # ---------------------------------
    # OUTPUT
    # ---------------------------------
    st.success("✅ Hasil Prediksi")

    st.write("🔹 Cluster (Ensemble):", cluster_ensemble)
    st.write("🔹 Cluster Final (Logistic Regression):", cluster_final)
    st.write("🔥 Prediksi Spending Score:", spending_score)

    # ---------------------------------
    # KESIMPULAN OTOMATIS
    # ---------------------------------
    if spending_score < 40:
        conclusion = "Pengguna diprediksi memiliki tingkat pengeluaran RENDAH."
    elif spending_score < 70:
        conclusion = "Pengguna diprediksi memiliki tingkat pengeluaran MENENGAH."
    else:
        conclusion = "Pengguna diprediksi memiliki tingkat pengeluaran TINGGI."

    st.info("📌 Kesimpulan: " + conclusion)

    # ---------------------------------
    # DATA REFERENSI
    # ---------------------------------
    st.subheader("📍 Data Referensi Terdekat")
    st.dataframe(nearest_data.to_frame().T)

# =================================
# VISUALISASI PCA
# =================================
st.subheader("📊 Visualisasi Cluster (PCA)")

fig, ax = plt.subplots(figsize=(7, 5))
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

# =================================
# DISTRIBUSI CLUSTER
# =================================
st.subheader("📈 Jumlah Data per Cluster")
st.bar_chart(df["Cluster_LogReg"].value_counts())
