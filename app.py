import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Ensemble Clustering Mobil", layout="wide")

st.title("🚗 Ensemble Clustering Dataset Mobil")

df = pd.read_csv("hasil_clustering.csv")

st.subheader("Preview Dataset")
st.dataframe(df.head())

st.subheader("Visualisasi Cluster (PCA)")
fig, ax = plt.subplots()
scatter = ax.scatter(
    df['PCA1'], df['PCA2'],
    c=df['Cluster_LogReg'], cmap='viridis'
)
ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
st.pyplot(fig)

st.subheader("Jumlah Data per Cluster")
st.bar_chart(df['Cluster_LogReg'].value_counts())
