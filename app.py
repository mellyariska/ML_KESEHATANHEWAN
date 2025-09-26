import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# =====================
# Load Data
# =====================
@st.cache_data
def load_data():
    df = pd.read_excel("data_mencit_tendik_alfinsuhanda.xlsx") # langsung baca file lokal
    return df

df = load_data()

# =====================
# Dashboard Title
# =====================
st.set_page_config(page_title="Dashboard Kesehatan Tikus", layout="wide")
st.title("📊 Dashboard Analisis Data Kesehatan Tikus")

st.sidebar.header("Navigasi")
menu = st.sidebar.radio("Pilih Visualisasi:", [
    "Tabel Data",
    "Boxplot & Violin",
    "Scatter Plot",
    "Tren Berat Badan",
    "Heatmap Korelasi",
    "PCA 2D",
    "t-SNE 2D"
])

# =====================
# Visualisasi
# =====================

if menu == "Tabel Data":
    st.subheader("📋 Dataset Kesehatan Tikus")
    st.dataframe(df)

elif menu == "Boxplot & Violin":
    st.subheader("📦 Boxplot & 🎻 Violin Plot")
    numeric_cols = ["Suhu (°C)", "Kelembaban (%)", "CO₂ (ppm)", 
                    "Amonia (ppm)", "Berat (g)", 
                    "Denyut Jantung (bpm)", "Aktivitas (unit)"]
    
    for col in numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Boxplot {col}")
            fig, ax = plt.subplots()
            sns.boxplot(x="Label Kesehatan", y=col, data=df, ax=ax)
            st.pyplot(fig)
        with col2:
            st.write(f"Violin Plot {col}")
            fig, ax = plt.subplots()
            sns.violinplot(x="Label Kesehatan", y=col, data=df, ax=ax)
            st.pyplot(fig)

elif menu == "Scatter Plot":
    st.subheader("Scatter Plot Berat vs Denyut Jantung")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Berat (g)", y="Denyut Jantung (bpm)", hue="Label Kesehatan", ax=ax)
    st.pyplot(fig)

elif menu == "Tren Berat Badan":
    st.subheader("📈 Tren Berat Badan Tikus per Hari")
    fig, ax = plt.subplots(figsize=(10,6))
    for tikus_id, subset in df.groupby("ID_Tikus"):
        ax.plot(subset["Hari"], subset["Berat (g)"], alpha=0.7)
    ax.set_xlabel("Hari")
    ax.set_ylabel("Berat (g)")
    ax.set_title("Tren Berat Badan Tikus per Hari")
    st.pyplot(fig)

elif menu == "Heatmap Korelasi":
    st.subheader("🔥 Heatmap Korelasi Fitur")
    fig, ax = plt.subplots(figsize=(10,6))
    corr = df.select_dtypes(include="number").corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

elif menu == "PCA 2D":
    st.subheader("🧭 PCA 2D Visualisasi Fitur")
    features = ["Suhu (°C)", "Kelembaban (%)", "CO₂ (ppm)", 
                "Amonia (ppm)", "Berat (g)", 
                "Denyut Jantung (bpm)", "Aktivitas (unit)"]
    X = df[features]
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(pcs, columns=["PCA1", "PCA2"])
    df_pca["Label Kesehatan"] = df["Label Kesehatan"]
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="Label Kesehatan", ax=ax)
    st.pyplot(fig)

elif menu == "t-SNE 2D":
    st.subheader("🌐 t-SNE 2D Visualisasi Fitur")
    features = ["Suhu (°C)", "Kelembaban (%)", "CO₂ (ppm)", 
                "Amonia (ppm)", "Berat (g)", 
                "Denyut Jantung (bpm)", "Aktivitas (unit)"]
    X = df[features]
    X_scaled = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_results = tsne.fit_transform(X_scaled)
    df_tsne = pd.DataFrame(tsne_results, columns=["tSNE1", "tSNE2"])
    df_tsne["Label Kesehatan"] = df["Label Kesehatan"]

    fig, ax = plt.subplots()
    sns.scatterplot(data=df_tsne, x="tSNE1", y="tSNE2", hue="Label Kesehatan", ax=ax)
    st.pyplot(fig)

