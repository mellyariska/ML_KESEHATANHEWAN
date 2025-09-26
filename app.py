# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib
import io

# Optional: SMOTE (if imblearn is installed)
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except Exception:
    SMOTE_AVAILABLE = False

sns.set(style="whitegrid")

# ------------------------------
# Utility functions
# ------------------------------
@st.cache_data
def load_data_from_excel(path):
    return pd.read_excel(path)

def compute_metrics(y_true, y_pred, average="weighted"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return acc, prec, rec, f1

def plot_grouped_bar(df_metrics):
    metrics = ["Akurasi", "Presisi", "Recall", "F1-Score"]
    labels = df_metrics["Algoritma"].values
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(9,5))
    for i, m in enumerate(metrics):
        ax.bar(x + i*width, df_metrics[m].values, width, label=m)

    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0,1)
    ax.set_ylabel("Nilai")
    ax.set_title("Perbandingan Performa Algoritma")
    ax.legend()
    st.pyplot(fig)

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ------------------------------
# Streamlit app layout
# ------------------------------
st.set_page_config(page_title="Dashboard Prediksi Kesehatan Hewan", layout="wide")
st.title("Dashboard Prediksi Kesehatan Hewan Percobaan (Machine Learning)")

# Sidebar: data upload / load default
st.sidebar.header("Data & Pengaturan")
uploaded_file = st.sidebar.file_uploader("Unggah file Excel (.xlsx) atau CSV (.csv)", type=["xlsx", "csv"])
use_default = st.sidebar.checkbox("Gunakan dataset contoh yang sudah tersedia (default path)", value=True)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"File {uploaded_file.name} berhasil dimuat.")
    except Exception as e:
        st.sidebar.error("Gagal membaca file: " + str(e))
        st.stop()
else:
    if use_default:
        default_path = "/mnt/data/data_mencit_tendik_alfinsuhanda.xlsx"
        try:
            df = load_data_from_excel(default_path)
            st.sidebar.info(f"Memuat file default: {default_path}")
        except Exception:
            st.sidebar.warning("File default tidak ditemukan. Silakan unggah file Anda.")
            st.stop()
    else:
        st.sidebar.info("Silakan unggah file data terlebih dahulu.")
        st.stop()

# Quick view of data
st.subheader("Preview Data")
st.dataframe(df.head())

# Ensure expected column names exist:
expected_cols = ["ID_Tikus", "Hari", "Suhu (°C)", "Kelembaban (%)", "CO₂ (ppm)",
                 "Amonia (ppm)", "Berat (g)", "Denyut Jantung (bpm)", "Aktivitas (unit)",
                 "Label Kesehatan"]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    st.error(f"Kolom berikut tidak ditemukan di dataset: {missing}. Silakan sesuaikan nama kolom atau unggah dataset yang benar.")
    st.stop()

# Sidebar: feature selection & preprocessing options
st.sidebar.subheader("Fitur dan Preprocessing")
default_features = ["Suhu (°C)", "Kelembaban (%)", "CO₂ (ppm)", "Amonia (ppm)", "Berat (g)", "Denyut Jantung (bpm)", "Aktivitas (unit)"]
features = st.sidebar.multiselect("Pilih fitur (X)", default_features, default=default_features)
label_col = st.sidebar.selectbox("Pilih kolom label (y)", options=df.columns.tolist(), index=df.columns.tolist().index("Label Kesehatan"))

scaler_choice = st.sidebar.selectbox("Skaler fitur", ["StandardScaler", "MinMaxScaler", "None"], index=0)
test_size = st.sidebar.slider("Proporsi test set", 0.05, 0.5, 0.2, step=0.05)
random_state = st.sidebar.number_input("Random state", value=42)

# Balancing option
st.sidebar.subheader("Penanganan Imbalance")
balancing = st.sidebar.selectbox("Pilih strategi penyeimbangan", ["Tidak", "Oversampling SMOTE (jika tersedia)"], index=0)
if balancing != "Tidak" and not SMOTE_AVAILABLE:
    st.sidebar.warning("SMOTE tidak tersedia di environment (imblearn belum diinstall). Pilih 'Tidak' atau install imblearn.")

# Sidebar: model hyperparameters minimal
st.sidebar.subheader("Model")
use_rf = st.sidebar.checkbox("Gunakan Random Forest", value=True)
use_svm = st.sidebar.checkbox("Gunakan SVM", value=True)
use_ann = st.sidebar.checkbox("Gunakan ANN (MLPClassifier)", value=True)

# Train button
if st.sidebar.button("Latih & Evaluasi Model"):

    # Prepare data
    X = df[features].copy()
    y = df[label_col].copy()

    # Encode label if not numeric
    if y.dtype == "object" or y.dtype.name == "category":
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        label_names = le.classes_
    else:
        y_enc = y.values
        label_names = np.unique(y)

    # Handle missing values (simple)
    if X.isnull().any().any():
        st.info("Menangani missing values dengan imputasi mean.")
        X = X.fillna(X.mean())

    # Scaling
    if scaler_choice == "StandardScaler":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaler_choice == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=test_size, random_state=int(random_state), stratify=y_enc)

    # Optional balancing
    if balancing == "Oversampling SMOTE" and SMOTE_AVAILABLE:
        sm = SMOTE(random_state=int(random_state))
        X_train, y_train = sm.fit_resample(X_train, y_train)
        st.success("SMOTE applied to training set.")

    models = {}
    results = []

    if use_rf:
        rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=int(random_state))
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        acc, prec, rec, f1 = compute_metrics(y_test, y_pred)
        results.append(["Random Forest", acc, prec, rec, f1])
        models["Random Forest"] = (rf, y_pred)

    if use_svm:
        svm = SVC(kernel="rbf", probability=True, random_state=int(random_state))
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        acc, prec, rec, f1 = compute_metrics(y_test, y_pred)
        results.append(["SVM", acc, prec, rec, f1])
        models["SVM"] = (svm, y_pred)

    if use_ann:
        ann = MLPClassifier(hidden_layer_sizes=(64,32,16), max_iter=500, random_state=int(random_state))
        ann.fit(X_train, y_train)
        y_pred = ann.predict(X_test)
        acc, prec, rec, f1 = compute_metrics(y_test, y_pred)
        results.append(["ANN", acc, prec, rec, f1])
        models["ANN"] = (ann, y_pred)

    # Results table
    df_results = pd.DataFrame(results, columns=["Algoritma", "Akurasi", "Presisi", "Recall", "F1-Score"])
    st.subheader("Perbandingan Performa Algoritma")
    st.dataframe(df_results.style.format({"Akurasi":"{:.3f}", "Presisi":"{:.3f}", "Recall":"{:.3f}", "F1-Score":"{:.3f}"}))

    # Plot grouped bar chart
    plot_grouped_bar(df_results)

    # Show best model confusion matrix & classification report
    best_idx = df_results["F1-Score"].idxmax()
    best_model_name = df_results.loc[best_idx, "Algoritma"]
    st.subheader(f"Analisis Detil Model Terbaik: {best_model_name}")
    best_model, best_y_pred = models[best_model_name]

    # Classification report
    report = classification_report(y_test, best_y_pred, target_names=[str(n) for n in label_names], zero_division=0)
    st.text("Classification Report:\n" + report)

    # Confusion matrix
    cm = confusion_matrix(y_test, best_y_pred)
    plot_confusion_matrix(cm, [str(n) for n in label_names])

    # Save model option
    save_opt = st.checkbox(f"Simpan model {best_model_name} ke file .pkl")
    if save_opt:
        buf = io.BytesIO()
        joblib.dump(best_model, buf)
        buf.seek(0)
        st.download_button(label=f"Download {best_model_name}.pkl", data=buf, file_name=f"{best_model_name}.pkl")

# ------------------------------
# Visualization tab
# ------------------------------
st.sidebar.subheader("Visualisasi Eksplorasi Data")
if st.sidebar.button("Tampilkan Semua Visualisasi"):
    st.header("Visualisasi Eksplorasi Data")

    # 1. Distribusi label
    st.subheader("Distribusi Label Kesehatan")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="Label Kesehatan", order=df["Label Kesehatan"].value_counts().index, palette="Set2", ax=ax)
    ax.set_title("Distribusi Label Kesehatan")
    ax.set_ylabel("Jumlah Tikus")
    ax.set_xlabel("Kategori Kesehatan")
    plt.xticks(rotation=25)
    st.pyplot(fig)

    # 2. Boxplot berat per label
    st.subheader("Boxplot Berat Badan per Label Kesehatan")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.boxplot(x="Label Kesehatan", y="Berat (g)", data=df, ax=ax)
    ax.set_title("Boxplot Berat Badan per Label Kesehatan")
    plt.xticks(rotation=25)
    st.pyplot(fig)

    # 3. Violin Plot denyut jantung per label
    st.subheader("Violin Plot Denyut Jantung per Label Kesehatan")
    fig, ax = plt.subplots(figsize=(8,4))
    sns.violinplot(x="Label Kesehatan", y="Denyut Jantung (bpm)", data=df, inner="quart", ax=ax)
    ax.set_title("Violin Plot Denyut Jantung per Label Kesehatan")
    plt.xticks(rotation=25)
    st.pyplot(fig)

    # 4. Scatter Berat vs Denyut Jantung
    st.subheader("Scatter Plot Berat vs Denyut Jantung")
    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(data=df, x="Berat (g)", y="Denyut Jantung (bpm)", hue="Label Kesehatan", alpha=0.7, ax=ax)
    ax.set_title("Scatter Plot Berat vs Denyut Jantung")
    st.pyplot(fig)

    # 5. Tren berat per hari (beberapa tikus contoh)
    st.subheader("Tren Berat Badan per Hari (Contoh beberapa tikus)")
    sample_ids = df["ID_Tikus"].unique()[:12]  # tampilkan 12 tikus pertama supaya tidak penuh
    fig, ax = plt.subplots(figsize=(10,6))
    for tid in sample_ids:
        dsub = df[df["ID_Tikus"] == tid]
        ax.plot(dsub["Hari"], dsub["Berat (g)"], label=tid, alpha=0.8)
    ax.set_xlabel("Hari")
    ax.set_ylabel("Berat (g)")
    ax.set_title("Tren Berat Badan Tikus per Hari (sample)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), ncol=1)
    st.pyplot(fig)

    # 6. Heatmap korelasi
    st.subheader("Heatmap Korelasi antar Fitur")
    fig, ax = plt.subplots(figsize=(8,6))
    corr = df[["Suhu (°C)", "Kelembaban (%)", "CO₂ (ppm)", "Amonia (ppm)", "Berat (g)", "Denyut Jantung (bpm)", "Aktivitas (unit)"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
    ax.set_title("Heatmap Korelasi antar Fitur")
    st.pyplot(fig)

    # 7. PCA 2D
    st.subheader("PCA 2D Visualisasi")
    features = ["Suhu (°C)", "Kelembaban (%)", "CO₂ (ppm)", "Amonia (ppm)", "Berat (g)", "Denyut Jantung (bpm)", "Aktivitas (unit)"]
    X = df[features].values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(7,6))
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df["Label Kesehatan"], alpha=0.7, ax=ax)
    ax.set_title("PCA 2D")
    st.pyplot(fig)

    # 8. t-SNE 2D
    st.subheader("t-SNE 2D Visualisasi")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    X_tsne = tsne.fit_transform(X_scaled)
    fig, ax = plt.subplots(figsize=(7,6))
    sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue=df["Label Kesehatan"], alpha=0.8, ax=ax)
    ax.set_title("t-SNE 2D")
    st.pyplot(fig)

st.sidebar.markdown("---")
st.sidebar.info("Setelah latihan model selesai, gunakan panel `Simpan model` untuk mengunduh file .pkl dan integrasikan ke dashboard operasional (mis. Streamlit atau API Flask).")
