# main.py
"""
Pipeline K-Means Customer Segmentation
Usage: python main.py
Output:
 - outputs/elbow.png
 - outputs/silhouette_scores.txt
 - outputs/cluster_plot.png
 - outputs/customers_with_cluster.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ---------------------------
# Konfigurasi awal
# ---------------------------
DATA_PATH = "data/Mall_Customers.csv"
OUT_DIR = "outputs"
RANDOM_STATE = 42

# Buat folder jika belum ada
os.makedirs("data", exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# 1. Load Dataset
# ---------------------------
def load_data(path=DATA_PATH):
    """
    Mencoba memuat dataset lokal jika sudah ada.
    Jika belum, otomatis mengunduh dari URL GitHub mirror.
    """
    if os.path.exists(path):
        print(f"✅ Dataset ditemukan secara lokal di {path}")
        df = pd.read_csv(path)
    else:
        print("⚠️ Dataset lokal tidak ditemukan. Mencoba mengunduh dari URL GitHub...")
        url = "https://raw.githubusercontent.com/Ankit152/Customer-Segmentation-Tutorial/master/Mall_Customers.csv"
        try:
            df = pd.read_csv(url)
            print("✅ Dataset berhasil diunduh dari GitHub.")
            # Simpan salinannya agar tidak perlu download lagi
            df.to_csv(path, index=False)
            print(f"💾 Dataset disimpan ke {path}")
        except Exception as e:
            print("❌ Gagal mengunduh dataset. Pastikan koneksi internet aktif.")
            print("Error:", e)
            raise e
    return df

# ---------------------------
# 2. Quick EDA
# ---------------------------
def quick_eda(df):
    print("=== INFO ===")
    print(df.info())
    print("\n=== DESCRIBE ===")
    print(df.describe(include='all'))
    # Simple pairplot may be slow; we'll show head
    print("\n=== HEAD ===")
    print(df.head())

# ---------------------------
# 3. Preprocessing
# ---------------------------
def preprocess(df):
    # Keep relevant features
    # The dataset column names may vary; check them
    cols = [c for c in df.columns]
    print("Columns:", cols)
    # Typical columns: 'CustomerID','Gender','Age','Annual Income (k$)','Spending Score (1-100)'
    use_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    for c in use_cols:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' not found in dataset.")
    X = df[use_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled

# ---------------------------
# 4. Elbow method
# ---------------------------
def elbow_method(X_scaled, k_max=10):
    inertia = []
    K = range(1, k_max+1)
    for k in K:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X_scaled)
        inertia.append(km.inertia_)
    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(K, inertia, 'bo-', linewidth=2)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.xticks(K)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "elbow.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved elbow plot to {out_path}")
    plt.close()
    return K, inertia

# ---------------------------
# 5. Silhouette scores
# ---------------------------
def silhouette_scores(X_scaled, k_min=2, k_max=10):
    scores = {}
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores[k] = score
        print(f"k={k} -> silhouette={score:.4f}")
    # Save to file
    out_file = os.path.join(OUT_DIR, "silhouette_scores.txt")
    with open(out_file, "w") as f:
        for k, s in scores.items():
            f.write(f"k={k}: {s:.6f}\n")
    print(f"Saved silhouette scores to {out_file}")
    return scores

# ---------------------------
# 6. Fit final KMeans and save results
# ---------------------------
def fit_final_and_save(df_original, X_scaled, best_k):
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=50)
    labels = km.fit_predict(X_scaled)
    df_result = df_original.copy()
    df_result['Cluster'] = labels
    out_csv = os.path.join(OUT_DIR, "customers_with_cluster.csv")
    df_result.to_csv(out_csv, index=False)
    print(f"Saved cluster assignments to {out_csv}")
    return km, df_result

# ---------------------------
# 7. PCA visualization
# ---------------------------
def plot_clusters_pca(X_scaled, labels, best_k):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8,6))
    palette = sns.color_palette("viridis", best_k)
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=labels, palette=palette, legend='full', s=50)
    plt.title(f'Customer Segments (k={best_k}) - PCA projection')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(title='Cluster')
    plt.tight_layout()
    out_fig = os.path.join(OUT_DIR, "cluster_plot.png")
    plt.savefig(out_fig, dpi=150)
    print(f"Saved cluster plot to {out_fig}")
    plt.close()

# ---------------------------
# 8. Cluster interpretation
# ---------------------------
def cluster_summary(df_with_cluster):
    cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    summary = df_with_cluster.groupby('Cluster')[cols].agg(['count','mean','std', 'min','max'])
    print("\nCluster Summary:\n", summary)
    out_summary = os.path.join(OUT_DIR, "cluster_summary.csv")
    summary.to_csv(out_summary)
    print(f"Saved cluster summary to {out_summary}")

# ---------------------------
# Main
# ---------------------------
def main():
    df = load_data()
    quick_eda(df)
    X, X_scaled = preprocess(df)
    # Elbow
    elbow_method(X_scaled, k_max=10)
    # Silhouette
    scores = silhouette_scores(X_scaled, k_min=2, k_max=10)
    # Choose best k by max silhouette
    best_k = max(scores, key=scores.get)
    print(f"\n=> Best k by silhouette score: {best_k} (score={scores[best_k]:.4f})")
    # Fit final model & save
    km_model, df_result = fit_final_and_save(df, X_scaled, best_k)
    # Plot clusters (PCA)
    plot_clusters_pca(X_scaled, df_result['Cluster'], best_k)
    # Summary
    cluster_summary(df_result)
    print("\nAll done. Check the outputs/ folder for plots and CSVs.")

if __name__ == "__main__":
    main()
