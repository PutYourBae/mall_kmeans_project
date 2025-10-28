# 🧠 Penerapan Algoritma K-Means untuk Segmentasi Pelanggan pada Data Transaksi Penjualan

## 📋 Deskripsi Proyek
Proyek ini bertujuan untuk menerapkan algoritma **K-Means Clustering** dalam melakukan **segmentasi pelanggan (customer segmentation)** berdasarkan data demografis dan perilaku belanja pada dataset **Mall Customers**.

Tujuan akhirnya adalah mengelompokkan pelanggan ke dalam beberapa segmen dengan karakteristik serupa untuk membantu pengambilan keputusan pemasaran yang lebih efektif.

---

## 🧰 Teknologi yang Digunakan
- **Python 3.8+**
- **Visual Studio Code**
- **Library Utama:**
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

---

## 📁 Struktur Folder
```bash
mall_kmeans_project/
├─ data/
│ └─ Mall_Customers.csv ← Dataset
├─ outputs/
│ ├─ elbow.png ← Grafik metode Elbow
│ ├─ silhouette_scores.txt ← Nilai Silhouette tiap k
│ ├─ cluster_plot.png ← Visualisasi hasil klaster
│ ├─ customers_with_cluster.csv ← Data + label klaster
│ └─ cluster_summary.csv ← Statistik tiap klaster
├─ main.py ← Script utama
├─ requirements.txt ← Library yang dibutuhkan
└─ README.md ← Dokumentasi proyek
```
---

## ⚙️ Cara Menjalankan Proyek

### 1️⃣ Clone / Salin Repository

```bash
git clone https://github.com/<username>/mall_kmeans_project.git
cd mall_kmeans_project
```
### 2️⃣ Buat Virtual Environment
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
# atau
source .venv/bin/activate  # Mac / Linux
```
### 3️⃣ Install Dependency
```bash
pip install -r requirements.txt
```
### 4️⃣ Jalankan Script Utama
```bash
python main.py
```
### 5️⃣ Lihat Hasil di Folder outputs/
```bash
- 📉 elbow.png → Menentukan jumlah cluster optimal

- 🧩 silhouette_scores.txt → Nilai evaluasi tiap k

- 🎨 cluster_plot.png → Visualisasi hasil klaster

- 📊 customers_with_cluster.csv → Data pelanggan + label klaster

- 📈 cluster_summary.csv → Statistik per klaster
```
---
## 📊 Hasil Singkat

- Jumlah klaster optimal ditentukan menggunakan metode Elbow dan Silhouette Score.

- Dataset berhasil dibagi menjadi beberapa segmen pelanggan dengan karakteristik berbeda berdasarkan usia, pendapatan, dan skor pengeluaran.

- Visualisasi hasil dapat dilihat pada file outputs/cluster_plot.png.
---
### 🧩 Metodologi (CRISP-DM)
---
1. Business Understanding – memahami kebutuhan bisnis dalam membedakan segmen pelanggan.

2. Data Understanding – eksplorasi dataset Mall Customers.

3. Data Preparation – pemilihan dan normalisasi fitur (Age, Income, Spending Score).

4. Modeling – penerapan algoritma K-Means untuk menemukan kelompok pelanggan.

5. Evaluation – menentukan jumlah cluster optimal dengan Elbow dan Silhouette.

6. Deployment – visualisasi hasil dan penyimpanan data ke CSV.
---
### 📈 Contoh Visualisasi
---
🔹 Metode Elbow

🔹 Hasil Klaster Pelanggan

---
### ✍️ Pengembang
---
Nama: Agustian Putra Sukarya

NIM: 17230804

Mata Kuliah: Machine Learning

---
### 📚 Referensi
---
Han, J., Kamber, M., & Pei, J. (2012). Data Mining: Concepts and Techniques.

Scikit-learn Documentation: https://scikit-learn.org

Dataset: Mall Customers Dataset – Kaggle / GitHub Mirror.