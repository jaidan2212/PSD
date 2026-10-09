# Tahap 3 - Eksperimen Clustering & Analisis Silhouette

Pada tahap ini, kita akan melakukan eksperimen clustering secara ekstensif menggunakan algoritma **K-Means**. Tujuan utama dari tahap ini adalah untuk membandingkan performa dari 8 representasi data hasil reduksi dimensi (dari Tahap 2) dan menentukan konfigurasi terbaik berdasarkan metrik evaluasi yang valid.

## Konsep Utama

Untuk mengevaluasi kualitas cluster yang terbentuk, kita menggunakan dua konsep dan metrik utama:

**1. K-Means dan Within-Cluster Sum of Squares (WCSS)**
Algoritma K-Means bertujuan meminimalkan varians di dalam masing-masing cluster. Metrik yang dioptimalkan oleh K-Means disebut sebagai *Inertia* atau *Within-Cluster Sum of Squares (WCSS)*, yang dirumuskan sebagai:

$$ WCSS = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2 $$

di mana:
- $k$ adalah jumlah cluster,
- $C_i$ adalah himpunan titik data dalam cluster ke-$i$,
- $x$ adalah titik data, dan
- $\mu_i$ adalah centroid dari cluster ke-$i$.

**2. Silhouette Coefficient**
Meskipun WCSS berguna (sering digunakan pada *Elbow Method*), nilai ini akan selalu menurun saat jumlah cluster ($k$) bertambah. Oleh karena itu, kita menggunakan **Silhouette Score** sebagai metrik evaluasi utama. Silhouette mengukur seberapa dekat setiap titik dengan cluster-nya sendiri dibandingkan dengan cluster tetangga terdekat. Rumusnya adalah:

$$ s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))} $$

di mana:
- $a(i)$ adalah rata-rata jarak antara titik $i$ dengan semua titik lain dalam cluster yang sama (kepadatan/kohesi),
- $b(i)$ adalah rata-rata jarak minimum antara titik $i$ dengan semua titik pada cluster terdekat (separasi),
- Nilai $s(i)$ berkisar dari -1 hingga 1. Nilai mendekati 1 menunjukkan clustering yang sangat baik.

## Desain Eksperimen

Eksperimen dirancang agar komprehensif dengan membandingkan seluruh dataset dan jumlah cluster.

| Parameter | Deskripsi |
| :--- | :--- |
| **Dataset** | 8 representasi (Linear/Poly pada dimensi 204, 203, 74, 37) |
| **Jumlah Cluster ($k$)** | $k = 2, 3, \dots, 10$ |
| **Algoritma** | K-Means |
| **Parameter K-Means** | `n_init=20`, `random_state=42` |
| **Total Skenario (Run)** | 8 dataset $\times$ 9 nilai $k$ = 72 run eksperimen |
| **Metrik Utama** | Silhouette Score |

---

## 3.1 Import & Load Data Tahap 2

```python
import warnings
warnings.filterwarnings('ignore')
import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from sklearn.decomposition import PCA as PCA_viz

OUTPUT_DIR = 'output'
plt.rcParams['figure.figsize'] = (13, 5)
sns.set_theme(style='whitegrid', font_scale=1.05)

# ==========================================
# 3.1 LOAD DATA TAHAP 2
# ==========================================
print("=== 3.1 Memuat Data Tahap 2 ===")
with open(f'{OUTPUT_DIR}/tahap2_pca.pkl', 'rb') as f:
    d = pickle.load(f)

representasi = d['representasi']
meta_linear, meta_poly = d['meta_linear'], d['meta_poly']
X_linear, X_poly = d['X_linear'], d['X_poly']
print(f"Data dimuat. Total representasi: {len(representasi)}")
```

```text
=== 3.1 Memuat Data Tahap 2 ===
Data dimuat. Total representasi: 8
```

---

## 3.2 Jalankan 72 Eksperimen Clustering

Pada tahap ini, model K-Means dilatih sebanyak 72 kali untuk mengevaluasi seluruh skenario yang telah dirancang.

```python
# ==========================================
# 3.2 JALANKAN 72 EKSPERIMEN CLUSTERING
# ==========================================
print("\n=== 3.2 Menjalankan Eksperimen Clustering ===")
K_RANGE = range(2, 11)
N_INIT = 20
RANDOM_STATE = 42

hasil, labels_map = [], {}
for nama_data, X in representasi.items():
    best_sil, best_k = -1, 2
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=N_INIT, init='k-means++', random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        
        # Urutkan label cluster agar konsisten (Cluster 0 selalu yang terbanyak)
        urut = np.argsort(-np.bincount(labels, minlength=k), kind='stable')
        remap = np.empty(k, dtype=int); remap[urut] = np.arange(k)
        labels = remap[labels]
        
        sil = silhouette_score(X, labels)
        iner = km.inertia_
        labels_map[(nama_data, k)] = labels
        hasil.append({'dataset': nama_data, 'k': k, 'silhouette': round(sil, 6), 'inertia': round(iner, 4)})
        if sil > best_sil: best_sil, best_k = sil, k
    print(f' {nama_data:<15} k_terbaik = {best_k} silhouette = {best_sil:.4f}')

df_hasil = pd.DataFrame(hasil)
```

```text
=== 3.2 Menjalankan Eksperimen Clustering ===
 Linear-204      k_terbaik = 2 silhouette = 0.4909
 Linear-203      k_terbaik = 2 silhouette = 0.4909
 Linear-74       k_terbaik = 2 silhouette = 0.4021
 Linear-37       k_terbaik = 2 silhouette = 0.4021
 Poly-204        k_terbaik = 2 silhouette = 0.5375
 Poly-203        k_terbaik = 2 silhouette = 0.5375
 Poly-74         k_terbaik = 3 silhouette = 0.4543
 Poly-37         k_terbaik = 3 silhouette = 0.4543
```

---

## 3.3 Visualisasi Silhouette Score vs k (semua dataset)

Grafik garis (line chart) ini memvisualisasikan bagaimana skor Silhouette berubah seiring dengan bertambahnya jumlah cluster ($k$) pada masing-masing dari 8 dataset.

```python
print("\nMembuat 8 visualisasi grafik...")
PALETTE = {'Linear-204': '#0D47A1', 'Linear-203': '#1565C0', 'Linear-74': '#1976D2', 'Linear-37': '#90CAF9',
           'Poly-204': '#B71C1C', 'Poly-203': '#C62828', 'Poly-74': '#E53935', 'Poly-37': '#FFCDD2'}
CLUSTER_COLORS = ['#E53935','#1E88E5', '#43A047', '#FB8C00', '#8E24AA']

# 1. Grafik Silhouette Semua
fig, axes = plt.subplots(2, 4, figsize=(22, 11))
for idx, nama_data in enumerate(representasi):
    sub = df_hasil[df_hasil['dataset'] == nama_data]
    color = PALETTE.get(nama_data, '#555')
    ax = axes.flatten()[idx]
    ax.plot(sub['k'], sub['silhouette'], 'o-', color=color, linewidth=2.5, markersize=7)
    best = sub.loc[sub['silhouette'].idxmax()]
    ax.scatter(best['k'], best['silhouette'], s=280, color='gold', edgecolors=color, linewidth=2, zorder=5, marker='*')
    ax.set_title(nama_data, fontweight='bold')
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/07_silhouette_semua.png', dpi=150); plt.close()
```

![Silhouette Score vs k](output/07_silhouette_semua.png)

---

## 3.4 Tabel Perbandingan & Pemilihan Konfigurasi Terbaik

Berdasarkan seluruh hasil eksperimen, data diekstraksi ke dalam bentuk tabel untuk memudahkan perbandingan antar konfigurasi.

```python
# ==========================================
# 3.4 PEMILIHAN KONFIGURASI TERBAIK
# ==========================================
best_row = df_hasil.loc[df_hasil['silhouette'].idxmax()]
BEST_DATASET = best_row['dataset']
BEST_K = int(best_row['k'])
BEST_SIL = best_row['silhouette']
BEST_LABELS = labels_map[(BEST_DATASET, BEST_K)]

print(f'\n>> TERBAIK GLOBAL: {BEST_DATASET} | k={BEST_K} | Sil={BEST_SIL:.4f}')
```

```text
>> TERBAIK GLOBAL: Poly-204 | k=2 | Sil=0.5375
```

---

## 3.4b Cluster Terbaik untuk Setiap Dimensi (204 -> 203 -> 74 -> 37)

Visualisasi ini menunjukkan perbandingan skor Silhouette terbaik pada masing-masing tingkatan reduksi dimensi.

```python
# ==========================================
# 3.4b CLUSTER TERBAIK PER DIMENSI
# ==========================================
DIMENSI = [204, 203, 74, 37]
METODE = ['Linear', 'Poly']
BEST_PER_DIM = {}

for metode in METODE:
    for dim in DIMENSI:
        nama = f'{metode}-{dim}'
        if nama not in representasi: continue
        sub = df_hasil[df_hasil['dataset'] == nama]
        row = sub.loc[sub['silhouette'].idxmax()]
        k = int(row['k'])
        lab = labels_map[(nama, k)]
        BEST_PER_DIM[(metode, dim)] = {
            'dataset': nama, 'metode': metode, 'dimensi': dim, 'k': k,
            'silhouette': float(row['silhouette']), 'inertia': float(row['inertia']),
            'labels': lab, 'ukuran': np.bincount(lab, minlength=k).tolist()
        }

# 2. Grafik Silhouette Per Dimensi
fig, ax = plt.subplots(figsize=(11, 5))
for metode, color in [('Linear', '#1565C0'), ('Poly', '#B71C1C')]:
    pts = [(i, BEST_PER_DIM[(metode, d)]) for i, d in enumerate(DIMENSI)]
    ax.plot([i for i, p in pts], [p['silhouette'] for i, p in pts], 'o-', color=color, lw=2.5, label=metode)
ax.set_xticks(range(4)); ax.set_xticklabels([f'{d}\ndim' for d in DIMENSI])
ax.legend(); plt.savefig(f'{OUTPUT_DIR}/07c_silhouette_per_dimensi.png', dpi=150); plt.close()
```

![Cluster Terbaik per Dimensi](output/07c_silhouette_per_dimensi.png)

---

## 3.5 Heatmap Perbandingan Silhouette - Semua Dimensi

Heatmap menyajikan pandangan yang lebih terpusat dan mempermudah kita melihat performa setiap kombinasi dataset dan jumlah cluster melalui intensitas warna.

```python
# 3. Heatmap
pivot = df_hasil.pivot(index='dataset', columns='k', values='silhouette')
urutan = ['Linear-204', 'Linear-203', 'Linear-74', 'Linear-37', 'Poly-204', 'Poly-203', 'Poly-74', 'Poly-37']
pivot = pivot.reindex(urutan)
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.3, ax=ax)
plt.savefig(f'{OUTPUT_DIR}/07b_heatmap_silhouette.png', dpi=150); plt.close()
```

![Heatmap Perbandingan Silhouette](output/07b_heatmap_silhouette.png)

---

## 3.6 Silhouette Plot - Konfigurasi Terbaik

Untuk konfigurasi terbaik (Poly-204, $k=2$), kita menggambar plot profil Silhouette secara detail untuk memvalidasi kepadatan dan pemisahan per cluster.

```python
# 4. Silhouette Plot Terbaik
X_best = representasi[BEST_DATASET]
sil_vals = silhouette_samples(X_best, BEST_LABELS)
fig, ax = plt.subplots(figsize=(10, 6))
y_lower = 10
for i in range(BEST_K):
    v = np.sort(sil_vals[BEST_LABELS == i])
    size_i = len(v)
    y_upper = y_lower + size_i
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, v, facecolor=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], alpha=0.8)
    y_lower = y_upper + 5
ax.axvline(BEST_SIL, color='crimson', ls='--')
plt.savefig(f'{OUTPUT_DIR}/08_silhouette_plot.png', dpi=150); plt.close()
```

![Silhouette Plot](output/08_silhouette_plot.png)

---

## 3.7 Elbow Method - Konfirmasi k Terbaik

Metode Elbow (berdasarkan perhitungan nilai *Inertia*/WCSS) di-plot untuk mengonfirmasi pilihan $k=2$ sebagai titik siku-siku dari model K-Means.

```python
# 5. Elbow Plot
sub_best = df_hasil[df_hasil['dataset'] == BEST_DATASET]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sub_best['k'], sub_best['inertia'], 'o-', color='#1565C0', lw=2.5)
ax.axvline(BEST_K, color='crimson', ls='--')
plt.savefig(f'{OUTPUT_DIR}/09_elbow_silhouette.png', dpi=150); plt.close()
```

![Elbow Method](output/09_elbow_silhouette.png)

---

## 3.8 Distribusi Cluster & Scatter PCA 2D

Langkah terakhir adalah memproyeksikan hasil clustering terbaik ke dalam 2 dimensi (menggunakan PCA dari representasi terbaik) agar sebaran titik-titik antar cluster dapat dilihat secara spasial.

```python
# 6. Scatter PCA 2D
pca2d = PCA_viz(n_components=2, random_state=42)
X_viz = pca2d.fit_transform(X_best)
fig, ax = plt.subplots(figsize=(8, 6))
for c in range(BEST_K):
    mask = BEST_LABELS == c
    ax.scatter(X_viz[mask, 0], X_viz[mask, 1], s=100, color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], label=f'Cluster {c}')
ax.legend()
plt.savefig(f'{OUTPUT_DIR}/10_scatter_cluster.png', dpi=150); plt.close()

# ==========================================
# 3.10 SIMPAN DATA UNTUK TAHAP 4
# ==========================================
save_data = {
    'BEST_DATASET': BEST_DATASET, 'BEST_K': BEST_K, 'BEST_SIL': BEST_SIL,
    'BEST_LABELS': BEST_LABELS, 'CLUSTER_COLORS': CLUSTER_COLORS,
    'meta_linear': meta_linear, 'meta_poly': meta_poly,
    'X_linear': X_linear, 'X_poly': X_poly, 'df_hasil': df_hasil,
    'BEST_PER_DIM': BEST_PER_DIM, 'DIMENSI': DIMENSI
}

save_path = f'{OUTPUT_DIR}/tahap3_clustering.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(save_data, f)
```

```text
Membuat 8 visualisasi grafik...
Semua grafik tersimpan di folder 'output'!
Hasil Tahap 3 disimpan di: output/tahap3_clustering.pkl
-> Lanjut ke Tahap 4: Visualisasi Peta Segmentasi
```
