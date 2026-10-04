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
# Kode Python dimasukkan di sini
```

---

## 3.2 Jalankan 72 Eksperimen Clustering

Pada tahap ini, model K-Means dilatih sebanyak 72 kali untuk mengevaluasi seluruh skenario yang telah dirancang.

```python
# Kode Python dimasukkan di sini
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

>> TERBAIK GLOBAL: Poly-204 | k=2 | Sil=0.5375
```

---

## 3.3 Visualisasi Silhouette Score vs k (semua dataset)

Grafik garis (line chart) ini memvisualisasikan bagaimana skor Silhouette berubah seiring dengan bertambahnya jumlah cluster ($k$) pada masing-masing dari 8 dataset.

```python
# Kode Python dimasukkan di sini
```

![Silhouette Score vs k](output/07_silhouette_semua.png)

---

## 3.4 Tabel Perbandingan & Pemilihan Konfigurasi Terbaik

Berdasarkan seluruh hasil eksperimen, data diekstraksi ke dalam bentuk tabel untuk memudahkan perbandingan antar konfigurasi.

```python
# Kode Python dimasukkan di sini
```

---

## 3.4b Cluster Terbaik untuk Setiap Dimensi (204 -> 203 -> 74 -> 37)

Visualisasi ini menunjukkan perbandingan skor Silhouette terbaik pada masing-masing tingkatan reduksi dimensi.

```python
# Kode Python dimasukkan di sini
```

![Cluster Terbaik per Dimensi](output/07c_silhouette_per_dimensi.png)

---

## 3.5 Heatmap Perbandingan Silhouette - Semua Dimensi

Heatmap menyajikan pandangan yang lebih terpusat dan mempermudah kita melihat performa setiap kombinasi dataset dan jumlah cluster melalui intensitas warna.

```python
# Kode Python dimasukkan di sini
```

![Heatmap Perbandingan Silhouette](output/07b_heatmap_silhouette.png)

---

## 3.6 Silhouette Plot - Konfigurasi Terbaik

Untuk konfigurasi terbaik (Poly-204, $k=2$), kita menggambar plot profil Silhouette secara detail untuk memvalidasi kepadatan dan pemisahan per cluster.

```python
# Kode Python dimasukkan di sini
```

![Silhouette Plot](output/08_silhouette_plot.png)

---

## 3.7 Elbow Method - Konfirmasi k Terbaik

Metode Elbow (berdasarkan perhitungan nilai *Inertia*/WCSS) di-plot untuk mengonfirmasi pilihan $k=2$ sebagai titik siku-siku dari model K-Means.

```python
# Kode Python dimasukkan di sini
```

![Elbow Method](output/09_elbow_silhouette.png)

---

## 3.8 Distribusi Cluster & Scatter PCA 2D

Langkah terakhir adalah memproyeksikan hasil clustering terbaik ke dalam 2 dimensi (menggunakan PCA dari representasi terbaik) agar sebaran titik-titik antar cluster dapat dilihat secara spasial.

```python
# Kode Python dimasukkan di sini
```

![Distribusi Cluster & Scatter PCA 2D](output/10_scatter_cluster.png)

```text
Membuat 8 visualisasi grafik...
Semua grafik tersimpan di folder 'output'!
Hasil Tahap 3 disimpan di: output/tahap3_clustering.pkl
-> Lanjut ke Tahap 4: Visualisasi Peta Segmentasi
```
