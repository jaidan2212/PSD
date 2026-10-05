# Pemodelan K-Means & Visualisasi Peta Segmentasi

## Bagian 1: K-Means Clustering (Polynomial)

### Dokumentasi Workflow (KNIME)
Workflow di atas merupakan rancangan proses klasterisasi menggunakan algoritma K-Means yang membandingkan empat skenario utama berdasarkan pengolahan data **polynomial**: **tanpa reduksi dimensi**, dan **dengan reduksi dimensi (PCA)** menjadi 203 fitur, 74 fitur, dan 37 fitur. Masing-masing skenario diuji dengan jumlah klaster (k) sebanyak 3, 5, dan 7.

![Workflow KNIME](output/path_gambar_workflow_knime.png)

Berikut adalah penjelasan fungsi untuk setiap node yang digunakan:

1. **MySQL Connector** Berfungsi untuk membangun koneksi dari KNIME ke server database MySQL menggunakan kredensial (host, port, database, username, password) yang sesuai.
2. **DB Table Selector** Menyeleksi atau memilih tabel spesifik di dalam database MySQL yang berisi data mentah polinomial.
3. **DB Reader** Mengeksekusi query dari DB Table Selector dan menarik (import) data mentah tersebut dari database ke dalam memori/environment KNIME agar dapat diproses pada node-node selanjutnya.
4. **Column Filter** Menyeleksi fitur-fitur yang akan digunakan dalam pemodelan. Pada workflow ini, digunakan secara spesifik untuk **menghilangkan kolom ID** karena tidak memiliki nilai analitik untuk proses klasterisasi.
5. **PCA (Principal Component Analysis)** Teknik reduksi dimensi yang digunakan untuk menyederhanakan kompleksitas data berdimensi tinggi. Karena penggunaan fitur polynomial akan melipatgandakan jumlah kolom, PCA sangat krusial untuk mengatasi *curse of dimensionality*. Pada workflow ini, PCA dijalankan dalam tiga skenario reduksi:
   - Reduksi ke **203 Fitur**
   - Reduksi ke **74 Fitur**
   - Reduksi ke **37 Fitur**
   
   Tujuannya adalah mencari keseimbangan antara efisiensi komputasi dan retensi informasi (variansi) dari data asli.
6. **k-Means** Algoritma inti yang bertugas mengelompokkan data. Pada workflow ini, K-Means dijalankan untuk keempat variasi input data (tanpa PCA, PCA 203, PCA 74, PCA 37) dan masing-masing diuji pembentukan **k = 3, 5, dan 7 klaster**.
7. **Silhouette Coefficient** Metrik evaluasi yang digunakan untuk mengukur seberapa baik setiap titik data dikelompokkan ke dalam klasternya sendiri dibandingkan dengan klaster lain.
8. **Table View** Node visualisasi yang menampilkan nilai Silhouette Coefficient dalam format tabel interaktif. Berdasarkan gambar, rata-rata *Silhouette Coefficient* dicatatkan secara langsung di atas node Table View untuk memudahkan observasi.

### Hasil Evaluasi per Skenario

Pada bagian ini, evaluasi pembentukan klaster diukur menggunakan rata-rata *Silhouette Coefficient* pada masing-masing bereksperimen. Menariknya, hasil yang diperoleh sangat konsisten di setiap variasi PCA maupun tanpa PCA:

#### 1. Tanpa Reduksi Dimensi (PCA)

**A. Klastering 3 Kelas (k = 3)**
- **Silhouette Coefficient**: 0.471

*Tabel Silhouette Coefficient:*
![Table View Tanpa PCA k=3](output/path_gambar_table_tanpa_pca_k3.png)

*Visualisasi Scatter Plot:*
![Scatter Plot Tanpa PCA k=3](output/path_gambar_scatter_tanpa_pca_k3.png)

**B. Klastering 5 Kelas (k = 5)**
- **Silhouette Coefficient**: 0.486

*Tabel Silhouette Coefficient:*
![Table View Tanpa PCA k=5](output/path_gambar_table_tanpa_pca_k5.png)

*Visualisasi Scatter Plot:*
![Scatter Plot Tanpa PCA k=5](output/path_gambar_scatter_tanpa_pca_k5.png)

**C. Klastering 7 Kelas (k = 7)**
- **Silhouette Coefficient**: 0.537

*Tabel Silhouette Coefficient:*
![Table View Tanpa PCA k=7](output/path_gambar_table_tanpa_pca_k7.png)

*Visualisasi Scatter Plot:*
![Scatter Plot Tanpa PCA k=7](output/path_gambar_scatter_tanpa_pca_k7.png)

#### 2. Reduksi Dimensi dengan PCA (203 Fitur)

**A. Klastering 3 Kelas (k = 3)**
- **Silhouette Coefficient**: 0.471

*Tabel Silhouette Coefficient:*
![Table View PCA 203 k=3](output/path_gambar_table_pca203_k3.png)

*Visualisasi Scatter Plot:*
![Scatter Plot PCA 203 k=3](output/path_gambar_scatter_pca203_k3.png)

**B. Klastering 5 Kelas (k = 5)**
- **Silhouette Coefficient**: 0.486

*Tabel Silhouette Coefficient:*
![Table View PCA 203 k=5](output/path_gambar_table_pca203_k5.png)

*Visualisasi Scatter Plot:*
![Scatter Plot PCA 203 k=5](output/path_gambar_scatter_pca203_k5.png)

**C. Klastering 7 Kelas (k = 7)**
- **Silhouette Coefficient**: 0.537

*Tabel Silhouette Coefficient:*
![Table View PCA 203 k=7](output/path_gambar_table_pca203_k7.png)

*Visualisasi Scatter Plot:*
![Scatter Plot PCA 203 k=7](output/path_gambar_scatter_pca203_k7.png)

#### 3. Reduksi Dimensi dengan PCA (74 Fitur)

**A. Klastering 3 Kelas (k = 3)**
- **Silhouette Coefficient**: 0.471

*Tabel Silhouette Coefficient:*
![Table View PCA 74 k=3](output/path_gambar_table_pca74_k3.png)

*Visualisasi Scatter Plot:*
![Scatter Plot PCA 74 k=3](output/path_gambar_scatter_pca74_k3.png)

**B. Klastering 5 Kelas (k = 5)**
- **Silhouette Coefficient**: 0.486

*Tabel Silhouette Coefficient:*
![Table View PCA 74 k=5](output/path_gambar_table_pca74_k5.png)

*Visualisasi Scatter Plot:*
![Scatter Plot PCA 74 k=5](output/path_gambar_scatter_pca74_k5.png)

**C. Klastering 7 Kelas (k = 7)**
- **Silhouette Coefficient**: 0.537

*Tabel Silhouette Coefficient:*
![Table View PCA 74 k=7](output/path_gambar_table_pca74_k7.png)

*Visualisasi Scatter Plot:*
![Scatter Plot PCA 74 k=7](output/path_gambar_scatter_pca74_k7.png)

#### 4. Reduksi Dimensi dengan PCA (37 Fitur)

**A. Klastering 3 Kelas (k = 3)**
- **Silhouette Coefficient**: 0.471

*Tabel Silhouette Coefficient:*
![Table View PCA 37 k=3](output/path_gambar_table_pca37_k3.png)

*Visualisasi Scatter Plot:*
![Scatter Plot PCA 37 k=3](output/path_gambar_scatter_pca37_k3.png)

**B. Klastering 5 Kelas (k = 5)**
- **Silhouette Coefficient**: 0.486

*Tabel Silhouette Coefficient:*
![Table View PCA 37 k=5](output/path_gambar_table_pca37_k5.png)

*Visualisasi Scatter Plot:*
![Scatter Plot PCA 37 k=5](output/path_gambar_scatter_pca37_k5.png)

**C. Klastering 7 Kelas (k = 7)**
- **Silhouette Coefficient**: 0.537

*Tabel Silhouette Coefficient:*
![Table View PCA 37 k=7](output/path_gambar_table_pca37_k7.png)

*Visualisasi Scatter Plot:*
![Scatter Plot PCA 37 k=7](output/path_gambar_scatter_pca37_k7.png)

### Kesimpulan

Berdasarkan rancangan eksperimen klasterisasi K-Means dengan data polynomial di atas, dapat ditarik beberapa kesimpulan utama:

1. **Jumlah Klaster Optimal (k=7):** Pembagian data menjadi 7 klaster secara konsisten memberikan nilai rata-rata *Silhouette Coefficient* tertinggi yaitu **0.537**, mengungguli pembagian 3 klaster (0.471) dan 5 klaster (0.486). Hal ini menandakan bahwa data memiliki struktur natural yang lebih cocok dipartisi menjadi 7 kelompok.
2. **Efektivitas PCA yang Ekstrem:** Penggunaan PCA untuk mereduksi dimensi hingga ke 37 fitur (dari total fitur polynomial awal) **sama sekali tidak mengubah nilai Silhouette Coefficient**. Nilai yang didapat identik dengan penggunaan data asli (tanpa PCA). Ini membuktikan bahwa 37 komponen utama sudah sangat cukup merepresentasikan variansi penting pada data, sehingga kita bisa menghemat resource komputasi secara signifikan tanpa mengorbankan kualitas klasterisasi.

---

## Bagian 2: Tahap 4 - Visualisasi Peta Segmentasi Wilayah

Tahap terakhir ini menampilkan hasil clustering di atas **peta interaktif** Indonesia menggunakan library **Folium** (berbasis Leaflet.js). Setiap wilayah divisualisasikan dengan warna berbeda sesuai clusternya.

### Mengapa Peta?

Peta adalah cara paling intuitif untuk memahami **pola spasial** clustering:
- Apakah wilayah yang berdekatan cenderung masuk ke cluster yang sama?
- Apakah ada pola regional (Jawa, Kalimantan, dll.)?
- Wilayah mana yang memiliki karakteristik polutan paling unik (outlier)?

### Strategi Geocoding

Data tidak memiliki kolom koordinat, sehingga kita perlu **geocoding** — mengubah nama daerah menjadi koordinat (latitude, longitude).

Strategi yang digunakan (prioritas berurutan):
1. **Lookup hardcoded** — koordinat yang sudah diverifikasi manual
2. **Nominatim (OpenStreetMap)** — geocoding otomatis via geopy
3. **Fallback** — koordinat pusat Jawa Timur + jitter acak (jika gagal)

> **⚠️ Warning**
> Hasil geocoding otomatis bisa tidak akurat untuk nama daerah yang ambigu. Verifikasi koordinat dengan membuka peta interaktif yang dihasilkan.

### Peta untuk Setiap Dimensi

Notebook ini memetakan hasil clustering terbaik di **setiap dimensi** (204 → 203 → 74 → 37) untuk interpolasi Linear dan Polynomial, bukan hanya satu konfigurasi terbaik global. Peta interaktif memiliki satu layer per dimensi, dan peta statis menampilkan semuanya berdampingan.

```python
import folium
import pandas as pd
import requests

# 1. Siapkan data wilayah dan hasil clustering (berdasarkan evaluasi K-Means terbaik)
# Data disederhanakan untuk contoh (Cluster 0: Mayoritas, Cluster 1: Sidoarjo)
data_cluster = {
    'Gresik': 0, 'Bangkalan': 0, 'Lamongan': 0, 'Pamekasan': 0,
    'Tuban': 0, 'Sumenep': 0, 'Surabaya': 0, 'Sidoarjo': 1,
    'Nganjuk': 0, 'Nunukan': 0, 'Sampang': 0, 'Jombang': 0,
    'Manado': 0, 'Madiun': 0, 'Ngawi': 0, 'Sukabumi': 0
}

# 2. Load GeoJSON wilayah (Gunakan public repo atau file lokal .geojson)
geojson_url = "https://raw.githubusercontent.com/superpikar/indonesia-geojson/master/indonesia-edit.geojson"
indonesia_geojson = requests.get(geojson_url).json()

# 3. Inisialisasi peta di koordinat pusat Jawa Timur
m = folium.Map(location=[-7.5360639, 112.2384017], zoom_start=7, tiles='OpenStreetMap')

# 4. Fungsi pewarnaan poligon (Choropleth) berdasarkan cluster
def get_color(feature):
    # Sesuaikan 'name' dengan key/atribut nama daerah pada file GeoJSON Anda
    nama_wilayah = feature['properties'].get('name', '')
    
    # Mencocokkan nama wilayah di GeoJSON dengan data cluster
    for wilayah, cluster in data_cluster.items():
        if wilayah.lower() in str(nama_wilayah).lower():
            if cluster == 0:
                return '#E53935' # Merah untuk Cluster 0
            elif cluster == 1:
                return '#1E88E5' # Biru untuk Cluster 1
    
    # Wilayah tanpa data / di luar jangkauan (Abu-abu)
    return '#cccccc' 

# 5. Tambahkan layer GeoJSON ke peta
folium.GeoJson(
    indonesia_geojson,
    name="Peta Segmentasi Polutan",
    style_function=lambda feature: {
        'fillColor': get_color(feature),
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.7 if get_color(feature) != '#cccccc' else 0.1
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['name'], 
        aliases=['Wilayah:'], 
        localize=True
    )
).add_to(m)

# Tambahkan kontrol layer
folium.LayerControl().add_to(m)

# 6. Simpan peta ke bentuk HTML interaktif
m.save('output/peta_clustering_interaktif.html')
```

### 4.4 Peta Interaktif Folium — Satu Layer per Dimensi

Peta interaktif ini menggunakan **Leaflet.js** via Folium, dengan fitur:
- **Satu layer untuk setiap dimensi** (Linear dan Poly × 204, 203, 74, 37). Pilih layer di kontrol kanan atas (default: konfigurasi terbaik global)
- **Marker berwarna** per cluster (dapat diklik untuk detail)
- **Pilihan basemap** (OpenStreetMap, Esri Street, Citra Satelit)
- **Ringkasan hasil** semua dimensi di pojok kiri bawah
- **Tooltip** saat hover

Aktifkan **satu layer pada satu waktu** agar marker tidak saling menumpuk.

<iframe src="_static/peta_clustering_interaktif.html" width="100%" height="600px" style="border:none; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"></iframe>

[Buka Peta di Tab Baru (Layar Penuh)](_static/peta_clustering_interaktif.html)

### 4.5 Peta Statis (Matplotlib) — untuk Laporan

Cluster terbaik di setiap dimensi (baris atas Linear, baris bawah Polynomial), semua dengan skala peta yang sama agar mudah dibandingkan.

![Peta Statis Matplotlib](output/12_peta_statis.png)

### 4.6 Ringkasan Lengkap Analisis

```text
=================================================================
 RINGKASAN LENGKAP ANALISIS CLUSTERING SEGMENTASI POLUTAN
=================================================================

[TAHAP 1] DATA & EDA
 Sumber     : MySQL — basisda1_PSD-A-Interpolasi
 Tabel      : ekstraksi_fitur_linear + ekstraksi_fitur_polynomial
 Ukuran     : 37 wilayah × 204 fitur (68 per polutan)

[TAHAP 2] REDUKSI DIMENSI
 Standardisasi : StandardScaler (mean=0, std=1)
 Urutan        : 204 → 203 → 74 → 37
 204 → 203     : seleksi fitur (buang 1 fitur paling redundan)
 203 → 74      : seleksi fitur (buang 129 fitur paling redundan)
 74 → 37       : PCA (37 komponen)

[TAHAP 3] EKSPERIMEN CLUSTERING
 Algoritma : K-Means (k-means++, n_init=20)
 Variasi   : 8 dataset × k=2..10 = 72 eksperimen
 Evaluasi  : Silhouette Coefficient
 Terbaik global: Poly-204, k=2, silhouette=0.5375

 Cluster terbaik di setiap dimensi:
  Linear-204  k=2  sil=0.4909  ukuran: 34 / 3
  Linear-203  k=2  sil=0.4909  ukuran: 34 / 3
  Linear-74   k=2  sil=0.4021  ukuran: 33 / 4
  Linear-37   k=2  sil=0.4021  ukuran: 33 / 4
  Poly-204    k=2  sil=0.5375  ukuran: 35 / 2
  Poly-203    k=2  sil=0.5375  ukuran: 35 / 2
  Poly-74     k=3  sil=0.4543  ukuran: 35 / 1 / 1
  Poly-37     k=3  sil=0.4543  ukuran: 35 / 1 / 1

[TAHAP 4] VISUALISASI PETA
 Geocoding : Nominatim (OpenStreetMap) + hardcoded
 Peta interaktif : output/peta_clustering_interaktif.html
 Peta statis     : output/12_peta_statis.png
 Peta per dimensi: output/13_peta_per_dimensi.png

DISTRIBUSI WILAYAH PER CLUSTER:
-----------------------------------------------------------------

Cluster 0 (35 wilayah) ─ warna: #E53935
 • Baron Nganjuk
 • Nunukan
 • Sreseh, Sampang
 • Manyar, Gresik
 • Kamal, Bangkalan
 • Kedungpring Lamongan
 • Gresik Kota, Gresik
 • Waru, Pamekasan
 • Paciran, Lamongan
 • Kertosono, Nganjuk
 • Menganti, Gresik
 • Bandung - Jogoroto, Jombang
 • Banyu Ajuh, Perumnas, Kamal
 • Widang, Tuban
 • Kwanyar, Bangkalan
 • sambeng, lamongan
 • Kec. Kalianget, Sumenep
 • Cerme, Gresik
 • Tikala, Manado
 • Kerek, Tuban
 • Kwanyar, Bangkalan
 • Wonokromo, Surabaya
 • Asemrowo, Surabaya
 • Kota Sumenep, Sumenep
 • Socah, Bangkalan
 • Pilangkenceng, Madiun
 • Tanah Merah, Bangkalan
 • Labang, Bangkalan
 • Widodaren, Ngawi
 • Bangkalan, Bangkalan
 • Warudoyong, Kota Sukabumi
 • Kamal, Bangkalan
 • Banyuajuh kamal, Bangkalan
 • Dukun, Gresik
 • Kecamatan Bangkalan,Bangkalan

Cluster 1 (2 wilayah) ─ warna: #1E88E5
 • Jabon, Sidoarjo
 • Sidoarjo, Wonoayu

=================================================================
 OUTPUT FILES (folder output/)
=================================================================
 01_distribusi_fitur.png                       76.5 KB
 02_heatmap_korelasi.png                      208.8 KB
 03_linear_vs_poly.png                        156.2 KB
 04_standardisasi.png                          92.7 KB
 04a_batas_pca.png                             85.7 KB
 04c_pipeline_reduksi.png                      38.2 KB
 05_seleksi_fitur.png                          53.5 KB
 06_pca_tahap3.png                            113.2 KB
 07_silhouette_semua.png                      275.3 KB
 07b_heatmap_silhouette.png                   171.0 KB
 07c_silhouette_per_dimensi.png                81.5 KB
 08_silhouette_plot.png                        50.2 KB
 08b_silhouette_plot_per_dimensi.png          129.0 KB
 09_elbow_silhouette.png                      100.0 KB
 10_scatter_cluster.png                       138.6 KB
 11_profil_cluster.png                        119.7 KB
 12_peta_statis.png                           122.5 KB
 13_peta_per_dimensi.png                      194.3 KB
 koordinat_cache.json                           2.2 KB
 peta_clustering_interaktif.html             1072.6 KB
 ringkasan_eksperimen.csv                       0.3 KB
 ringkasan_per_dimensi.csv                      0.5 KB
 tahap1_data.pkl                              131.1 KB
 tahap2_pca.pkl                               434.3 KB
 tahap3_clustering.pkl                        138.4 KB

[SELESAI]
```
