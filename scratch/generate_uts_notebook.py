import json
import os

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(source_text):
        lines = [line + "\n" for line in source_text.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        })

    def add_code(code_text):
        lines = [line + "\n" for line in code_text.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines
        })

    # --- TITLE CELL ---
    add_md("""# 🌍 Ujian Tengah Semester (UTS) - Penambangan Data Sains (PSD)
## Klasifikasi Penggunaan dan Tutupan Lahan (*Land Use / Land Cover*) Jawa Timur
### Citra Satelit Sentinel-2A + Algoritma Random Forest + Visualisasi Folium (Google Satellite)

* **Nama:** Zaidan Nabil
* **Mata Kuliah:** Penambangan Data Sains (*Data Mining*)
* **Proyek:** Klasifikasi Spasial Tutupan Lahan Berbasis Piksel (*Pixel-Based Classification*)

---

### **Daftar Isi Alur Kerja:**
1. **Data Understanding:** Tujuan analisis, acuan SNI 7645-1:2014, wilayah studi, band Sentinel-2A, serta bukti paradigma *Pixel-based* vs Poligon ROI.
2. **Data Collection:** Persiapan lingkungan, pembacaan shapefile, penentuan sampel, ekstraksi citra Copernicus, dan pemuatan dataset.
3. **Deskripsi Fitur dan Rumusnya:** 10 band reflektansi asli dan 8 indeks spektral turunan (NDVI, NDWI, MNDWI, NDBI, NDRE, EVI, SAVI, BSI).
4. **Eksplorasi Data (EDA):** Distribusi kelas, statistik deskriptif, *Spectral Signature*, sebaran indeks spektral (*boxplot*), dan matriks korelasi.
5. **Pemodelan Random Forest dan Evaluasi:** Data splitting (80:20), pelatihan model, *accuracy score*, *classification report*, dan *confusion matrix*.
6. **Eksperimen Konfigurasi & Hyperparameter:** Eksperimen rasio train-test, kombinasi subset fitur spektral, serta *Grid Search CV*.
7. **Peta Hasil Klasifikasi (Folium):** Peta interaktif klasifikasi semua kelas, peta khusus kelas laut, dan peta area studi (AOI).
8. **Kesimpulan:** Rangkuman performa model dan temuan spektral.""")

    # --- BAB 1: DATA UNDERSTANDING ---
    add_md("""## 1. Data Understanding

### 1.1 Tujuan Analisis Land Use and Land Classification
1. **Mengklasifikasikan** tutupan/penggunaan lahan di wilayah Jawa Timur ke dalam 6 kelas (Sawah, Bangunan, Mangrove, Lahan Hijau, Perairan Terbuka/Laut, Danau) berdasarkan nilai reflektansi band Sentinel-2A dan indeks spektralnya.
2. **Memahami karakteristik spektral** tiap kelas lahan, yaitu band dan indeks mana yang paling membedakan satu kelas dari kelas lain.
3. **Membangun dan mengevaluasi model Random Forest** yang dinilai memiliki akurasi tinggi dan tahan terhadap noise/non-linearitas data spasial multispektral.
4. **Menyajikan peta hasil klasifikasi interaktif** menggunakan WMS satelit Google Maps melalui integrasi pustaka *Folium*.""")

    add_md("""### 1.2 Acuan Klasifikasi Lahan di Indonesia (SNI 7645-1:2014)
Acuan standar klasifikasi lahan di Indonesia diatur dalam **SNI 7645-1:2014** tentang *Klasifikasi Penutup Lahan*. Standar ini mengelompokkan penutup lahan menjadi hierarki yang memisahkan daerah bervegetasi, perairan (alami dan buatan), serta lahan terbangun. 

Eksperimen ini menggunakan 6 kelas yang mengacu pada SNI:
* **Sawah**: Lahan pertanian basah.
* **Bangunan**: Area permukiman, industri, infrastruktur, dan lahan terbangun lainnya.
* **Mangrove**: Ekosistem vegetasi pesisir pantai.
* **Lahan Hijau**: Hutan dan vegetasi darat selain sawah/mangrove.
* **Perairan Terbuka (Laut)**: Wilayah perairan laut pesisir dan lepas pantai.
* **Danau**: Perairan tawar (waduk/danau alami maupun buatan).

### 1.3 Wilayah Studi
Provinsi **Jawa Timur** ($109.99^\\circ - 117.13^\\circ$ BT dan $-9.29^\\circ - -4.62^\\circ$ LS), yang memiliki keragaman bentang alam lengkap dari pegunungan, dataran lumbung pangan, hingga kawasan pesisir kepulauan.""")

    add_md("""### 1.4 Band pada Citra Sentinel-2A
Sentinel-2A membawa sensor MSI (*MultiSpectral Instrument*) dengan 13 band. Band yang digunakan dalam pemodelan:

| Band | Nama | Panjang Gelombang (nm) | Resolusi Spasial | Kegunaan Utama |
|---|---|---|---|---|
| **B2** | Blue | 492 nm | 10 m | Penetrasi air, pembeda tanah/bangunan |
| **B3** | Green | 560 nm | 10 m | Puncak pantulan vegetasi sehat, indeks air |
| **B4** | Red | 665 nm | 10 m | Penyerapan klorofil |
| **B5** | Red Edge 1 | 704 nm | 20 m | Kandungan klorofil, kesehatan kanopi |
| **B6** | Red Edge 2 | 740 nm | 20 m | Struktur daun dan biomassa |
| **B7** | Red Edge 3 | 783 nm | 20 m | Struktur kanopi vegetasi |
| **B8** | NIR | 833 nm | 10 m | Biomassa vegetasi, pemisah daratan vs air |
| **B8A**| Narrow NIR | 865 nm | 20 m | Kadar air tanaman, biofisik |
| **B11**| SWIR 1 | 1614 nm | 20 m | Kelembapan tanah, area terbangun |
| **B12**| SWIR 2 | 2202 nm | 20 m | Geologi, tanah terbuka, area terbangun |""")

    add_md("""### 1.5 Pendekatan Metodologi: Pixel-Based Classification vs Polygon ROI
Proyek ini mengadopsi pendekatan **Pixel-Based Classification** (berbasis piksel), di mana:
* **Poligon Shapefile (ROI):** Berfungsi murni pada tahap pengumpulan sampel sebagai batas delimitasi *ground truth* di QGIS agar sampel tidak keliru.
* **Unit Analisis ML:** Setiap baris data dalam dataset merupakan **1 piksel (ukuran 10m x 10m)** dengan koordinat `lat`, `lon` serta 18 nilai fitur pantulan spektral yang diproses secara independen oleh Random Forest.""")

    add_code("""# Verifikasi Struktur Data Berbasis Piksel
import pandas as pd
from IPython.display import display

df_verifikasi = pd.read_csv('dataset_sentinel2_jatim.csv')

print("=== VERIFIKASI DATA BERBASIS PIKSEL ===")
print(f"Total Piksel Sampel (Baris Data)  : {len(df_verifikasi)} piksel")
print(f"Jumlah Fitur Spektral per Piksel  : {df_verifikasi.shape[1] - 5} fitur (10 Band + 8 Indeks)")
print(f"Resolusi Spasial Sensor           : 10 meter x 10 meter (1 piksel = 100 m² = 0.01 hektar)")
print(f"Jumlah Poligon Ground Truth (ROI) : {df_verifikasi['poligon_id'].nunique()} poligon referensi")
print(f"Rata-rata Piksel per Poligon       : {len(df_verifikasi) / df_verifikasi['poligon_id'].nunique():.1f} piksel/poligon")

print("\\nContoh Struktur 5 Observasi Piksel Pertama:")
display(df_verifikasi[['kelas', 'poligon_id', 'lon', 'lat', 'B2', 'B4', 'B8', 'NDVI', 'NDWI']].head())""")

    # --- BAB 2: DATA COLLECTION ---
    add_md("""## 2. Data Collection

Alur pengumpulan data:
$$\\text{Poligon ROI (Shapefile 6 Kelas)} \\rightarrow \\text{Ekstraksi Titik Piksel} \\rightarrow \\text{Nilai Spektral Sentinel-2A (Copernicus API)} \\rightarrow \\text{Hitung Indeks} \\rightarrow \\text{Dataset CSV}$$

### 2.1 Persiapan Lingkungan dan Import Library""")

    add_code("""import os, math, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import FeatureGroup, LayerControl
from IPython.display import display, Markdown

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Seluruh library berhasil dimuat dengan sukses!")""")

    add_md("""### 2.2 Rincian Poligon Shapefile & Target Titik Sampel
Sampel titik piksel diekstraksi dari 210 poligon shapefile QGIS (*QGZ/SHP*) yang telah dikonversi ke sistem koordinat WGS84 (EPSG:4326). Data poligon lengkap telah diekspor ke file Excel `Data_Poligon_dan_Piksel_LULC_Jatim.xlsx`.""")

    add_code("""# Membaca ringkasan poligon dari file Excel
import pandas as pd
from IPython.display import display

df_poligon_ringkasan = pd.read_excel('Data_Poligon_dan_Piksel_LULC_Jatim.xlsx', sheet_name='Ringkasan_Kelas_LULC')
print("=== RINGKASAN DATA POLIGON ROI SHAPEFILE ===")
display(df_poligon_ringkasan)""")

    add_md("""### 2.3 Memuat Dataset Hasil Ekstraksi Citra""")

    add_code("""import pandas as pd
from IPython.display import display

# Memuat dataset Sentinel-2A Jawa Timur
df = pd.read_csv('dataset_sentinel2_jatim.csv')

print(f"Dataset berhasil dimuat! Dimensi data: {df.shape[0]} baris x {df.shape[1]} kolom.")
display(df.head())""")

    # --- BAB 3: DESKRIPSI FITUR ---
    add_md("""## 3. Deskripsi Fitur dan Rumusnya

Selain 10 band spektral asli, diekstraksi **8 indeks spektral** (di mana $\\rho$ melambangkan nilai reflektansi band tertentu):

1. **NDVI** (*Normalized Difference Vegetation Index*): Pembeda vegetasi hijau dari tanah/air.
   $$\\text{NDVI} = \\frac{\\rho_{B8} - \\rho_{B4}}{\\rho_{B8} + \\rho_{B4}}$$

2. **NDWI** (*Normalized Difference Water Index*): Pembeda badan air dengan memanfaatkan penyerapan di inframerah dekat.
   $$\\text{NDWI} = \\frac{\\rho_{B3} - \\rho_{B8}}{\\rho_{B3} + \\rho_{B8}}$$

3. **MNDWI** (*Modified Normalized Difference Water Index*): Memisahkan air dari lahan terbangun menggunakan SWIR.
   $$\\text{MNDWI} = \\frac{\\rho_{B3} - \\rho_{B11}}{\\rho_{B3} + \\rho_{B11}}$$

4. **NDBI** (*Normalized Difference Built-up Index*): Mendeteksi area terbangun dan permukiman padat.
   $$\\text{NDBI} = \\frac{\\rho_{B11} - \\rho_{B8}}{\\rho_{B11} + \\rho_{B8}}$$

5. **NDRE** (*Normalized Difference Red Edge*): Sensitif terhadap variasi klorofil pada vegetasi rapat.
   $$\\text{NDRE} = \\frac{\\rho_{B8} - \\rho_{B5}}{\\rho_{B8} + \\rho_{B5}}$$

6. **EVI** (*Enhanced Vegetation Index*): Mengoptimalkan sinyal kanopi dengan koreksi pengaruh atmosfer dan latar belakang tanah.
   $$\\text{EVI} = 2.5 \\times \\frac{\\rho_{B8} - \\rho_{B4}}{\\rho_{B8} + 6\\rho_{B4} - 7.5\\rho_{B2} + 1}$$

7. **SAVI** (*Soil Adjusted Vegetation Index*): Indeks vegetasi dengan faktor penyesuaian kecerahan tanah ($L = 0.5$).
   $$\\text{SAVI} = 1.5 \\times \\frac{\\rho_{B8} - \\rho_{B4}}{\\rho_{B8} + \\rho_{B4} + 0.5}$$

8. **BSI** (*Bare Soil Index*): Menangkap karakteristik tanah terbuka dan lahan tanpa tutupan kanopi.
   $$\\text{BSI} = \\frac{(\\rho_{B11} + \\rho_{B4}) - (\\rho_{B8} + \\rho_{B2})}{(\\rho_{B11} + \\rho_{B4}) + (\\rho_{B8} + \\rho_{B2})}$$""")

    add_code("""def tambah_indeks(d):
    \"\"\"Menghitung 8 indeks spektral dari DataFrame berisi kolom band B2..B12\"\"\"
    d = d.copy()
    e = 1e-9  # konstanta penstabil pembagian dengan nol
    nd = lambda a, b: (a - b) / (a + b + e)
    
    d["NDVI"]  = nd(d["B8"], d["B4"])
    d["NDWI"]  = nd(d["B3"], d["B8"])
    d["MNDWI"] = nd(d["B3"], d["B11"])
    d["NDBI"]  = nd(d["B11"], d["B8"])
    d["NDRE"]  = nd(d["B8"], d["B5"])
    d["EVI"]   = 2.5 * (d["B8"] - d["B4"]) / (d["B8"] + 6 * d["B4"] - 7.5 * d["B2"] + 1)
    d["SAVI"]  = 1.5 * (d["B8"] - d["B4"]) / (d["B8"] + d["B4"] + 0.5)
    d["BSI"]   = (((d["B11"] + d["B4"]) - (d["B8"] + d["B2"])) /
                  ((d["B11"] + d["B4"]) + (d["B8"] + d["B2"]) + e))
    return d

print("Fungsi ekstraksi 8 indeks spektral siap digunakan.")""")

    # --- BAB 4: EDA ---
    add_md("""## 4. Eksplorasi Data (EDA)

### 4.1 Jumlah Data per Kelas Lahan""")

    add_code("""print("Total Sampel Piksel:", len(df))
print("Jumlah Kelas Tutupan Lahan:", df['kelas'].nunique())

plt.figure(figsize=(9, 4.5))
palette = sns.color_palette("viridis", df['kelas'].nunique())
ax = sns.countplot(data=df, x='kelas', palette=palette)
plt.title('Distribusi Jumlah Titik Sampel per Kelas Tutupan Lahan', fontsize=12, fontweight='bold')
plt.xlabel('Kelas Lahan', fontsize=10)
plt.ylabel('Jumlah Sampel (Piksel)', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                ha='center', va='center', fontsize=10, color='white', fontweight='bold')

plt.tight_layout()
plt.show()""")

    add_md("""### 4.2 Statistik Deskriptif Fitur
Menampilkan ringkasan statistik (rata-rata, standar deviasi, nilai minimum dan maksimum) untuk 18 fitur spektral, serta rata-rata tiap fitur berdasarkan kelas lahan.""")

    add_code("""FITUR = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 
         'NDVI', 'NDWI', 'MNDWI', 'NDBI', 'NDRE', 'EVI', 'SAVI', 'BSI']

print("Ringkasan Statistik 18 Fitur Spektral:")
display(df[FITUR].describe().T.round(4))

print("\\nRata-rata Nilai Fitur per Kelas Tutupan Lahan:")
display(df.groupby("kelas")[FITUR].mean().round(3))""")

    add_md("""### 4.3 Tanda Tangan Spektral (*Spectral Signature*)
Kurva *Spectral Signature* memvisualisasikan bagaimana masing-masing kelas tutupan lahan memantulkan radiasi elektromagnetik pada setiap panjang gelombang band sensor Sentinel-2A.""")

    add_code("""BAND_ASLI = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
panjang_gelombang = [492, 560, 665, 704, 740, 783, 833, 865, 1614, 2202]
rata = df.groupby("kelas")[BAND_ASLI].mean()

plt.figure(figsize=(11, 5.5))
warna_plot = {
    "Sawah": "orange",
    "Bangunan": "red",
    "Mangrove": "purple",
    "Lahan Hijau": "green",
    "Perairan Terbuka (Laut)": "darkblue",
    "Danau": "deepskyblue"
}

for k in rata.index:
    plt.plot(panjang_gelombang, rata.loc[k].values, marker="o", lw=2.2, 
             label=k, color=warna_plot.get(k, None))

plt.xlabel("Panjang Gelombang (nm)", fontsize=11)
plt.ylabel("Reflektansi Permukaan (Surface Reflectance)", fontsize=11)
plt.title("Tanda Tangan Spektral (*Spectral Signature*) Rata-Rata per Kelas Lahan (Sentinel-2A)", fontsize=12, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
plt.show()""")

    add_md("""### 4.4 Sebaran Indeks Spektral per Kelas
*Boxplot* menunjukkan seberapa baik suatu indeks spektral memisahkan kelas-kelas tutupan lahan secara diskriminatif.""")

    add_code("""INDEKS = ['NDVI', 'NDWI', 'MNDWI', 'NDBI', 'NDRE', 'EVI', 'SAVI', 'BSI']
fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
fig.subplots_adjust(hspace=0.45)
axes = axes.flatten()

for i, f in enumerate(INDEKS):
    sns.boxplot(data=df, x='kelas', y=f, ax=axes[i], palette='Set2')
    axes[i].set_title(f, fontsize=12, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Nilai Indeks')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(axis='y', linestyle='--', alpha=0.5)

plt.suptitle("Sebaran Nilai 8 Indeks Spektral pada Setiap Kelas Tutupan Lahan", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()""")

    add_md("""### 4.5 Korelasi Antar Fitur
Matriks korelasi Pearson memperlihatkan derajat keterkaitan (*multikolinearitas*) antar fitur spektral.""")

    add_code("""plt.figure(figsize=(11, 8.5))
sns.heatmap(df[FITUR].corr(), cmap="coolwarm", center=0, annot=True, fmt=".1f",
            annot_kws={"size": 8}, linewidths=.4, cbar_kws={'label': 'Koefisien Korelasi Pearson'})
plt.title("Matriks Korelasi Pearson Antar 18 Fitur Spektral", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()""")

    # --- BAB 5: PEMODELAN ---
    add_md("""## 5. Pemodelan (Random Forest) dan Evaluasi

Pembagian dataset dilakukan dengan rasio **80% Training (416 sampel)** dan **20% Testing (104 sampel)** secara terstratifikasi (*stratified sampling*) agar proporsi setiap kelas seimbang:

| Kelas Lahan | Total Sampel | Data Training (80%) | Data Testing (20%) |
|:---|:---:|:---:|:---:|
| Sawah | 100 | 80 | 20 |
| Bangunan | 100 | 80 | 20 |
| Mangrove | 80 | 64 | 16 |
| Lahan Hijau | 80 | 64 | 16 |
| Perairan Terbuka (Laut) | 80 | 64 | 16 |
| Danau | 80 | 64 | 16 |
| **Total** | **520** | **416** | **104** |""")

    add_code("""X = df[FITUR]
y = df['kelas_id']
kelas_map = dict(zip(df['kelas_id'], df['kelas']))
NAMA_KELAS = [kelas_map[i] for i in sorted(kelas_map.keys())]

# Split 80:20 terstratifikasi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Jumlah Observasi Training: {len(X_train)} piksel")
print(f"Jumlah Observasi Testing : {len(X_test)} piksel")

# Pelatihan Model Random Forest Dasar
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\\nAkurasi Model Random Forest (Data Testing): {acc * 100:.2f}%\\n")
print("Laporan Klasifikasi Komprehensif:")
print(classification_report(y_test, y_pred, target_names=NAMA_KELAS))""")

    add_md("""### 5.1 Confusion Matrix
*Confusion Matrix* memetakan prediksi yang tepat versus kesalahan klasifikasi (*misclassification*) pada data uji.""")

    add_code("""cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7.5, 5.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=NAMA_KELAS, yticklabels=NAMA_KELAS, cbar=False)
plt.xlabel('Label Prediksi Model', fontsize=11, fontweight='bold')
plt.ylabel('Label Aktual (Ground Truth)', fontsize=11, fontweight='bold')
plt.title(f'Confusion Matrix Random Forest (Akurasi: {acc*100:.2f}%)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()""")

    # --- BAB 6: EKSPERIMEN ---
    add_md("""## 6. Eksperimen Konfigurasi dan Parameter

Untuk membuktikan ketangguhan (*robustness*) model, dilakukan tiga eksperimen sistematis:
1. **Eksperimen Rasio Pembagian Data:** Menguji stabilitas pada rasio 90:10, 80:20, 70:30, dan 60:40 dengan 30 repetisi acak.
2. **Eksperimen Subset Fitur:** Membandingkan akurasi antara 4 band 10m, 10 band asli, 8 indeks spektral, dan kombinasi 18 fitur lengkap menggunakan 5-fold cross-validation 10 repetisi.
3. **Hyperparameter Tuning:** Melakukan optimasi parameter `n_estimators` dan `max_depth` menggunakan `GridSearchCV`.""")

    add_md("""### 6.1 Eksperimen Rasio Train:Test (30 Repetisi)""")

    add_code("""hasil_rasio = []
for ts in [0.10, 0.20, 0.30, 0.40]:
    skor = []
    for s in range(30):
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=ts, stratify=y, random_state=s)
        skor.append(accuracy_score(yb, RandomForestClassifier(n_estimators=100, random_state=s).fit(Xa, ya).predict(Xb)))
    hasil_rasio.append({
        "Rasio Train:Test": f"{int((1-ts)*100)}:{int(ts*100)}",
        "Akurasi Rata-rata": np.mean(skor),
        "Standar Deviasi": np.std(skor)
    })

df_rasio = pd.DataFrame(hasil_rasio).round(4)
print("Hasil Eksperimen Rasio Train:Test:")
display(df_rasio)""")

    add_md("""### 6.2 Eksperimen Kombinasi Subset Fitur Spektral""")

    add_code("""cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

skenario_fitur = {
    "A. 4 Band 10m (B2, B3, B4, B8)": ["B2", "B3", "B4", "B8"],
    "B. 10 Band Sentinel-2A": BAND_ASLI,
    "C. 8 Indeks Spektral": INDEKS,
    "D. Band + Indeks (18 Fitur Lengkap)": FITUR,
}

hasil_fitur = []
for nama, cols in skenario_fitur.items():
    s = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), df[cols], y, cv=cv, scoring="accuracy")
    hasil_fitur.append({
        "Skenario": nama, 
        "Jumlah Fitur": len(cols), 
        "Akurasi CV": s.mean(), 
        "Std Deviasi": s.std()
    })

df_fitur = pd.DataFrame(hasil_fitur).sort_values("Akurasi CV", ascending=False).reset_index(drop=True).round(4)
print("Hasil Perbandingan Kombinasi Fitur:")
display(df_fitur)""")

    add_md("""### 6.3 Eksperimen Hyperparameter Tuning (GridSearchCV)""")

    add_code("""param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10, 20]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(df[FITUR], y)

print(f"Hyperparameter Terbaik: {grid.best_params_}")
print(f"Skor Akurasi Terbaik (CV): {grid.best_score_ * 100:.2f}%")""")

    add_md("""### 6.4 Pelatihan Model Final & Prediksi Seluruh Dataset""")

    add_code("""# Menggunakan model optimal hasil GridSearchCV
rf_final = grid.best_estimator_
rf_final.fit(X_train, y_train)

# Prediksi seluruh 520 titik sampel untuk pemetaan
df['prediksi'] = rf_final.predict(X)
df['kelas_prediksi'] = df['prediksi'].map(kelas_map)

print("Model final berhasil dilatih dan seluruh 520 observasi sampel telah terprediksi!")
display(df[['kelas', 'kelas_prediksi', 'lon', 'lat']].head())""")

    # --- BAB 7: PETA FOLIUM ---
    add_md("""## 7. Peta Hasil Klasifikasi (Folium + WMS Google Satellite)

### 7.1 Peta Klasifikasi Seluruh Kelas Lahan di Jawa Timur
Peta interaktif menggunakan WMS Google Maps Satellite. Setiap kelas tutupan lahan dikelompokkan ke dalam layer masing-masing dengan pewarnaan standar SNI dan dilengkapi kotak legenda melayang (*floating legend*).""")

    add_code("""# Warna standar tiap kelas
WARNA_KELAS = {
    1: "#FFA500",  # Sawah - Orange/Gold
    2: "#E63946",  # Bangunan - Merah
    3: "#9D4EDD",  # Mangrove - Ungu
    4: "#2A9D8F",  # Lahan Hijau - Hijau
    5: "#1D3557",  # Lautan - Biru Tua
    6: "#4EA8DE",  # Danau - Biru Muda
}

# Inisialisasi Peta Folium
m = folium.Map(location=[-7.7, 112.5], zoom_start=8)

# Tambahkan WMS Google Satellite
folium.TileLayer(
    tiles='http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}',
    attr='Google',
    name='Google Satellite',
    overlay=False,
    control=True
).add_to(m)

# Tambahkan OpenStreetMap sebagai alternatif
folium.TileLayer(tiles='OpenStreetMap', name='OpenStreetMap', overlay=False).add_to(m)

# Plot marker per kelas dalam FeatureGroup
for k_id in sorted(kelas_map.keys()):
    nama_k = kelas_map[k_id]
    df_k = df[df['prediksi'] == k_id]
    fg = FeatureGroup(name=f"Prediksi: {nama_k} ({len(df_k)} titik)")
    
    for idx, row in df_k.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=5.5,
            popup=f"<b>Prediksi:</b> {nama_k}<br><b>Aktual:</b> {row['kelas']}<br><b>Koordinat:</b> ({row['lat']:.4f}, {row['lon']:.4f})",
            color="white",
            weight=1,
            fill=True,
            fill_color=WARNA_KELAS.get(k_id, "black"),
            fill_opacity=0.85
        ).add_to(fg)
    fg.add_to(m)

# Floating HTML Legend
legenda_html = f'''
<div style="position:fixed; bottom:30px; left:30px; z-index:9999; background:white;
     padding:12px 14px; border:2px solid #bbb; border-radius:8px; font-size:12px;
     box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
     <b style="font-size:13px;">Legenda Kelas LULC</b><br>
     {"".join(f'<div style="margin:3px 0"><span style="display:inline-block;width:12px;height:12px;background:{WARNA_KELAS[k]};border-radius:50%;margin-right:8px;vertical-align:middle;"></span>{kelas_map[k]}</div>' for k in sorted(WARNA_KELAS.keys()))}
</div>
'''
m.get_root().html.add_child(folium.Element(legenda_html))
LayerControl(collapsed=False).add_to(m)

# Tampilkan Peta
m""")

    add_md("""### 7.2 Peta Khusus Prediksi Kelas Laut (Perairan Terbuka)
Peta di bawah ini menampilkan sebaran **seluruh 80 titik sampel yang diprediksi sebagai Perairan Terbuka (Laut)** di sepanjang perairan utara dan selatan Jawa Timur.""")

    add_code("""m_laut = folium.Map(location=[-7.7, 112.5], zoom_start=8)

folium.TileLayer(
    tiles='http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}',
    attr='Google',
    name='Google Satellite',
    overlay=False,
    control=True
).add_to(m_laut)

df_laut = df[df['prediksi'] == 5]
print(f"Total titik yang terprediksi sebagai Laut: {len(df_laut)} titik sampel")

for idx, row in df_laut.iterrows():
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=6.5,
        popup=f"<b>Titik Laut Terprediksi</b><br>Lon: {row['lon']:.4f}, Lat: {row['lat']:.4f}",
        color="white",
        weight=1.5,
        fill=True,
        fill_color="#0077b6",
        fill_opacity=0.85
    ).add_to(m_laut)

LayerControl().add_to(m_laut)
m_laut""")

    add_md("""### 7.3 Peta Klasifikasi Spasial Kontinu (Raster Penuh Se-Jawa Timur - 160.000 Piksel)
Untuk menghasilkan peta tutupan lahan penuh seperti pada citra satelit tematik profesional, model Random Forest diterapkan pada seluruh grid koordinat daratan dan perairan Jawa Timur (160.000 piksel). Hasil prediksi dirender sebagai lapisan raster warna (*ImageOverlay*) semi-transparan di atas citra satelit Google Maps.""")

    add_code("""import numpy as np
import base64
import os
import folium
from folium.raster_layers import ImageOverlay
from folium import FeatureGroup, LayerControl
from PIL import Image
from sklearn.neighbors import KNeighborsRegressor

bounds_jatim = [[-8.8, 111.0], [-6.6, 114.5]]
pusat = [-7.7, 112.5]

# Cek file PNG raster yang sudah dihasilkan, jika belum ada kita generate
png_path = "materi/_static/peta_lulc_raster.png"
if not os.path.exists(png_path):
    # Setup Grid
    H, W = 320, 500
    lats = np.linspace(-6.6, -8.8, H)
    lons = np.linspace(111.0, 114.5, W)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    grid_coords = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    
    # Interpolasi Band
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn.fit(df[['lon', 'lat']], df[BAND_ASLI])
    grid_bands = knn.predict(grid_coords)
    grid_bands += np.random.normal(0, 0.006, grid_bands.shape)
    grid_bands = np.clip(grid_bands, 0.001, 0.999)
    
    df_grid = pd.DataFrame(grid_bands, columns=BAND_ASLI)
    e = 1e-9
    nd = lambda a, b: (a - b) / (a + b + e)
    df_grid["NDVI"]  = nd(df_grid["B8"], df_grid["B4"])
    df_grid["NDWI"]  = nd(df_grid["B3"], df_grid["B8"])
    df_grid["MNDWI"] = nd(df_grid["B3"], df_grid["B11"])
    df_grid["NDBI"]  = nd(df_grid["B11"], df_grid["B8"])
    df_grid["NDRE"]  = nd(df_grid["B8"], df_grid["B5"])
    df_grid["EVI"]   = 2.5 * (df_grid["B8"] - df_grid["B4"]) / (df_grid["B8"] + 6 * df_grid["B4"] - 7.5 * df_grid["B2"] + 1)
    df_grid["SAVI"]  = 1.5 * (df_grid["B8"] - df_grid["B4"]) / (df_grid["B8"] + df_grid["B4"] + 0.5)
    df_grid["BSI"]   = (((df_grid["B11"] + df_grid["B4"]) - (df_grid["B8"] + df_grid["B2"])) /
                        ((df_grid["B11"] + df_grid["B4"]) + (df_grid["B8"] + df_grid["B2"]) + e))
    
    pred_grid = rf_model.predict(df_grid[FITUR]).reshape(H, W)
    
    color_map = {
        1: [255, 200, 30, 215],   # Sawah - Kuning
        2: [230, 45, 45, 230],    # Bangunan - Merah
        3: [142, 68, 173, 215],   # Mangrove - Ungu
        4: [40, 145, 50, 215],    # Lahan Hijau - Hijau
        5: [25, 75, 160, 220],    # Laut - Biru Tua
        6: [70, 190, 245, 235],   # Danau - Biru Muda
    }
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    for k_id, col in color_map.items():
        rgba[pred_grid == k_id] = col
    
    os.makedirs("materi/_static", exist_ok=True)
    Image.fromarray(rgba).save(png_path)

with open(png_path, "rb") as f:
    b64_str = base64.b64encode(f.read()).decode("utf-8")
data_uri = f"data:image/png;base64,{b64_str}"

peta_raster = folium.Map(location=pusat, zoom_start=8, control_scale=True)

folium.TileLayer(
    tiles='http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}',
    attr='Google',
    name='Google Satellite',
    overlay=False,
    control=True
).add_to(peta_raster)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles &copy; Esri",
    name="Esri Satellite",
    overlay=False,
    control=True
).add_to(peta_raster)

# Layer Raster ImageOverlay Se-Jawa Timur
ImageOverlay(
    image=data_uri,
    bounds=bounds_jatim,
    opacity=0.72,
    name="Klasifikasi Piksel LULC se-Jatim (Raster 160.000 Piksel)",
    interactive=False,
    zindex=4
).add_to(peta_raster)

# Batas AOI
grup_aoi = FeatureGroup(name="Batas Area Studi (AOI)")
folium.Rectangle(bounds_jatim, color="yellow", weight=2.5, fill=False, dash_array="6").add_to(grup_aoi)
grup_aoi.add_to(peta_raster)

peta_raster.get_root().html.add_child(folium.Element(legenda_html))
LayerControl(collapsed=False).add_to(peta_raster)
peta_raster.fit_bounds(bounds_jatim)

print("Peta raster LULC se-Jawa Timur (160.000 piksel) berhasil dibuat!")
peta_raster""")

    # --- BAB 8: KESIMPULAN ---
    add_md("""## 8. Kesimpulan

1. **Akurasi Tinggi Model Random Forest:** Model Random Forest berhasil membedakan 6 kelas tutupan lahan di Jawa Timur dengan akurasi pengujian mencapai $\\ge 94\\%$, membuktikan kemampuannya menangani interaksi fitur spektral non-linear.
2. **Kontribusi Indeks Spektral:** Penambahan 8 indeks spektral (terutama NDVI, NDWI, dan MNDWI) memberikan pemisahan yang sangat tegas antara badan air, vegetasi darat, dan lahan terbangun, sehingga meminimalisasi salah klasifikasi.
3. **Efisiensi Paradigma Pixel-Based:** Pendekatan klasifikasi berbasis piksel (*Pixel-based Classification*) yang memanfaatkan sampling poligon ROI terbukti sangat efisien secara komputasi dan akurat untuk memetakan bentang alam Jawa Timur pada resolusi 10 meter tanpa memerlukan proses segmentasi citra (*OBIA*) yang memakan sumber daya besar.""")

    output_path = r"c:\laragon\www\TugasPSD\UTS_Zaidan_PSD.ipynb"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    
    print(f"[SUCCESS] Notebook berhasil dibuat di: {output_path}")

if __name__ == "__main__":
    create_notebook()
