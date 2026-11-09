# Ekstraksi Fitur Citra Digital: Momen Warna RGB, Konversi Grayscale 8-Bit, Fitur Statistik, Tekstur (GLCM & Tamura), dan Morfologi Geometri

**Mata Kuliah:** Pengolahan Sinyal Digital (PSD)  
**Topik:** Pengolahan Citra Digital (Digital Image Processing & Feature Extraction)  
**Referensi Utama:**  
- *Teknik Ekstraksi Fitur Citra*, Dr. Mulaab, S.Si., M.Kom. (Teknik Informatika, Universitas Trunojoyo Madura)  
- *Image Feature Extraction Techniques: A Comprehensive Review*, S. Hallur & A. Gavade, *Franklin Open* (Elsevier, 2025)  
**Penulis:** Muhammad Zaidan Nabil Rafi (NIM: 240411100068)  
**Sampel Citra Fokus:** `24-068 Berantakan 1.jpeg`  
**Dataset Pendukung:** Direktori `Foto citra/` (6 Sampel Citra Berantakan vs Rapi)  
**Tanggal:** November 2026  

---

## Daftar Isi

1. Pendahuluan: Urgensi dan Taksonomi Ekstraksi Fitur Citra
   - 1.1 Mengapa Ekstraksi Fitur Penting?
   - 1.2 Domain Aplikasi Ekstraksi Fitur
   - 1.3 Taksonomi Empat Kategori Fitur Citra
2. Karakteristik dan Representasi Citra Sampel (`24-068 Berantakan 1.jpeg`)
   - 2.1 Dimensi Spasial dan Tensor 3D
   - 2.2 Dekomposisi Kanal Warna RGB
3. Materi Khusus: Transformasi Citra RGB ke Grayscale 8-Bit
   - 3.1 Rasionalitas Konversi ke Grayscale
   - 3.2 Landasan Fisiologis Penglihatan Manusia
   - 3.3 Formulasi Matematis Metode Konversi
   - 3.4 Kuantisasi dan Representasi Diskrit 8-Bit
   - 3.5 Visualisasi Perbandingan Metode Konversi
4. Kategori 1: Fitur Warna (Color Features)
   - 4.1 Momen Warna (Color Moments) Orde 1 hingga 4
   - 4.2 Histogram Warna (Color Histogram)
   - 4.3 Rata-rata RGB (Average RGB)
   - 4.4 Analisis Kompleksitas Komputasi dan Memori
5. Kategori 2: Fitur Statistik Intensitas Global (Statistical Features)
   - 5.1 Mean dan Varians
   - 5.2 Standar Deviasi (Simpangan Baku / Kontras Global)
   - 5.3 Skewness (Kemiringan Asimetri)
   - 5.4 Kurtosis (Kepuncakan Distribusi)
   - 5.5 Root Mean Square (RMS) dan Total Energy
   - 5.6 Entropi Shannon (Ketidakteraturan Sinyal)
   - 5.7 Smoothness (Tingkat Kehalusan Relatif)
6. Kategori 3: Fitur Tekstur Spasial (Spatial Texture Features)
   - 6.1 Gray-Level Co-occurrence Matrix (GLCM) Haralick
   - 6.2 Simulasi Pembentukan GLCM Langkah demi Langkah (Matriks $3 \times 3$)
   - 6.3 Lima Atribut Tekstur GLCM (Kontras, Energi, Korelasi, Entropi, Homogenitas)
   - 6.4 Enam Atribut Tekstur Tamura (Persepsi Visual Manusia)
7. Kategori 4: Fitur Bentuk dan Geometri (Shape & Geometry Features)
   - 7.1 Binarisasi dan Deteksi Region of Interest (ROI)
   - 7.2 Luas (Area) dan Keliling (Perimeter)
   - 7.3 Sentroid / Pusat Massa
   - 7.4 Kemiringan Garis (Slope)
   - 7.5 Indeks Irregularitas (Irregularity Index)
   - 7.6 Diameter Ekuivalen
   - 7.7 Convex Area dan Solidity
8. Simulasi Perhitungan Manual Terpadu (Sub-Matriks $5 \times 5$)
   - 8.1 Data Matriks Asli Patch $5 \times 5$
   - 8.2 Konversi Manual RGB ke Grayscale 8-Bit
   - 8.3 Perhitungan Statistik Manual (Mean, Varians, Standar Deviasi, RMS, Smoothness)
9. Hasil Ekstraksi Fitur Aktual pada Citra `24-068 Berantakan 1`
   - 9.1 Tabel Parameter Momen Warna RGB
   - 9.2 Tabel Fitur Statistik Citra Grayscale 8-Bit
   - 9.3 Tabel Fitur Tekstur GLCM Haralick
   - 9.4 Tabel Fitur Morfologi Geometri ROI
10. Implementasi Pipeline Komputasi dengan Python
11. Analisis Komparatif: Citra Berantakan vs Rapi & Strategi Fusi Fitur
    - 11.1 Evaluasi Fitur pada 6 Sampel Dataset
    - 11.2 Rangkuman Kelebihan, Kekurangan, Efisiensi, dan Akurasi
    - 11.3 Strategi Fusi Multi-Fitur
12. Kesimpulan
13. Referensi

---

## 1. Pendahuluan: Urgensi dan Taksonomi Ekstraksi Fitur Citra

### 1.1 Mengapa Ekstraksi Fitur Penting?

Dalam bidang **Pengolahan Sinyal Digital (PSD)** dan **Computer Vision**, citra digital beresolusi tinggi menyimpan data yang sangat besar. Sebagai contoh, citra beresolusi $1600 \times 1200$ piksel dengan format RGB menyimpan $5.760.000$ nilai numerik mentah (*raw pixels*). Memproses jutaan piksel ini secara langsung untuk tugas klasifikasi atau pengenalan pola sangat tidak efisien, membutuhkan memori masif, dan rentan terhadap *curse of dimensionality*.

**Ekstraksi Fitur (*Feature Extraction*)** adalah proses matematis untuk mentransformasikan data citra mentah menjadi sekumpulan representasi numerik yang ringkas (*compact*), informatif, dan memiliki daya pisah (*discriminative power*) yang tinggi, dengan tetap mempertahankan karakteristik esensial dari citra tersebut.

```
+------------------+       Ekstraksi Fitur       +-------------------+       Model Keputusan       +----------------------+
| Citra Mentah     | ------------------------>   | Vektor Fitur (x)  | ------------------------>   | Keputusan / Prediksi |
| 5.760.000 Piksel |   Reduksi Dimensi Spasial   | d-Dimensi Terpilih|     ML / Deep Learning      | (Berantakan vs Rapi) |
+------------------+                             +-------------------+                             +----------------------+
```

### 1.2 Domain Aplikasi Ekstraksi Fitur

Ekstraksi fitur citra menjadi fondasi utama dalam berbagai sistem cerdas modern:
1. **Pengenalan Wajah & Biometrik:** Mengidentifikasi individu berdasarkan proporsi geometris wajah, jarak antar-landmark, dan tekstur kulit.
2. **Analisis Deteksi Emosi (Affective Computing):** Mendeteksi perubahan mikro-ekspresi, kerutan wajah (fitur tekstur), dan perubahan rona wajah (fitur warna).
3. **Diagnostik Citra Medis:** Menganalisis tekstur jaringan pada CT-Scan, MRI, atau rontgen untuk mendeteksi massa tumor, lesi, atau kelainan patologis.
4. **Keselamatan Pengemudi (Driver Monitoring System):** Mengamati kedipan mata, orientasi kepala (*head pose*), dan kelelahan pengemudi secara *real-time*.
5. **Klasifikasi Pemandangan & Tata Guna Lahan:** Membedakan area vegetasi, perairan, dan pemukiman pada citra satelit/drone.

### 1.3 Taksonomi Empat Kategori Fitur Citra

Merujuk pada tinjauan ilmiah komprehensif oleh **Hallur & Gavade (Franklin Open, Elsevier 2025)** serta silabus materi perkuliahan *Teknik Ekstraksi Fitur Citra* oleh **Dr. Mulaab (UTM)**, fitur citra digital diklasifikasikan ke dalam 4 kategori utama:

```
                                      TAKSONOMI FITUR CITRA DIGITAL
                                                    |
       +--------------------+-----------------------+------------------------+----------------------+
       |                    |                                                |                      |
[1. Fitur Warna]     [2. Fitur Statistik]                             [3. Fitur Tekstur]     [4. Fitur Geometri]
  - Momen Warna        - Mean (μ) & Varians (σ²)                        - GLCM Haralick        - Area & Perimeter
  - Color Histogram    - Standar Deviasi (σ)                              * Contrast             - Centroid (xu, yu)
  - Average RGB        - Skewness (γ₁) & Kurtosis (γ₂)                    * Energy               - Slope (m)
                       - Root Mean Square (RMS)                           * Correlation          - Irregularity Index
                       - Total Energy (E)                                 * Entropy              - Equiv. Diameter
                       - Entropi Shannon (H)                              * Homogeneity          - Convex Area
                       - Smoothness (Q)                                 - Tamura Features        - Solidity (S)
                                                                          * Coarseness, Contrast,
                                                                            Directionality, Roughness
```

---

## 2. Karakteristik dan Representasi Citra Sampel (`24-068 Berantakan 1.jpeg`)

### 2.1 Dimensi Spasial dan Tensor 3D

Citra yang dijadikan objek pengujian utama pada penelitian ini adalah:
* **Nama Berkas:** `24-068 Berantakan 1.jpeg` (Koleksi NIM 240411100068)
* **Kondisi Objek:** Area meja/ruang dalam kondisi tidak tertata (*Berantakan*)
* **Resolusi Horizontal (Lebar / $W$):** $1600$ piksel
* **Resolusi Vertikal (Tinggi / $H$):** $1200$ piksel
* **Jumlah Kanal ($C$):** $3$ saluran warna (Red, Green, Blue)
* **Total Piksel Spasial per Kanal:** $N = W \times H = 1600 \times 1200 = 1.920.000$ piksel
* **Total Elemen Tensor Citra:** $1.920.000 \times 3 = 5.760.000$ elemen bernilai $[0, 255]$

### 2.2 Dekomposisi Kanal Warna RGB

Citra digital warna tersusun atas superposisi 3 bidang intensitas:
1. **Kanal Merah ($I_R$):** Membawa komponen spektral merah ($\approx 700\text{ nm}$).
2. **Kanal Hijau ($I_G$):** Membawa komponen spektral hijau ($\approx 546\text{ nm}$).
3. **Kanal Biru ($I_B$):** Membawa komponen spektral biru ($\approx 435\text{ nm}$).

![Dekomposisi Kanal RGB](output/citra_rgb_decomposition.png)

---

## 3. Materi Khusus: Transformasi Citra RGB ke Grayscale 8-Bit

### 3.1 Rasionalitas Konversi ke Grayscale

Banyak algoritma analisis tekstur (seperti GLCM dan Tamura), deteksi tepi (Sobel, Canny), serta segmentasi morfologi yang dirancang bekerja pada **sinyal intensitas tunggal 2D** ($I(x, y)$), bukan tensor 3D. 

Alasan konversi RGB ke Grayscale:
1. **Reduksi Dimensi Data:** Mengurangi volume data sebesar $66.7\%$ (dari 3 matriks menjadi 1 matriks), mempercepat pemrosesan algoritma secara signifikan.
2. **Fokus pada Informasi Struktural:** Mata manusia dan sistem penglihatan komputer mengenali objek, tepi, dan tekstur terutama melalui kontras luminansi (terang-gelap), bukan melalui nilai kromatisitas warna murni.
3. **Penyederhanaan Komputasi Matriks:** Matriks $1$ saluran memudahkan operasi konvolusi spasial, penapisan filter frekuensi, dan penentuan ambang batas (*thresholding*).

### 3.2 Landasan Fisiologis Penglihatan Manusia

Retina mata manusia memiliki sel fotoreseptor kerucut (*cone cells*) dengan sensitivitas yang tidak merata terhadap panjang gelombang cahaya:
* Sel kerucut tipe **M (Medium)** paling peka terhadap spektrum **Hijau** ($\approx 530\text{ nm}$).
* Sel kerucut tipe **L (Long)** peka terhadap spektrum **Merah** ($\approx 560\text{ nm}$).
* Sel kerucut tipe **S (Short)** paling sedikit jumlahnya dan kurang peka terhadap spektrum **Biru** ($\approx 420\text{ nm}$).

Oleh karena itu, konversi ke grayscale tidak boleh sekadar merata-ratakan nilai R, G, dan B secara seimbang, melainkan harus memberikan bobot pembobotan persepsi (*perceptual luminance weighting*).

```
Sensitivitas Spektral Mata Manusia:
Hijau (~59%) >>> Merah (~30%) >>> Biru (~11%)
```

### 3.3 Formulasi Matematis Metode Konversi

Terdapat beberapa metode utama untuk mengonversi citra warna RGB menjadi citra monokromatik (Grayscale):

#### 1. Metode Luminansi Standar ITU-R BT.601 (NTSC/PAL)
Metode pembobotan psikofisiologis standar yang diadopsi secara luas dalam penyiaran televisi dan pemrosesan citra digital:

$$Y_{\text{BT.601}} = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

Bobot hijau ($0.587$) paling tinggi karena mata manusia paling sensitif terhadap cahaya hijau.

#### 2. Metode Luminansi Standar ITU-R BT.709 (HDTV / sRGB)
Standar yang disesuaikan untuk monitor modern dan HDTV:

$$Y_{\text{BT.709}} = 0.2126 \cdot R + 0.7152 \cdot G + 0.0722 \cdot B$$

#### 3. Metode Rata-Rata Aritmatika Sederhana (Average Method)
Metode tanpa pembobotan, menghitung rata-rata nilai ketiga kanal:

$$Y_{\text{Average}} = \frac{R + G + B}{3}$$

Kelemahan metode ini adalah mengabaikan sensitivitas visual mata manusia, sehingga area berwarna hijau terang tampak lebih gelap dari aslinya, dan area biru tampak terlalu terang.

#### 4. Metode Desaturasi / Lightness
Mengambil nilai tengah antara intensitas saluran tertinggi dan terendah:

$$Y_{\text{Lightness}} = \frac{\max(R, G, B) + \min(R, G, B)}{2}$$

---

### 3.4 Kuantisasi dan Representasi Diskrit 8-Bit

Citra grayscale standar menggunakan kedalaman bit **8-bit per piksel** (`uint8`). Artinya, hasil perhitungan kontinu $Y$ harus dibulatkan (*rounding*) ke bilangan bulat terdekat dan dibatasi dalam rentang $[0, 255]$:

$$I_{\text{gray}}(i, j) = \text{clip}\left( \text{round}\left( Y(i, j) \right), 0, 255 \right)$$

* $I_{\text{gray}} = 0 \rightarrow$ Hitam mutlak (tidak ada cahaya).
* $I_{\text{gray}} = 128 \rightarrow$ Abu-abu netral (*mid-gray*).
* $I_{\text{gray}} = 255 \rightarrow$ Putih terang maksimal.

---

### 3.5 Visualisasi Perbandingan Metode Konversi

Berikut adalah visualisasi hasil konversi citra nyata `24-068 Berantakan 1.jpeg` ke grayscale 8-bit menggunakan Luminansi BT.601 vs Rata-Rata Sederhana:

![Perbandingan Konversi RGB ke Grayscale](output/citra_rgb_to_grayscale_comparison.png)

> **Analisis Gambar:**
> Panel paling kanan menampilkan selisih absolut $|Y_{\text{BT.601}} - Y_{\text{Average}}|$. Terlihat area dengan perbedaan intensitas hingga $8 - 15$ level keabuan, terutama pada objek berwarna merah hangat dan kayu di mana pembobotan mata manusia menghasilkan luminansi yang lebih alami dan bergradasi halus.

---

## 4. Kategori 1: Fitur Warna (Color Features)

### 4.1 Momen Warna (Color Moments) Orde 1 hingga 4

Momen warna adalah deskriptor statistik yang mencirikan distribusi probabilitas intensitas warna di dalam citra. Untuk setiap kanal warna $C \in \{R, G, B\}$ dengan dimensi $M \times N$:

#### 1. Momen Pertama: Mean ($\mu_C$)
Mengukur tendensi sentral atau tingkat kecerahan rata-rata kanal warna:

$$\mu_C = \frac{1}{M \times N} \sum_{i=1}^{M} \sum_{j=1}^{N} C(i, j)$$

#### 2. Momen Kedua: Varians ($\sigma_C^2$) dan Standar Deviasi ($\sigma_C$)
Mengukur sebaran atau dispersi warna di sekitar nilai rata-rata:

$$\sigma_C = \sqrt{\frac{1}{M \times N} \sum_{i=1}^{M} \sum_{j=1}^{N} \left( C(i, j) - \mu_C \right)^2}$$

#### 3. Momen Ketiga: Skewness ($\gamma_{1, C}$)
Mengukur tingkat asimetri distribusi warna terhadap nilai mean:

$$\text{Skewness}_C = \frac{1}{M \times N} \sum_{i=1}^{M} \sum_{j=1}^{N} \left( \frac{C(i, j) - \mu_C}{\sigma_C} \right)^3$$

* $\text{Skewness} > 0$: Ekor distribusi menjulur ke kanan (didominasi area gelap dengan sedikit highlight terang).
* $\text{Skewness} < 0$: Ekor distribusi menjulur ke kiri (didominasi area terang dengan sedikit bayangan gelap).
* $\text{Skewness} \approx 0$: Distribusi warna simetris seimbang di sekitar mean.

#### 4. Momen Keempat: Kurtosis ($\gamma_{2, C}$)
Mengukur ketajaman puncak (*tailedness/peakedness*) dari distribusi warna:

$$\text{Kurtosis}_C = \left[ \frac{1}{M \times N} \sum_{i=1}^{M} \sum_{j=1}^{N} \left( \frac{C(i, j) - \mu_C}{\sigma_C} \right)^4 \right] - 3$$

Pengurangan angka $3$ menghasilkan *Excess Kurtosis* (distribusi normal Gaussian bernilai $0$):
* $\text{Kurtosis} > 0$ (*Leptokurtik*): Distribusi berpuncak tajam dan berekor tebal (banyak nilai ekstrem/outlier).
* $\text{Kurtosis} < 0$ (*Platikurtik*): Distribusi berpuncak mendatar dan berekor tipis (sebaran nilai merata).

---

### 4.2 Histogram Warna (Color Histogram)

Histogram warna mencatat frekuensi kemunculan setiap nilai intensitas warna dalam bin tertentu. Jika digunakan $B$ bin per kanal, maka citra warna 3D dapat direpresentasikan sebagai histogram bersama (*joint histogram*) berdimensi $B \times B \times B$.

### 4.3 Rata-rata RGB (Average RGB)

Rata-rata RGB merefleksikan tanda tangan warna global (*global chromatic signature*) dari citra:

$$\mu_{\text{RGB}} = \frac{\mu_R + \mu_G + \mu_B}{3}$$

---

### 4.4 Analisis Kompleksitas Komputasi dan Memori

Tabel berikut menyajikan perbandingan efisiensi penyimpanan antara Momen Warna dan Histogram Warna (mengacu pada data empiris Hallur & Gavade, Elsevier 2025):

| Tipe Fitur Warna | Resolusi / Bin | Dimensi Vektor Fitur | Konsumsi Memori (`float32`) | Karakteristik Utama |
| :--- | :---: | :---: | :---: | :--- |
| **Color Moments (Mean, Std, Skew)** | $3$ kanal $\times 3$ momen | **$9$ fitur** | **$36$ Byte (Tetap)** | Sangat cepat, hemat memori, ideal untuk IoT & *real-time* |
| **Color Moments (Orde 1-4)** | $3$ kanal $\times 4$ momen | **$12$ fitur** | **$48$ Byte (Tetap)** | Representasi distribusi warna sangat lengkap |
| **Color Histogram ($8^3$)** | $8$ bin per kanal | $512$ bin | $2.048$ Byte ($2$ KB) | Menangkap detail kromatik, sensitif pencahayaan |
| **Color Histogram ($16^3$)** | $16$ bin per kanal | $4.096$ bin | $16.384$ Byte ($16$ KB) | Dimensi tinggi, memerlukan normalisasi |
| **Color Histogram ($32^3$)** | $32$ bin per kanal | $32.768$ bin | $131.072$ Byte (~$128$ KB) | Sangat berat, rentan *overfitting* pada ML |

---

## 5. Kategori 2: Fitur Statistik Intensitas Global (Statistical Features)

Fitur statistik mengevaluasi sifat non-deterministik dan sebaran tingkat keabuan pada citra grayscale $I(x, y)$ berukuran $M \times N$ ($N_{\text{total}} = M \times N$):

### 5.1 Mean ($\mu$) dan Varians ($\sigma^2$)
$$\mu = \frac{1}{N_{\text{total}}} \sum_{x=1}^{M} \sum_{y=1}^{N} I(x, y)$$

$$\text{Variance} = \sigma^2 = \frac{1}{N_{\text{total}}} \sum_{x=1}^{M} \sum_{y=1}^{N} \left( I(x, y) - \mu \right)^2$$

### 5.2 Standar Deviasi ($\sigma$ / Kontras Statistik)
Standar deviasi merupakan ukuran utama dari kontras intensitas global citra:

$$\text{Contrast}_{\text{stat}} = \sigma = \sqrt{\frac{1}{N_{\text{total}}} \sum_{x=1}^{M} \sum_{y=1}^{N} \left( I(x, y) - \mu \right)^2}$$

### 5.3 Root Mean Square (RMS)
Mengukur energi magnitudo rata-rata dari sinyal kecerahan citra:

$$\text{RMS} = \sqrt{\frac{1}{M \times N} \sum_{x=1}^{M} \sum_{y=1}^{N} I(x, y)^2}$$

### 5.4 Total Energy ($E$)
Konsentrasi kuadratik energi intensitas piksel secara kumulatif:

$$E = \sum_{x=1}^{M} \sum_{y=1}^{N} I(x, y)^2$$

### 5.5 Entropi Shannon ($H$)
Mengukur tingkat keacakan, ketidakteraturan (*randomness*), atau kompleksitas informasi dalam citra:

$$H = - \sum_{i=0}^{L-1} p(i) \log_2 p(i)$$

Dimana $L$ adalah jumlah tingkat keabuan ($L = 256$ untuk 8-bit), dan $p(i)$ adalah probabilitas kemunculan piksel dengan intensitas $i$. Nilai entropi yang mendekati $8$ bit menandakan tekstur visual yang sangat kaya dan tidak berpola tunggal.

### 5.6 Smoothness (Tingkat Kehalusan Relatif / $Q$)
Mengukur kehalusan relatif permukaan intensitas citra ($Q \in [0, 1)$):

$$Q = 1 - \frac{1}{1 + \sigma^2}$$

* Jika varians $\sigma^2 = 0$ (bidang warna polos sempurna), maka $Q = 0$ (sangat halus).
* Semakin besar varians kontras citra, nilai $Q$ akan mendekati $1$ (kasar/bertekstur).

---

## 6. Kategori 3: Fitur Tekstur Spasial (Spatial Texture Features)

### 6.1 Gray-Level Co-occurrence Matrix (GLCM) Haralick

Tekstur tidak dapat dijelaskan hanya melalui statistik piksel independen, melainkan membutuhkan hubungan spasial antar-piksel tetangga. **GLCM** mentabulasi seberapa sering pasangan piksel dengan intensitas keabuan tertentu muncul pada jarak offset $d$ dan orientasi sudut $\theta$ tertentu ($\theta = 0^\circ$ horizontal, $45^\circ$ diagonal, $90^\circ$ vertikal, atau $135^\circ$).

```
Offset Spasial Horizontal (d=1, θ=0°):
[ Piksel Acuan (x, y) ]  ---- d=1 ---->  [ Piksel Tetangga (x, y+1) ]
```

---

### 6.2 Simulasi Pembentukan GLCM Langkah demi Langkah (Matriks $3 \times 3$)

Untuk memahami cara kerja GLCM secara intuitif, berikut disajikan simulasi pembentukan matriks GLCM menggunakan contoh resmi dari materi perkuliahan UTM:

Diberikan citra grayscale sederhana berukuran $3 \times 3$ piksel dengan 3 tingkat keabuan ($0, 1, 2$):

$$\text{Citra Input } I = \begin{bmatrix} 0 & 1 & 1 \\ 0 & 2 & 1 \\ 2 & 2 & 0 \end{bmatrix}$$

Parameter offset: Jarak $d = 1$, arah horizontal kanan $\theta = 0^\circ$.

#### Langkah 1: Observasi Pasangan Piksel Horisontal
* **Baris 1:** $(0, 1)$ dan $(1, 1)$
* **Baris 2:** $(0, 2)$ dan $(2, 1)$
* **Baris 3:** $(2, 2)$ dan $(2, 0)$

*(Piksel terakhir di ujung kanan kolom tidak memiliki tetangga kanan, sehingga tidak berpasangan).*  
Total pasangan teramati = $6$ pasangan.

#### Langkah 2: Tabulasi Matriks Frekuensi GLCM Mentah
Baris matriks menyatakan nilai piksel asal ($i$), dan kolom menyatakan nilai piksel tetangga ($j$):

$$\text{GLCM Mentah} = \begin{bmatrix} 
0 & 1 & 1 \\ 
0 & 1 & 0 \\ 
1 & 1 & 1 
\end{bmatrix}$$

#### Langkah 3: Normalisasi Matriks Menjadi Probabilitas $P(i, j)$
Setiap elemen dibagi dengan total frekuensi ($6$):

$$P(i, j) = \begin{bmatrix} 
0 & 1/6 & 1/6 \\ 
0 & 1/6 & 0 \\ 
1/6 & 1/6 & 1/6 
\end{bmatrix} \approx \begin{bmatrix} 
0.0000 & 0.1667 & 0.1667 \\ 
0.0000 & 0.1667 & 0.0000 \\ 
0.1667 & 0.1667 & 0.1667 
\end{bmatrix}$$

Total penjumlahan $\sum_{i} \sum_{j} P(i, j) = 6 \times \frac{1}{6} = 1.0$.

---

### 6.3 Lima Atribut Tekstur GLCM (Haralick Features)

Dari matriks probabilitas $P(i, j)$, dihitung 5 fitur tekstur utama:

#### 1. Contrast (Kontras Spasial)
Mengukur variasi intensitas lokal antar-piksel bertetangga. Nilai tinggi menunjukkan transisi yang tajam (kasar):

$$\text{Contrast}_{\text{GLCM}} = \sum_{i=1}^{N} \sum_{j=1}^{N} (i - j)^2 P(i, j)$$

*Pada contoh $3 \times 3$:*
$$\text{Contrast} = (0-1)^2(1/6) + (0-2)^2(1/6) + (1-1)^2(1/6) + (2-0)^2(1/6) + (2-1)^2(1/6) + (2-2)^2(1/6)$$
$$\text{Contrast} = \frac{1 + 4 + 0 + 4 + 1 + 0}{6} = \frac{10}{6} \approx \mathbf{1.6667}$$

#### 2. Energy / Angular Second Moment (ASM)
Mengukur keseragaman (*uniformity*) atau keteraturan tekstur:

$$\text{Energy} = \sum_{i=1}^{N} \sum_{j=1}^{N} P(i, j)^2$$

*Pada contoh $3 \times 3$:* $\text{Energy} = 6 \times \left(\frac{1}{6}\right)^2 = \frac{6}{36} = \frac{1}{6} \approx \mathbf{0.1667}$

#### 3. Homogeneity / Inverse Difference Moment (IDM)
Mengukur kedekatan sebaran elemen GLCM terhadap garis diagonal utama:

$$\text{Homogeneity} = \sum_{i=1}^{N} \sum_{j=1}^{N} \frac{P(i, j)}{1 + |i - j|}$$

*Pada contoh $3 \times 3$:*
$$\text{Homogeneity} = \frac{1}{6} \left( \frac{1}{1+1} + \frac{1}{1+2} + \frac{1}{1+0} + \frac{1}{1+2} + \frac{1}{1+1} + \frac{1}{1+0} \right) = \frac{1}{6} \left( 0.5 + 0.333 + 1.0 + 0.333 + 0.5 + 1.0 \right) \approx \mathbf{0.6111}$$

#### 4. Entropy GLCM
Mengukur tingkat ketidakteraturan atau keacakan susunan spasial:

$$\text{Entropy}_{\text{GLCM}} = - \sum_{i=1}^{N} \sum_{j=1}^{N} P(i, j) \log_2 P(i, j)$$

*Pada contoh $3 \times 3$:* $\text{Entropy} = - 6 \times \left( \frac{1}{6} \log_2 \frac{1}{6} \right) = \log_2(6) \approx \mathbf{2.5850}$

#### 5. Correlation (Korelasi Linier Spasial)
Mengukur kebergantungan linier nilai keabuan antar-piksel bertetangga ($\rho \in [-1, 1]$):

$$\text{Correlation} = \sum_{i=1}^{N} \sum_{j=1}^{N} \frac{(i - \mu_i)(j - \mu_j) P(i, j)}{\sigma_i \sigma_j}$$

*Pada contoh $3 \times 3$:* $\text{Correlation} \approx \mathbf{-0.3153}$

![Analisis Tekstur GLCM](output/citra_glcm_texture_analysis.png)

---

### 6.4 Enam Atribut Tekstur Tamura (Persepsi Visual Manusia)

Metode Tamura merumuskan tekstur citra berdasarkan eksperimen psikofisika persepsi penglihatan manusia:

1. **Coarseness (Kekasaran Butir):** Menilai skala ukuran elemen tekstur fundamental (butiran halus vs pola kasar besar).
2. **Contrast (Kontras Tamura):**
   $$\text{Contrast}_{\text{Tamura}} = \frac{\sigma}{\mu_4^{1/4}}, \quad \text{dimana } \mu_4 = \frac{1}{M \times N} \sum \sum (I(i, j) - \mu)^4$$
3. **Directionality (Arah Pola):** Mengukur derajat orientasi garis atau gradien tepi dalam citra.
4. **Line-likeness (Kemiripan Garis):** Mengukur seberapa besar tekstur menyerupai pola garis lurus beraturan.
5. **Regularity (Keteraturan):** Menilai keseragaman pengulangan elemen tekstur di seluruh bidang citra:
   $$\text{Regularity} = \frac{1}{R} \left( \sigma_{\text{coarseness}} + \sigma_{\text{contrast}} + \sigma_{\text{directionality}} \right)$$
6. **Roughness (Kekasaran Permukaan):**
   $$\text{Roughness} = \text{Coarseness} + \text{Contrast}$$

---

## 7. Kategori 4: Fitur Bentuk dan Geometri (Shape & Geometry Features)

Fitur bentuk digunakan untuk mendeskripsikan morfologi dan ukuran fisik objek dalam citra setelah dilakukan proses segmentasi biner (pemisahan *foreground* objek dari *background*).

![Ekstraksi Fitur Geometris](output/citra_segmentasi_geometri.png)

### 7.1 Binarisasi dan Deteksi Region of Interest (ROI)
Citra grayscale dikonversi menjadi citra biner hitam-putih ($B(x, y) \in \{0, 1\}$) menggunakan algoritma penentuan ambang batas optimal **Otsu's Thresholding**. Pada citra `24-068 Berantakan 1.jpeg`, nilai ambang Otsu optimal terhitung pada nilai $T = 112$.

### 7.2 Luas (Area / $A$)
Jumlah piksel yang menyusun objek biner:

$$A = \sum_{i=1}^{M} \sum_{j=1}^{N} B(i, j)$$

### 7.3 Keliling (Perimeter / $P$)
Panjang batas terluar (*boundary pixels*) yang mengelilingi objek.

### 7.4 Sentroid / Pusat Massa ($x_u, y_u$)
Titik pusat gravitasi geometri dari objek:

$$x_u = \frac{\sum x \cdot B(x, y)}{A}, \quad y_u = \frac{\sum y \cdot B(x, y)}{A}$$

### 7.5 Indeks Irregularitas (Irregularity Index / $II$)
Mengukur deviasi bentuk objek dari bentuk lingkaran sempurna ($II \ge 1$):

$$II = \frac{P^2}{4 \pi A}$$

* Untuk lingkaran sempurna: $II = 1.0$.
* Semakin rumit, bergerigi, atau tak beraturan bentuk objek, nilai $II$ akan semakin besar.

### 7.6 Diameter Ekuivalen ($D$)
Diameter lingkaran imajiner yang memiliki luas sama dengan luas objek:

$$D = 2 \sqrt{\frac{A}{\pi}}$$

### 7.7 Convex Area ($A_{\text{convex}}$) dan Solidity ($S$)
* **Convex Area ($A_{\text{convex}}$):** Luas poligon cembung terkecil (*convex hull*) yang dapat membungkus seluruh objek.
* **Solidity ($S$):** Tingkat kepadatan atau keutuhan bentuk objek:
  $$S = \frac{A}{A_{\text{convex}}}$$
  Nilai $S = 1.0$ menunjukkan objek cembung pejal tanpa cekungan. Nilai $S < 1$ menunjukkan objek memiliki lekukan atau rongga.

---

## 8. Simulasi Perhitungan Manual Terpadu (Sub-Matriks $5 \times 5$)

Berikut adalah simulasi perhitungan manual lengkap pada sub-matriks $5 \times 5$ ($N = 25$ piksel) yang diambil dari citra riil `24-068 Berantakan 1.jpeg` pada koordinat baris $y \in [600, 604]$ dan kolom $x \in [800, 804]$:

### 8.1 Data Matriks Asli Patch $5 \times 5$

| Koordinat | Matriks Kanal Merah ($R$) | Matriks Kanal Hijau ($G$) | Matriks Kanal Biru ($B$) |
| :---: | :---: | :---: | :---: |
| Baris 0 | `[166, 166, 165, 165, 164]` | `[162, 162, 161, 161, 160]` | `[153, 153, 152, 152, 151]` |
| Baris 1 | `[160, 160, 159, 159, 159]` | `[156, 156, 155, 155, 155]` | `[147, 147, 146, 146, 146]` |
| Baris 2 | `[161, 161, 161, 160, 160]` | `[157, 157, 157, 156, 156]` | `[148, 148, 148, 147, 147]` |
| Baris 3 | `[162, 162, 162, 161, 161]` | `[158, 158, 158, 157, 157]` | `[149, 149, 149, 148, 148]` |
| Baris 4 | `[158, 158, 157, 157, 157]` | `[154, 154, 153, 153, 153]` | `[145, 145, 144, 144, 144]` |

### 8.2 Konversi Manual RGB ke Grayscale 8-Bit

Menggunakan formula luminansi ITU-R BT.601: $Y = \text{round}(0.299R + 0.587G + 0.114B)$:

* Piksel $(0, 0)$: $0.299(166) + 0.587(162) + 0.114(153) = 49.634 + 95.094 + 17.442 = 162.17 \rightarrow \mathbf{162}$
* Piksel $(0, 4)$: $0.299(164) + 0.587(160) + 0.114(151) = 49.036 + 93.920 + 17.214 = 160.17 \rightarrow \mathbf{160}$
* Piksel $(4, 4)$: $0.299(157) + 0.587(153) + 0.114(144) = 46.943 + 89.811 + 16.416 = 153.17 \rightarrow \mathbf{153}$

Matriks Grayscale 8-bit Terkuantisasi ($5 \times 5$):
$$\text{Patch Grayscale} = \begin{bmatrix}
162 & 162 & 161 & 161 & 160 \\
156 & 156 & 155 & 155 & 155 \\
157 & 157 & 157 & 156 & 156 \\
158 & 158 & 158 & 157 & 157 \\
154 & 154 & 153 & 153 & 153
\end{bmatrix}$$

### 8.3 Perhitungan Statistik Manual

1. **Jumlah Nilai ($\sum I_{\text{gray}}$):**
   $$(162+162+161+161+160) + (156+156+155+155+155) + (157+157+157+156+156) + (158+158+158+157+157) + (154+154+153+153+153)$$
   $$= 806 + 777 + 783 + 788 + 767 = \mathbf{3921}$$

2. **Mean Grayscale ($\mu_{\text{gray}}$):**
   $$\mu_{\text{gray}} = \frac{3921}{25} = \mathbf{156.8400}$$

3. **Jumlah Selisih Kuadrat ($\sum (I - \mu)^2$):**
   $$\sum_{k=1}^{25} (I_k - 156.84)^2 = \mathbf{175.3600}$$

4. **Varians ($\sigma^2$) dan Standar Deviasi ($\sigma$):**
   $$\sigma^2 = \frac{175.3600}{25} = \mathbf{7.0144}$$
   $$\sigma = \sqrt{7.0144} \approx \mathbf{2.6485}$$

5. **Root Mean Square (RMS):**
   $$\text{RMS} = \sqrt{\frac{\sum I^2}{25}} = \sqrt{\frac{615145}{25}} = \sqrt{24605.8} \approx \mathbf{156.8624}$$

6. **Smoothness ($Q$):**
   $$Q = 1 - \frac{1}{1 + \sigma^2} = 1 - \frac{1}{1 + 7.0144} = 1 - \frac{1}{8.0144} = 1 - 0.124775 = \mathbf{0.875225}$$

---

## 9. Hasil Ekstraksi Fitur Aktual pada Citra `24-068 Berantakan 1`

Berikut adalah tabel hasil ekstraksi fitur menyeluruh pada citra penuh `24-068 Berantakan 1.jpeg` ($1.920.000$ piksel):

### 9.1 Tabel Parameter Momen Warna RGB

| Kanal Warna | Mean ($\mu$) | Standar Deviasi ($\sigma$) | Varians ($\sigma^2$) | Skewness ($\gamma_1$) | Excess Kurtosis ($\gamma_2$) | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Kanal Merah ($R$)** | **$134.8988$** | **$56.2021$** | $3158.67$ | $-0.5372$ | $-0.4682$ | $0$ | $255$ |
| **Kanal Hijau ($G$)** | **$133.4873$** | **$55.0876$** | $3034.64$ | $-0.5367$ | $-0.4026$ | $0$ | $255$ |
| **Kanal Biru ($B$)** | **$128.8982$** | **$52.1753$** | $2722.26$ | $-0.5481$ | $-0.3809$ | $0$ | $255$ |
| **Rata-rata Gabungan RGB** | **$132.4281$** | **$54.5749$** | $2978.42$ | $-0.5410$ | $-0.4180$ | $0$ | $255$ |

### 9.2 Tabel Fitur Statistik Citra Grayscale 8-Bit (BT.601)

| Parameter Statistik | Simbol Formula | Nilai Terhitung | Keterangan Fisis |
| :--- | :---: | :---: | :--- |
| **Tingkat Kecerahan Rata-Rata** | $\mu$ | **$133.3659$** | Berada tepat di tengah rentang 8-bit (pencahayaan optimal) |
| **Varians Intensitas** | $\sigma^2$ | **$2997.6434$** | Dispersi energi pencahayaan luas |
| **Standar Deviasi (Kontras Global)** | $\sigma$ | **$54.7507$** | Kontras tinggi, gradasi terang-gelap dinamis |
| **Asimetri Distribusi (Skewness)** | $\gamma_1$ | **$-0.5417$** | Menceng ke kiri (*slightly left-skewed*) |
| **Kepuncakan (Excess Kurtosis)** | $\gamma_2$ | **$-0.4213$** | Platikurtik (kurva lebih landai dari Gaussian) |
| **Root Mean Square (RMS)** | $\text{RMS}$ | **$144.1669$** | Magnitudo intensitas efektif |
| **Total Energy Sinyal** | $E$ | **$3.9905 \times 10^{10}$** | Akumulasi kuadrat intensitas seluruh piksel |
| **Entropi Informasi Shannon** | $H$ | **$7.5174\text{ bit}$** | Mendekati batas maksimum ($8.0$), kaya detail tekstur |
| **Tingkat Kehalusan Relatif** | $Q$ | **$0.999667$** | Mendekati $1.0$ (citra bertekstur heterogen/kasar) |
| **Kontras Tamura** | $\text{Contrast}_{\text{Tamura}}$ | **$0.7891$** | Menilai kontras tekstur berbasis persepsi optik |

### 9.3 Tabel Fitur Tekstur GLCM Haralick (Kuantisasi 16 Level, $d=1, \theta=0^\circ$)

| Atribut GLCM | Nilai Terhitung | Interpretasi Tekstural |
| :--- | :---: | :--- |
| **Contrast** | **$0.3430$** | Fluktuasi intensitas antar-piksel bertetangga relatif rapat pada skala lokal |
| **Energy (ASM)** | **$0.0736$** | Nilai rendah menunjukkan tekstur acak/kompleks (*tidak seragam*) |
| **Homogeneity (IDM)**| **$0.9138$** | Mayoritas transisi piksel berdekatan terkonsentrasi di dekat diagonal |
| **Correlation** | **$0.9853$** | Ketergantungan linier spasial sangat kuat antar piksel horizontal |
| **Entropy GLCM** | **$4.5046$** | Keacakan spasial pasangan piksel tergolong tinggi |

### 9.4 Tabel Fitur Morfologi Geometri ROI Utama (Segmentasi Otsu $T=112$)

| Parameter Geometri | Simbol | Nilai Terukur | Satuan |
| :--- | :---: | :---: | :--- |
| **Luas Objek (Area)** | $A$ | $1.070.266,5$ | piksel ($55.7\%$ luas bidang citra) |
| **Keliling (Perimeter)** | $P$ | $16.250,0$ | piksel |
| **Koordinat Sentroid** | $(x_u, y_u)$ | $(828.8, 810.4)$ | piksel (berada di area tengah bawah) |
| **Indeks Irregularitas** | $II$ | $19.6338$ | Bentuk sangat tak beraturan ($II \gg 1.0$) |
| **Diameter Ekuivalen** | $D$ | $1167.3$ | piksel |
| **Luas Convex Hull** | $A_{\text{convex}}$ | $1.342.584,5$ | piksel |
| **Solidity** | $S$ | $0.7972$ | Memiliki lekukan kontur signifikan akibat barang berserakan |

---

## 10. Implementasi Pipeline Komputasi dengan Python

Berikut skrip modular Python lengkap untuk mengekstrak seluruh fitur di atas secara otomatis:

```python
"""
Pipeline Terintegrasi Ekstraksi Fitur Citra Digital
Mata Kuliah: Pengolahan Sinyal Digital (PSD)
Penulis: Muhammad Zaidan Nabil Rafi (240411100068)
"""

import cv2
import numpy as np
import scipy.stats as stats

def extract_all_image_features(image_path):
    # 1. Akuisisi Citra
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"File tidak ditemukan: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape
    total_pixels = h * w
    
    # 2. Pemisahan Kanal RGB
    r, g, b = rgb[:,:,0].astype(float), rgb[:,:,1].astype(float), rgb[:,:,2].astype(float)
    
    # Momen Warna RGB
    color_features = {
        'mean_r': float(np.mean(r)), 'std_r': float(np.std(r)), 'skew_r': float(stats.skew(r.ravel())),
        'mean_g': float(np.mean(g)), 'std_g': float(np.std(g)), 'skew_g': float(stats.skew(g.ravel())),
        'mean_b': float(np.mean(b)), 'std_b': float(np.std(b)), 'skew_b': float(stats.skew(b.ravel())),
        'mean_rgb_combined': float((np.mean(r) + np.mean(g) + np.mean(b)) / 3.0),
        'global_std_rgb': float(np.std(rgb))
    }
    
    # 3. Konversi ke Grayscale 8-bit (ITU-R BT.601)
    gray = np.round(0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    g_flt = gray.astype(float)
    
    # 4. Fitur Statistik Intensitas Global
    mu = float(np.mean(g_flt))
    var = float(np.var(g_flt))
    sigma = float(np.std(g_flt))
    skew = float(stats.skew(g_flt.ravel()))
    kurt = float(stats.kurtosis(g_flt.ravel())) # excess kurtosis
    rms = float(np.sqrt(np.mean(g_flt**2)))
    energy_total = float(np.sum(g_flt**2))
    smoothness = float(1.0 - (1.0 / (1.0 + var)))
    
    # Entropi Shannon
    counts, _ = np.histogram(gray, bins=256, range=(0, 256))
    p = counts[counts > 0] / float(total_pixels)
    shannon_entropy = float(-np.sum(p * np.log2(p)))
    
    # Tamura Contrast
    mu4 = float(np.mean((g_flt - mu)**4))
    tamura_contrast = float(sigma / (mu4 ** 0.25))
    
    stat_features = {
        'mean_gray': mu, 'var_gray': var, 'std_gray': sigma,
        'skew_gray': skew, 'kurtosis_gray': kurt, 'rms': rms,
        'energy_total': energy_total, 'shannon_entropy': shannon_entropy,
        'smoothness_q': smoothness, 'tamura_contrast': tamura_contrast
    }
    
    # 5. Fitur Tekstur Spasial GLCM (16 Level, d=1, θ=0°)
    levels = 16
    q_gray = (gray // (256 // levels)).astype(int)
    src = q_gray[:, :-1].ravel()
    dst = q_gray[:, 1:].ravel()
    glcm_counts = np.bincount(src * levels + dst, minlength=levels*levels).reshape((levels, levels)).astype(float)
    P = glcm_counts / np.sum(glcm_counts)
    
    i_idx, j_idx = np.indices((levels, levels))
    glcm_contrast = float(np.sum((i_idx - j_idx)**2 * P))
    glcm_energy = float(np.sum(P**2))
    glcm_homogeneity = float(np.sum(P / (1.0 + np.abs(i_idx - j_idx))))
    mask = P > 0
    glcm_entropy = float(-np.sum(P[mask] * np.log2(P[mask])))
    
    p_i = np.sum(P, axis=1)
    p_j = np.sum(P, axis=0)
    mu_i = np.sum(np.arange(levels) * p_i)
    mu_j = np.sum(np.arange(levels) * p_j)
    sig_i = np.sqrt(np.sum((np.arange(levels) - mu_i)**2 * p_i))
    sig_j = np.sqrt(np.sum((np.arange(levels) - mu_j)**2 * p_j))
    glcm_correlation = float(np.sum((i_idx - mu_i) * (j_idx - mu_j) * P) / (sig_i * sig_j))
    
    glcm_features = {
        'glcm_contrast': glcm_contrast, 'glcm_energy': glcm_energy,
        'glcm_homogeneity': glcm_homogeneity, 'glcm_entropy': glcm_entropy,
        'glcm_correlation': glcm_correlation
    }
    
    # 6. Fitur Morfologi Geometri (Otsu Thresholding)
    thresh_val, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest_c))
    perimeter = float(cv2.arcLength(largest_c, True))
    M = cv2.moments(largest_c)
    cx = float(M['m10'] / M['m00']) if M['m00'] != 0 else 0.0
    cy = float(M['m01'] / M['m00']) if M['m00'] != 0 else 0.0
    ii = float((perimeter**2) / (4 * np.pi * area)) if area > 0 else 0.0
    equiv_diam = float(2 * np.sqrt(area / np.pi)) if area > 0 else 0.0
    convex_area = float(cv2.contourArea(cv2.convexHull(largest_c)))
    solidity = float(area / convex_area) if convex_area > 0 else 0.0
    
    geom_features = {
        'otsu_thresh': float(thresh_val), 'area': area, 'perimeter': perimeter,
        'centroid_x': cx, 'centroid_y': cy, 'irregularity_index': ii,
        'equiv_diameter': equiv_diam, 'convex_area': convex_area, 'solidity': solidity
    }
    
    return {**color_features, **stat_features, **glcm_features, **geom_features}

# Contoh Eksekusi
if __name__ == '__main__':
    path = r'Foto citra/24-068 Berantakan 1.jpeg'
    results = extract_all_image_features(path)
    for k, v in results.items():
        print(f"{k:20s}: {v:12.4f}")
```

---

## 11. Analisis Komparatif: Citra Berantakan vs Rapi & Strategi Fusi Fitur

### 11.1 Evaluasi Fitur pada 6 Sampel Dataset

Tabel berikut membandingkan fitur statistik dan warna antar sampel citra dalam dataset:

| Nama File | Kategori | Mean R | Mean G | Mean B | Mean RGB | Std R | Std G | Std B | Std Global | Entropi (bit) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`24-068 Berantakan 1.jpeg`** | **Berantakan** | **134.90** | **133.49** | **128.90** | **132.43** | **56.20** | **55.09** | **52.18** | **54.57** | **7.52** |
| `24-068 Berantakan 2.jpeg` | Berantakan | 163.02 | 160.17 | 151.98 | 158.39 | 58.43 | 60.53 | 59.05 | 59.53 | 7.61 |
| `24-068 Berantakan 3.jpeg` | Berantakan | 165.02 | 154.04 | 145.38 | 154.81 | 56.48 | 59.73 | 62.33 | 60.10 | 7.58 |
| `24-068 Rapi 1.jpeg` | Rapi | 124.46 | 125.32 | 120.72 | 123.50 | 49.36 | 54.91 | 56.32 | 53.65 | 7.39 |
| `24-068 Rapi 2.jpeg` | Rapi | 130.51 | 133.04 | 130.78 | 131.44 | 49.96 | 55.11 | 58.48 | 54.64 | 7.42 |
| `24-068 Rapi 3.jpeg` | Rapi | 146.69 | 149.35 | 143.92 | 146.65 | 44.28 | 48.36 | 49.93 | 47.64 | 7.15 |

![Komparasi Dataset](output/citra_komparasi_dataset.png)

---

### 11.2 Rangkuman Kelebihan, Kekurangan, Efisiensi, dan Akurasi

Merujuk pada sintesis komparatif **Table 5 pada Hallur & Gavade (Franklin Open 2025)**:

| No | Kategori Fitur | Metode Ekstraksi | Kelebihan (*Merits*) | Kekurangan (*Demerits*) | Domain Aplikasi Utama | Efisiensi Komputasi | Akurasi Diskriminatif |
| :-: | :--- | :--- | :--- | :--- | :--- | :-: | :-: |
| 1 | **Warna** | **Color Moments** | Komputasi sangat cepat, vektor berukuran kecil (9 float), invarian terhadap rotasi | Kehilangan informasi spasial, sensitif terhadap perubahan cahaya ekstrem | *Image retrieval*, pengenalan objek cepat | **Tinggi** | **Moderat** |
| 2 | **Warna** | **Color Histogram** | Menangkap detail kromatisitas kaya, implementasi mudah | Dimensi fitur tinggi, sensitif pergeseran intensitas | CBIR, klasifikasi pemandangan | **Moderat** | **Moderat** |
| 3 | **Warna** | **Average RGB** | Komputasi instan $O(N)$, representasi global ringkas | Kehilangan detail warna lokal dan distribusi | Kategorisasi awal citra | **Sangat Tinggi** | **Rendah** |
| 4 | **Tekstur** | **GLCM Haralick** | Menangkap relasi spasial piksel, deskriptor tekstur sangat kuat | Memerlukan kuantisasi level keabuan, lambat jika tanpa vektorisasi | Analisis citra medis, klasifikasi material | **Rendah** | **Tinggi** |
| 5 | **Tekstur** | **Tamura Features**| Memodelkan persepsi visual manusia (kasar/halus, kontras, arah) | Terbatas pada jenis tekstur naturalis tertentu | Analisis permukaan, inspeksi visual industri | **Moderat** | **Moderat** |
| 6 | **Statistik** | **Mean & Std Dev** | Sederhana, mengukur kecerahan global dan kontras secara objektif | Tidak membedakan struktur geometris objek | Kategorisasi citra dasar, prapemrosesan | **Sangat Tinggi**| **Rendah-Moderat** |
| 7 | **Statistik** | **Entropy Shannon**| Mengukur ketidakteraturan sinyal dan kompleksitas informasi | Komputasi memerlukan histogram frekuensi | Segmentasi citra, kompresi data | **Moderat** | **Tinggi** |
| 8 | **Statistik** | **RMS & Energy** | Mengukur konsentrasi energi kuadratik sinyal secara efektif | Sensitif terhadap noise pencahayaan | *Image quality assessment* | **Tinggi** | **Moderat** |
| 9 | **Statistik** | **Skewness & Kurtosis**| Menangkap asimetri dan kepuncakan distribusi (ekor tebal/outlier) | Interpretasi non-linear, rentan noise ekstrem | Deteksi anomali citra | **Moderat** | **Moderat** |
| 10 | **Geometri** | **Area & Perimeter** | Intuitif, mengukur dimensi fisik objek biner secara presisi | Kehilangan struktur internal objek | Analisis bentuk, pengenalan siluet | **Tinggi** | **Moderat** |
| 11 | **Geometri** | **Centroid & Solidity**| Mengetahui posisi spasial dan derajat kepadatan bentuk objek | Membutuhkan tahap segmentasi biner yang akurat | Pelacakan objek (*tracking*), biometrik | **Moderat** | **Moderat** |

---

### 11.3 Strategi Fusi Multi-Fitur

Tidak ada satu fitur tunggal yang sempurna untuk semua kondisi citra. Oleh karena itu, pendekatan terbaik adalah menggabungkan beberapa fitur (*multi-feature fusion*):

1. **Feature-Level Fusion (Early Fusion):**  
   Menggabungkan vektor fitur warna, statistik, dan tekstur menjadi satu vektor gabungan:
   $$\mathbf{x}_{\text{fused}} = \left[ \mu_R, \sigma_R, \dots, \mu_{\text{gray}}, \sigma_{\text{gray}}, H, \text{Contrast}_{\text{GLCM}}, \text{Energy}_{\text{GLCM}}, \dots \right]^T$$
   Kelebihan: Model machine learning (seperti SVM atau Random Forest) dapat mempelajari korelasi silang antar-fitur secara simultan.
2. **Decision-Level Fusion (Late Fusion):**  
   Melatih klasifier terpisah pada masing-masing modalitas fitur (misal: satu model untuk warna, satu untuk tekstur), lalu menggabungkan hasil prediksi melalui *soft voting* atau pembobotan probabilitas Bayesian.

---

## 12. Kesimpulan

1. **Konversi RGB ke Grayscale 8-Bit:** Merupakan langkah krusial dalam pemrosesan citra digital yang mereduksi dimensi data sebesar $66.7\%$ ($3$ kanal menjadi $1$ kanal) tanpa menghilangkan karakteristik spasial dan kontras. Metode pembobotan Luminansi ITU-R BT.601 ($Y = 0.299R + 0.587G + 0.114B$) menghasilkan representasi keabuan yang paling sesuai dengan sensitivitas fisiologis mata manusia.
2. **Karakteristik Citra `24-068 Berantakan 1.jpeg`:**
   - **Momen Warna RGB:** Rata-rata kecerahan per kanal terukur $\mu_R = 134.90$, $\mu_G = 133.49$, $\mu_B = 128.90$, dengan Mean Gabungan RGB $\mu_{\text{RGB}} = 132.43$. Nilai $\mu_R > \mu_G > \mu_B$ mencerminkan pencahayaan lampu ruangan hangat (*warm lighting*).
   - **Fitur Statistik Grayscale:** Standar deviasi $\sigma = 54.75$ dan Entropi Shannon $H = 7.52\text{ bit}$ membuktikan dinamika kontras yang tinggi dan variasi tekstur visual yang kaya akibat benda-benda yang tersebar acak.
   - **Fitur Tekstur Spasial GLCM:** Menghasilkan kontras lokal $0.3430$, korelasi spasial tinggi $0.9853$, dan energi $0.0736$, mencerminkan struktur permukaan yang kompleks dan heterogen.
   - **Morfologi Geometris:** Binarisasi Otsu ($T=112$) menghasilkan Indeks Irregularitas tinggi ($II = 19.63$) dan Solidity $0.7972$, menegaskan ketidakteraturan bentuk tumpukan objek pada kondisi "Berantakan".
3. **Validasi Matematis & Implementasi:** Simulasi perhitungan manual pada sub-matriks $5 \times 5$ dan contoh GLCM $3 \times 3$ membuktikan ketepatan alur komputasi diskrit di balik algoritma Python modern.

---

## 13. Referensi

1. Hallur, S., & Gavade, A. (2025). Image feature extraction techniques: A comprehensive review. *Franklin Open*, 12, 100366. Elsevier. https://doi.org/10.1016/j.fraope.2025.100366
2. Mulaab. *Teknik Ekstraksi Fitur Citra*. Jurusan Teknik Informatika, Fakultas Teknik, Universitas Trunojoyo Madura (UTM).
3. Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural features for image classification. *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-3(6), 610–621.
4. Tamura, H., Mori, S., & Yamawaki, T. (1978). Textural features corresponding to visual perception. *IEEE Transactions on Systems, Man, and Cybernetics*, 8(6), 460–473.
5. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
6. International Telecommunication Union. (2011). *Studio encoding parameters of digital television for standard 4:3 and wide-screen 16:9 aspect ratios* (Recommendation ITU-R BT.601-7).
7. Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62–66.
