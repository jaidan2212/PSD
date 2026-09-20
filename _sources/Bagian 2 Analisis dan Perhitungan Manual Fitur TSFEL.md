# Bagian 2: Analisis dan Perhitungan Manual Fitur TSFEL

## 1. Teori Dasar

### 1.1 Fitur `median_diff`

Fitur **`median_diff(signal)`** merupakan fitur pada **domain temporal** yang menghitung nilai median dari selisih antara dua titik data yang berurutan pada suatu sinyal *time-series*. Secara matematis, selisih antar data dapat dituliskan sebagai:

$$
d_i = x_{i+1} - x_i
$$

dengan:

* $x_i$ = nilai sinyal pada titik ke-$i$.
* $d_i$ = selisih antara dua titik yang berurutan.
* $n$ = jumlah titik data.

Selanjutnya, seluruh nilai selisih tersebut diurutkan dan dicari nilai tengahnya atau **median**.

Secara umum:

$$
\text{median\_diff} = \operatorname{median}(x_{i+1}-x_i)
$$

Implementasi TSFEL menggunakan operasi `np.diff(signal)` untuk memperoleh selisih antar titik, kemudian menggunakan `np.median()` untuk mendapatkan nilai mediannya.

### Interpretasi pada Data Kualitas Udara

Pada data kualitas udara, fitur `median_diff` dapat digunakan untuk menggambarkan **perubahan tipikal atau perubahan tengah** konsentrasi suatu polutan dari satu pengamatan ke pengamatan berikutnya.

Untuk polutan **CO (Karbon Monoksida)**:

* Nilai `median_diff` **positif** menunjukkan bahwa perubahan konsentrasi CO secara tipikal cenderung meningkat.
* Nilai `median_diff` **negatif** menunjukkan kecenderungan penurunan.
* Nilai yang mendekati **0** menunjukkan bahwa perubahan antar pengamatan relatif kecil atau cenderung stabil.

Kelebihan penggunaan median adalah sifatnya yang relatif tidak mudah dipengaruhi oleh satu perubahan yang sangat besar dibandingkan dengan penggunaan rata-rata (*mean*).

---

### 1.2 Fitur `median_frequency`

Fitur **`median_frequency(signal, fs)`** merupakan fitur pada **domain frekuensi** yang digunakan untuk menentukan frekuensi yang membagi akumulasi magnitudo spektrum menjadi sekitar **50% bagian di bawahnya dan 50% bagian di atasnya**.

Dalam implementasi TSFEL, sinyal terlebih dahulu diubah dari **domain waktu** ke **domain frekuensi** menggunakan *Fast Fourier Transform* (FFT). TSFEL menggunakan magnitudo dari hasil FFT:

$$
F_k = |\operatorname{FFT}(x)_k|
$$

Kemudian dilakukan perhitungan frekuensi yang bersesuaian dengan setiap komponen FFT:

$$
f_k = \operatorname{rfftfreq}(N, d=1/fs)
$$

Selanjutnya, nilai magnitudo dijumlahkan secara kumulatif:

$$
C_k = \sum_{i=0}^{k}F_i
$$

Frekuensi median ditentukan sebagai frekuensi pertama yang memenuhi:

$$
C_k > 0.50 \times C_{\text{total}}
$$

Dengan demikian:

$$
f_{\text{median}} = f_k
$$

TSFEL secara spesifik mendokumentasikan `median_frequency` sebagai fitur yang mencari **50% dari total spektrum menggunakan cumulative sum (cumsum)**. Implementasinya menggunakan magnitudo FFT, bukan kuadrat magnitudo FFT.

### Interpretasi pada Data Kualitas Udara

Pada data *time-series* kualitas udara, `median_frequency` menggambarkan **distribusi perubahan sinyal CO pada domain frekuensi**.

Jika nilai median frequency relatif rendah, sebagian besar magnitudo spektrum berada pada frekuensi rendah. Hal ini dapat menunjukkan bahwa perubahan konsentrasi CO lebih banyak terjadi secara perlahan.

Sebaliknya, nilai median frequency yang lebih tinggi menunjukkan bahwa distribusi spektrum relatif lebih banyak berada pada frekuensi yang lebih tinggi, yang berkaitan dengan perubahan sinyal yang lebih cepat.

Perlu diperhatikan bahwa nilai fitur ini sangat bergantung pada **frekuensi sampling (`fs`)** dan panjang data yang digunakan.

---

# 2. Perhitungan Manual (Coret-coretan)

Data sampel konsentrasi **CO (Karbon Monoksida)** yang digunakan adalah:

$$
signal = [0.021,\ 0.025,\ 0.024,\ 0.029,\ 0.031]
$$

Jumlah data:

$$
N = 5
$$

Frekuensi sampling yang digunakan:

$$
fs = 1
$$

---

## 2.1 Perhitungan Manual `median_diff`

### Langkah 1 — Menghitung selisih antar titik

Selisih setiap dua titik yang berurutan dihitung menggunakan:

$$
d_i = x_{i+1}-x_i
$$

Data:

$$
[0.021,\ 0.025,\ 0.024,\ 0.029,\ 0.031]
$$

Maka:

**Selisih pertama:**

$$
d_1 = 0.025 - 0.021 = 0.004
$$

**Selisih kedua:**

$$
d_2 = 0.024 - 0.025 = -0.001
$$

**Selisih ketiga:**

$$
d_3 = 0.029 - 0.024 = 0.005
$$

**Selisih keempat:**

$$
d_4 = 0.031 - 0.029 = 0.002
$$

Sehingga diperoleh:

$$
d = [0.004,\ -0.001,\ 0.005,\ 0.002]
$$

### Langkah 2 — Mengurutkan nilai selisih

Nilai selisih kemudian diurutkan dari yang terkecil hingga terbesar:

$$
[-0.001,\ 0.002,\ 0.004,\ 0.005]
$$

### Langkah 3 — Menentukan median

Karena terdapat **4 nilai**, median merupakan rata-rata dari dua nilai yang berada di tengah:

$$
\text{Median} =
\frac{0.002+0.004}{2}
$$

$$
\text{Median} = \frac{0.006}{2}
$$

$$
\boxed{\text{median\_diff}=0.003}
$$

Jadi, hasil perhitungan manual fitur `median_diff` adalah:

**`0.003`**

---

# 2.2 Perhitungan Manual `median_frequency`

Perhitungan FFT secara manual dapat menjadi panjang apabila dilakukan menggunakan persamaan kompleks untuk setiap frekuensi. Oleh karena itu, proses berikut menjelaskan **alur matematis yang dilakukan oleh algoritma TSFEL** secara bertahap.

### Langkah 1 — Sinyal awal

Digunakan sinyal:

$$
x=[0.021,\ 0.025,\ 0.024,\ 0.029,\ 0.031]
$$

dengan:

$$
N=5
$$

dan:

$$
fs=1
$$

---

### Langkah 2 — Mengubah sinyal dari domain waktu ke domain frekuensi

Transformasi dilakukan menggunakan **Fast Fourier Transform (FFT)**.

Secara umum, DFT dapat dituliskan:

$$
X_k = \sum_{n=0}^{N-1}x_n e^{-j2\pi kn/N}
$$

FFT merupakan algoritma yang digunakan untuk menghitung transformasi tersebut secara lebih efisien.

TSFEL kemudian mengambil magnitudo dari hasil FFT:

$$
F_k=|X_k|
$$

Untuk sinyal dengan $N=5$, `rfft` menghasilkan frekuensi:

$$
f=[0,\ 0.2,\ 0.4]
$$

Dengan $fs=1$, resolusi frekuensinya adalah:

$$
\Delta f=\frac{fs}{N}
$$

$$
\Delta f=\frac{1}{5}=0.2
$$

---

### Langkah 3 — Menghitung magnitudo FFT

Untuk data:

$$
[0.021,\ 0.025,\ 0.024,\ 0.029,\ 0.031]
$$

magnitudo FFT yang diperoleh secara numerik adalah:

$$
F \approx
[0.130000,\ 0.009780,\ 0.008022]
$$

Sehingga pasangan frekuensi dan magnitudonya adalah:

| Frekuensi | Magnitudo FFT |
| --------: | ------------: |
|       0.0 |      0.130000 |
|       0.2 |      0.009780 |
|       0.4 |      0.008022 |

> **Catatan:** Pada implementasi `median_frequency` TSFEL, yang dijumlahkan adalah **magnitudo FFT**, yaitu $|FFT|$. Jadi, tidak dilakukan pengkuadratan magnitudo menjadi $|FFT|^2$ pada fitur ini.

---

### Langkah 4 — Menghitung magnitudo kumulatif

Magnitudo kemudian dijumlahkan secara kumulatif (*cumulative sum*):

$$
C_0=0.130000
$$

$$
C_1=0.130000+0.009780
$$

$$
C_1\approx0.139780
$$

$$
C_2=0.139780+0.008022
$$

$$
C_2\approx0.147802
$$

Sehingga:

$$
C=[0.130000,\ 0.139780,\ 0.147802]
$$

Total magnitudo:

$$
C_{\text{total}}\approx0.147802
$$

---

### Langkah 5 — Menentukan batas 50%

Median frequency menggunakan batas 50% dari total magnitudo:

$$
0.50\times C_{\text{total}}
$$

$$
=0.50\times0.147802
$$

$$
\approx0.073901
$$

Jadi, batas median adalah:

$$
\boxed{0.073901}
$$

---

### Langkah 6 — Menentukan frekuensi median

Sekarang bandingkan magnitudo kumulatif dengan batas 50%:

| Frekuensi | Magnitudo | Kumulatif | Melewati 50%? |
| --------: | --------: | --------: | :-----------: |
|       0.0 |  0.130000 |  0.130000 |     **Ya**    |
|       0.2 |  0.009780 |  0.139780 |       Ya      |
|       0.4 |  0.008022 |  0.147802 |       Ya      |

Pada frekuensi pertama, yaitu:

$$
f=0
$$

nilai kumulatif sudah lebih besar daripada 50% total magnitudo:

$$
0.130000 > 0.073901
$$

Oleh karena itu:

$$
\boxed{\text{median\_frequency}=0.0}
$$

Jadi, berdasarkan algoritma yang digunakan TSFEL, hasil `median_frequency` untuk data sampel tersebut adalah:

**`0.0`**

---

# 3. Pembuktian dengan Python

Pembuktian dilakukan menggunakan **NumPy**, **SciPy**, dan **TSFEL**. NumPy digunakan untuk pengolahan array dan perhitungan numerik, SciPy digunakan untuk membantu proses FFT secara independen, sedangkan TSFEL digunakan untuk menghitung fitur menggunakan fungsi aslinya.

```python
import numpy as np
import scipy
from scipy.fft import rfft, rfftfreq
import tsfel.feature_extraction.features as tsfel_features

# ============================================================
# DATA SAMPEL
# ============================================================

signal = np.array([0.021, 0.025, 0.024, 0.029, 0.031])
fs = 1

print("===================================================")
print("DATA SAMPEL CO")
print("===================================================")
print("Signal :", signal)
print("fs     :", fs)


# ============================================================
# 1. PEMBUKTIAN median_diff
# ============================================================

print("\n===================================================")
print("1. PEMBUKTIAN FITUR median_diff")
print("===================================================")

# ----- PERHITUNGAN MANUAL -----

# Menghitung selisih antar titik yang berurutan
diff_manual = np.diff(signal)

# Mengurutkan nilai difference
diff_sorted = np.sort(diff_manual)

# Menghitung median
hasil_manual_diff = np.median(diff_manual)

# ----- PERHITUNGAN TSFEL -----

hasil_tsfel_diff = tsfel_features.median_diff(signal)

print("Sinyal asli       :", signal)
print("Difference        :", diff_manual)
print("Difference urut   :", diff_sorted)

print(f"Hasil Manual      : {hasil_manual_diff:.5f}")
print(f"Hasil TSFEL       : {hasil_tsfel_diff:.5f}")

print(
    "Status            :",
    "SAMA PERSIS" if np.isclose(
        hasil_manual_diff,
        hasil_tsfel_diff
    ) else "BERBEDA"
)


# ============================================================
# 2. PEMBUKTIAN median_frequency
# ============================================================

print("\n===================================================")
print("2. PEMBUKTIAN FITUR median_frequency")
print("===================================================")

# ----- PERHITUNGAN MANUAL SESUAI ALGORITMA TSFEL -----

# FFT sinyal
fft_values = np.abs(rfft(signal))

# Deret frekuensi
freqs = rfftfreq(len(signal), d=1/fs)

# Magnitudo kumulatif
cum_magnitude = np.cumsum(fft_values)

# Total magnitudo
total_magnitude = cum_magnitude[-1]

# Batas 50%
threshold = total_magnitude * 0.50

# Mencari indeks pertama yang melewati 50%
median_freq_index = np.where(
    cum_magnitude > threshold
)[0][0]

# Frekuensi median
hasil_manual_freq = freqs[median_freq_index]


# ----- PERHITUNGAN TSFEL -----

hasil_tsfel_freq = tsfel_features.median_frequency(
    signal,
    fs=fs
)


# ----- MENAMPILKAN HASIL -----

print("Deret Frekuensi   :", freqs)
print("Magnitudo FFT     :", np.round(fft_values, 6))
print("Magnitudo Kumulatif:",
      np.round(cum_magnitude, 6))

print(f"Total Magnitudo   : {total_magnitude:.6f}")
print(f"Batas 50%         : {threshold:.6f}")

print(f"Hasil Manual      : {hasil_manual_freq:.5f}")
print(f"Hasil TSFEL       : {hasil_tsfel_freq:.5f}")

print(
    "Status            :",
    "SAMA PERSIS" if np.isclose(
        hasil_manual_freq,
        hasil_tsfel_freq
    ) else "BERBEDA"
)


# ============================================================
# KESIMPULAN PEMBUKTIAN
# ============================================================

print("\n===================================================")
print("KESIMPULAN")
print("===================================================")

print(
    f"median_diff      : {hasil_manual_diff:.5f} "
    f"== {hasil_tsfel_diff:.5f}"
)

print(
    f"median_frequency : {hasil_manual_freq:.5f} "
    f"== {hasil_tsfel_freq:.5f}"
)
```

## Hasil yang Diharapkan

Berdasarkan perhitungan manual dan implementasi algoritma TSFEL, diperoleh:

| Fitur              | Hasil Manual | Hasil TSFEL |
| ------------------ | -----------: | ----------: |
| `median_diff`      |  **0.00300** | **0.00300** |
| `median_frequency` |  **0.00000** | **0.00000** |

Dengan demikian, kedua hasil perhitungan manual dapat dibuktikan konsisten dengan fungsi yang tersedia pada TSFEL untuk data sampel yang digunakan.

---

# Bagian 1: Eksplorasi Data (EDA) Time Series pada Setiap Polutan

### 1. Pengantar EDA Time Series

**Exploratory Data Analysis (EDA)** pada data *time series* dilakukan sebagai tahap awal untuk memahami karakteristik, pola, serta kualitas data sebelum digunakan dalam proses pemodelan *machine learning*. Pada data kualitas udara, EDA digunakan untuk mengamati perubahan konsentrasi polutan dari waktu ke waktu, mengidentifikasi nilai yang tidak wajar, serta memastikan bahwa data memiliki struktur yang sesuai untuk analisis lebih lanjut.

Polutan yang dianalisis meliputi **Karbon Monoksida (CO)**, **Nitrogen Dioksida (NO₂)**, dan **Sulfur Dioksida (SO₂)**. Karena data memiliki dimensi waktu, urutan pengamatan sangat penting dalam analisis. Oleh karena itu, data perlu diurutkan berdasarkan tanggal sebelum dilakukan proses pembersihan dan analisis.

Secara umum, tahapan EDA yang dilakukan meliputi:

* **Memuat dan memeriksa data** serta memastikan kolom waktu memiliki format yang benar.
* **Mengidentifikasi anomali atau kesalahan pembacaan sensor**.
* **Mendeteksi outlier** menggunakan metode IQR.
* **Melakukan imputasi nilai yang hilang** berdasarkan informasi waktu.
* **Membandingkan data mentah dan data bersih** melalui visualisasi.

#### Contoh kode: Membaca dan menyiapkan data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Mengatur tampilan grafik
sns.set_theme(style="whitegrid")

# Membaca data
df = pd.read_csv('data_co_kamal_fix2.csv')

# Menyesuaikan nama kolom tanggal
df = df.rename(columns={'Tanggal': 'date', 'time': 'date'})

# Mengubah kolom date menjadi format datetime
df['date'] = pd.to_datetime(df['date'])

# Mengurutkan data berdasarkan waktu
df = df.sort_values('date').reset_index(drop=True)

# Mengubah nilai polutan menjadi numerik
df['CO'] = pd.to_numeric(df['CO'], errors='coerce')

# Melihat informasi awal data
print(df.head())
print(df.info())
```

---

### 2. Identifikasi Anomali (Error Sensor)

Berdasarkan pemeriksaan terhadap **data mentah**, ditemukan nilai anomali berupa angka **`-9999`** pada beberapa pengukuran polutan.

Nilai `-9999` tersebut **bukan merupakan konsentrasi polutan yang sebenarnya**, melainkan kode yang digunakan untuk menunjukkan adanya **error atau kegagalan sensor dalam melakukan pembacaan**.

Apabila nilai tersebut dibiarkan sebagai nilai numerik biasa, maka akan menyebabkan beberapa permasalahan:

* **Mean** menjadi tidak representatif karena nilai `-9999` sangat jauh dari nilai pengukuran normal.
* **Standard deviation** menjadi terdistorsi karena penyebaran data terlihat jauh lebih besar.
* Nilai `-9999` dapat dianggap sebagai **outlier ekstrem** oleh metode statistik.
* Pola *time series* menjadi tidak natural karena muncul penurunan nilai yang tidak menggambarkan kondisi atmosfer sebenarnya.

Oleh karena itu, sebelum melakukan analisis lebih lanjut, nilai `-9999` perlu diidentifikasi dan diperlakukan sebagai **data tidak valid**.

#### Contoh kode: Mengecek jumlah error `-9999`

```python
# Menghitung jumlah nilai -9999
jumlah_error = (df['CO'] == -9999).sum()

print("Jumlah nilai -9999:", jumlah_error)
```

Untuk melihat posisi nilai error berdasarkan waktu:

```python
# Menampilkan data yang memiliki nilai -9999
data_error = df[df['CO'] == -9999]

print(data_error[['date', 'CO']].head(10))
```

Untuk mengetahui persentase data yang mengalami error:

```python
persentase_error = (jumlah_error / len(df)) * 100

print(f"Persentase data error: {persentase_error:.2f}%")
```

---

### 3. Strategi Pembersihan Data (Imputasi)

Pembersihan data dilakukan secara bertahap untuk memastikan data yang digunakan pada proses selanjutnya memiliki kualitas yang lebih baik.

Tahapan utama yang digunakan adalah:

**Data Mentah → Penghapusan Kode Error → Deteksi Outlier IQR → Imputasi Time Interpolation → Data Bersih**

---

#### 3.1 Mengubah Nilai `-9999` Menjadi Null (NaN)

Tahap pertama adalah mengganti nilai `-9999` dengan **`NaN` (*Not a Number*)**.

Pengubahan ini bertujuan agar nilai tersebut tidak dianggap sebagai konsentrasi polutan yang valid. Dengan menjadi `NaN`, nilai tersebut dapat dikenali sebagai data yang hilang dan kemudian diproses menggunakan metode imputasi.

#### Contoh kode: Mengubah `-9999` menjadi `NaN`

```python
# Mengubah nilai error -9999 menjadi NaN
df['CO'] = df['CO'].replace(-9999, np.nan)
df['CO'] = df['CO'].replace(-9999.0, np.nan)

# Mengecek jumlah nilai NaN
print("Jumlah NaN setelah penggantian error:",
      df['CO'].isna().sum())
```

Setelah proses tersebut, nilai `-9999` tidak lagi dianggap sebagai nilai pengukuran.

Untuk memastikan hasilnya:

```python
print(df[df['CO'].isna()][['date', 'CO']].head())
```

---

#### 3.2 Mendeteksi Outlier Tambahan Menggunakan IQR

Tidak semua data ekstrem berbentuk `-9999`. Oleh karena itu, dilakukan pemeriksaan tambahan menggunakan metode **Interquartile Range (IQR)**.

IQR dihitung berdasarkan:

**IQR = Q3 − Q1**

dengan:

* **Q1** = kuartil pertama atau persentil ke-25.
* **Q3** = kuartil ketiga atau persentil ke-75.

Kemudian ditentukan batas:

**Batas Bawah = Q1 − 1,5 × IQR**

**Batas Atas = Q3 + 1,5 × IQR**

Nilai yang berada di luar batas tersebut dianggap sebagai **outlier** dan diubah menjadi `NaN`.

#### Contoh kode: Deteksi outlier menggunakan IQR

```python
# Menghitung Q1 dan Q3
Q1 = df['CO'].quantile(0.25)
Q3 = df['CO'].quantile(0.75)

# Menghitung IQR
IQR = Q3 - Q1

# Menentukan batas bawah dan atas
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Q1 =", Q1)
print("Q3 =", Q3)
print("IQR =", IQR)
print("Batas bawah =", lower_bound)
print("Batas atas =", upper_bound)
```

Selanjutnya, nilai yang berada di luar batas tersebut diubah menjadi `NaN`:

```python
# Mendeteksi dan mengubah outlier menjadi NaN
kondisi_outlier = (
    (df['CO'] < lower_bound) |
    (df['CO'] > upper_bound)
)

df.loc[kondisi_outlier, 'CO'] = np.nan

# Menghitung jumlah data yang menjadi NaN
print("Jumlah NaN setelah deteksi outlier:",
      df['CO'].isna().sum())
```

Tahap ini membuat data ekstrem diperlakukan sama seperti data yang tidak valid, yaitu sebagai nilai yang perlu diestimasi kembali.

---

#### 3.3 Imputasi Menggunakan Time Interpolation

Setelah nilai error sensor dan outlier diubah menjadi `NaN`, dilakukan proses **imputasi berbasis waktu (*time interpolation*)**.

Metode ini memanfaatkan informasi waktu untuk memperkirakan nilai yang hilang berdasarkan nilai yang tersedia sebelum dan sesudahnya. Dengan demikian, nilai hasil imputasi tetap mengikuti **tren perubahan data sepanjang waktu**.

Secara sederhana:

**Nilai sebelum → NaN → Nilai sesudah**

akan menghasilkan:

**Nilai sebelum → Nilai hasil interpolasi → Nilai sesudah**

Pendekatan ini lebih sesuai untuk data *time series* karena perubahan nilai tidak dianggap berdiri sendiri, tetapi berkaitan dengan waktu pengamatan.

#### Contoh kode: Time interpolation

```python
# Menjadikan date sebagai index
df_clean = df.set_index('date')

# Melakukan interpolasi berdasarkan waktu
df_clean['CO'] = df_clean['CO'].interpolate(method='time')

# Mengisi NaN yang masih tersisa di awal/akhir data
df_clean['CO'] = df_clean['CO'].ffill().bfill()

# Mengecek apakah masih ada nilai kosong
print("Jumlah NaN akhir:", df_clean['CO'].isna().sum())
```

Penggunaan **`ffill()`** (*forward fill*) dan **`bfill()`** (*backward fill*) dilakukan sebagai langkah tambahan apabila masih terdapat nilai kosong pada bagian awal atau akhir deret waktu yang tidak dapat diinterpolasi dari dua titik waktu.

Dengan demikian, data akhir diharapkan tidak lagi memiliki nilai `-9999`, outlier yang terdeteksi, maupun nilai kosong yang tidak tertangani.

---

### 4. Visualisasi Data

Setelah proses pembersihan selesai, dilakukan visualisasi untuk membandingkan **data mentah** dengan **data bersih**.

Visualisasi tersebut bertujuan untuk memperlihatkan secara langsung perbedaan kondisi data sebelum dan sesudah proses pembersihan, terutama pengaruh nilai `-9999`, outlier, serta hasil interpolasi terhadap pola *time series*.

#### Contoh kode: Menyimpan data mentah dan membuat data bersih

```python
def clean_polutan(file_name, target_pollutant):

    # Membaca data
    df = pd.read_csv(file_name)

    # Menyesuaikan nama kolom tanggal
    df = df.rename(columns={'Tanggal': 'date', 'time': 'date'})

    # Mengubah date menjadi datetime
    df['date'] = pd.to_datetime(df['date'])

    # Mengurutkan data berdasarkan waktu
    df = df.sort_values('date').reset_index(drop=True)

    # Mengubah kolom polutan menjadi numerik
    df[target_pollutant] = pd.to_numeric(
        df[target_pollutant],
        errors='coerce'
    )

    # Menyimpan data mentah
    raw_data = df[target_pollutant].copy()

    # Mengubah -9999 menjadi NaN
    df[target_pollutant] = df[target_pollutant].replace(
        [-9999, -9999.0],
        np.nan
    )

    # Menghitung IQR
    Q1 = df[target_pollutant].quantile(0.25)
    Q3 = df[target_pollutant].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Mengubah outlier menjadi NaN
    df.loc[
        (df[target_pollutant] < lower_bound) |
        (df[target_pollutant] > upper_bound),
        target_pollutant
    ] = np.nan

    # Time interpolation
    df_clean = (
        df.set_index('date')
        .interpolate(method='time')
        .ffill()
        .bfill()
    )

    return df, df_clean, raw_data
```

Kemudian data dapat divisualisasikan sebagai berikut:

```python
def plot_eda_polutan(file_name, target_pollutant):

    # Memanggil proses cleaning
    df, df_clean, raw_data = clean_polutan(
        file_name,
        target_pollutant
    )

    # Membuat dua grafik
    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 8),
        sharex=True
    )

    # Grafik data mentah
    axes[0].plot(
        df['date'],
        raw_data,
        color='red',
        alpha=0.7
    )

    axes[0].set_title(
        f'Data Mentah {target_pollutant} '
        f'(Perhatikan anomali sensor di angka -9999)',
        fontsize=14,
        fontweight='bold'
    )

    axes[0].set_ylabel(
        'Konsentrasi',
        fontsize=12
    )

    # Grafik data bersih
    axes[1].plot(
        df_clean.index,
        df_clean[target_pollutant],
        color='green',
        alpha=0.9
    )

    axes[1].set_title(
        f'Data Bersih {target_pollutant} '
        f'(Setelah Pembersihan & Interpolasi Waktu)',
        fontsize=14,
        fontweight='bold'
    )

    axes[1].set_ylabel(
        'Konsentrasi',
        fontsize=12
    )

    axes[1].set_xlabel(
        'Tanggal',
        fontsize=12
    )

    plt.tight_layout()
    plt.show()
```

---

### 5. Visualisasi EDA untuk Setiap Polutan

Setelah fungsi pembersihan dan visualisasi dibuat, proses dapat diterapkan pada masing-masing polutan.

#### **Karbon Monoksida (CO)**

```python
print("Menampilkan EDA untuk Karbon Monoksida (CO)...")

plot_eda_polutan(
    'data_co_kamal_fix2.csv',
    'CO'
)
```

#### **Sulfur Dioksida (SO₂)**

```python
print("Menampilkan EDA untuk Sulfur Dioksida (SO2)...")

plot_eda_polutan(
    'data_so2_kamal_fix2.csv',
    'SO2'
)
```

#### **Nitrogen Dioksida (NO₂)**

Apabila file data NO₂ tersedia, analisis dapat dilakukan dengan cara yang sama:

```python
print("Menampilkan EDA untuk Nitrogen Dioksida (NO2)...")

plot_eda_polutan(
    'data_no2_kamal_fix2.csv',
    'NO2'
)
```

---

### 6. Interpretasi Visualisasi

Grafik hasil EDA digunakan untuk memperlihatkan perubahan pola data sebelum dan sesudah proses *data cleaning*.

Pada **grafik data mentah**, nilai error sensor `-9999` dapat terlihat sebagai penurunan ekstrem yang tidak sesuai dengan pola konsentrasi polutan pada waktu tersebut. Nilai tersebut menunjukkan bahwa sensor gagal memberikan pembacaan yang valid.

Setelah dilakukan proses **penggantian `-9999` menjadi `NaN`, deteksi outlier menggunakan IQR, serta interpolasi berbasis waktu**, grafik data bersih diharapkan menunjukkan pola yang lebih kontinu.

Hasil visualisasi tersebut memberikan gambaran bahwa:

* **Anomali sensor tidak lagi diperlakukan sebagai nilai konsentrasi aktual.**
* **Nilai ekstrem yang terdeteksi sebagai outlier telah ditangani.**
* **Nilai yang hilang diisi berdasarkan hubungan temporal antarobservasi.**
* **Pola perubahan konsentrasi polutan dari waktu ke waktu tetap dipertahankan.**

Dengan demikian, data yang telah melalui proses pembersihan menjadi lebih sesuai untuk digunakan pada **tahap ekstraksi fitur, analisis lebih lanjut, dan pemodelan Machine Learning**.

---
### Kesimpulan

Berdasarkan analisis yang telah dilakukan, fitur **`median_diff`** digunakan untuk mengukur perubahan tengah dari nilai sinyal antar titik waktu yang berurutan. Pada data CO, fitur ini menghasilkan nilai **0.003**, yang menunjukkan bahwa perubahan tengah konsentrasi CO antar pengamatan adalah sebesar 0.003 satuan.

Sementara itu, fitur **`median_frequency`** bekerja pada domain frekuensi dengan melakukan transformasi FFT, mengambil magnitudo spektrum, menghitung akumulasi magnitudo, kemudian menentukan frekuensi ketika akumulasi tersebut melewati 50% dari total magnitudo. Pada data sampel CO dengan $fs=1$, diperoleh nilai **0.0 Hz**.

Hasil perhitungan manual untuk kedua fitur tersebut sesuai dengan hasil yang diberikan oleh fungsi TSFEL, sehingga proses perhitungan dapat digunakan sebagai pembuktian matematis terhadap implementasi fitur pada library TSFEL.
