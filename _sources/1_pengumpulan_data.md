# Bab 1: Pengumpulan Data, Preprocessing, dan Ekstraksi Fitur *Time-Series* NO2

Bab ini membahas tahapan awal dan paling fundamental dalam alur kerja analisis kualitas udara: mulai dari pengumpulan data konsentrasi Nitrogen Dioksida (NO2), pembersihan data dari nilai-nilai anomali, hingga transformasi data mentah menjadi representasi fitur numerik yang siap digunakan untuk tahap pemodelan pada bab-bab selanjutnya.

| Atribut | Keterangan |
|---|---|
| **Penulis** | Muhammad Zaidan Nabil Rafi |
| **Lokasi Studi** | Kecamatan Kamal |
| **Rentang Waktu Data** | 31 Agustus 2025 – 31 Agustus 2026 |
| **Target Polutan** | Nitrogen Dioksida (NO2) |
| **Sumber Data** | Citra Satelit Sentinel-5P (Instrumen TROPOMI) |

---

## 1.1 Pendahuluan

Nitrogen Dioksida (NO2) merupakan salah satu polutan udara utama yang erat kaitannya dengan aktivitas pembakaran bahan bakar fosil, baik dari sektor transportasi maupun industri. Konsentrasi NO2 yang tinggi di atmosfer tidak hanya berdampak pada kualitas udara, tetapi juga menjadi indikator penting dalam memantau dinamika aktivitas manusia di suatu wilayah.

Pada studi ini, data konsentrasi NO2 dikumpulkan menggunakan citra satelit **Sentinel-5P**, sebuah misi penginderaan jauh milik European Space Agency (ESA) yang dilengkapi instrumen TROPOMI (*TROPOspheric Monitoring Instrument*) untuk memantau komposisi atmosfer secara harian dengan resolusi spasial yang relatif tinggi. Wilayah kajian yang menjadi fokus dalam bab ini adalah **Kecamatan Kamal**, dengan rentang waktu pengamatan dari **31 Agustus 2025 hingga 31 Agustus 2026**, atau setara dengan satu tahun penuh data deret waktu (*time-series*).

Data mentah hasil ekstraksi citra satelit umumnya masih mengandung beberapa permasalahan, seperti nilai-nilai ekstrem akibat gangguan atmosferik (misalnya tutupan awan) maupun kekosongan data pada tanggal-tanggal tertentu. Oleh karena itu, sebelum data dapat digunakan untuk analisis lebih lanjut atau pemodelan prediktif, diperlukan tahapan **preprocessing** yang matang serta proses **ekstraksi fitur** yang representatif. Kedua tahapan inilah yang menjadi inti pembahasan pada bab ini.

## 1.2 Pembersihan Data (Preprocessing)

Tahap preprocessing bertujuan untuk memastikan data deret waktu NO2 bersih dari anomali dan bebas dari nilai kosong (*missing value*), tanpa mengorbankan struktur kronologis data. Terdapat dua proses utama yang dilakukan secara berurutan: **deteksi outlier** dan **imputasi nilai hilang**.

### 1.2.1 Deteksi Outlier dengan Metode IQR

Deteksi outlier dilakukan menggunakan metode **Interquartile Range (IQR)**, sebuah pendekatan statistik non-parametrik yang umum digunakan karena tidak mengasumsikan distribusi data tertentu. Metode ini bekerja dengan menghitung sebaran nilai berdasarkan kuartil pertama (Q1) dan kuartil ketiga (Q3), lalu menetapkan batas atas dan batas bawah kewajaran data sebagai berikut:

$$
IQR = Q3 - Q1
$$

$$
\text{Batas Bawah} = Q1 - 1.5 \times IQR \qquad \text{Batas Atas} = Q3 + 1.5 \times IQR
$$

Setiap nilai konsentrasi NO2 yang berada di luar rentang batas bawah dan batas atas tersebut dikategorikan sebagai outlier. Penting untuk dicatat bahwa pada studi ini, nilai outlier **tidak dihapus** dari dataset (yang akan merusak kontinuitas deret waktu), melainkan **diubah menjadi `NaN` (Not a Number)**. Pendekatan ini memungkinkan struktur temporal data tetap utuh dan siap untuk diperbaiki pada tahap selanjutnya.

### 1.2.2 Imputasi Missing Value dengan Interpolasi Waktu

Setelah nilai-nilai outlier ditandai sebagai `NaN`, langkah berikutnya adalah mengisi kekosongan tersebut melalui proses **imputasi**. Metode yang digunakan adalah **Interpolasi Waktu (Time Interpolation)**, yaitu teknik estimasi nilai yang hilang berdasarkan posisi temporalnya di antara titik-titik data yang valid, dengan mempertimbangkan jarak waktu antar-observasi secara proporsional.

```{tip}
Interpolasi berbasis waktu (`method='time'`) berbeda dengan interpolasi linear biasa karena metode ini memperhitungkan indeks tanggal aktual, sehingga lebih akurat digunakan pada data deret waktu yang intervalnya tidak selalu seragam.
```

Untuk mengantisipasi kemungkinan nilai `NaN` yang tersisa di bagian awal atau akhir deret waktu (di mana interpolasi tidak dapat dilakukan karena tidak ada titik pembanding), dilakukan proses tambahan berupa *forward fill* (`ffill`) dan *backward fill* (`bfill`) sebagai penyempurna.

### 1.2.3 Visualisasi Tren Kualitas Udara
Untuk memberikan gambaran yang lebih jelas mengenai dinamika konsentrasi NO2 di Kecamatan Kamal, berikut adalah visualisasi data deret waktu sebelum dan sesudah tahap *preprocessing* (deteksi *outlier* dan imputasi):

![Grafik Tren NO2 Kamal](grafik_tren_no2.png)

Melalui visualisasi ini, kita dapat melihat bagaimana metode IQR memotong lonjakan ekstrem yang tidak masuk akal, dan bagaimana interpolasi menambal kekosongan data dengan tren garis yang mulus, sehingga sinyal NO2 siap diekstrak pada tahap selanjutnya.

## 1.3 Ekstraksi Fitur *Time-Series* dengan TSFEL

Setelah data NO2 bersih dan lengkap, tahap selanjutnya adalah mengekstraksi karakteristik numerik dari sinyal deret waktu tersebut menggunakan **TSFEL (Time Series Feature Extraction Library)**. TSFEL merupakan pustaka Python yang dirancang untuk mengekstraksi fitur-fitur representatif dari sinyal *time-series* secara otomatis, mencakup tiga domain utama, yaitu domain statistik, domain temporal, dan domain spektral.

Proses ekstraksi yang dilakukan pada sinyal polutan NO2 ini tidak hanya mengambil nilai rata-rata, melainkan membedah pola data ke dalam **tiga domain utama**, yaitu:

1. **Domain Statistik (Statistical Domain):** Mengekstrak karakteristik dasar dari sebaran data. Contoh fitur yang diambil antara lain: nilai rata-rata (*mean*), varians (*variance*), deviasi standar (*std*), hingga kemencengan (*skewness*) dan keruncingan (*kurtosis*) dari distribusi polusi udara.
2. **Domain Temporal (Temporal Domain):** Mengukur dinamika perubahan nilai NO2 seiring berjalannya waktu. Contoh fiturnya meliputi: autokorelasi (*autocorr*), jarak antar puncak sinyal (*pk_pk_distance*), perubahan energi sinyal, dan kemiringan tren (*slope*).
3. **Domain Spektral & Fraktal (Spectral & Fractal Domain):** Mengubah sinyal waktu menjadi sinyal frekuensi untuk melihat pola musiman atau siklus tersembunyi. Contoh fiturnya adalah: entropi spektral (*spectral_entropy*), energi gelombang (*wavelet_energy*), dan dimensi fraktal (*fractal_dimension*) yang mengukur tingkat kekasaran fluktuasi polusi.

Ketiga domain ini digabungkan secara komprehensif sehingga algoritma kita nantinya bisa "memahami" sifat dari udara di Kamal secara utuh.

### 1.3.1 Mengapa 68 Fitur, Bukan 156?

Secara *default*, konfigurasi bawaan TSFEL akan menghasilkan **156 fitur**, karena beberapa fungsi ekstraksi (seperti `mfcc`, koefisien *wavelet*, atau `fft_mean_coeff`) secara alami mengembalikan **banyak nilai sekaligus** (berupa vektor atau array), bukan satu nilai tunggal per fungsi.

Pada studi ini, kebutuhan standar analisis mengharuskan setiap fungsi ekstraksi menghasilkan **tepat satu nilai representatif (skalar)**, sehingga total fitur akhir yang dihasilkan berjumlah **tepat 68 fitur** — satu fitur untuk setiap fungsi dasar TSFEL yang digunakan. Untuk mencapai hal ini, dilakukan modifikasi pada proses ekstraksi melalui sebuah fungsi khusus bernama `to_scalar()`, yang bertugas meringkas keluaran berupa vektor/array menjadi satu nilai tunggal menggunakan rata-rata (`nanmean`). Dengan pendekatan ini, setiap fungsi fitur — baik yang secara alami sudah menghasilkan skalar maupun yang menghasilkan array — akan tetap berkontribusi sebagai **satu kolom fitur** pada dataset akhir.

Pendekatan ini juga memanfaatkan modul `inspect` untuk secara otomatis memeriksa apakah suatu fungsi fitur memerlukan parameter frekuensi sampling (`fs`) atau tidak, sehingga proses ekstraksi dapat dilakukan secara dinamis untuk seluruh daftar fungsi tanpa perlu penanganan khusus satu per satu.

### 1.3.2 Implementasi Kode

Berikut adalah implementasi lengkap proses preprocessing dan ekstraksi fitur yang telah dijelaskan pada bagian sebelumnya:

```python
import pandas as pd
import numpy as np
import inspect
import tsfel
import tsfel.feature_extraction.features as tsfel_features
import warnings
warnings.filterwarnings('ignore')

# 1. PREPROCESSING DATA KAMAL
# Membaca data mentah hasil ekstraksi citra Sentinel-5P untuk Kecamatan Kamal
df = pd.read_csv('data_no2_kamal.csv')
df['Tanggal'] = pd.to_datetime(df['Tanggal'])
# Menggabungkan observasi dengan tanggal yang sama (rata-rata harian)
df = df.groupby('Tanggal').mean().reset_index()
# Memastikan data terurut secara kronologis, syarat wajib untuk interpolasi waktu
df = df.sort_values('Tanggal').reset_index(drop=True)

target_pollutant = 'NO2'
df[target_pollutant] = pd.to_numeric(df[target_pollutant], errors='coerce')

# Deteksi Outlier (Ubah jadi NaN)
# Menghitung kuartil dan rentang antar-kuartil (IQR) dari konsentrasi NO2
Q1 = df[target_pollutant].quantile(0.25)
Q3 = df[target_pollutant].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Nilai di luar batas bawah/atas ditandai sebagai NaN, bukan dihapus,
# agar struktur deret waktu tetap utuh
df.loc[(df[target_pollutant] < lower_bound) | (df[target_pollutant] > upper_bound), target_pollutant] = np.nan

# Imputasi Missing Value dengan Interpolasi Waktu
# Mengisi nilai NaN berdasarkan posisi temporalnya, lalu menambal sisa NaN
# di ujung deret waktu dengan forward fill & backward fill
df_clean = df.set_index('Tanggal').interpolate(method='time').ffill().bfill()

fs = 1
signal_1d = df_clean[target_pollutant].astype(float).values

# 2. EKSTRAKSI 68 FITUR TSFEL
# Daftar 68 fungsi fitur dasar TSFEL (domain statistik, temporal, dan spektral)
FEATURE_LIST = """abs_energy auc autocorr average_power calc_centroid calc_max calc_mean
calc_median calc_min calc_std calc_var dfa distance ecdf ecdf_percentile ecdf_percentile_count
ecdf_slope entropy fundamental_frequency higuchi_fractal_dimension hist_mode human_range_energy
hurst_exponent interq_range kurtosis lempel_ziv lpcc max_frequency max_power_spectrum
maximum_fractal_length mean_abs_deviation mean_abs_diff mean_diff median_abs_deviation
median_abs_diff median_diff median_frequency mfcc mse negative_turning neighbourhood_peaks
petrosian_fractal_dimension pk_pk_distance positive_turning power_bandwidth rms skewness slope
spectral_centroid spectral_decrease spectral_distance spectral_entropy spectral_kurtosis
spectral_positive_turning spectral_roll_off spectral_roll_on spectral_skewness spectral_slope
spectral_spread spectral_variation spectrogram_mean_coeff sum_abs_diff wavelet_abs_mean
wavelet_energy wavelet_entropy wavelet_std wavelet_var zero_cross""".split()

# Fungsi wajib untuk membatasi fitur menjadi skalar (tepat 68 fitur)
# Fungsi TSFEL yang mengembalikan array/vektor (mis. mfcc, wavelet) diringkas
# menjadi satu nilai representatif menggunakan rata-rata (nanmean)
def to_scalar(result):
    if isinstance(result, dict) and "values" in result:
        result = result["values"]
    if isinstance(result, (list, tuple, np.ndarray)):
        arr = np.asarray(result, dtype=float)
        return float(np.nanmean(arr))
    return float(result)

# Mengekstraksi satu fitur, sambil memeriksa otomatis apakah fungsi tersebut
# membutuhkan parameter frekuensi sampling (fs)
def extract_one(fn_name, signal, fs):
    fn = getattr(tsfel_features, fn_name)
    params = inspect.signature(fn).parameters
    if "fs" in params:
        result = fn(signal, fs)
    else:
        result = fn(signal)
    return to_scalar(result)

# Melakukan ekstraksi untuk seluruh 68 fungsi fitur pada sinyal NO2
row = {}
for fn_name in FEATURE_LIST:
    row[fn_name] = extract_one(fn_name, signal_1d, fs)

# Menyimpan hasil akhir: satu baris berisi tepat 68 kolom fitur
extracted_features_final = pd.DataFrame([row])
extracted_features_final.to_csv('fitur_68_kamal.csv', index=False)
```

```{note}
Perlu digarisbawahi bahwa pendekatan ekstraksi di atas **berbeda** dari pemanggilan standar `tsfel.time_series_features_extractor()` yang secara default menghasilkan 156 fitur. Dengan memanggil setiap fungsi dari `tsfel.feature_extraction.features` secara langsung dan meringkasnya melalui `to_scalar()`, jumlah fitur akhir dapat dikontrol secara presisi menjadi **68 fitur**, sesuai standar kebutuhan analisis pada studi ini.
```

## 1.4 Ringkasan Bab

Melalui tahapan yang telah dijabarkan pada bab ini, data konsentrasi NO2 di Kecamatan Kamal — yang mencakup periode 31 Agustus 2025 hingga 31 Agustus 2026 — berhasil ditransformasikan dari data mentah yang masih mengandung outlier dan nilai kosong, menjadi sebuah representasi fitur numerik yang bersih, konsisten, dan siap pakai. Secara ringkas, alur kerja yang telah dilalui adalah sebagai berikut:

1. **Deteksi outlier** menggunakan metode IQR, dengan nilai anomali ditandai sebagai `NaN`.
2. **Imputasi** nilai `NaN` melalui interpolasi waktu, disempurnakan dengan `ffill` dan `bfill`.
3. **Ekstraksi fitur** deret waktu menggunakan TSFEL yang dimodifikasi agar menghasilkan tepat **68 fitur skalar**.

Hasil akhir dari proses ini disimpan dalam berkas `fitur_68_kamal.csv`, yang akan menjadi masukan (*input*) utama pada tahap analisis dan pemodelan pada bab-bab selanjutnya.