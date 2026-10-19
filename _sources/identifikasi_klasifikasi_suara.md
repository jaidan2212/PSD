# Identifikasi dan Klasifikasi Suara Menggunakan Fitur Audio

**Mata Kuliah:** Pengolahan Sinyal Digital (PSD)  
**Program Studi:** Sains Data / Teknik Informatika  
**Oleh:** Muhammad Zaidan Nabil Rafi (240411100068)  
**Repositori:** `jaidan2212/PSD`  
**Tanggal:** Oktober 2026

---

## Daftar Isi

1. Pendahuluan
2. Tujuan Praktikum
3. Dasar Teori (Sinyal Audio Digital, RMS, Spectral Centroid, ZCR, MFCC)
4. Perangkat dan Library
5. Dataset dan Karakteristik Sampel
6. Tahapan Pengolahan Audio
7. Implementasi Kode Python
8. Hasil Ekstraksi Fitur
9. Visualisasi dan Analisis
10. Proses Identifikasi dan Klasifikasi Suara
11. Evaluasi Hasil
12. Pembahasan
13. Kesimpulan
14. Referensi

---

## 1. Pendahuluan

Suara merupakan gelombang mekanik yang merambat melalui medium dan dapat direpresentasikan sebagai sinyal digital setelah melalui proses *sampling* dan *quantization*. Dalam domain Pengolahan Sinyal Digital (PSD), analisis suara bertujuan untuk mengekstrak informasi kuantitatif dari sinyal audio sehingga karakteristik suatu suara dapat diidentifikasi dan dibedakan secara sistematis.

Setiap jenis suara — baik suara manusia, alat musik, suara alam, maupun suara mesin — memiliki karakteristik yang unik. Perbedaan ini tercermin pada distribusi energi, pola frekuensi, dan bentuk spektral sinyal. Dengan mengekstrak fitur-fitur terukur dari sinyal audio, kita dapat merepresentasikan karakteristik tersebut secara numerik untuk selanjutnya digunakan dalam proses identifikasi maupun klasifikasi.

Dokumentasi ini menjelaskan proses lengkap mulai dari pemuatan audio, ekstraksi empat fitur utama (**RMS**, **Spectral Centroid**, **ZCR**, dan **MFCC**), hingga klasifikasi menggunakan algoritma *machine learning*. Seluruh implementasi menggunakan Python beserta library pendukung yang relevan.

---

## 2. Tujuan Praktikum

1. Memahami konsep dasar sinyal audio digital dan representasinya dalam domain waktu serta domain frekuensi.
2. Memahami pengertian, rumus matematika, dan cara menghitung fitur RMS, Spectral Centroid, ZCR, dan MFCC.
3. Melakukan ekstraksi fitur dari sampel audio nyata menggunakan library `librosa`.
4. Menyusun tabel fitur yang merangkum karakteristik setiap sampel audio.
5. Membuat visualisasi yang informatif untuk membandingkan karakteristik antarsampel atau antarkelas suara.
6. Melakukan klasifikasi suara berdasarkan fitur yang diekstrak, apabila label kelas tersedia dan jumlah data memadai.
7. Menginterpretasikan dan menganalisis hasil fitur serta evaluasi model secara kritis.

---

## 3. Dasar Teori

### 3.1 Konsep Dasar Sinyal Audio Digital

Suara analog diubah menjadi sinyal digital melalui dua tahap utama:

- **Sampling:** pengambilan nilai amplitudo sinyal analog pada interval waktu tertentu. Jumlah sampel yang diambil per detik disebut *sample rate* (satuan: Hz). Sebagai contoh, *sample rate* 22050 Hz berarti sinyal diukur 22.050 kali dalam satu detik.
- **Quantization:** pembulatan nilai amplitudo hasil sampling ke level diskrit terdekat. Kedalaman bit (*bit depth*) menentukan jumlah level kuantisasi yang tersedia; misalnya, 16-bit menghasilkan 65.536 level.

Setelah proses ini, sinyal audio menjadi deretan angka (*array of numbers*) yang dapat diproses secara komputasional. Setiap elemen pada array tersebut merepresentasikan amplitudo sinyal pada satu titik waktu.

Konsep penting lainnya:

| Istilah | Penjelasan |
|---|---|
| **Sample rate (Fs/sr)** | Jumlah sampel per detik (Hz). Menentukan frekuensi maksimum yang dapat direpresentasikan (Nyquist = Fs/2). |
| **Mono vs Stereo** | Mono: 1 kanal audio. Stereo: 2 kanal (kiri dan kanan). Untuk analisis fitur, umumnya digunakan sinyal mono. |
| **Frame** | Potongan kecil sinyal audio (biasanya 20–50 ms). Analisis fitur sering dilakukan per frame (*short-time analysis*) karena sinyal audio dianggap stasioner dalam durasi pendek. |
| **Hop length** | Jumlah sampel yang digeser antar-frame berurutan. Dengan overlap, informasi transisi antar-frame tidak hilang. |
| **Domain waktu** | Representasi sinyal sebagai fungsi amplitudo terhadap waktu. |
| **Domain frekuensi** | Representasi sinyal sebagai fungsi magnitudo terhadap frekuensi, diperoleh melalui transformasi Fourier. |

### 3.2 RMS (Root Mean Square)

#### Pengertian dan Fungsi

RMS (*Root Mean Square*) adalah ukuran statistik yang menghitung akar kuadrat rata-rata dari kuadrat seluruh nilai sampel dalam suatu frame audio. Secara fisik, RMS merepresentasikan **tingkat energi** atau **loudness** relatif dari sinyal audio.

Nilai RMS memberikan gambaran seberapa "keras" suatu suara secara kuantitatif, tanpa terpengaruh oleh tanda positif atau negatif dari amplitudo sinyal.

#### Rumus Matematika

Untuk sinyal diskrit $x[n]$ dengan panjang $N$ sampel dalam satu frame:

$$
\text{RMS} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x[n]^2}
$$

Di mana:
- $x[n]$ = nilai amplitudo pada sampel ke-$n$
- $N$ = jumlah sampel dalam frame

#### Cara Membaca Hasil RMS

- **RMS tinggi** → suara memiliki energi besar, terdengar lebih keras (contoh: teriakan, suara mesin berat).
- **RMS rendah** → suara memiliki energi kecil, terdengar lebih pelan atau lembut (contoh: bisikan, suara angin).
- RMS dihitung per frame sehingga menghasilkan kurva energi terhadap waktu. Untuk merangkum satu file audio menjadi satu nilai, digunakan statistik seperti **mean** dan **standar deviasi** dari seluruh frame.

#### Membedakan Karakteristik Suara

Perbedaan nilai RMS antar-sampel dapat membantu membedakan:
- Suara yang bersifat *sustained* (energi relatif konstan, standar deviasi rendah) vs suara *impulsif* (terdapat puncak energi tajam, standar deviasi tinggi).
- Suara keras vs suara lembut, meskipun hal ini juga dipengaruhi kondisi perekaman (jarak mikrofon, *gain*, dll.).

> **Catatan:** RMS saja tidak cukup untuk menentukan jenis suara secara pasti. RMS perlu dikombinasikan dengan fitur lain untuk identifikasi yang lebih akurat.

### 3.3 Spectral Centroid

#### Pengertian dan Fungsi

Spectral Centroid (pusat spektral) adalah fitur yang menggambarkan **pusat massa** dari distribusi energi pada spektrum frekuensi sinyal audio. Secara intuitif, Spectral Centroid mengindikasikan apakah suara cenderung terdengar **"terang" (bright)** atau **"redup/gelap" (dark/dull)**.

Fitur ini dihitung pada domain frekuensi menggunakan spektrum magnitudo yang diperoleh dari Short-Time Fourier Transform (STFT).

#### Rumus Matematika

$$
\text{Spectral Centroid} = \frac{\sum_{k=0}^{K-1} f(k) \cdot |X(k)|}{\sum_{k=0}^{K-1} |X(k)|}
$$

Di mana:
- $f(k)$ = frekuensi yang berkorespondensi dengan bin ke-$k$ (dalam Hz)
- $|X(k)|$ = magnitudo spektrum pada bin ke-$k$
- $K$ = jumlah bin frekuensi

Rumus ini pada dasarnya adalah **rata-rata tertimbang frekuensi**, dengan bobot berupa magnitudo spektrum pada setiap bin frekuensi.

#### Hubungan dengan Frekuensi

Satuan Spectral Centroid adalah **Hertz (Hz)**, sama seperti frekuensi. Nilai ini menunjukkan frekuensi "pusat" di mana energi spektral terkumpul:

- **Spectral Centroid tinggi** (misalnya > 3000 Hz): energi terkonsentrasi pada frekuensi tinggi → suara terdengar tajam dan cerah (contoh: simbal, suara "sss", kicau burung).
- **Spectral Centroid rendah** (misalnya < 1000 Hz): energi terkonsentrasi pada frekuensi rendah → suara terdengar berat dan dalam (contoh: bass drum, suara "uum", gemuruh).

#### Interpretasi dalam Konteks Sampel

Dengan membandingkan rata-rata Spectral Centroid antar-sampel, kita dapat mengobservasi:
- Sampel mana yang memiliki komponen frekuensi tinggi lebih dominan.
- Sampel mana yang secara spektral lebih "gelap" atau "terang".
- Variasi kecerahan suara dari waktu ke waktu di dalam satu rekaman (menggunakan grafik Spectral Centroid per frame).

### 3.4 ZCR (Zero-Crossing Rate)

#### Pengertian dan Fungsi

Zero-Crossing Rate (ZCR) mengukur **seberapa sering sinyal audio berubah tanda** (dari positif ke negatif atau sebaliknya) dalam suatu frame. Secara teknis, setiap kali sinyal melewati garis nol (*zero line*), terjadi satu *zero-crossing*.

ZCR merupakan fitur sederhana pada domain waktu yang dapat memberikan indikasi kasar tentang frekuensi dominan dan karakteristik sinyal.

#### Rumus Matematika

$$
\text{ZCR} = \frac{1}{2N} \sum_{n=1}^{N-1} |\text{sign}(x[n]) - \text{sign}(x[n-1])|
$$

Di mana:
- $x[n]$ = nilai sampel pada indeks $n$
- $\text{sign}(x)$ = fungsi tanda ($+1$ jika $x > 0$, $-1$ jika $x < 0$, $0$ jika $x = 0$)
- $N$ = jumlah sampel dalam frame
- Faktor $\frac{1}{2}$ menormalkan hasil karena setiap *crossing* menghasilkan selisih absolut sebesar 2

Beberapa implementasi menggunakan versi tanpa faktor $\frac{1}{2N}$ dan hanya menghitung jumlah total *zero-crossing*. Library `librosa` menghitung ZCR sebagai rasio per frame.

#### Interpretasi Nilai ZCR

- **ZCR tinggi**: sinyal sering berubah tanda → umumnya mengindikasikan komponen frekuensi tinggi yang dominan, atau sinyal yang bersifat *noisy*. Contoh: suara desisan (*unvoiced speech* seperti "s", "f", "sh"), white noise, suara gesekan.
- **ZCR rendah**: sinyal jarang berubah tanda → umumnya mengindikasikan komponen frekuensi rendah yang dominan, atau sinyal periodik yang halus. Contoh: vokal "aaa", nada bass, dengungan.

#### Membedakan Karakteristik Suara

ZCR efektif untuk membedakan:
- **Voiced vs unvoiced speech**: segmen ucapan bersuara (vokal) cenderung memiliki ZCR lebih rendah dibanding segmen tak bersuara (konsonan frikatif).
- **Suara bernada vs noise**: suara yang bersifat periodik (bernada) memiliki ZCR lebih teratur dan rendah dibanding noise.
- **Perbedaan dasar antar-jenis suara**: suara alam seperti angin atau hujan cenderung memiliki ZCR tinggi, sementara nada alat musik yang dalam memiliki ZCR rendah.

### 3.5 MFCC (Mel-Frequency Cepstral Coefficients)

#### Pengertian dan Fungsi

MFCC adalah sekumpulan koefisien yang merepresentasikan **karakteristik spektral** suara menggunakan **skala Mel**. Skala Mel adalah skala frekuensi non-linear yang dirancang untuk meniru cara sistem pendengaran manusia mempersepsikan perbedaan frekuensi — manusia lebih sensitif terhadap perbedaan frekuensi pada rentang rendah dibanding rentang tinggi.

MFCC merupakan salah satu fitur yang paling banyak digunakan dalam pengenalan suara (*speech recognition*), identifikasi pembicara (*speaker identification*), dan klasifikasi suara (*sound classification*) karena kemampuannya mengekstrak informasi yang relevan secara perseptual.

#### Tahapan Umum Ekstraksi MFCC

Proses ekstraksi MFCC terdiri dari beberapa langkah:

1. **Pre-emphasis:** menerapkan filter *high-pass* untuk memperkuat frekuensi tinggi yang cenderung memiliki energi lebih kecil.
2. **Framing:** membagi sinyal menjadi frame-frame pendek (tipikal 20–40 ms) dengan overlap.
3. **Windowing:** mengalikan setiap frame dengan fungsi window (contoh: Hamming window) untuk mengurangi efek diskontinuitas di tepi frame.
4. **FFT (Fast Fourier Transform):** mengubah setiap frame dari domain waktu ke domain frekuensi untuk mendapatkan spektrum daya.
5. **Mel Filterbank:** menerapkan sekumpulan filter segitiga (*triangular filters*) yang tersebar secara merata pada skala Mel. Filter ini menjumlahkan energi di masing-masing sub-band frekuensi.
6. **Logaritma:** mengambil logaritma dari energi tiap filter, sesuai dengan persepsi loudness manusia yang bersifat logaritmik.
7. **DCT (Discrete Cosine Transform):** mentransformasi log-energi menjadi koefisien *cepstral*. Hasil DCT menghasilkan koefisien-koefisien MFCC.

#### Mengapa MFCC Sering Digunakan

- **Berbasis persepsi manusia:** skala Mel menyesuaikan resolusi frekuensi dengan cara manusia mendengar.
- **Reduksi dimensi:** dari ribuan bin frekuensi menjadi belasan koefisien yang kompak namun informatif.
- **Dekorelasi fitur:** DCT menghasilkan koefisien yang cenderung tidak berkorelasi, sehingga lebih efektif untuk model klasifikasi.
- **Terbukti efektif:** digunakan secara luas dalam berbagai sistem pengenalan suara dan musik.

#### Arti Koefisien MFCC

Setiap koefisien MFCC merepresentasikan aspek berbeda dari bentuk spektral (*spectral envelope*):

- **MFCC ke-0 (C₀):** berkaitan dengan energi total sinyal; kadang diabaikan dalam analisis.
- **MFCC ke-1 (C₁):** menggambarkan kemiringan (*tilt*) spektral secara umum — apakah energi lebih terpusat di frekuensi rendah atau tinggi.
- **MFCC ke-2 hingga ke-12 (C₂–C₁₂):** menangkap detail bentuk spektral yang semakin halus, merepresentasikan variasi lokal pada *spectral envelope*.
- Koefisien bernomor lebih tinggi menangkap detail spektral yang semakin halus, tetapi juga semakin rentan terhadap noise.

#### Mengapa 13 Koefisien

Secara konvensi, 13 koefisien MFCC pertama (C₀ hingga C₁₂, atau C₁ hingga C₁₃ tergantung konvensi library) dianggap cukup untuk merepresentasikan karakteristik spektral suara yang relevan secara perseptual. Koefisien di atas 13 biasanya menangkap detail yang terlalu halus dan lebih banyak berisi noise dibanding informasi yang berguna.

#### MFCC Bukan Nilai Tunggal

> **Penting:** MFCC **bukan satu angka tunggal** per file audio. Untuk setiap frame audio, dihasilkan **satu vektor berisi 13 koefisien** (jika menggunakan `n_mfcc=13`). Karena satu file audio terdiri dari banyak frame, hasil MFCC berupa **matriks** berdimensi `(n_mfcc, jumlah_frame)`.
>
> Untuk merangkum MFCC menjadi representasi per sampel audio (satu baris pada tabel fitur), setiap koefisien perlu diringkas menggunakan statistik deskriptif, misalnya **mean** dan **standar deviasi**, sehingga menghasilkan 26 nilai (13 mean + 13 std) untuk setiap file audio.

#### Visualisasi MFCC

MFCC biasanya divisualisasikan sebagai **heatmap** dengan:
- Sumbu horizontal: waktu (frame)
- Sumbu vertikal: nomor koefisien MFCC
- Warna: nilai koefisien

Visualisasi ini memungkinkan kita melihat bagaimana karakteristik spektral berubah sepanjang durasi audio.

---

## 4. Perangkat dan Library

### Lingkungan Pengembangan

| Komponen | Keterangan |
|---|---|
| Bahasa | Python 3.x |
| Editor / IDE | Jupyter Notebook, VS Code, atau Google Colab |
| Sistem Operasi | Windows / Linux / macOS |

### Library Python

| Library | Versi Minimum | Fungsi |
|---|---|---|
| `librosa` | ≥ 0.9.0 | Membaca file audio, ekstraksi fitur (RMS, Spectral Centroid, ZCR, MFCC) |
| `numpy` | ≥ 1.21 | Operasi numerik pada array dan matriks |
| `pandas` | ≥ 1.3 | Menyusun dan mengelola tabel (DataFrame) |
| `matplotlib` | ≥ 3.4 | Visualisasi grafik dan plot |
| `scikit-learn` | ≥ 1.0 | Standardisasi fitur, pembagian data, algoritma klasifikasi, metrik evaluasi |
| `soundfile` | ≥ 0.10 | Backend untuk `librosa` dalam membaca file audio (terutama WAV) |

### Instalasi

```bash
pip install librosa numpy pandas matplotlib scikit-learn soundfile
```

Jika menggunakan Google Colab, `librosa` dan library lainnya dapat diinstal langsung:

```bash
!pip install librosa soundfile
```

---

## 5. Dataset dan Karakteristik Sampel

### Struktur Folder Dataset

Sampel audio harus diletakkan di dalam folder `dataset_audio/` pada *root* proyek. Terdapat **dua skenario** yang didukung oleh kode implementasi:

#### Skenario A: Dengan Label Kelas (Supervised)

Jika data memiliki label, buat subfolder untuk setiap kelas:

```
TugasPSD/
└── dataset_audio/
    ├── kelas_a/
    │   ├── sampel_01.wav
    │   ├── sampel_02.wav
    │   └── ...
    ├── kelas_b/
    │   ├── sampel_01.wav
    │   ├── sampel_02.wav
    │   └── ...
    └── kelas_c/
        ├── sampel_01.wav
        └── ...
```

Nama subfolder akan digunakan sebagai **label kelas** secara otomatis oleh kode. Contoh nama subfolder yang dapat digunakan disesuaikan dengan isi data aktual, misalnya: `suara_manusia`, `alat_musik`, `suara_alam`, `suara_mesin`, dsb.

#### Skenario B: Tanpa Label Kelas (Unsupervised)

Jika data tidak memiliki label, letakkan semua file audio langsung di dalam `dataset_audio/`:

```
TugasPSD/
└── dataset_audio/
    ├── rekaman_01.wav
    ├── rekaman_02.wav
    ├── rekaman_03.mp3
    └── ...
```

Pada skenario ini, klasifikasi terawasi (*supervised*) tidak dapat dilakukan. Sebagai alternatif, pengelompokan dapat dilakukan menggunakan metode *clustering* (K-Means), namun hasil cluster memerlukan interpretasi manual.

### Format Audio yang Didukung

Library `librosa` mendukung berbagai format audio melalui backend `soundfile` dan `audioread`:

- **WAV** (direkomendasikan — *lossless*, paling stabil)
- **MP3** (memerlukan `ffmpeg` atau `audioread` terinstal)
- **FLAC** (*lossless compressed*)
- **OGG/Vorbis**

> **Rekomendasi:** Gunakan format **WAV** untuk menghindari masalah kompatibilitas. Jika sampel dalam format MP3, pastikan `ffmpeg` terinstal di sistem.

### Catatan tentang Dataset

- **Jumlah sampel** dan **nama kelas** akan diidentifikasi secara otomatis oleh kode dari isi folder.
- **Metadata** (durasi, sample rate, jumlah kanal) akan dibaca langsung dari setiap file audio.
- Kode menyertakan penanganan untuk file yang gagal dibaca atau berisi data kosong.
- Hasil fitur sangat dipengaruhi oleh: sumber suara, kondisi perekaman (mikrofon, jarak, lingkungan), durasi, sample rate, dan tingkat noise latar belakang.

---

## 6. Tahapan Pengolahan Audio

Berikut adalah alur kerja pengolahan audio secara berurutan beserta penjelasan mengapa setiap tahap diperlukan:

| No | Tahap | Penjelasan |
|---|---|---|
| 1 | **Memuat dan memeriksa sampel audio** | Membaca file audio dan memastikan file valid serta dapat diproses. File yang korup atau kosong harus diidentifikasi sejak awal. |
| 2 | **Membaca metadata audio** | Memeriksa durasi, sample rate, dan jumlah kanal setiap file. Informasi ini penting untuk memahami karakteristik dasar data dan menentukan apakah prapemrosesan diperlukan. |
| 3 | **Prapemrosesan** | Mengonversi stereo ke mono (jika ada) dan menyamakan sample rate agar seluruh sampel memiliki representasi yang konsisten. Librosa secara default melakukan konversi mono dan resampling ke 22050 Hz. |
| 4 | **Menampilkan sinyal dalam domain waktu** | Memvisualisasikan waveform untuk melihat pola amplitudo secara visual — apakah sinyal bersifat periodik, impulsif, atau stokastik. |
| 5 | **Ekstraksi RMS** | Menghitung tingkat energi per frame. Memberikan informasi tentang seberapa keras atau lembut suara pada setiap segmen waktu. |
| 6 | **Ekstraksi Spectral Centroid** | Menghitung pusat distribusi energi frekuensi per frame. Mengindikasikan kecerahan (*brightness*) suara. |
| 7 | **Ekstraksi ZCR** | Menghitung laju persimpangan nol per frame. Memberikan indikasi kasar tentang frekuensi dominan dan apakah sinyal bersifat tonal atau *noisy*. |
| 8 | **Ekstraksi MFCC** | Mengekstrak 13 koefisien MFCC per frame yang merepresentasikan *spectral envelope* berdasarkan skala Mel. |
| 9 | **Penggabungan fitur ke dalam tabel** | Merangkum fitur per frame menjadi statistik (mean, std) untuk setiap sampel audio, lalu menggabungkan seluruh sampel ke dalam satu DataFrame. |
| 10 | **Visualisasi dan analisis** | Membuat grafik perbandingan fitur antar-sampel atau antar-kelas untuk mengidentifikasi pola dan perbedaan karakteristik. |
| 11 | **Klasifikasi** | Jika label tersedia, melatih model klasifikasi (SVM / Random Forest) dan mengevaluasi hasilnya. Jika tidak, menggunakan K-Means sebagai alternatif eksplorasi. |
| 12 | **Kesimpulan** | Menyusun rangkuman berdasarkan hasil nyata dari analisis dan evaluasi. |

---

## 7. Implementasi Kode Python

Seluruh kode di bawah ini dirancang untuk dijalankan secara berurutan, baik di dalam Jupyter Notebook maupun sebagai skrip Python. Kode bersifat modular sehingga setiap bagian dapat dijalankan dan dipahami secara independen.

### 7.1 Impor Library dan Konfigurasi

```python
import os
import warnings
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path

# Opsional: untuk klasifikasi
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

# Konfigurasi
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11

# ============================================================
# KONFIGURASI LOKASI DATASET
# Sesuaikan path ini dengan lokasi folder audio Anda.
# ============================================================
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_audio')
# Jika di Jupyter Notebook, gunakan:
# DATASET_DIR = './dataset_audio'

TARGET_SR = 22050  # Sample rate target (Hz)
N_MFCC = 13        # Jumlah koefisien MFCC
```

**Penjelasan:**
- `DATASET_DIR`: lokasi folder yang berisi sampel audio. Sesuaikan dengan lokasi aktual di sistem Anda.
- `TARGET_SR`: sample rate target. Librosa secara default melakukan resampling ke 22050 Hz, cukup untuk analisis umum (frekuensi Nyquist = 11025 Hz).
- `N_MFCC`: jumlah koefisien MFCC yang diekstrak. 13 merupakan nilai konvensi standar.

### 7.2 Pemindaian dan Pemuatan Dataset

```python
def scan_dataset(dataset_dir):
    """
    Memindai folder dataset dan mendeteksi apakah data memiliki label kelas
    (berbasis subfolder) atau tidak.

    Returns:
        file_list: list of dict, setiap dict berisi 'path' dan 'label' (atau None).
        has_labels: bool, True jika data terstruktur dalam subfolder kelas.
        class_names: list of str, daftar nama kelas (kosong jika tanpa label).
    """
    supported_ext = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    file_list = []
    has_labels = False
    class_names = []

    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] Folder '{dataset_dir}' tidak ditemukan.")
        print("Pastikan folder 'dataset_audio/' ada dan berisi file audio.")
        return file_list, has_labels, class_names

    # Periksa apakah ada subfolder yang berisi audio (= ada label kelas)
    subdirs = [d for d in os.listdir(dataset_dir)
               if os.path.isdir(os.path.join(dataset_dir, d))]

    for subdir in sorted(subdirs):
        subdir_path = os.path.join(dataset_dir, subdir)
        audio_files = [f for f in os.listdir(subdir_path)
                       if os.path.splitext(f)[1].lower() in supported_ext]
        if audio_files:
            has_labels = True
            class_names.append(subdir)
            for f in sorted(audio_files):
                file_list.append({
                    'path': os.path.join(subdir_path, f),
                    'label': subdir
                })

    # Jika tidak ada subfolder berisi audio, cari langsung di root
    if not has_labels:
        root_files = [f for f in os.listdir(dataset_dir)
                      if os.path.isfile(os.path.join(dataset_dir, f))
                      and os.path.splitext(f)[1].lower() in supported_ext]
        for f in sorted(root_files):
            file_list.append({
                'path': os.path.join(dataset_dir, f),
                'label': None
            })

    return file_list, has_labels, class_names


# Eksekusi pemindaian
file_list, has_labels, class_names = scan_dataset(DATASET_DIR)

print(f"Total file audio ditemukan : {len(file_list)}")
print(f"Label kelas tersedia       : {'Ya' if has_labels else 'Tidak'}")
if has_labels:
    print(f"Kelas yang terdeteksi      : {class_names}")
    for cls in class_names:
        count = sum(1 for f in file_list if f['label'] == cls)
        print(f"  - {cls}: {count} sampel")
```

**Penjelasan:**
- Fungsi `scan_dataset` secara otomatis mendeteksi apakah dataset menggunakan struktur subfolder (berlabel) atau datar (tanpa label).
- Hanya file dengan ekstensi audio yang didukung yang diproses.
- Jika folder tidak ditemukan, pesan error yang jelas ditampilkan.

### 7.3 Membaca Metadata Audio

```python
def get_audio_metadata(filepath, target_sr=TARGET_SR):
    """
    Membaca file audio dan mengembalikan sinyal, sample rate, serta metadata.

    Returns:
        dict berisi: 'signal', 'sr', 'duration', 'n_samples', 'filename', 'success', 'error'
    """
    filename = os.path.basename(filepath)
    try:
        # librosa.load secara default: mono=True, sr=22050
        y, sr = librosa.load(filepath, sr=target_sr, mono=True)

        if y is None or len(y) == 0:
            return {
                'filename': filename, 'signal': None, 'sr': None,
                'duration': 0, 'n_samples': 0,
                'success': False, 'error': 'Sinyal kosong'
            }

        return {
            'filename': filename,
            'signal': y,
            'sr': sr,
            'duration': librosa.get_duration(y=y, sr=sr),
            'n_samples': len(y),
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'filename': filename, 'signal': None, 'sr': None,
            'duration': 0, 'n_samples': 0,
            'success': False, 'error': str(e)
        }


# Memuat seluruh file audio
audio_data = []
for item in file_list:
    meta = get_audio_metadata(item['path'])
    meta['label'] = item['label']
    meta['filepath'] = item['path']
    audio_data.append(meta)

# Pisahkan yang berhasil dan gagal
audio_valid = [a for a in audio_data if a['success']]
audio_failed = [a for a in audio_data if not a['success']]

print(f"\nBerhasil dimuat : {len(audio_valid)} file")
print(f"Gagal dimuat    : {len(audio_failed)} file")

if audio_failed:
    print("\nFile yang gagal dimuat:")
    for f in audio_failed:
        print(f"  - {f['filename']}: {f['error']}")

# Tampilkan metadata
if audio_valid:
    print("\n--- Metadata Sampel Audio ---")
    meta_df = pd.DataFrame([{
        'Filename': a['filename'],
        'Durasi (s)': round(a['duration'], 2),
        'Sample Rate': a['sr'],
        'Jumlah Sampel': a['n_samples'],
        'Label': a['label'] if a['label'] else '-'
    } for a in audio_valid])
    print(meta_df.to_string(index=False))
```

**Penjelasan:**
- `librosa.load()` secara default mengonversi audio ke **mono** dan melakukan **resampling** ke `sr=22050`.
- Fungsi ini menangani file yang gagal dibaca atau berisi data kosong tanpa menghentikan eksekusi keseluruhan.

### 7.4 Visualisasi Waveform

```python
def plot_waveform(signal, sr, title="Waveform", ax=None):
    """Menampilkan bentuk gelombang (waveform) sinyal audio."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    librosa.display.waveshow(signal, sr=sr, ax=ax, alpha=0.7)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Waktu (detik)')
    ax.set_ylabel('Amplitudo')
    ax.grid(True, alpha=0.3)


# Tampilkan waveform untuk setiap sampel (maks 6 pertama agar tidak terlalu banyak)
n_display = min(len(audio_valid), 6)
if n_display > 0:
    fig, axes = plt.subplots(n_display, 1, figsize=(14, 3 * n_display))
    if n_display == 1:
        axes = [axes]

    for i, a in enumerate(audio_valid[:n_display]):
        label_text = f" [{a['label']}]" if a['label'] else ""
        plot_waveform(a['signal'], a['sr'],
                      title=f"Waveform: {a['filename']}{label_text}",
                      ax=axes[i])

    plt.tight_layout()
    plt.savefig('output_waveforms.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Waveform disimpan ke: output_waveforms.png")
```

**Penjelasan:**
- Waveform menampilkan perubahan amplitudo sinyal terhadap waktu.
- Dari waveform, kita dapat mengamati: durasi efektif suara, pola periodik/non-periodik, segmen sunyi (*silence*), dan variasi amplitudo.
- Grafik disimpan sebagai file PNG untuk keperluan dokumentasi.

### 7.5 Ekstraksi RMS

```python
def extract_rms(signal, sr):
    """
    Menghitung RMS (Root Mean Square) per frame.
    Returns: array RMS per frame, times (untuk plotting).
    """
    rms = librosa.feature.rms(y=signal)[0]  # shape: (n_frames,)
    times = librosa.times_like(rms, sr=sr)
    return rms, times


# Plot RMS untuk setiap sampel
if audio_valid:
    n_display = min(len(audio_valid), 6)
    fig, axes = plt.subplots(n_display, 1, figsize=(14, 3 * n_display))
    if n_display == 1:
        axes = [axes]

    for i, a in enumerate(audio_valid[:n_display]):
        rms_vals, rms_times = extract_rms(a['signal'], a['sr'])
        label_text = f" [{a['label']}]" if a['label'] else ""

        axes[i].plot(rms_times, rms_vals, color='tab:red', linewidth=1.2)
        axes[i].set_title(f"RMS: {a['filename']}{label_text}",
                          fontsize=13, fontweight='bold')
        axes[i].set_xlabel('Waktu (detik)')
        axes[i].set_ylabel('RMS')
        axes[i].grid(True, alpha=0.3)
        axes[i].fill_between(rms_times, rms_vals, alpha=0.2, color='tab:red')

    plt.tight_layout()
    plt.savefig('output_rms.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Grafik RMS disimpan ke: output_rms.png")
```

**Penjelasan:**
- `librosa.feature.rms()` menghitung RMS untuk setiap frame audio.
- `fill_between` pada grafik membantu memvisualisasikan area di bawah kurva energi.
- Frame dengan RMS tinggi menunjukkan segmen audio yang keras.

### 7.6 Ekstraksi Spectral Centroid

```python
def extract_spectral_centroid(signal, sr):
    """
    Menghitung Spectral Centroid per frame.
    Returns: array Spectral Centroid per frame (Hz), times.
    """
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    times = librosa.times_like(centroid, sr=sr)
    return centroid, times


# Plot Spectral Centroid
if audio_valid:
    n_display = min(len(audio_valid), 6)
    fig, axes = plt.subplots(n_display, 1, figsize=(14, 3 * n_display))
    if n_display == 1:
        axes = [axes]

    for i, a in enumerate(audio_valid[:n_display]):
        cent_vals, cent_times = extract_spectral_centroid(a['signal'], a['sr'])
        label_text = f" [{a['label']}]" if a['label'] else ""

        axes[i].plot(cent_times, cent_vals, color='tab:blue', linewidth=1.2)
        axes[i].set_title(f"Spectral Centroid: {a['filename']}{label_text}",
                          fontsize=13, fontweight='bold')
        axes[i].set_xlabel('Waktu (detik)')
        axes[i].set_ylabel('Spectral Centroid (Hz)')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_spectral_centroid.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Grafik Spectral Centroid disimpan ke: output_spectral_centroid.png")
```

**Penjelasan:**
- `librosa.feature.spectral_centroid()` mengembalikan nilai pusat spektral dalam satuan Hz.
- Pola grafik menunjukkan bagaimana "kecerahan" suara berubah sepanjang waktu.
- Lonjakan ke atas menunjukkan segmen dengan dominasi frekuensi tinggi.

### 7.7 Ekstraksi ZCR

```python
def extract_zcr(signal):
    """
    Menghitung Zero-Crossing Rate per frame.
    Returns: array ZCR per frame.
    """
    zcr = librosa.feature.zero_crossing_rate(y=signal)[0]
    return zcr


# Plot ZCR
if audio_valid:
    n_display = min(len(audio_valid), 6)
    fig, axes = plt.subplots(n_display, 1, figsize=(14, 3 * n_display))
    if n_display == 1:
        axes = [axes]

    for i, a in enumerate(audio_valid[:n_display]):
        zcr_vals = extract_zcr(a['signal'])
        zcr_times = librosa.times_like(zcr_vals, sr=a['sr'])
        label_text = f" [{a['label']}]" if a['label'] else ""

        axes[i].plot(zcr_times, zcr_vals, color='tab:green', linewidth=1.2)
        axes[i].set_title(f"ZCR: {a['filename']}{label_text}",
                          fontsize=13, fontweight='bold')
        axes[i].set_xlabel('Waktu (detik)')
        axes[i].set_ylabel('Zero-Crossing Rate')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_zcr.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Grafik ZCR disimpan ke: output_zcr.png")
```

**Penjelasan:**
- `librosa.feature.zero_crossing_rate()` menghitung rasio *zero-crossing* per frame.
- Segmen dengan ZCR tinggi biasanya mengandung komponen *noise* atau frekuensi tinggi.
- Segmen tonal (bernada) cenderung memiliki ZCR lebih rendah dan stabil.

### 7.8 Ekstraksi MFCC

```python
def extract_mfcc(signal, sr, n_mfcc=N_MFCC):
    """
    Mengekstrak MFCC (Mel-Frequency Cepstral Coefficients).
    Returns: matriks MFCC berdimensi (n_mfcc, n_frames).
    """
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return mfcc


# Visualisasi MFCC sebagai heatmap
if audio_valid:
    n_display = min(len(audio_valid), 6)
    fig, axes = plt.subplots(n_display, 1, figsize=(14, 4 * n_display))
    if n_display == 1:
        axes = [axes]

    for i, a in enumerate(audio_valid[:n_display]):
        mfcc_vals = extract_mfcc(a['signal'], a['sr'])
        label_text = f" [{a['label']}]" if a['label'] else ""

        img = librosa.display.specshow(mfcc_vals, x_axis='time', sr=a['sr'],
                                       ax=axes[i], cmap='coolwarm')
        axes[i].set_title(f"MFCC Heatmap: {a['filename']}{label_text}",
                          fontsize=13, fontweight='bold')
        axes[i].set_ylabel('Koefisien MFCC')
        axes[i].set_xlabel('Waktu (detik)')
        fig.colorbar(img, ax=axes[i], format='%+2.0f')

    plt.tight_layout()
    plt.savefig('output_mfcc_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Heatmap MFCC disimpan ke: output_mfcc_heatmap.png")
```

**Penjelasan:**
- `librosa.feature.mfcc()` mengembalikan matriks berdimensi `(n_mfcc, n_frames)`.
- Heatmap MFCC menampilkan variasi setiap koefisien sepanjang waktu.
- Colormap `coolwarm` memudahkan identifikasi pola: nilai positif tinggi (merah) dan negatif (biru).
- Baris paling bawah (MFCC ke-0 atau ke-1) merepresentasikan aspek spektral paling kasar (energi keseluruhan), sementara baris atas menangkap detail yang lebih halus.

### 7.9 Penggabungan Fitur ke dalam Tabel

```python
def extract_all_features(signal, sr, n_mfcc=N_MFCC):
    """
    Mengekstrak seluruh fitur (RMS, Spectral Centroid, ZCR, MFCC)
    dan merangkumnya menjadi satu dictionary menggunakan mean dan std.

    Returns:
        dict berisi nama fitur -> nilai (float).
    """
    features = {}

    # --- RMS ---
    rms = librosa.feature.rms(y=signal)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))

    # --- Spectral Centroid ---
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    features['centroid_mean'] = float(np.mean(centroid))
    features['centroid_std'] = float(np.std(centroid))

    # --- ZCR ---
    zcr = librosa.feature.zero_crossing_rate(y=signal)[0]
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std'] = float(np.std(zcr))

    # --- MFCC ---
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    for j in range(n_mfcc):
        features[f'mfcc_{j+1}_mean'] = float(np.mean(mfcc[j]))
        features[f'mfcc_{j+1}_std'] = float(np.std(mfcc[j]))

    return features


# Ekstrak fitur dari seluruh sampel valid
rows = []
for a in audio_valid:
    try:
        feats = extract_all_features(a['signal'], a['sr'])
        feats['filename'] = a['filename']
        feats['label'] = a['label'] if a['label'] else '-'
        feats['durasi_detik'] = round(a['duration'], 2)
        rows.append(feats)
    except Exception as e:
        print(f"[WARNING] Gagal mengekstrak fitur dari {a['filename']}: {e}")

# Buat DataFrame
df_features = pd.DataFrame(rows)

# Atur ulang urutan kolom agar lebih rapi
meta_cols = ['filename', 'label', 'durasi_detik']
feature_cols = [c for c in df_features.columns if c not in meta_cols]
df_features = df_features[meta_cols + sorted(feature_cols)]

print("\n--- Tabel Fitur Audio ---")
print(f"Dimensi tabel: {df_features.shape[0]} baris x {df_features.shape[1]} kolom")
print(f"\nDaftar kolom fitur ({len(feature_cols)} fitur):")
for c in sorted(feature_cols):
    print(f"  - {c}")

# Tampilkan beberapa baris pertama
print("\n--- Preview Tabel (5 baris pertama) ---")
print(df_features.head().to_string(index=False))
```

**Penjelasan:**
- Fungsi `extract_all_features()` mengekstrak keempat fitur dan merangkumnya menggunakan **mean** dan **standard deviation** per sampel.
- Total fitur per sampel: 2 (RMS) + 2 (Centroid) + 2 (ZCR) + 26 (MFCC: 13×2) = **32 fitur**.
- Kolom metadata (`filename`, `label`, `durasi_detik`) diletakkan di awal tabel agar mudah dibaca.

### 7.10 Menyimpan Tabel Fitur ke CSV

```python
# Simpan ke CSV
output_csv = 'hasil_ekstraksi_fitur_audio.csv'
df_features.to_csv(output_csv, index=False)
print(f"\nTabel fitur disimpan ke: {output_csv}")
```

**Penjelasan:**
- File CSV memudahkan pembacaan ulang hasil tanpa perlu mengulang proses ekstraksi.
- File ini juga dapat dibuka di Excel atau Google Sheets untuk keperluan presentasi.

---

## 8. Hasil Ekstraksi Fitur

> **Catatan:** Bagian ini akan menampilkan hasil aktual setelah kode dijalankan pada sampel audio yang tersedia. Tabel dan nilai di bawah ini akan terisi oleh output dari kode pada Bagian 7.9 dan 7.10.

Hasil ekstraksi fitur berupa tabel (DataFrame) dengan struktur kolom sebagai berikut:

| Kolom | Deskripsi |
|---|---|
| `filename` | Nama file audio |
| `label` | Label kelas (dari nama subfolder, atau `-` jika tanpa label) |
| `durasi_detik` | Durasi audio dalam detik |
| `rms_mean` | Rata-rata RMS dari seluruh frame |
| `rms_std` | Standar deviasi RMS dari seluruh frame |
| `centroid_mean` | Rata-rata Spectral Centroid (Hz) dari seluruh frame |
| `centroid_std` | Standar deviasi Spectral Centroid dari seluruh frame |
| `zcr_mean` | Rata-rata Zero-Crossing Rate dari seluruh frame |
| `zcr_std` | Standar deviasi ZCR dari seluruh frame |
| `mfcc_1_mean` ... `mfcc_13_mean` | Rata-rata setiap koefisien MFCC |
| `mfcc_1_std` ... `mfcc_13_std` | Standar deviasi setiap koefisien MFCC |

Tabel hasil aktual akan tersimpan di file `hasil_ekstraksi_fitur_audio.csv` setelah kode dijalankan.

---

## 9. Visualisasi dan Analisis

### 9.1 Ringkasan Visualisasi yang Dihasilkan

Kode pada Bagian 7 menghasilkan visualisasi berikut:

| No | Visualisasi | File Output | Informasi yang Dibaca |
|---|---|---|---|
| 1 | Waveform | `output_waveforms.png` | Pola amplitudo, durasi efektif, segmen sunyi |
| 2 | Grafik RMS vs Waktu | `output_rms.png` | Distribusi energi sepanjang waktu |
| 3 | Grafik Spectral Centroid vs Waktu | `output_spectral_centroid.png` | Perubahan kecerahan spektral |
| 4 | Grafik ZCR vs Waktu | `output_zcr.png` | Pola *zero-crossing*, indikasi tonal vs *noisy* |
| 5 | Heatmap MFCC | `output_mfcc_heatmap.png` | Perubahan karakteristik spektral per frame |

### 9.2 Perbandingan Fitur Antar-Sampel

Visualisasi berikut membandingkan rata-rata fitur utama di antara seluruh sampel audio:

```python
if len(df_features) > 1:
    # --- Bar chart: perbandingan fitur utama ---
    main_features = ['rms_mean', 'centroid_mean', 'zcr_mean']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, feat in enumerate(main_features):
        bars = axes[idx].bar(
            range(len(df_features)),
            df_features[feat],
            color=plt.cm.Set2(range(len(df_features))),
            edgecolor='black', linewidth=0.5
        )
        axes[idx].set_title(f'Perbandingan {feat}', fontsize=13, fontweight='bold')
        axes[idx].set_xlabel('Sampel Audio')
        axes[idx].set_ylabel(feat)
        axes[idx].set_xticks(range(len(df_features)))
        axes[idx].set_xticklabels(df_features['filename'], rotation=45,
                                   ha='right', fontsize=8)
        axes[idx].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_perbandingan_fitur.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Grafik perbandingan fitur disimpan ke: output_perbandingan_fitur.png")
```

### 9.3 Perbandingan MFCC Rata-Rata Antar-Sampel

```python
if len(df_features) > 1:
    mfcc_mean_cols = [f'mfcc_{j+1}_mean' for j in range(N_MFCC)]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(1, N_MFCC + 1)
    width = 0.8 / len(df_features)

    for i, (_, row) in enumerate(df_features.iterrows()):
        mfcc_vals = [row[c] for c in mfcc_mean_cols]
        label_text = f"{row['filename']}"
        if row['label'] != '-':
            label_text += f" [{row['label']}]"
        offset = (i - len(df_features) / 2 + 0.5) * width
        ax.bar(x + offset, mfcc_vals, width=width, label=label_text)

    ax.set_title('Perbandingan Rata-rata MFCC Antar-Sampel',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Nomor Koefisien MFCC')
    ax.set_ylabel('Nilai Rata-rata')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{j}' for j in range(1, N_MFCC + 1)])
    ax.legend(fontsize=8, loc='best')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('output_mfcc_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Grafik MFCC comparison disimpan ke: output_mfcc_comparison.png")
```

**Interpretasi:**
- Dari grafik perbandingan, kita dapat mengobservasi perbedaan karakteristik suara antar-sampel.
- Sampel dengan `centroid_mean` lebih tinggi menunjukkan dominasi frekuensi tinggi.
- Pola MFCC yang berbeda antar-sampel mengindikasikan perbedaan *spectral envelope* yang signifikan.

### 9.4 Perbandingan Fitur Antar-Kelas (Jika Label Tersedia)

```python
if has_labels and len(df_features) > 1:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    features_to_compare = [
        ('rms_mean', 'RMS Mean', 'tab:red'),
        ('centroid_mean', 'Spectral Centroid Mean (Hz)', 'tab:blue'),
        ('zcr_mean', 'ZCR Mean', 'tab:green'),
        ('mfcc_1_mean', 'MFCC-1 Mean', 'tab:purple')
    ]

    for ax, (feat, title, color) in zip(axes.flatten(), features_to_compare):
        class_data = []
        class_labels = []
        for cls in class_names:
            vals = df_features[df_features['label'] == cls][feat].values
            class_data.append(vals)
            class_labels.append(cls)

        bp = ax.boxplot(class_data, labels=class_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.set_title(f'{title} per Kelas', fontsize=12, fontweight='bold')
        ax.set_ylabel(feat)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Perbandingan Distribusi Fitur Antar-Kelas',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('output_fitur_per_kelas.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Grafik distribusi fitur per kelas disimpan ke: output_fitur_per_kelas.png")
```

**Penjelasan:**
- Boxplot menampilkan distribusi fitur untuk setiap kelas: median, kuartil, dan outlier.
- Kelas yang terpisah secara jelas pada suatu fitur menunjukkan bahwa fitur tersebut diskriminatif untuk membedakan kelas tersebut.

---

## 10. Proses Identifikasi dan Klasifikasi Suara

### 10.1 Perbedaan Ekstraksi Fitur dan Klasifikasi

> **Penting:** RMS, Spectral Centroid, ZCR, dan MFCC adalah **fitur** — mereka merepresentasikan karakteristik suara secara numerik. Keempatnya **bukan** algoritma klasifikasi. Fitur-fitur ini menjadi **input** bagi algoritma klasifikasi (seperti SVM atau Random Forest) yang bertugas mempelajari pola dan membuat keputusan.

Analogi sederhana:
- **Fitur** = atribut yang mendeskripsikan seseorang (tinggi, berat, warna rambut)
- **Klasifikasi** = proses mengelompokkan orang berdasarkan atribut tersebut

### 10.2 Klasifikasi Terawasi (Supervised) — Jika Label Tersedia

Bagian ini hanya dijalankan jika dataset memiliki label kelas (struktur subfolder) **dan** jumlah sampel per kelas mencukupi.

```python
if has_labels:
    print("=" * 60)
    print("KLASIFIKASI TERAWASI (SUPERVISED CLASSIFICATION)")
    print("=" * 60)

    # Periksa kelayakan data untuk klasifikasi
    class_counts = df_features['label'].value_counts()
    print(f"\nDistribusi kelas:")
    print(class_counts.to_string())

    min_class_count = class_counts.min()
    n_classes = len(class_counts)

    if len(df_features) < 6 or min_class_count < 2:
        print(f"\n[PERINGATAN] Jumlah data terlalu sedikit untuk klasifikasi yang reliabel.")
        print(f"  - Total sampel : {len(df_features)}")
        print(f"  - Kelas terkecil: {min_class_count} sampel")
        print(f"  - Minimal disarankan: ≥ 5 sampel per kelas untuk evaluasi dasar.")
        print(f"\nEkstraksi fitur tetap dilakukan. Klasifikasi akan dijalankan")
        print(f"dengan catatan bahwa hasilnya memiliki keterbatasan signifikan.")

    # ====================================================
    # Persiapan fitur dan target
    # ====================================================
    feature_cols_only = [c for c in df_features.columns
                         if c not in ['filename', 'label', 'durasi_detik']]

    X = df_features[feature_cols_only].values
    y = df_features['label'].values

    # Standardisasi fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\nDimensi fitur (X): {X_scaled.shape}")
    print(f"Jumlah kelas: {n_classes}")

    # ====================================================
    # Strategi evaluasi berdasarkan jumlah data
    # ====================================================
    if len(df_features) >= 10 and min_class_count >= 3:
        # Cukup data: gunakan train-test split
        print("\n--- Strategi: Train-Test Split (80:20) ---")

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Data latih  : {len(X_train)} sampel")
        print(f"Data uji    : {len(X_test)} sampel")

        # --- SVM ---
        print("\n--- Model: SVM (Support Vector Machine) ---")
        svm_model = SVC(kernel='rbf', random_state=42)
        svm_model.fit(X_train, y_train)
        y_pred_svm = svm_model.predict(X_test)

        print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_svm, zero_division=0))

        # Confusion Matrix SVM
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_svm = confusion_matrix(y_test, y_pred_svm, labels=class_names)
        disp_svm = ConfusionMatrixDisplay(cm_svm, display_labels=class_names)
        disp_svm.plot(ax=ax, cmap='Blues', values_format='d')
        ax.set_title('Confusion Matrix — SVM', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('output_confusion_matrix_svm.png', dpi=150, bbox_inches='tight')
        plt.show()

        # --- Random Forest ---
        print("\n--- Model: Random Forest ---")
        rf_model = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)

        print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_rf, zero_division=0))

        # Confusion Matrix Random Forest
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_rf = confusion_matrix(y_test, y_pred_rf, labels=class_names)
        disp_rf = ConfusionMatrixDisplay(cm_rf, display_labels=class_names)
        disp_rf.plot(ax=ax, cmap='Greens', values_format='d')
        ax.set_title('Confusion Matrix — Random Forest',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('output_confusion_matrix_rf.png', dpi=150, bbox_inches='tight')
        plt.show()

        # Feature Importance (Random Forest)
        importances = rf_model.feature_importances_
        feat_importance = pd.Series(importances, index=feature_cols_only)
        feat_importance = feat_importance.sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        feat_importance.tail(15).plot(kind='barh', ax=ax, color='tab:olive')
        ax.set_title('Top 15 Feature Importance — Random Forest',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('output_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show()

    elif len(df_features) >= 6:
        # Data terbatas: gunakan Cross-Validation
        print("\n--- Strategi: Stratified K-Fold Cross-Validation ---")
        n_splits = min(min_class_count, 5)
        print(f"Jumlah fold: {n_splits}")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # SVM
        svm_model = SVC(kernel='rbf', random_state=42)
        svm_scores = cross_val_score(svm_model, X_scaled, y, cv=skf, scoring='accuracy')
        print(f"\nSVM - Accuracy per fold: {svm_scores}")
        print(f"SVM - Mean Accuracy: {svm_scores.mean():.4f} ± {svm_scores.std():.4f}")

        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_scores = cross_val_score(rf_model, X_scaled, y, cv=skf, scoring='accuracy')
        print(f"\nRandom Forest - Accuracy per fold: {rf_scores}")
        print(f"Random Forest - Mean Accuracy: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")

        print(f"\n[CATATAN] Dengan data yang terbatas ({len(df_features)} sampel),")
        print(f"hasil evaluasi memiliki varians tinggi dan belum tentu")
        print(f"merepresentasikan performa generalisasi model secara akurat.")

    else:
        print("\n[INFO] Data terlalu sedikit untuk klasifikasi yang bermakna.")
        print("Silakan tambah sampel audio untuk evaluasi klasifikasi.")
```

**Penjelasan alur klasifikasi:**

1. **Persiapan fitur:** kolom metadata dikeluarkan, hanya kolom fitur numerik yang digunakan.
2. **Standardisasi:** `StandardScaler` menormalisasi setiap fitur agar memiliki mean=0 dan std=1. Ini penting karena fitur memiliki skala yang sangat berbeda (RMS ~ 0.01, Centroid ~ 3000 Hz).
3. **Pembagian data:** menggunakan `train_test_split` dengan `stratify` agar distribusi kelas proporsional di data latih dan uji. Rasio 80:20 digunakan jika data cukup.
4. **Model SVM:** menggunakan kernel RBF yang efektif untuk data berdimensi tinggi dengan sampel terbatas.
5. **Model Random Forest:** ensemble dari banyak *decision tree*, robust terhadap overfitting dan memberikan informasi *feature importance*.
6. **Evaluasi:** Accuracy, Precision, Recall, F1-Score, dan Confusion Matrix.
7. **Penanganan data terbatas:** jika data sedikit, digunakan *Stratified K-Fold Cross-Validation*.

> **Peringatan kebocoran data:** `StandardScaler` di-*fit* pada seluruh data sebelum split dalam contoh cross-validation. Untuk implementasi yang lebih ketat, standardisasi sebaiknya dilakukan di dalam *pipeline* agar hanya data latih yang digunakan untuk menghitung parameter scaler.

### 10.3 Pengelompokan Tanpa Label (Unsupervised) — Jika Label Tidak Tersedia

```python
if not has_labels and len(df_features) >= 3:
    from sklearn.cluster import KMeans

    print("=" * 60)
    print("PENGELOMPOKAN TANPA LABEL (UNSUPERVISED CLUSTERING)")
    print("=" * 60)
    print("\n[INFO] Label kelas tidak tersedia. Menggunakan K-Means sebagai")
    print("alternatif eksplorasi untuk mengelompokkan sampel audio.")
    print("Hasil cluster BUKAN label kelas yang sebenarnya dan memerlukan")
    print("interpretasi atau validasi tambahan.\n")

    feature_cols_only = [c for c in df_features.columns
                         if c not in ['filename', 'label', 'durasi_detik']]

    X = df_features[feature_cols_only].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Tentukan jumlah cluster (dicoba 2-5, atau kurang jika data sedikit)
    max_k = min(len(df_features) - 1, 5)
    max_k = max(max_k, 2)
    inertias = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    # Elbow plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
    ax.set_title('Elbow Method — Menentukan Jumlah Cluster Optimal',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Jumlah Cluster (K)')
    ax.set_ylabel('Inertia')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('output_elbow_plot.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Gunakan K=2 sebagai default, atau sesuaikan setelah melihat elbow
    chosen_k = 2
    print(f"\nMenggunakan K = {chosen_k} cluster (sesuaikan berdasarkan elbow plot)")

    kmeans_final = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    df_features['cluster'] = kmeans_final.fit_predict(X_scaled)

    print("\nHasil pengelompokan:")
    print(df_features[['filename', 'cluster']].to_string(index=False))

    print("\n[CATATAN] Cluster yang dihasilkan belum tentu merepresentasikan")
    print("kelas suara yang bermakna. Diperlukan interpretasi manual dengan")
    print("mendengarkan sampel di setiap cluster dan menganalisis fitur-fiturnya.")
```

**Penjelasan:**
- K-Means mengelompokkan data berdasarkan kedekatan fitur, tanpa memerlukan label.
- **Elbow method** membantu menentukan jumlah cluster optimal dengan melihat titik di mana penurunan *inertia* mulai melambat.
- Hasil cluster **bukan** label kelas yang sebenarnya. Untuk memahami makna setiap cluster, perlu dilakukan analisis manual terhadap sampel-sampel di dalamnya.

---

## 11. Evaluasi Hasil

> **Catatan:** Bagian ini berisi template evaluasi. Isi dengan hasil aktual setelah kode dijalankan.

### 11.1 Metrik Evaluasi (Klasifikasi Terawasi)

Jika klasifikasi terawasi berhasil dijalankan, metrik berikut dilaporkan:

| Metrik | Definisi |
|---|---|
| **Accuracy** | Proporsi prediksi yang benar dari seluruh prediksi |
| **Precision** | Dari semua yang diprediksi sebagai kelas X, berapa proporsi yang benar |
| **Recall** | Dari semua sampel kelas X yang sebenarnya, berapa yang berhasil diprediksi |
| **F1-Score** | Rata-rata harmonik antara Precision dan Recall |
| **Confusion Matrix** | Tabel yang menunjukkan distribusi prediksi benar dan salah per kelas |

### 11.2 Keterbatasan Evaluasi

Beberapa keterbatasan yang perlu diperhatikan:

1. **Jumlah sampel sedikit:** jika total sampel kurang dari 20–30, evaluasi bersifat *high-variance* — perbedaan kecil pada data uji dapat mengubah akurasi secara signifikan.
2. **Distribusi kelas tidak seimbang:** jika satu kelas memiliki jauh lebih banyak sampel, accuracy bisa menyesatkan. F1-Score per kelas lebih informatif dalam kasus ini.
3. **Kondisi perekaman:** perbedaan mikrofon, jarak, atau noise latar belakang antar-sampel dapat menyebabkan model belajar fitur yang tidak relevan (*confounding variables*).
4. **Overfitting:** dengan data sedikit dan fitur yang banyak (32 fitur), model berpotensi menghafal data latih tanpa kemampuan generalisasi yang baik.

---

## 12. Pembahasan

> **Catatan:** Bagian ini harus diisi berdasarkan hasil aktual yang diperoleh dari menjalankan kode pada sampel audio. Berikut adalah kerangka pembahasan yang dapat digunakan.

### 12.1 Analisis Karakteristik Fitur

Setelah mengekstrak fitur dari seluruh sampel, analisis yang dapat dilakukan meliputi:

- **RMS:** apakah terdapat perbedaan signifikan pada tingkat energi antar-sampel atau antar-kelas? Sampel mana yang paling keras dan paling lembut? Apakah perbedaan ini konsisten dengan persepsi pendengaran saat mendengarkan langsung?

- **Spectral Centroid:** sampel mana yang memiliki pusat frekuensi tertinggi (suara paling "terang")? Apakah pola ini sesuai dengan jenis sumber suaranya?

- **ZCR:** sampel mana yang memiliki *zero-crossing rate* tertinggi? Apakah ini konsisten dengan ekspektasi (misalnya suara gesekan/desisan seharusnya memiliki ZCR tinggi)?

- **MFCC:** koefisien MFCC mana yang menunjukkan perbedaan paling signifikan antar-kelas? Apakah pola MFCC secara visual (dari heatmap) terlihat berbeda antar-jenis suara?

### 12.2 Efektivitas Fitur untuk Identifikasi Suara

- Tidak ada satu fitur tunggal yang dapat menentukan jenis suara secara pasti.
- Kombinasi keempat fitur (RMS, Spectral Centroid, ZCR, MFCC) memberikan representasi yang lebih kaya dibanding penggunaan salah satu fitur saja.
- MFCC umumnya menjadi fitur yang paling diskriminatif karena merepresentasikan *spectral envelope* secara komprehensif.

### 12.3 Analisis Hasil Klasifikasi (Jika Ada)

Jika klasifikasi berhasil dilakukan, analisis hasil meliputi:
- Kelas mana yang paling mudah dan paling sulit dibedakan? Mengapa?
- Apakah terjadi *confusion* antar-kelas tertentu? Apa penyebabnya?
- Fitur mana yang paling berpengaruh berdasarkan *feature importance*?
- Apakah performa model sudah memadai untuk tujuan identifikasi?

### 12.4 Faktor yang Mempengaruhi Hasil

Hasil ekstraksi fitur dan klasifikasi dipengaruhi oleh:

1. **Karakteristik sumber suara:** setiap jenis suara memiliki spektrum dan pola temporal yang berbeda.
2. **Kondisi perekaman:** mikrofon, jarak perekaman, ruangan, dan perangkat berpengaruh pada kualitas sinyal.
3. **Durasi sampel:** sampel yang terlalu pendek mungkin tidak cukup representatif; sampel yang terlalu panjang memiliki variabilitas internal yang tinggi.
4. **Sample rate:** menentukan frekuensi maksimum yang dapat dianalisis.
5. **Noise latar belakang:** noise dapat mengaburkan fitur yang sebenarnya dari sumber suara utama.
6. **Jumlah dan kualitas data:** semakin banyak dan beragam data per kelas, semakin baik model dapat belajar membedakan kelas.

---

## 13. Kesimpulan

> **Catatan:** Kesimpulan harus ditulis berdasarkan hasil aktual yang diperoleh dari menjalankan kode. Berikut adalah kerangka kesimpulan yang dapat disesuaikan.

Berdasarkan analisis yang telah dilakukan, dapat disimpulkan bahwa:

1. **Sinyal audio digital** dapat direpresentasikan sebagai deretan nilai numerik yang memungkinkan analisis komputasional terhadap karakteristik suara.

2. **Empat fitur audio** telah berhasil diekstrak dari sampel yang tersedia:
   - **RMS** merepresentasikan tingkat energi atau *loudness* suara.
   - **Spectral Centroid** menggambarkan kecerahan spektral suara berdasarkan pusat distribusi energi frekuensi.
   - **ZCR** mengindikasikan karakter tonal vs *noisy* berdasarkan laju persimpangan nol sinyal.
   - **MFCC** memberikan representasi *spectral envelope* yang komprehensif menggunakan skala Mel.

3. **Ekstraksi fitur dan klasifikasi adalah dua proses yang berbeda.** Fitur-fitur di atas merepresentasikan karakteristik suara, sementara klasifikasi dilakukan oleh algoritma *machine learning* (SVM, Random Forest) yang menggunakan fitur sebagai input.

4. **Kombinasi keempat fitur** memberikan representasi yang lebih baik dibanding penggunaan satu fitur saja. Tidak ada satu fitur tunggal yang dapat menentukan jenis suara secara pasti.

5. Hasil aktual dari klasifikasi dan analisis fitur akan dilaporkan setelah kode dijalankan pada dataset audio yang tersedia.

> Isi poin-poin di atas dengan **hasil dan angka spesifik** setelah kode dijalankan pada data aktual. Jangan menyimpulkan keberhasilan atau kegagalan model tanpa bukti dari hasil yang benar-benar diperoleh.

---

## 14. Referensi

1. Müller, M. (2015). *Fundamentals of Music Processing: Audio, Analysis, Algorithms, Applications*. Springer International Publishing. ISBN: 978-3-319-21944-8.

2. McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and Music Signal Analysis in Python. *Proceedings of the 14th Python in Science Conference*, 18-25.

3. Davis, S. B., & Mermelstein, P. (1980). Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 28(4), 357-366.

4. Logan, B. (2000). Mel Frequency Cepstral Coefficients for Music Modeling. *Proceedings of the 1st International Symposium on Music Information Retrieval (ISMIR)*.

5. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.

6. Librosa Documentation. https://librosa.org/doc/latest/index.html

7. Zheng, F., Zhang, G., & Song, Z. (2001). Comparison of Different Implementations of MFCC. *Journal of Computer Science and Technology*, 16(6), 582-589.

8. Peeters, G. (2004). A Large Set of Audio Features for Sound Description (Similarity and Classification) in the CUIDADO Project. *IRCAM Technical Report*.

---

*Dokumen ini disusun untuk keperluan mata kuliah Pengolahan Sinyal Digital (PSD). Seluruh kode dapat dijalankan secara mandiri setelah dependensi diinstal dan file audio dimasukkan ke dalam folder `dataset_audio/`.*
