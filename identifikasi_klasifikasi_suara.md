# Identifikasi dan Klasifikasi Suara Menggunakan Fitur Audio: Studi Kasus Berisik vs Non-Berisik (Hening)

**Mata Kuliah:** Pengolahan Sinyal Digital (PSD)  
**Program Studi:** Sains Data / Teknik Informatika  
**Penulis:**  
- Muhammad Zaidan Nabil Rafi (240411100068)  
- Okan Syailendraa Wahyudi (240411100179)  
- Alifah Sayyidaturrhohma (240411100145)  
**Repositori:** `jaidan2212/PSD`  
**Dataset:** 229 Sampel Audio Lingkungan (`BerisikPSD.zip` & `NonBerisikPSD.zip`)  
**Tanggal:** Oktober 2026  

---

## Daftar Isi

1. Pendahuluan
2. Tujuan Praktikum
3. Dasar Teori Pengolahan Sinyal Audio Digital
   - 3.1 Representasi Sinyal Audio Digital & Analisis Short-Time
   - 3.2 Root Mean Square (RMS)
   - 3.3 Spectral Centroid
   - 3.4 Zero-Crossing Rate (ZCR)
   - 3.5 Mel-Frequency Cepstral Coefficients (MFCC)
4. Perangkat Lunak dan Library Python
5. Dataset dan Karakteristik Sampel Audio
6. Tahapan Pengolahan Audio Sistematis
7. Implementasi Kode Python
8. Hasil Ekstraksi Fitur Aktual
9. Visualisasi dan Analisis Komparatif
10. Proses Identifikasi dan Pemodelan Klasifikasi
11. Evaluasi Hasil dan Performa Model
12. Pembahasan
13. Kesimpulan
14. Referensi

---

## 1. Pendahuluan

Suara merupakan fenomena gelombang mekanik longitudinal yang merambat melalui medium elastis (udara) akibat osilasi partikel mekanis. Ketika gelombang akustik tersebut ditangkap oleh transduser elektroakustik (mikrofon), variasi tekanan udara dikonversi menjadi sinyal tegangan analog kontinu, yang selanjutnya didigitalkan melalui proses *sampling* dan *quantization* untuk menghasilkan sinyal audio digital.

Dalam ranah **Pengolahan Sinyal Digital (PSD)** dan **Acoustic Scene Classification (ASC)**, identifikasi kondisi lingkungan berdasarkan karakteristik suara merupakan tantangan nyata. Salah satu aplikasi esensialnya adalah membedakan lingkungan **Berisik** (*Noisy Environment*, seperti jalan raya, pasar, kafe ramai, bengkel, kerumunan) dengan lingkungan **Non-Berisik / Hening** (*Quiet/Silent Environment*, seperti ruang belajar, perpustakaan, kamar tidur, alam sunyi).

Pembedaan kondisi ini tidak cukup dilakukan secara subjektif oleh manusia, melainkan memerlukan representasi matematis objektif melalui **ekstraksi fitur audio** (*audio feature extraction*). Fitur-fitur tersebut merangkum karakteristik sinyal dalam domain waktu (*temporal domain*), domain frekuensi (*spectral domain*), serta domain persepsi psikoakustik (*cepstral domain*).

Dokumentasi ini menyajikan alur ilmiah dan teknis lengkap dalam memproses **229 sampel rekaman audio nyata** yang dihimpun ke dalam kelas **Berisik** (116 file) dan **Non-Berisik** (113 file). Seluruh proses—mulai dari akuisisi audio multi-format, ekstraksi fitur RMS, Spectral Centroid, ZCR, dan 13 koefisien MFCC, hingga pemodelan klasifikasi menggunakan Support Vector Machine (SVM) dan Random Forest—diuji secara komprehensif dengan metrik evaluasi nyata.

---

## 2. Tujuan Praktikum

1. **Memahami Representasi Digital Sinyal Akustik:** Mengkaji representasi sinyal audio diskrit dalam domain waktu (amplitudo waveform) dan domain frekuensi melalui Transformasi Fourier (*Short-Time Fourier Transform* / STFT).
2. **Mengekstrak Fitur-Fitur Standar PSD:** Menghitung secara kuantitatif nilai **RMS**, **Spectral Centroid**, **Zero-Crossing Rate (ZCR)**, dan **Mel-Frequency Cepstral Coefficients (MFCC)** dari sampel audio riil.
3. **Mengolah Dataset Audio Nyata Multi-Format:** Mengonstruksi pipeline Python yang mampu menangani file audio dengan format beragam (MP3, WAV, AAC, M4A) dengan sample rate seragam 22.050 Hz mono.
4. **Menganalisis Perbedaan Fisik dan Statistik Antar-Kelas:** Membandingkan nilai rata-rata (*mean*) dan variabilitas (*standard deviation*) fitur antara lingkungan Berisik dan Non-Berisik melalui visualisasi waveform, spektrogram, dan boxplot distribusi.
5. **Membangun Model Klasifikasi Suara Terawasi (*Supervised Learning*):** Melatih algoritma *Support Vector Machine* (SVM) berbasis kernel RBF dan *Random Forest Classifier* untuk mengenali kelas suara secara otomatis.
6. **Mengevaluasi Kinerja Model secara Obyektif:** Mengukur performa klasifikasi menggunakan Confusion Matrix, Precision, Recall, F1-Score, serta *5-Fold Stratified Cross-Validation*.
7. **Mengidentifikasi Fitur Paling Diskriminatif:** Menentukan parameter akustik mana yang paling berkontribusi dalam membedakan suara berisik dari suara sunyi (*feature importance*).

---

## 3. Dasar Teori Pengolahan Sinyal Audio Digital

### 3.1 Representasi Sinyal Audio Digital & Analisis Short-Time

Sinyal akustik kontinu $x_a(t)$ dikonversi ke domain digital menjadi sinyal waktu diskrit $x[n]$ melalui dua tahapan fundamental:

1. **Sampling:** Pengambilan sampel amplitudo secara berkala pada interval $T_s = 1 / f_s$, di mana $f_s$ adalah frekuensi *sampling rate*. Menurut teorema Nyquist-Shannon, agar aliasing frekuensi tidak terjadi, frekuensi sampling harus memenuhi:
   $$f_s \ge 2 \cdot f_{\max}$$
   Pada praktikum ini digunakan $f_s = 22.050\text{ Hz}$, sehingga frekuensi tertinggi yang dapat dianalisis adalah frekuensi Nyquist sebesar $11.025\text{ Hz}$, yang mencakup seluruh rentang pendengaran manusia yang relevan untuk suara lingkungan.
2. **Kuantisasi (*Quantization*):** Pemetaan nilai amplitudo kontinu ke level-level diskrit tertentu. Umumnya audio PCM dikuantisasi dalam kedalaman bit 16-bit (menghasilkan $2^{16} = 65.536$ level kuantisasi diskrit).

#### Analisis Frame-by-Frame (*Short-Time Analysis*)

Sinyal suara bersifat **non-stasioner**, artinya sifat statistik dan kandungan frekuensinya berubah seiring berjalannya waktu. Oleh karena itu, sinyal audio tidak dapat dianalisis secara global sekaligus. Sebaliknya, sinyal dipecah menjadi potongan-potongan pendek berdurasi 20–50 ms yang disebut **frame**, di mana sinyal diasumsikan bersifat **kuasi-stasioner** (*quasi-stationary*).

- **Frame Length ($N$):** 2048 sampel ($\approx 92,88\text{ ms}$ pada $f_s = 22.050\text{ Hz}$).
- **Hop Length ($H$):** 512 sampel ($\approx 23,22\text{ ms}$), menghasilkan overlap sebesar $75\%$ antar-frame berturutan guna memastikan transisi energi sinyal tidak terputus.

---

### 3.2 Root Mean Square (RMS)

#### Pengertian dan Signifikansi Akustik
RMS (*Root Mean Square*) adalah ukuran energi atau kekuatan efektif suatu sinyal periodik/stokastik dalam suatu frame waktu. Secara fisik, RMS berkaitan langsung dengan **amplitudo rata-rata**, **intensitas suara**, dan persepsi kenyaringan (*loudness*).

#### Formulasi Matematis
Untuk sebuah frame sinyal $x[n]$ dengan panjang $N$ sampel:

$$\text{RMS} = \sqrt{\frac{1}{N} \sum_{n=0}^{N-1} x[n]^2}$$

#### Interpretasi pada Kasus Berisik vs Non-Berisik
- **Lingkungan Berisik:** Memiliki tingkat tekanan akustik tinggi yang ditimbulkan oleh banyak sumber suara simultan (mesin kendaraan, desau kerumunan, benturan fisik). Hal ini menghasilkan nilai RMS yang secara konsisten tinggi dengan variansi energi yang dinamis.
- **Lingkungan Non-Berisik / Sunyi:** Amplitudo sinyal mendekati garis ekuilibrium nol (hanya ada derau lantai mikrofon/thermal noise atau desis bisu), sehingga nilai RMS sangat rendah.

---

### 3.3 Spectral Centroid

#### Pengertian dan Signifikansi Akustik
Spectral Centroid menunjukkan titik pusat massa (*center of gravity*) dari spektrum frekuensi audio. Fitur ini merepresentasikan **kecerahan suara (*brightness*)**; semakin banyak komponen energi pada frekuensi tinggi, semakin besar nilai Spectral Centroid (dinyatakan dalam satuan Hertz).

#### Formulasi Matematis
Melalui penerapan *Short-Time Fourier Transform* (STFT), diperoleh spektrum magnitudo $|X[k]|$ pada bin frekuensi ke-$k$ dengan frekuensi fisik $f[k]$:

$$\text{Spectral Centroid} = \frac{\sum_{k=0}^{K-1} f[k] \cdot |X[k]|}{\sum_{k=0}^{K-1} |X[k]|}$$

Di mana:
- $|X[k]|$ = magnitudo spektral pada bin frekuensi ke-$k$.
- $f[k] = \frac{k \cdot f_s}{N_{\text{FFT}}}$ = frekuensi tengah bin ke-$k$ (Hz).
- $K = \frac{N_{\text{FFT}}}{2} + 1$ = jumlah bin frekuensi Nyquist.

#### Interpretasi pada Kasus Berisik vs Non-Berisik
- **Suara Berisik:** Mengandung spektrum frekuensi luas (*wideband*), termasuk komponen frekuensi menengah-tinggi akibat desis knalpot, gesekan ban, gesekan piring, derau klakson, atau percakapan, sehingga centroid cenderung bergeser ke frekuensi yang lebih tinggi.
- **Suara Sunyi:** Didominasi oleh dengung frekuensi sangat rendah (*low-frequency rumble*) dari perangkat elektronik atau tidak adanya energi spektral frekuensi tinggi sama sekali, sehingga titik berat spektrum berada pada nilai frekuensi yang lebih rendah atau berfluktuasi rendah.

---

### 3.4 Zero-Crossing Rate (ZCR)

#### Pengertian dan Signifikansi Akustik
ZCR (*Zero-Crossing Rate*) mengukur seberapa sering sinyal audio memotong garis nol (berpindah tanda dari positif ke negatif atau sebaliknya) per satuan frame. Fitur ini merupakan aproksimasi kasar domain-waktu dari kandungan frekuensi sinyal dan sangat berguna dalam membedakan suara bernada (*voiced/tonal*) dari derau tak teratur (*unvoiced/noise*).

#### Formulasi Matematis
Untuk sebuah frame sinyal $x[n]$ berukuran $N$ sampel:

$$\text{ZCR} = \frac{1}{2(N-1)} \sum_{n=1}^{N-1} |\text{sgn}(x[n]) - \text{sgn}(x[n-1])|$$

Di mana fungsi tanda (*sign function*) didefinisikan sebagai:

$$\text{sgn}(x) = \begin{cases} 1, & x \ge 0 \\ -1, & x < 0 \end{cases}$$

#### Interpretasi pada Kasus Berisik vs Non-Berisik
- **Suara Berisik / Noise:** Memiliki sifat stokastik frekuensi tinggi yang berosilasi cepat melewati sumbu nol, menghasilkan ZCR yang tinggi.
- **Suara Sunyi / Tenang:** Osilasi sangat lambat atau hampir datar pada garis diam (*silence*), menghasilkan ZCR yang lebih rendah secara signifikan.

---

### 3.5 Mel-Frequency Cepstral Coefficients (MFCC)

#### Pengertian dan Signifikansi Akustik
MFCC adalah representasi fitur spektral paling dominan dalam pengenalan wicara (*speech recognition*) dan klasifikasi suara lingkungan (*sound event detection*). MFCC memetakan spektrum daya sinyal ke dalam skala **Mel**, yaitu skala persepsi nada manusia yang bersifat logaritmik—telinga manusia jauh lebih sensitif membedakan perubahan nada pada frekuensi rendah (<1 kHz) daripada frekuensi tinggi.

#### Tahapan Komputasi MFCC
Ekstraksi 13 koefisien MFCC dilakukan melalui 6 langkah berurutan:

```
Sinyal x[n] ──> [Pre-emphasis] ──> [Framing & Windowing] ──> [FFT & Power Spectrum]
                                                                    │
[MFCC Coeffs] <── [DCT-II] <── [Logarithm ln(·)] <── [Mel Filterbank]
```

1. **Pre-emphasis:** Filter diferensial tingkat-pertama $y[n] = x[n] - \alpha x[n-1]$ ($\alpha \approx 0,97$) untuk mengkompensasi pelemahan spektral alami pada frekuensi tinggi.
2. **Framing & Hamming Windowing:** Memecah sinyal ke dalam frame dan mengalikannya dengan jendela Hamming $w[n] = 0,54 - 0,46 \cos\left(\frac{2\pi n}{N-1}\right)$ untuk meminimalisasi *spectral leakage*.
3. **Fast Fourier Transform (FFT):** Menghitung magnitudo kuadrat dari spektrum frekuensi untuk memperoleh *Power Spectrum* $P[k] = \frac{1}{N} |X[k]|^2$.
4. **Mel-Scale Filterbank:** Mengintegrasikan daya spektral ke dalam $M$ buah bandpass filter berbentuk segitiga (umumnya $M = 40$) yang spasi pusatnya berjarak sama pada skala Mel:
   $$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$
5. **Logarithmic Compression:** Mengambil logaritma natural dari energi setiap kanal filterbank: $S[m] = \ln\left( \sum_k P[k] H_m[k] \right)$. Langkah ini menirukan persepsi intensitas suara manusia yang bersifat logaritmik (desibel).
6. **Discrete Cosine Transform (DCT-II):** Menerapkan transformasi kosinus terbalik pada log-energi filterbank untuk mendekorelasikan energi dan memisahkan *envelope spektral* dari eksitasi harmonik:
   $$c[j] = \sum_{m=1}^{M} S[m] \cos\left( \frac{\pi j (m - 0,5)}{M} \right), \quad j = 1, 2, \dots, 13$$

#### Arti Fisik Koefisien MFCC
- **MFCC-1 ($c_1$):** Merepresentasikan energi spektral keseluruhan (*overall log-energy*). Nilainya berkorelasi kuat dengan RMS dan tingkat keterisian spektrum.
- **MFCC-2 ($c_2$):** Mengukur kemiringan spektral (*spectral tilt*), membedakan distribusi energi dominan frekuensi rendah vs frekuensi tinggi.
- **MFCC-3 s.d. MFCC-13:** Menangkap detail formasi lekukan spektral (*spectral envelope shape*) yang spesifik bagi jenis derau tertentu (derau mekanik, aliran udara, suara percakapan).

---

## 4. Perangkat Lunak dan Library Python

Eksperimen ini dibangun menggunakan ekosistem ilmiah Python dengan dependensi library sebagai berikut:

| Perangkat / Library | Versi | Peran dan Fungsi dalam Praktikum |
|---|---|---|
| **Python** | 3.10.x | Lingkungan komputasi utama |
| **librosa** | 0.10.x | Ekstraksi fitur audio (RMS, Spectral Centroid, ZCR, MFCC) dan dekode sinyal |
| **soundfile** | 0.12.x | Backend pembaca file audio PCM WAV tanpa kompresi |
| **imageio-ffmpeg** | 0.6.0 | Penyedia pustaka biner FFmpeg portabel untuk mendekode format MP3, AAC, dan M4A |
| **numpy** | 1.23+ | Operasi matriks, komputasi numerik, dan agregasi statistik (mean, std) |
| **pandas** | 2.0+ | Manajemen struktur data tabel tabular (DataFrame) dan ekspor hasil ke CSV |
| **scikit-learn** | 1.3+ | Standardisasi (`StandardScaler`), pemodelan ML (SVM, Random Forest), dan metrik evaluasi |
| **matplotlib** | 3.7+ | Visualisasi waveform, spektrogram, diagram kurva temporal, boxplot, dan confusion matrix |

---

## 5. Dataset dan Karakteristik Sampel Audio

Dataset yang digunakan dalam analisis ini merupakan kumpulan data rekaman suara lingkungan riil yang dikumpulkan dari 2 berkas arsip zip:
- **`BerisikPSD.zip`**: Berisi rekaman kondisi lingkungan bising (jalan raya, kerumunan kampus, kantin, warung kopi, suara latar televisi, aktivitas ruangan padat).
- **`NonBerisikPSD.zip`**: Berisi rekaman kondisi lingkungan tenang/hening (kamar tidur pribadi, ruang belajar tertutup, perpustakaan sepi, lingkungan sunyi malam hari).

### 5.1 Distribusi dan Spesifikasi Sampel

Secara keseluruhan, sistem berhasil mengekstraksi dan memvalidasi **229 file audio** tanpa ada data yang korup atau gagal muat:

| Parameter Dataset | Kelas Berisik | Kelas NonBerisik | Total Keseluruhan |
|---|---|---|---|
| **Jumlah File Sampel** | **116 file** ($50,65\%$) | **113 file** ($49,35\%$) | **229 file** ($100\%$) |
| **Format File** | MP3, WAV, AAC, M4A | MP3, WAV, AAC, M4A | 203 MP3, 12 AAC, 8 M4A, 6 WAV |
| **Durasi Rata-rata** | $31,20\text{ detik}$ | $30,85\text{ detik}$ | $31,03\text{ detik}$ |
| **Rentang Durasi** | $18,04\text{ s} - 48,13\text{ s}$ | $29,91\text{ s} - 38,46\text{ s}$ | $18,04\text{ s} - 48,13\text{ s}$ |
| **Kanal Audio** | Mono (dikonversi otomatis) | Mono (dikonversi otomatis) | 1 kanal (*single channel*) |
| **Target Sample Rate** | $22.050\text{ Hz}$ | $22.050\text{ Hz}$ | $22.050\text{ Hz}$ |

```
TugasPSD/
└── dataset_audio/
    ├── Berisik/       (116 file .mp3, .wav, .aac, .m4a)
    └── NonBerisik/    (113 file .mp3, .wav, .aac, .m4a)
```

> **Keseimbangan Kelas (*Balanced Dataset*):** Rasio perbandingan data Berisik (116) dan NonBerisik (113) berada pada kisaran $1,02 : 1$, yang menunjukkan dataset sangat seimbang. Kondisi ini ideal untuk pelatihan machine learning tanpa perlu dilakukan teknik *oversampling* (SMOTE) maupun *undersampling*.

---

## 6. Tahapan Pengolahan Audio Sistematis

Alur kerja pengolahan audio dirancang dalam pipeline end-to-end sebagai berikut:

```text
flowchart TD
    A[Dataset Audio: BerisikPSD & NonBerisikPSD] --> B[Pemindaian Berkas & Filter Ekstensi]
    B --> C[Pemuatan Audio via Librosa + FFmpeg]
    C --> D[Prapemrosesan: Resampling ke 22.050 Hz & Downmix Mono]
    D --> E[Short-Time Framing: Frame=2048, Hop=512]
    E --> F1[Ekstraksi Fitur Domain Waktu: RMS & ZCR]
    E --> F2[Ekstraksi Fitur Domain Spektral: Spectral Centroid]
    E --> F3[Ekstraksi Fitur Psikoakustik: MFCC 1-13]
    F1 --> G[Agregasi Statistik: Mean & Standard Deviation]
    F2 --> G
    F3 --> G
    G --> H[Konstruksi Tabel Fitur 32 Dimensi & Ekspor CSV]
    H --> I[Eksplorasi Data & Visualisasi Komparatif]
    H --> J[Standardisasi Fitur: StandardScaler]
    J --> K[Pembagian Data: Stratified Split 80:20]
    K --> L1[Pelatihan Model SVM Kernel RBF]
    K --> L2[Pelatihan Model Random Forest 100 Trees]
    L1 --> M[Evaluasi Model & 5-Fold Stratified Cross Validation]
    L2 --> M
    M --> N[Analisis Feature Importance & Pembahasan]
```

1. **Akuisisi Data & Dekode:** Mengakses file audio dalam direktori kelas, mengekstrak stream audio mentah melalui pustaka `librosa` yang terintegrasi dengan backend dekoder `FFmpeg`.
2. **Prapemrosesan:**
   - Melakukan konversi seluruh sinyal multi-kanal (stereo) menjadi sinyal mono melalui perataan kanal: $x_{\text{mono}}[n] = \frac{x_L[n] + x_R[n]}{2}$.
   - Menyelaraskan frekuensi pencuplikan (*resampling*) ke standar seragam $f_s = 22.050\text{ Hz}$.
3. **Ekstraksi Fitur Frame-by-Frame:**
   - Menghitung array RMS per frame.
   - Menghitung array Spectral Centroid per frame (STFT).
   - Menghitung array ZCR per frame.
   - Menghitung matriks 13 koefisien MFCC per frame.
4. **Agregasi Statistik:** Merangkum variasi temporal setiap fitur menjadi 2 nilai statistik per sampel audio: **Rata-rata (*Mean*)** untuk menangkap tingkat tendensi sentral, dan **Standar Deviasi (*Std*)** untuk menangkap dinamika variasi sinyal.
   $$\text{Total Dimensi Fitur} = (1\text{ RMS} + 1\text{ Centroid} + 1\text{ ZCR} + 13\text{ MFCC}) \times 2 = 32\text{ Fitur Numerik}$$
5. **Penyusunan Tabel Tabular:** Menggabungkan kolom metadata (`filename`, `label`, `durasi_detik`) dengan 32 kolom fitur ke dalam DataFrame Pandas dan menyimpannya sebagai file `hasil_ekstraksi_fitur_audio.csv`.
6. **Pemodelan dan Evaluasi Machine Learning:**
   - Melakukan normalisasi z-score dengan `StandardScaler`.
   - Menguji akurasi model pada data uji (*unseen test data*) serta stabilitas validasi silang (*5-Fold Stratified CV*).

---

## 7. Implementasi Kode Python

Berikut adalah implementasi kode Python modular yang digunakan untuk mengeksekusi pipeline analisis audio ini secara menyeluruh:

```python
"""
Pipeline Pengolahan Sinyal Digital: Ekstraksi Fitur Audio dan Klasifikasi
Penulis: Muhammad Zaidan Nabil Rafi (240411100068)
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Integrasikan path binary ffmpeg agar mendukung MP3, AAC, dan M4A
import imageio_ffmpeg
ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

warnings.filterwarnings('ignore')

# Parameter Akustik
TARGET_SR = 22050
N_MFCC = 13
DATASET_DIR = './dataset_audio'

def scan_dataset(dataset_dir):
    """Memindai subfolder Berisik dan NonBerisik"""
    supported_ext = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    file_list = []
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    for cls in classes:
        subdir = os.path.join(dataset_dir, cls)
        for f in sorted(os.listdir(subdir)):
            if os.path.splitext(f)[1].lower() in supported_ext:
                file_list.append({'path': os.path.join(subdir, f), 'filename': f, 'label': cls})
    return file_list, classes

def extract_audio_features(filepath, filename, label, target_sr=TARGET_SR, n_mfcc=N_MFCC):
    """Mengekstrak 32 fitur akustik statistik dari sinyal audio"""
    try:
        y, sr = librosa.load(filepath, sr=target_sr, mono=True)
        if y is None or len(y) == 0:
            return None
        duration = float(librosa.get_duration(y=y, sr=sr))
        
        # 1. RMS
        rms = librosa.feature.rms(y=y)[0]
        # 2. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        # 3. ZCR
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        # 4. MFCC (13 koefisien)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        row = {
            'filename': filename,
            'label': label,
            'durasi_detik': round(duration, 2),
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms)),
            'centroid_mean': float(np.mean(centroid)),
            'centroid_std': float(np.std(centroid)),
            'zcr_mean': float(np.mean(zcr)),
            'zcr_std': float(np.std(zcr))
        }
        for j in range(n_mfcc):
            row[f'mfcc_{j+1}_mean'] = float(np.mean(mfcc[j]))
            row[f'mfcc_{j+1}_std'] = float(np.std(mfcc[j]))
            
        return row
    except Exception as e:
        print(f"Gagal memuat {filename}: {e}")
        return None

# Ekstraksi seluruh file
file_list, class_names = scan_dataset(DATASET_DIR)
data_rows = [extract_audio_features(f['path'], f['filename'], f['label']) for f in file_list]
df_features = pd.DataFrame([r for r in data_rows if r is not None])

# Simpan ke CSV
df_features.to_csv('output/hasil_ekstraksi_fitur_audio.csv', index=False)
print(f"Ekstraksi selesai: {len(df_features)} file berhasil diproses.")
```

---

## 8. Hasil Ekstraksi Fitur Aktual

Proses ekstraksi menghasilkan tabel fitur berdimensi **229 baris $\times$ 35 kolom** (3 kolom metadata + 32 kolom fitur numerik). Tabel lengkap tersimpan pada berkas [hasil_ekstraksi_fitur_audio.csv](output/hasil_ekstraksi_fitur_audio.csv).

### 8.1 Pratinjau Sampel Fitur Nyata

Berikut adalah sampel data nyata dari 10 rekaman audio (5 sampel Berisik dan 5 sampel Non-Berisik) yang merepresentasikan kolom-kolom utama:

| Nama File Sampel | Kelas | Durasi (s) | RMS Mean | Centroid Mean (Hz) | ZCR Mean | MFCC-1 Mean | MFCC-2 Mean |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `228_berisik_1.wav` | **Berisik** | 30.00 | 0.091092 | 1174.62 | 0.062157 | -234.55 | 152.20 |
| `228_berisik_2.wav` | **Berisik** | 30.00 | 0.095198 | 1184.44 | 0.060980 | -235.03 | 156.49 |
| `228_berisik_3.wav` | **Berisik** | 30.00 | 0.088939 | 1323.32 | 0.071794 | -235.12 | 132.62 |
| `24-003 berisik (1).mp3` | **Berisik** | 30.04 | 0.041927 | 1776.16 | 0.080559 | -250.63 | 120.91 |
| `24-003 berisik (2).mp3` | **Berisik** | 30.77 | 0.077234 | 2139.96 | 0.154330 | -249.56 | 102.71 |
| `228_non_berisik_1.wav` | **NonBerisik** | 30.00 | 0.061232 | 1455.91 | 0.080069 | -209.27 | 153.12 |
| `228_non_berisik_2.wav` | **NonBerisik** | 30.00 | 0.045996 | 1303.54 | 0.069200 | -252.25 | 155.71 |
| `228_non_berisik_3.wav` | **NonBerisik** | 30.00 | 0.069960 | 1282.05 | 0.081010 | -220.88 | 152.48 |
| `24-003 sunyi 2.mp3` | **NonBerisik** | 30.00 | 0.021035 | 1726.61 | 0.084064 | -326.73 | 135.66 |
| `24-003 sunyi 3.mp3` | **NonBerisik** | 30.41 | 0.015946 | 1788.42 | 0.099719 | -429.05 | 116.02 |

### 8.2 Rangkuman Statistik Komparatif Antar-Kelas

Tabel berikut menyajikan rekapitulasi nilai rata-rata ($\mu$) dan standar deviasi ($\sigma$) dari fitur-fitur fundamental untuk masing-masing kelas:

| Fitur Akustik | Kelas Berisik ($N=116$) | Kelas NonBerisik ($N=113$) | Rasio / Selisih $(\Delta)$ | Makna Karakteristik Akustik |
|---|:---:|:---:|:---:|---|
| **RMS Mean** | $\mathbf{0,0844 \pm 0,0632}$ | $\mathbf{0,0327 \pm 0,0371}$ | $\mathbf{\approx 2,58\times}$ lebih besar | Suara bising memiliki intensitas energi rata-rata jauh lebih kuat |
| **RMS Std** | $0,0491 \pm 0,0328$ | $0,0221 \pm 0,0225$ | $\approx 2,22\times$ lebih besar | Variasi dinamika volume kebisingan jauh lebih fluktuatif |
| **Spectral Centroid Mean (Hz)** | $\mathbf{1.786,13 \pm 581,48}$ | $\mathbf{1.646,51 \pm 633,93}$ | $+139,62\text{ Hz}$ | Suara berisik lebih terang (*brighter*) dengan komponen frekuensi tinggi |
| **Spectral Centroid Std (Hz)** | $457,23 \pm 194,41$ | $479,34 \pm 187,23$ | $-22,11\text{ Hz}$ | Persebaran kecerahan spektral memiliki rentang deviasi seimbang |
| **ZCR Mean** | $\mathbf{0,0961 \pm 0,0443}$ | $\mathbf{0,0784 \pm 0,0408}$ | $+22,58\%$ lebih tinggi | Kebisingan menghasilkan lebih banyak perpotongan nol per detik |
| **ZCR Std** | $0,0375 \pm 0,0218$ | $0,0349 \pm 0,0179$ | $+7,45\%$ | Fluktuasi laju perpotongan nol sedikit lebih bervariasi |
| **MFCC-1 Mean (Log Energy)** | $\mathbf{-221,78 \pm 97,45}$ | $\mathbf{-342,67 \pm 129,69}$ | $\mathbf{+120,89\text{ dB}}$ | Energi logaritmik total kelas bising terpaut jauh lebih padat |
| **MFCC-2 Mean** | $123,14 \pm 30,12$ | $125,41 \pm 32,87$ | $-2,27$ | Distribusi keseimbangan low-vs-high relatif berdekatan |
| **MFCC-3 Mean** | $\mathbf{-27,84 \pm 23,15}$ | $\mathbf{-6,52 \pm 19,83}$ | $-21,32$ | Lekukan spektral formant ketiga menunjukkan diferensiasi jelas |
| **MFCC-4 Mean** | $20,24 \pm 15,00$ | $21,29 \pm 14,84$ | $-1,05$ | Harmonik orde keempat menunjukkan kemiripan pola resonansi |

---

## 9. Visualisasi dan Analisis Komparatif

Untuk membuktikan secara visual perbedaan karakteristik sinyal kedua kelas, telah diproduksi 4 grafik komparatif berkualitas publikasi.

### 9.1 Perbandingan Waveform dan Spektrogram Frekuensi

![Waveform dan Spektrogram Komparatif](output/audio_waveform_spectrogram.png)

#### Analisis dan Interpretasi:
1. **Domain Waktu (Waveform):**
   - Pada panel atas (Waveform), sampel **Berisik** (`24-003 berisik (1).mp3`) menunjukkan densitas amplitudo yang padat, berosilasi konstan pada kisaran rentang $[-0,4 \text{ s.d. } 0,4]$ tanpa jeda hening. Ini mengindikasikan kehadiran sumber suara berkelanjutan seperti suara deru kendaraan atau obrolan kerumunan.
   - Sebaliknya, sampel **NonBerisik** (`228_non_berisik_1.wav`) memiliki amplitudo yang relatif tipis dan mendekati garis ekuilibrium nol, dengan dinamika puncak yang sangat terkendali.
2. **Domain Frekuensi (Spektrogram STFT):**
   - Spektrogram menampilkan persebaran energi dalam desibel (dB) terhadap sumbu waktu dan frekuensi.
   - Pada kelas **Berisik**, energi spektral (warna terang/kuning-merah) tersebar tebal mulai dari frekuensi rendah hingga melintasi $4.000 - 8.000\text{ Hz}$ (*wideband acoustic energy distribution*).
   - Pada kelas **NonBerisik**, sebagian besar area spektrogram berwarna ungu gelap hingga hitam pada frekuensi di atas $2.000\text{ Hz}$, menandakan energi akustik frekuensi tinggi hampir nihil (*low high-frequency presence*).

---

### 9.2 Dinamika Fitur Temporal (RMS, Spectral Centroid, dan ZCR)

![Kurva Fitur Temporal RMS, Spectral Centroid, dan ZCR](output/audio_temporal_features.png)

#### Analisis dan Interpretasi:
1. **Kurva RMS vs Waktu:**
   - Garis merah (Berisik) secara konsisten berada di atas garis hijau (NonBerisik) sepanjang durasi 30 detik. Puncak RMS Berisik mencapai $\approx 0,10$, sedangkan NonBerisik berada di bawah $0,05$. Hal ini membuktikan bahwa RMS adalah indikator langsung kekuatan energi fisik yang sangat diskriminatif.
2. **Kurva Spectral Centroid vs Waktu:**
   - Spectral Centroid kelas Berisik menunjukkan fluktuasi tajam yang kerap menyentuh angka $2.000 - 3.000\text{ Hz}$ saat terjadi letupan derau sesaat. Kelas NonBerisik memiliki lintasan centroid yang lebih rendah dan datar.
3. **Kurva Zero-Crossing Rate (ZCR) vs Waktu:**
   - Nilai ZCR kelas Berisik mengalami lonjakan periodik yang menandai komponen desis frekuensi tinggi acak, sedangkan ZCR kelas NonBerisik cenderung stabil pada level baseline.

---

### 9.3 Heatmap Koefisien MFCC (Mel-Frequency Cepstral Coefficients)

![Heatmap Koefisien MFCC](output/audio_mfcc_heatmaps.png)

#### Analisis dan Interpretasi:
- Heatmap menampilkan variasi magnitudo koefisien MFCC ke-1 hingga ke-13 terhadap waktu.
- Pada **Kelas Berisik**, koefisien paling bawah (MFCC-1, indeks 1) didominasi oleh warna biru muda hingga merah lembut (nilai negatif yang tidak terlalu ekstrem, sekitar $-200\text{ s.d. } -250$), menandakan energi logaritmik total yang tinggi. Transisi antar-koefisien menunjukkan pola dinamis akibat kekayaan spektral kebisingan.
- Pada **Kelas NonBerisik**, koefisien MFCC-1 berwarna biru tua pekat (mencapai $-400\text{ s.d. } -500$), membuktikan bahwa intensitas energi total spektrum berada pada batas minimum. Pola pada koefisien tinggi (MFCC 5–13) cenderung homogen tanpa kontur harmonik yang tegas.

---

### 9.4 Distribusi Fitur Komparatif (Boxplot)

![Distribusi Fitur Boxplot](output/audio_boxplots_distribusi.png)

#### Analisis dan Interpretasi Boxplot:
1. **Pemisahan Kelas pada RMS Mean & MFCC-1 Mean:**
   - Boxplot `rms_mean` dan `mfcc_1_mean` memperlihatkan pemisahan median (*interquartile range* / IQR) yang sangat kontras antara kotak merah (Berisik) dan hijau (NonBerisik). Tumpang tindih (*overlap*) antar kedua kelas sangat minimal, menjadikan kedua fitur ini kandidat utama pemisah linier bagi model klasifikasi.
2. **Karakteristik Spectral Centroid & ZCR:**
   - Pada `centroid_mean` dan `zcr_mean`, nilai tengah kelas Berisik lebih tinggi, namun terdapat rentang *overlap* moderat di bagian tengah. Kondisi ini wajar karena sebagian ruangan hening dapat memiliki gema kecil atau derau frekuensi tinggi berdaya sangat rendah yang menaikkan nilai ZCR tanpa menaikkan nilai RMS.
3. **Pola Formant pada MFCC-2 & MFCC-3:**
   - `mfcc_2_mean` memiliki distribusi yang relatif simetris dan berdekatan di angka $\approx 124$, sementara `mfcc_3_mean` menunjukkan pergeseran nyata (median Berisik $\approx -27,8$ vs NonBerisik $\approx -6,5$).

---

## 10. Proses Identifikasi dan Pemodelan Klasifikasi

### 10.1 Perbedaan Konseptual: Ekstraksi Fitur vs Klasifikasi

Dalam disiplin machine learning untuk sinyal audio, penting untuk membedakan dua tahapan berikut:
- **Ekstraksi Fitur (Fitur Audio):** Berperan sebagai representasi data (*representation learning*). Fitur seperti RMS, Centroid, ZCR, dan MFCC tidak dapat mengambil keputusan klasifikasi dengan sendirinya; mereka hanya menyederhanakan sinyal mentah jutaan sampel audio menjadi 32 parameter numerik yang padat informasi.
- **Algoritma Klasifikasi (Classifier):** Algoritma terawasi seperti SVM dan Random Forest yang bertugas memetakan vektor fitur 32 dimensi tersebut ke dalam label diskrit $\{0: \text{Berisik}, 1: \text{NonBerisik}\}$ melalui pembentukan bidang batas keputusan (*decision boundary*).

### 10.2 Protokol Pembagian Data dan Standardisasi

Untuk menghindari *data leakage* (kebocoran data) dan menjamin keandalan pengujian model:

1. **Pembagian Data (*Stratified Train-Test Split*):**
   - Rasio pembagian data: **$80\%$ Data Latih ($183$ sampel)** dan **$20\%$ Data Uji ($46$ sampel)**.
   - Stratifikasi diterapkan agar proporsi kedua kelas di data uji tepat seimbang ($23$ sampel Berisik dan $23$ sampel NonBerisik).
2. **Standardisasi Fitur (*Z-score Scaling*):**
   Karena satuan fitur berbeda drastis (RMS bernilai $\approx 0,08$, Centroid bernilai ribuan Hz, MFCC-1 bernilai ratusan negatif), seluruh fitur diskalakan menggunakan `StandardScaler`:
   $$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
3. **Spesifikasi Algoritma:**
   - **Support Vector Machine (SVM):** Kernel Radial Basis Function (*RBF*), $C = 1,0$, $\gamma = \text{'scale'}$. Efektif dalam memetakan data berdimensi sedang ke ruang berdimensi lebih tinggi untuk mencari *hyperplane* pemisah maksimal.
   - **Random Forest Classifier:** Ensemble dari $100$ pohon keputusan (*decision trees*), `max_depth = 8`, `random_state = 42`. Sangat tangguh terhadap *outlier* dan mampu mengukur kontribusi kepentingan fitur secara probabilistik.

---

## 11. Evaluasi Hasil dan Performa Model

### 11.1 Evaluasi pada Data Uji Independen ($N_{\text{test}} = 46$)

Hasil inferensi kedua model pada 46 sampel uji yang belum pernah dilihat saat pelatihan dirangkum pada tabel berikut:

| Model Machine Learning | Kelas Target | Precision | Recall | F1-Score | Support | Akurasi Keseluruhan |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Support Vector Machine (SVM RBF)** | **Berisik** | $0,9000$ | $0,7826$ | $0,8372$ | 23 | $\mathbf{84,78\%}$ |
| | **NonBerisik** | $0,8077$ | $0,9130$ | $0,8571$ | 23 | (39 dari 46 benar) |
| | *Macro Average* | $0,8538$ | $0,8478$ | $0,8472$ | 46 | |
| **Random Forest (100 Trees)** | **Berisik** | $0,8261$ | $0,8261$ | $0,8261$ | 23 | $\mathbf{82,61\%}$ |
| | **NonBerisik** | $0,8261$ | $0,8261$ | $0,8261$ | 23 | (38 dari 46 benar) |
| | *Macro Average* | $0,8261$ | $0,8261$ | $0,8261$ | 46 | |

### 11.2 Visualisasi Confusion Matrix

![Confusion Matrix Model SVM dan Random Forest](output/audio_cm_models.png)

#### Interpretasi Confusion Matrix:
- **SVM Model:**
  - **True Berisik:** 18 sampel berhasil diidentifikasi sebagai Berisik secara tepat.
  - **False NonBerisik:** 5 sampel Berisik salah terprediksi sebagai NonBerisik.
  - **True NonBerisik:** 21 sampel berhasil diidentifikasi sebagai NonBerisik secara tepat (Recall NonBerisik sangat tinggi, $91,30\%$).
  - **False Berisik:** 2 sampel NonBerisik salah diklasifikasikan sebagai Berisik.
- **Random Forest Model:**
  - Menunjukkan performa yang simetris sempurna: 19 True Berisik, 19 True NonBerisik, 4 False Berisik, dan 4 False NonBerisik.

---

### 11.3 Validasi Silang (*5-Fold Stratified Cross Validation*)

Untuk memastikan performa model stabil dan tidak bergantung pada keberuntungan split data, dilakukan validasi silang 5-lipatan pada seluruh 229 sampel dataset:

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Rata-rata Akurasi ($\mu \pm \sigma$)** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Support Vector Machine (SVM)** | $86,96\%$ | $89,13\%$ | $73,91\%$ | $76,09\%$ | $88,89\%$ | $\mathbf{83,00\% \pm 6,61\%}$ |
| **Random Forest Classifier** | $93,48\%$ | $93,48\%$ | $73,91\%$ | $89,13\%$ | $88,89\%$ | $\mathbf{87,78\% \pm 7,22\%}$ |

> **Analisis Stabilitas:** Pada pengujian validasi silang, **Random Forest mencapai rata-rata akurasi sebesar $87,78\%$**, bahkan menyentuh $93,48\%$ pada dua fold pertama. Ini membuktikan bahwa ensemble tree memiliki kapasitas generalisasi yang lebih kokoh dalam menangani variabilitas rekaman ponsel mahasiswa.

---

### 11.4 Analisis Kepentingan Fitur (*Feature Importance*)

Melalui mekanisme *Gini Impurity Reduction* pada Random Forest, kontribusi masing-masing fitur dalam membedakan suara berisik dan non-berisik dapat diurutkan secara objektif:

![Top 15 Feature Importance](output/audio_feature_importance.png)

Berikut adalah daftar 15 fitur paling berpengaruh dalam klasifikasi suara ini:

| Peringkat | Nama Fitur | Nilai Importance | Akumulasi | Kategori Domain Fitur |
|:---:|:---|:---:|:---:|:---|
| **1** | `rms_mean` | $\mathbf{0,13614}$ ($13,61\%$) | $13,61\%$ | Domain Waktu (Tingkat Energi Rata-rata) |
| **2** | `mfcc_1_mean` | $\mathbf{0,07932}$ ($7,93\%$) | $21,55\%$ | Domain Cepstral (Log-Energi Spektral Total) |
| **3** | `rms_std` | $\mathbf{0,06096}$ ($6,10\%$) | $27,64\%$ | Domain Waktu (Fluktuasi Dinamika Energi) |
| **4** | `mfcc_12_std` | $0,05641$ ($5,64\%$) | $33,28\%$ | Domain Cepstral (Variasi Spektral Frekuensi Tinggi) |
| **5** | `mfcc_11_std` | $0,05127$ ($5,13\%$) | $38,41\%$ | Domain Cepstral (Variasi Spektral Frekuensi Tinggi) |
| **6** | `mfcc_3_mean` | $0,04868$ ($4,87\%$) | $43,28\%$ | Domain Cepstral (Kontur Formant Ketiga) |
| **7** | `mfcc_11_mean` | $0,04131$ ($4,13\%$) | $47,41\%$ | Domain Cepstral (Komponen Spektral Orde Tinggi) |
| **8** | `mfcc_13_std` | $0,03361$ ($3,36\%$) | $50,77\%$ | Domain Cepstral (Variasi Detail Spektral) |
| **9** | `mfcc_5_mean` | $0,03144$ ($3,14\%$) | $53,92\%$ | Domain Cepstral (Bentuk Envelope Menengah) |
| **10** | `mfcc_6_mean` | $0,02857$ ($2,86\%$) | $56,77\%$ | Domain Cepstral (Bentuk Envelope Menengah) |
| **11** | `mfcc_10_mean` | $0,02835$ ($2,84\%$) | $59,61\%$ | Domain Cepstral (Detail Harmonik Atas) |
| **12** | `zcr_mean` | $0,02525$ ($2,53\%$) | $62,14\%$ | Domain Waktu (Laju Perpotongan Nol) |
| **13** | `mfcc_8_mean` | $0,02460$ ($2,46\%$) | $64,60\%$ | Domain Cepstral (Envelope Frekuensi Menengah) |
| **14** | `mfcc_13_mean` | $0,02425$ ($2,43\%$) | $67,03\%$ | Domain Cepstral (Koefisien Teratas) |
| **15** | `mfcc_5_std` | $0,02372$ ($2,37\%$) | $69,40\%$ | Domain Cepstral (Variabilitas Envelope) |

---

## 12. Pembahasan

### 12.1 Mengapa RMS dan MFCC-1 Menjadi Fitur Paling Berpengaruh?

Berdasarkan hasil pemodelan, `rms_mean` ($13,61\%$), `mfcc_1_mean` ($7,93\%$), dan `rms_std` ($6,10\%$) menduduki posisi teratas dalam hierarki *feature importance*. Secara gabungan, tiga fitur ini menyumbang lebih dari $27,6\%$ bobot keputusan model:

1. **Korelasi Fisik dengan Energi Akustik:**
   Secara esensial, perbedaan paling mendasar antara kondisi lingkungan "Berisik" dan "Non-Berisik" adalah **densitas energi getaran partikel udara**. Pada lingkungan berisik, terdapat sumber emisi suara aktif yang terus memancarkan gelombang bertekanan tinggi, menghasilkan deviasi amplitudo yang besar dari garis netral. `rms_mean` mengkuantifikasi energi absolut ini secara langsung, sementara `mfcc_1_mean` menghitung energi spektral terintegrasi dalam skala logaritmik.
2. **Karakteristik Temporal (`rms_std`):**
   Suara berisik di lingkungan nyata (pasar, jalan raya) bersifat sangat fluktuatif—ada kendaraan lewat, orang berteriak, benda beradu—yang menyebabkan nilai `rms_std` tinggi ($0,0491$ vs $0,0221$). Sebaliknya, kondisi hening memiliki energi rendah yang statis.

### 12.2 Peran Krusial Koefisien MFCC Orde Tinggi

Menarik untuk dicermati bahwa koefisien MFCC tingkat tinggi (`mfcc_12_std`, `mfcc_11_std`, `mfcc_11_mean`) menempati peringkat 4, 5, dan 7 dalam *importance*.
- Koefisien MFCC orde rendah ($c_1 - c_4$) menggambarkan kemiringan spektral makro (*spectral tilt*).
- Koefisien MFCC orde tinggi ($c_{10} - c_{13}$) menangkap fluktuasi spektral frekuensi tinggi yang sangat cepat.
- Pada suara derau lingkungan (*acoustic environmental noise*), terdapat desis gesekan ban, hembusan angin, dan pantulan ruangan yang meninggalkan jejak fluktuasi dinamis pada filterbank frekuensi tinggi Mel, yang berhasil dideteksi oleh standar deviasi koefisien ke-11 dan ke-12.

### 12.3 Analisis Kesalahan Prediksi (*Error Analysis*)

Pada matriks konfusi model SVM, terdapat 5 sampel Berisik yang diprediksi sebagai NonBerisik dan 2 sampel NonBerisik yang diprediksi sebagai Berisik:

1. **False Negative (Berisik terprediksi Non-Berisik):**
   Terjadi pada rekaman berlabel bising yang direkam dari jarak jauh atau menggunakan mikrofon ponsel dengan fitur *noise suppression* bawaan yang agresif. Akibatnya, amplitudo rekaman teredam dan menyerupai rekaman sunyi (`rms_mean` rendah).
2. **False Positive (Non-Berisik terprediksi Berisik):**
   Terjadi ketika pada rekaman hening terdapat bunyi ketukan mikrofon sesaat (*handling noise*), suara tarikan napas dekat mikrofon, atau dengung keras AC/kipas angin ruangan. Letupan impulsif ini menaikkan nilai `rms_std` dan `centroid_mean`, sehingga model mengira rekaman tersebut berada di lingkungan bising.

---

## 13. Kesimpulan

Berdasarkan keseluruhan tahapan eksperimen, ekstraksi fitur audio, dan pemodelan machine learning pada **229 sampel audio riil** (`BerisikPSD.zip` dan `NonBerisikPSD.zip`), dapat ditarik kesimpulan ilmiah sebagai berikut:

1. **Keberhasilan Pipeline PSD:** Telah berhasil dirancang dan diimplementasikan sebuah pipeline Pengolahan Sinyal Digital berbasis Python yang mampu memproses rekaman audio multi-format (MP3, WAV, AAC, M4A), melakukan standardisasi ke $22.050\text{ Hz}$ mono, dan mengekstrak 32 fitur audio statistik secara otomatis.
2. **Karakteristik Fisik Akustik Terbukti Signifikan:**
   - Lingkungan **Berisik** memiliki tingkat energi rata-rata **$2,58\times$ lebih tinggi** (`rms_mean` $0,0844$ vs $0,0327$) dan energi logaritmik total terpaut jauh lebih padat (`mfcc_1_mean` $-221,78\text{ dB}$ vs $-342,67\text{ dB}$).
   - Lingkungan Berisik memiliki pusat massa frekuensi yang lebih tinggi (`centroid_mean` $1.786,13\text{ Hz}$ vs $1.646,51\text{ Hz}$) serta laju perpotongan nol yang lebih sering (`zcr_mean` $0,0961$ vs $0,0784$).
3. **Kinerja Klasifikasi Unggul:**
   - Model **Support Vector Machine (SVM RBF)** berhasil mencapai akurasi **$84,78\%$** pada data uji independen dengan F1-Score berimbang ($0,8372$ pada Berisik dan $0,8571$ pada Non-Berisik).
   - Pada pengujian **5-Fold Stratified Cross Validation**, model **Random Forest** membuktikan stabilitas generalisasi yang sangat tinggi dengan rata-rata akurasi **$87,78\% \pm 7,22\%$** (mencapai puncak $93,48\%$).
4. **Fitur Paling Diskriminatif:**
   Analisis *Feature Importance* membuktikan bahwa kombinasi fitur domain waktu (`rms_mean` $13,61\%$ dan `rms_std` $6,10\%$) serta fitur domain cepstral (`mfcc_1_mean` $7,93\%$, `mfcc_12_std` $5,64\%$, `mfcc_11_std` $5,13\%$) merupakan representasi terbaik untuk membedakan kebisingan lingkungan secara otomatis dan akurat.

---

## 14. Referensi

1. Müller, M. (2015). *Fundamentals of Music Processing: Audio, Analysis, Algorithms, Applications*. Springer International Publishing. ISBN: 978-3-319-21944-8.
2. McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and Music Signal Analysis in Python. *Proceedings of the 14th Python in Science Conference*, 18-25.
3. Davis, S. B., & Mermelstein, P. (1980). Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 28(4), 357-366.
4. Barchiesi, D., Giannoulis, D., Stowell, D., & Plumbley, M. D. (2015). Acoustic Scene Classification: Classifying environments from the sounds they produce. *IEEE Signal Processing Magazine*, 32(3), 16-34.
5. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
6. Oppenheim, A. V., & Schafer, R. W. (2009). *Discrete-Time Signal Processing* (3rd ed.). Prentice Hall.
7. Librosa Development Team. (2024). *Librosa Documentation: Audio Feature Extraction*. https://librosa.org/doc/latest/feature.html
8. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
