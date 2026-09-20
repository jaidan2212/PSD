# Bagian 2: Analisis dan Perhitungan Manual Fitur TSFEL (Tugas Individu)

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

### Kesimpulan

Berdasarkan analisis yang telah dilakukan, fitur **`median_diff`** digunakan untuk mengukur perubahan tengah dari nilai sinyal antar titik waktu yang berurutan. Pada data CO, fitur ini menghasilkan nilai **0.003**, yang menunjukkan bahwa perubahan tengah konsentrasi CO antar pengamatan adalah sebesar 0.003 satuan.

Sementara itu, fitur **`median_frequency`** bekerja pada domain frekuensi dengan melakukan transformasi FFT, mengambil magnitudo spektrum, menghitung akumulasi magnitudo, kemudian menentukan frekuensi ketika akumulasi tersebut melewati 50% dari total magnitudo. Pada data sampel CO dengan $fs=1$, diperoleh nilai **0.0 Hz**.

Hasil perhitungan manual untuk kedua fitur tersebut sesuai dengan hasil yang diberikan oleh fungsi TSFEL, sehingga proses perhitungan dapat digunakan sebagai pembuktian matematis terhadap implementasi fitur pada library TSFEL.
