# Bagian 1: Eksplorasi Data (EDA) Time Series pada Setiap Polutan

Bagian ini menguraikan proses pengumpulan, pembersihan, dan visualisasi data konsentrasi polutan udara di wilayah studi. Tujuan utamanya adalah memperoleh tiga deret waktu (*time series*) yang bersih, kontinu, dan siap dianalisis lebih lanjut, serta memahami karakteristik awal masing-masing polutan sebelum masuk ke tahap pemodelan.

---

## 1.1 Pengumpulan Data melalui Google Earth Engine (GEE)

### Sumber dan Cakupan Data

Data konsentrasi polutan diperoleh dari citra satelit **Sentinel-5P (TROPOMI)** yang diakses melalui platform **Google Earth Engine (GEE)**. Sentinel-5P dipilih karena satelit ini secara khusus dirancang untuk pemantauan komposisi atmosfer harian dengan resolusi spasial yang memadai untuk analisis tingkat kecamatan.

Terdapat tiga parameter polutan yang diekstraksi:

| Polutan | Nama Lengkap | Satuan Pengukuran | Karakteristik Utama |
|---|---|---|---|
| **NO₂** | Nitrogen Dioksida | mol/m² | Indikator emisi kendaraan bermotor dan pembakaran bahan bakar fosil |
| **CO** | Karbon Monoksida | mol/m² | Indikator pembakaran tidak sempurna (kendaraan, pembakaran biomassa) |
| **SO₂** | Sulfur Dioksida | mol/m² | Indikator emisi industri dan bahan bakar berkandungan sulfur tinggi |

**Wilayah studi (*Area of Interest* / AOI)** yang digunakan adalah **Kecamatan Kamal, Kabupaten Bangkalan, Madura**. Wilayah ini dipilih karena posisinya sebagai gerbang masuk Pulau Madura melalui Jembatan Suramadu dan Pelabuhan Kamal, sehingga memiliki karakteristik lalu lintas dan aktivitas transportasi yang relevan untuk kajian kualitas udara.

Proses ekstraksi dilakukan dengan mereduksi nilai piksel dalam batas AOI menjadi satu nilai representatif per hari (agregasi spasial), kemudian mengekspor hasilnya ke dalam format **CSV** untuk diolah lebih lanjut di Python.

### Tantangan: Piksel Kosong akibat Tutupan Awan

Kendala teknis utama muncul pada tahap ekspor data. Sensor TROPOMI bekerja pada spektrum optik, sehingga **tutupan awan** akan menghalangi pengamatan permukaan dan menghasilkan piksel tanpa nilai (*null*). Permasalahan ini paling sering terjadi pada parameter **CO** dan **SO₂**.

Konsekuensinya bersifat struktural, bukan sekadar kosongnya satu baris data:

- Ketika seluruh piksel dalam AOI bernilai *null* pada tanggal tertentu, fungsi reduksi di GEE mengembalikan objek kosong.
- GEE menyusun *header* kolom CSV berdasarkan properti yang tersedia pada *feature* pertama. Bila properti tersebut tidak ada, **kolom polutan tidak ikut terekspor sama sekali**.
- Akibatnya, file CSV yang dihasilkan menjadi tidak konsisten — jumlah dan nama kolomnya dapat berbeda antar periode ekspor, sehingga tidak dapat langsung digabungkan.

### Solusi: Teknik *Dictionary Combine* dengan Nilai Default

Untuk mengatasi hal tersebut, script JavaScript di GEE dimodifikasi menggunakan pendekatan **"Dictionary Combine"**. Prinsip kerjanya adalah menjamin bahwa setiap *feature* harian selalu memiliki properti polutan, terlepas dari ada tidaknya hasil pengukuran satelit:

1. Hasil reduksi spasial harian dikonversi menjadi sebuah *dictionary*.
2. Dibuat *dictionary* cadangan (*fallback*) yang berisi nama band polutan dengan nilai default **`-9999`**.
3. Kedua *dictionary* digabungkan dengan aturan: **nilai asli diprioritaskan**, dan nilai default hanya dipakai bila kunci tersebut tidak ditemukan.

Dengan cara ini, struktur kolom CSV menjadi **konsisten dan terjamin** di sepanjang periode pengamatan. Nilai `-9999` berfungsi sebagai *sentinel value* — penanda eksplisit bahwa "pengamatan pada tanggal ini tidak tersedia", bukan sebagai nilai konsentrasi yang sebenarnya. Penanda inilah yang kemudian ditangani pada tahap *preprocessing*.

---

## 1.2 Data Preprocessing (Pembersihan Data Mentah)

Tahap ini dikerjakan di Python menggunakan pustaka **Pandas**, dengan tiga langkah utama sebagai berikut.

### a. Penggabungan Dataset

Ketiga file CSV (NO₂, CO, dan SO₂) dimuat dan disatukan ke dalam satu alur pemrosesan. Kolom tanggal dikonversi ke tipe data `datetime` dan ditetapkan sebagai indeks *DataFrame*. Langkah ini penting karena indeks bertipe waktu merupakan prasyarat bagi operasi *time series* pada tahap berikutnya, khususnya interpolasi berbasis waktu.

### b. Konversi *Sentinel Value* menjadi *Missing Value*

Nilai `-9999` yang sebelumnya disisipkan di GEE diubah menjadi **`NaN`** (*Not a Number*). Langkah ini krusial karena:

- Secara statistik, `-9999` adalah *outlier* ekstrem yang akan merusak perhitungan rata-rata, simpangan baku, dan skala visualisasi.
- Dengan mengubahnya menjadi `NaN`, Pandas dapat mengenali data tersebut sebagai **data hilang yang sah** dan memperlakukannya secara tepat pada operasi agregasi maupun interpolasi.

Singkatnya, `-9999` adalah solusi untuk masalah *ekspor*, sedangkan `NaN` adalah representasi yang benar untuk masalah *analisis*.

### c. Penanganan Data Hilang: *Time Interpolation*

Karena data yang hilang muncul secara tersebar dan tidak beraturan (bergantung pada kondisi cuaca harian), teknik pengisian yang digunakan adalah **interpolasi berbasis waktu** (`method='time'`).

Metode ini dipilih dengan pertimbangan berikut:

- **Mempertimbangkan jarak temporal.** Berbeda dengan interpolasi linear biasa yang hanya melihat urutan baris, interpolasi berbasis waktu memperhitungkan selisih hari antar observasi. Celah kosong selama 1 hari dan 10 hari karenanya diperlakukan berbeda secara proporsional.
- **Mempertahankan kontinuitas deret.** Konsentrasi polutan udara merupakan besaran yang berubah secara gradual, sehingga asumsi transisi halus antar titik pengamatan dapat dipertanggungjawabkan secara fisis.

Interpolasi kemudian dilengkapi dengan **`ffill` (*forward fill*)** dan **`bfill` (*backward fill*)** untuk menangani nilai kosong yang berada di **ujung awal dan ujung akhir** deret. Hal ini diperlukan karena interpolasi hanya dapat bekerja di antara dua titik data yang valid, sehingga data hilang di batas periode tidak akan tertangani tanpa langkah tambahan ini.

Hasil akhir tahap *preprocessing* adalah tiga deret waktu harian yang **lengkap, kontinu, dan bebas dari nilai anomali** `-9999`.

---

## 1.3 Visualisasi Time Series

### Rancangan Visualisasi

Visualisasi dibangun menggunakan **Matplotlib** dengan konfigurasi **tiga subplot yang ditumpuk secara vertikal** dan **berbagi satu sumbu X** (`sharex=True`).

Pemilihan format ini didasarkan pada dua alasan:

- **Perbedaan skala antar polutan.** Konsentrasi CO berada pada orde 10⁻² mol/m², sementara NO₂ dan SO₂ berada pada orde 10⁻⁴ mol/m². Jika ketiganya diplot pada satu sumbu Y yang sama, fluktuasi NO₂ dan SO₂ akan tampak sebagai garis datar dan kehilangan informasi.
- **Kemudahan perbandingan temporal.** Dengan sumbu X yang seragam, posisi vertikal setiap titik waktu tetap sejajar antar panel, sehingga lonjakan yang terjadi bersamaan pada polutan berbeda dapat dikenali secara visual.

Setiap panel diberi warna berbeda (*teal* untuk NO₂, oranye untuk CO, ungu untuk SO₂), judul spesifik, serta *grid* bergaris putus-putus untuk memudahkan pembacaan nilai.

### Hasil Visualisasi

Grafik yang dihasilkan menunjukkan **deret waktu yang tersambung mulus tanpa garis terputus maupun penurunan ekstrem** ke nilai `-9999`. Hal ini mengonfirmasi bahwa strategi penanganan data hilang — mulai dari penyisipan *sentinel value* di GEE hingga interpolasi di Pandas — telah berjalan sebagaimana dirancang.

---

## 1.4 Temuan Awal dari Eksplorasi Visual

Beberapa pola dapat diamati dari hasil visualisasi:

- **NO₂** menunjukkan variabilitas harian yang paling tinggi di antara ketiga polutan, dengan nilai dasar berkisar pada 0,00005–0,00010 mol/m² dan lonjakan sesekali yang mencapai lebih dari tiga kali nilai dasar. Pola berduri (*spiky*) semacam ini konsisten dengan karakteristik NO₂ yang memiliki umur atmosferik pendek dan sangat responsif terhadap sumber emisi lokal seperti lalu lintas kendaraan.
- **CO** bergerak dalam rentang yang jauh lebih sempit secara relatif (sekitar 0,020–0,045 mol/m²) dan tampak lebih stabil. Hal ini sejalan dengan sifat CO yang memiliki umur atmosferik lebih panjang, sehingga konsentrasinya lebih mencerminkan kondisi latar regional dibandingkan emisi sesaat.
- **SO₂** memperlihatkan fluktuasi di sekitar nilai nol, termasuk sejumlah **nilai negatif**. Nilai negatif ini bukan kesalahan pengolahan data, melainkan karakteristik retrieval Sentinel-5P: ketika konsentrasi SO₂ sangat rendah dan mendekati batas deteksi sensor, derau pengukuran dapat menghasilkan estimasi kolom yang bernilai negatif. Temuan ini mengindikasikan bahwa aktivitas industri penghasil SO₂ di sekitar wilayah studi relatif rendah.

---

## 1.5 Catatan dan Keterbatasan

Beberapa hal perlu dicatat sebagai batasan yang memengaruhi interpretasi dan tahap analisis selanjutnya:

1. **Perbedaan cakupan periode antar polutan.** Berdasarkan grafik, deret NO₂ tersedia untuk periode awal rentang pengamatan, sementara CO dan SO₂ tersedia untuk periode setelahnya, dengan irisan waktu yang terbatas. Konsekuensinya, **analisis korelasi atau perbandingan langsung antar polutan hanya valid pada periode yang beririsan**. Penyelarasan rentang tanggal pada saat ekstraksi di GEE disarankan apabila analisis multivariat menjadi bagian dari tahap berikutnya.
2. **Ruas garis lurus sebagai jejak interpolasi.** Segmen grafik yang tampak sangat halus dan menyerupai garis lurus panjang menandakan periode dengan tutupan awan berkepanjangan, di mana nilai sepenuhnya merupakan hasil estimasi. Ruas semacam ini tidak boleh diperlakukan sebagai hasil observasi aktual dan sebaiknya tidak dijadikan dasar penarikan kesimpulan.
3. **Interpolasi meredam variabilitas.** Data hasil interpolasi secara inheren memiliki simpangan baku yang lebih rendah dibandingkan data observasi murni. Nilai statistik deskriptif yang dihitung dari deret ini karenanya cenderung *underestimate* terhadap variabilitas yang sebenarnya.
4. **Agregasi spasial.** Satu nilai harian merepresentasikan rata-rata seluruh wilayah kecamatan, sehingga variasi konsentrasi antar lokasi di dalam Kecamatan Kamal tidak tertangkap dalam analisis ini.

Dengan catatan tersebut, dataset hasil tahap ini dinilai telah memenuhi syarat untuk dilanjutkan ke tahap analisis berikutnya.

# Bagian 2: Konsep Dasar dan Pembuktian Ekstraksi Fitur TSFEL

Pada bagian sebelumnya, deret waktu harian ketiga polutan telah dibersihkan dan divisualisasikan. Langkah berikutnya adalah mengubah deret waktu tersebut menjadi representasi numerik yang dapat dipelajari oleh algoritma *machine learning*, melalui proses **ekstraksi fitur** menggunakan pustaka **TSFEL (Time Series Feature Extraction Library)**.

Sebelum menjalankan proses ekstraksi 68 fitur secara otomatis, bagian ini terlebih dahulu membedah **landasan matematis** di balik fitur-fitur tersebut. Tujuannya adalah memastikan bahwa pustaka yang digunakan tidak diperlakukan sebagai *black box*, melainkan sebagai alat yang perilakunya dipahami dan dapat diverifikasi secara mandiri.

---

## 2.1 Mengapa Ekstraksi Fitur Diperlukan?

Data deret waktu memiliki karakteristik yang berbeda dari data tabular konvensional. Pada data tabular, satu baris merepresentasikan satu entitas dengan sejumlah atribut tetap. Pada deret waktu, satu entitas justru direpresentasikan oleh **ratusan hingga ribuan titik observasi berurutan** yang saling bergantung secara temporal.

Hal ini menimbulkan beberapa persoalan mendasar:

- **Ketidaksesuaian format.** Sebagian besar algoritma *machine learning* klasik (SVM, Random Forest, KNN, Regresi Logistik) mensyaratkan input berupa vektor fitur berdimensi tetap. Deret waktu mentah dengan panjang yang bervariasi tidak dapat langsung dimasukkan ke dalam model tersebut.
- **Dimensionalitas tinggi dan redundansi.** Memperlakukan setiap hari sebagai satu kolom fitur akan menghasilkan ruang fitur berdimensi sangat tinggi, sementara nilai antar hari yang berdekatan cenderung sangat berkorelasi sehingga informasinya tumpang tindih.
- **Informasi pola yang tersembunyi.** Nilai konsentrasi pada satu hari tertentu kurang bermakna bila dilihat sendirian. Yang lebih informatif justru **perilaku agregat** dari sekelompok hari: seberapa tinggi rata-ratanya, seberapa besar gejolaknya, dan berapa puncak tertingginya.

Ekstraksi fitur menjawab persoalan tersebut dengan cara **meringkas sebuah segmen deret waktu (*window*) menjadi sekumpulan nilai skalar** yang merangkum karakteristiknya. Secara formal, proses ini adalah pemetaan:

$$
f : \mathbb{R}^{N} \longrightarrow \mathbb{R}
$$

yaitu sebuah fungsi yang menerima segmen deret waktu sepanjang $N$ titik observasi dan mengembalikan satu nilai skalar tunggal. Dengan menerapkan sejumlah fungsi $f_1, f_2, \dots, f_k$ yang berbeda pada segmen yang sama, diperoleh vektor fitur:

$$
\mathbf{v} = \left[ f_1(X),\; f_2(X),\; \dots,\; f_k(X) \right]
$$

Vektor inilah yang menjadi satu baris pada dataset tabular final, siap digunakan sebagai input model.

TSFEL mengelompokkan fitur-fiturnya ke dalam tiga domain utama: **statistik** (distribusi nilai), **temporal** (perilaku terhadap waktu), dan **spektral** (kandungan frekuensi). Pembahasan pada bagian ini difokuskan pada **empat fitur dari domain statistik** yang paling fundamental dan paling mudah diverifikasi secara manual.

---

## 2.2 Notasi yang Digunakan

Agar pembahasan konsisten, berikut notasi yang dipakai di seluruh bagian ini:

- $X = \{x_1, x_2, x_3, \dots, x_N\}$ — segmen deret waktu konsentrasi polutan
- $x_i$ — nilai konsentrasi pada observasi ke-$i$
- $N$ — jumlah observasi dalam segmen
- $x_{(i)}$ — nilai ke-$i$ setelah data **diurutkan** dari terkecil ke terbesar (*order statistic*)

---

## 2.3 Empat Fitur Domain Statistik

### a. Mean (Rata-rata)

**Konsep dasar.** Mean adalah pusat massa dari sebaran data, diperoleh dengan menjumlahkan seluruh nilai lalu membaginya dengan banyaknya observasi. Mean merupakan ukuran pemusatan (*measure of central tendency*) yang paling umum digunakan.

**Rumus matematis:**

$$
\bar{x} = \frac{1}{N} \sum_{i=1}^{N} x_i
$$

**Kegunaan dalam konteks polusi udara.** Mean merepresentasikan **tingkat paparan rata-rata** (*baseline exposure*) selama periode pengamatan. Dalam kajian kualitas udara, nilai ini penting karena sebagian besar baku mutu lingkungan justru dinyatakan dalam bentuk rata-rata pada rentang waktu tertentu, misalnya rata-rata 24 jam atau rata-rata tahunan. Mean NO₂ yang tinggi pada suatu periode mengindikasikan beban emisi yang persisten, bukan sekadar kejadian sesaat.

**Keterbatasan.** Mean sangat sensitif terhadap nilai ekstrem. Satu hari dengan lonjakan konsentrasi yang sangat tinggi dapat menarik nilai mean ke atas, sehingga memberi kesan paparan yang lebih merata daripada kenyataannya.

---

### b. Median (Nilai Tengah)

**Konsep dasar.** Median adalah nilai yang berada tepat di tengah data setelah seluruh observasi diurutkan. Median membagi sebaran menjadi dua bagian sama besar: 50% data berada di bawahnya dan 50% di atasnya.

**Rumus matematis:**

$$
\tilde{x} =
\begin{cases}
x_{\left(\frac{N+1}{2}\right)}, & \text{jika } N \text{ ganjil} \\[8pt]
\dfrac{1}{2}\left( x_{\left(\frac{N}{2}\right)} + x_{\left(\frac{N}{2}+1\right)} \right), & \text{jika } N \text{ genap}
\end{cases}
$$

**Kegunaan dalam konteks polusi udara.** Median menggambarkan **kondisi udara pada hari yang tipikal**, tanpa terdistorsi oleh kejadian luar biasa. Karena bersifat *robust* terhadap *outlier*, median menjadi pelengkap penting bagi mean.

Perbandingan antara keduanya bahkan bersifat diagnostik terhadap bentuk distribusi:

- $\bar{x} \approx \tilde{x}$ → sebaran relatif simetris; fluktuasi harian berlangsung wajar.
- $\bar{x} > \tilde{x}$ → sebaran **menceng ke kanan** (*right-skewed*); mayoritas hari berkondisi baik, namun terdapat beberapa hari dengan lonjakan tajam. Pola ini justru khas pada data NO₂ yang dipengaruhi emisi sesaat dari lalu lintas.

---

### c. Maximum (Nilai Tertinggi)

**Konsep dasar.** Maximum adalah nilai terbesar yang tercatat dalam segmen. Berbeda dengan mean dan median yang bersifat agregat, fitur ini merekam satu peristiwa tunggal, yaitu titik puncak dari deret.

**Rumus matematis:**

$$
x_{\max} = \max_{1 \le i \le N} \; x_i
$$

**Kegunaan dalam konteks polusi udara.** Maximum merepresentasikan **skenario paparan terburuk** (*worst-case exposure*) pada periode tersebut. Fitur ini memiliki relevansi kesehatan yang berbeda dari mean: paparan akut jangka pendek pada konsentrasi tinggi dapat memicu gangguan pernapasan meskipun rata-rata periodenya masih tergolong aman. Dalam analisis, nilai maximum berguna untuk mendeteksi **episode pencemaran** (*pollution episode*) yang berpotensi terkait dengan kejadian spesifik seperti kemacetan luar biasa, pembakaran terbuka, atau kondisi meteorologi yang menghambat dispersi polutan.

---

### d. Standard Deviation (Standar Deviasi)

**Konsep dasar.** Standar deviasi mengukur seberapa jauh nilai-nilai dalam data tersebar di sekitar mean-nya. Nilainya diperoleh dengan menghitung rata-rata kuadrat simpangan terhadap mean (varians), kemudian menarik akarnya agar satuannya kembali sama dengan satuan data asli.

**Rumus matematis (populasi):**

$$
\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left( x_i - \bar{x} \right)^{2}}
$$

**Kegunaan dalam konteks polusi udara.** Standar deviasi mengukur **stabilitas atau volatilitas kualitas udara**. Interpretasinya:

- **Nilai kecil** → konsentrasi relatif konstan dari hari ke hari, mengindikasikan sumber emisi yang stabil dan kondisi dispersi atmosfer yang seragam.
- **Nilai besar** → konsentrasi berfluktuasi tajam, mengindikasikan adanya sumber emisi intermiten atau pengaruh faktor meteorologi yang berubah-ubah seperti arah angin dan curah hujan.

Fitur ini sangat diskriminatif karena dua periode dapat memiliki mean yang nyaris identik namun karakter yang sama sekali berbeda: satu periode tenang dan stabil, satu lagi penuh lonjakan yang saling meniadakan ketika dirata-ratakan. Perbedaan tersebut hanya tertangkap oleh ukuran penyebaran.

---

## 2.4 Simulasi Perhitungan Manual

Untuk membuktikan pemahaman atas rumus di atas, berikut simulasi perhitungan menggunakan data sampel fiktif berupa konsentrasi polutan selama 5 hari:

$$
X = [\,1,\; 3,\; 5,\; 7,\; 9\,], \qquad N = 5
$$

### Langkah 1 — Perhitungan Mean

Jumlahkan seluruh nilai, lalu bagi dengan jumlah observasi:

$$
\sum_{i=1}^{5} x_i = 1 + 3 + 5 + 7 + 9 = 25
$$

$$
\bar{x} = \frac{25}{5} = \mathbf{5{,}0}
$$

### Langkah 2 — Perhitungan Median

Data telah terurut menaik: $[1, 3, 5, 7, 9]$. Karena $N = 5$ bernilai **ganjil**, median adalah nilai pada posisi:

$$
\frac{N+1}{2} = \frac{5+1}{2} = 3
$$

Nilai pada posisi ke-3 adalah:

$$
\tilde{x} = x_{(3)} = \mathbf{5{,}0}
$$

> **Catatan interpretatif:** pada sampel ini $\bar{x} = \tilde{x} = 5{,}0$, yang menandakan distribusi data sempurna simetris. Kondisi ideal semacam ini jarang ditemui pada data polutan riil.

### Langkah 3 — Perhitungan Maximum

Telusuri seluruh nilai dan pilih yang terbesar:

$$
x_{\max} = \max(1, 3, 5, 7, 9) = \mathbf{9{,}0}
$$

### Langkah 4 — Perhitungan Standar Deviasi

**(a) Hitung simpangan setiap nilai terhadap mean $(x_i - \bar{x})$ dan kuadratkan:**

| $i$ | $x_i$ | $x_i - \bar{x}$ | $(x_i - \bar{x})^2$ |
|:---:|:-----:|:---------------:|:-------------------:|
| 1 | 1 | $1 - 5 = -4$ | 16 |
| 2 | 3 | $3 - 5 = -2$ | 4 |
| 3 | 5 | $5 - 5 = 0$ | 0 |
| 4 | 7 | $7 - 5 = 2$ | 4 |
| 5 | 9 | $9 - 5 = 4$ | 16 |
| | | **Total** | **40** |

**(b) Hitung varians** dengan membagi total kuadrat simpangan dengan $N$:

$$
\sigma^{2} = \frac{40}{5} = 8{,}0
$$

**(c) Tarik akar kuadrat** untuk memperoleh standar deviasi:

$$
\sigma = \sqrt{8} \approx \mathbf{2{,}8284271247}
$$

### Ringkasan Hasil Perhitungan Manual

| Fitur | Notasi | Hasil Manual |
|---|:---:|---:|
| Mean | $\bar{x}$ | 5,0000000000 |
| Median | $\tilde{x}$ | 5,0000000000 |
| Maximum | $x_{\max}$ | 9,0000000000 |
| Standard Deviation | $\sigma$ | 2,8284271247 |

---

## 2.5 Catatan Penting: Pembagi $N$ atau $N-1$?

Terdapat satu detail teknis yang kerap menjadi sumber perbedaan hasil dan perlu ditegaskan sebelum pembuktian dilakukan. Standar deviasi memiliki dua varian rumus yang berbeda pada penyebutnya:

| Varian | Rumus | Pembagi | Hasil pada sampel $X$ |
|---|---|:---:|---:|
| **Populasi** ($ddof = 0$) | $\sigma = \sqrt{\dfrac{1}{N}\sum (x_i - \bar{x})^2}$ | $N = 5$ | $\sqrt{8} \approx 2{,}8284$ |
| **Sampel** ($ddof = 1$) | $s = \sqrt{\dfrac{1}{N-1}\sum (x_i - \bar{x})^2}$ | $N - 1 = 4$ | $\sqrt{10} \approx 3{,}1623$ |

Varian sampel menggunakan pembagi $N-1$ (**koreksi Bessel**) untuk menghasilkan estimasi varians populasi yang tidak bias ketika data yang dimiliki hanya berupa sampel dari populasi yang lebih besar.

Hal ini relevan karena **TSFEL mengimplementasikan fitur standar deviasi dengan memanggil fungsi `numpy.std()` tanpa mengubah parameter bawaannya**, sedangkan nilai default `ddof` pada NumPy adalah $0$. Dengan demikian, TSFEL menghitung **standar deviasi populasi**. Sebagai perbandingan, metode `.std()` pada Pandas justru menggunakan default $ddof = 1$ sehingga akan menghasilkan angka yang berbeda untuk data yang sama persis.

Oleh karena itu, perhitungan manual pada Langkah 4 sengaja menggunakan pembagi $N = 5$ agar setara dengan konvensi yang dipakai TSFEL. Kesadaran atas detail semacam inilah yang membedakan penggunaan pustaka secara sadar dari penggunaan secara *black box*.

---

## 2.6 Menuju Pembuktian Komputasional

Keempat nilai pada tabel ringkasan di atas diperoleh sepenuhnya melalui penurunan rumus secara manual, tanpa bantuan pustaka apa pun. Nilai-nilai tersebut kini berfungsi sebagai **kebenaran acuan** (*ground truth*) untuk tahap verifikasi.

Pada blok kode berikut, array sampel yang sama, $X = [1, 3, 5, 7, 9]$, akan diproses menggunakan algoritma ekstraksi fitur TSFEL. Hasil keluaran pustaka kemudian dibandingkan secara langsung dengan hasil perhitungan manual. Apabila keduanya menghasilkan **nilai yang identik hingga tingkat presisi desimal**, maka terbukti bahwa:

1. Pemahaman terhadap formulasi matematis setiap fitur sudah tepat.
2. TSFEL mengimplementasikan definisi statistik baku, bukan varian rumus yang tersembunyi.
3. Ekstraksi 68 fitur pada data polutan riil di bagian selanjutnya dapat dijalankan dengan keyakinan penuh terhadap makna setiap kolom yang dihasilkan.

```python
import numpy as np
import tsfel

# 1. Menyiapkan data sampel yang sama persis dengan perhitungan manual
# X = [1, 3, 5, 7, 9]
data_contoh = np.array([1, 3, 5, 7, 9])

# 2. Mendefinisikan konfigurasi TSFEL (hanya 4 fitur statistik ini yang diekstrak)
cfg = {
    'statistical': {
        'Mean': {'use': 'yes'},
        'Median': {'use': 'yes'},
        'Max': {'use': 'yes'},
        'Standard deviation': {'use': 'yes'}
    }
}

# 3. Melakukan Ekstraksi menggunakan TSFEL
print("Mengekstrak fitur menggunakan TSFEL...")
hasil_tsfel = tsfel.time_series_features_extractor(cfg, data_contoh, verbose=0)

# 4. Menampilkan Perbandingan Output Manual vs Algoritma
print("\n=== PEMBUKTIAN MANUAL VS TSFEL ===")
print(f"Rata-rata (Mean)    | Manual: 5.0    | TSFEL: {hasil_tsfel['0_Mean'].values[0]}")
print(f"Median              | Manual: 5.0    | TSFEL: {hasil_tsfel['0_Median'].values[0]}")
print(f"Nilai Max           | Manual: 9.0    | TSFEL: {hasil_tsfel['0_Max'].values[0]}")
print(f"Standar Deviasi     | Manual: 2.828  | TSFEL: {hasil_tsfel['0_Standard deviation'].values[0]:.3f}")
```

**Output:**
```text
Mengekstrak fitur menggunakan TSFEL...

=== PEMBUKTIAN MANUAL VS TSFEL ===
Rata-rata (Mean)    | Manual: 5.0    | TSFEL: 5.0
Median              | Manual: 5.0    | TSFEL: 5.0
Nilai Max           | Manual: 9.0    | TSFEL: 9.0
Standar Deviasi     | Manual: 2.828  | TSFEL: 2.828
```

# Bagian 3: Machine Learning — Reduksi Dimensi dan K-Means Clustering

Setelah landasan matematis ekstraksi fitur diverifikasi pada bagian sebelumnya, tahap ini menerapkan TSFEL pada data polutan riil dan melanjutkannya ke analisis *unsupervised learning*. Tujuan akhirnya adalah mengidentifikasi **pola-pola rezim emisi** yang secara alami terbentuk dalam data, tanpa menggunakan label yang ditentukan sebelumnya.

Alur pemrosesan pada bagian ini terdiri atas empat tahap berurutan: *windowing* → standardisasi → reduksi dimensi → pengelompokan.

---

## 3.1 Dari Deret Waktu ke Matriks Fitur: Strategi *Windowing*

Ekstraksi fitur tidak diterapkan pada keseluruhan deret waktu sekaligus, karena hal itu hanya akan menghasilkan **satu baris sampel** dan tidak memungkinkan analisis pengelompokan. Sebagai gantinya, deret waktu dipotong menjadi segmen-segmen berurutan menggunakan teknik ***windowing*** dengan `window_size = 14`.

**Rasionalisasi pemilihan jendela 14 hari:**

- **Kecukupan statistik.** Setiap jendela harus memuat titik observasi yang memadai agar fitur statistik seperti standar deviasi, kurtosis, dan *skewness* dapat dihitung secara stabil. Jendela yang terlalu pendek menghasilkan estimasi yang bergejolak dan tidak dapat diandalkan.
- **Relevansi periodik.** Rentang 14 hari setara dengan dua siklus mingguan penuh, sehingga mampu menangkap kontras antara hari kerja dan akhir pekan — pola yang sangat relevan pada polutan berbasis transportasi seperti NO₂.
- **Resolusi temporal yang memadai.** Jendela yang terlalu panjang akan meratakan (*smoothing out*) episode pencemaran jangka pendek dan mengurangi jumlah sampel secara drastis.

Karena parameter `overlap` dibiarkan pada nilai bawaan (nol), setiap jendela bersifat **saling lepas** (*non-overlapping*). Pilihan ini penting secara metodologis: jendela yang bertumpang tindih akan berbagi observasi yang sama sehingga menciptakan ketergantungan artifisial antar sampel, yang pada gilirannya membuat klaster tampak lebih kohesif daripada kenyataannya.

Hasil transformasi ini adalah **matriks fitur** berbentuk:

$$
\mathbf{X} \in \mathbb{R}^{n \times p}, \qquad p = 68
$$

dengan $n$ adalah jumlah jendela yang terbentuk dan $p = 68$ adalah jumlah fitur per jendela. Deret waktu yang semula bersifat sekuensial kini telah berubah menjadi **data tabular konvensional**, di mana setiap baris merepresentasikan "karakter dua mingguan" dari kualitas udara.

---

## 3.2 Mengapa Standardisasi Bersifat Wajib?

### Permasalahan: Heterogenitas Skala Antar Fitur

Ke-68 fitur yang dihasilkan TSFEL berasal dari tiga domain berbeda dan memiliki **satuan serta rentang nilai yang sangat tidak seragam**. Sebagai ilustrasi pada data NO₂:

| Jenis Fitur | Orde Nilai Tipikal |
|---|---|
| Mean, Median, Max (mol/m²) | $\sim 10^{-4}$ |
| Variance (mol²/m⁴) | $\sim 10^{-9}$ |
| Entropy, Autocorrelation | $\sim 10^{0}$ (rentang 0–1) |
| Zero Crossing Rate, jumlah puncak | $\sim 10^{0}$ hingga $10^{1}$ |
| Spectral energy / FFT magnitude | dapat mencapai $\sim 10^{2}$ ke atas |

Rentang perbedaan yang mencapai **lebih dari sepuluh orde magnitudo** ini menimbulkan konsekuensi serius bagi kedua algoritma yang digunakan.

### Dampak pada K-Means

K-Means bekerja dengan meminimalkan jumlah kuadrat jarak Euklides antara setiap titik dan pusat klasternya:

$$
J = \sum_{k=1}^{K} \sum_{\mathbf{x}_i \in C_k} \left\lVert \mathbf{x}_i - \boldsymbol{\mu}_k \right\rVert^{2}
$$

Dalam perhitungan jarak Euklides, kontribusi setiap fitur bersifat **aditif dan proporsional terhadap besaran nilainya**:

$$
d(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{j=1}^{p} (a_j - b_j)^2}
$$

Tanpa standardisasi, selisih pada fitur berorde $10^{2}$ akan menghasilkan kontribusi kuadrat yang jauh lebih besar dibandingkan selisih pada fitur berorde $10^{-4}$. Akibatnya, **struktur klaster praktis ditentukan oleh satu atau dua fitur berskala besar saja**, sementara informasi dari puluhan fitur lain menjadi tidak berpengaruh sama sekali. Ini bukan sekadar bias, melainkan kegagalan metodologis: algoritma seolah-olah memproses 68 fitur, padahal secara efektif hanya membaca segelintir di antaranya.

### Dampak pada PCA

Persoalan pada PCA bersifat lebih fundamental. PCA mencari arah dalam ruang fitur yang **memaksimalkan varians**. Namun varians adalah besaran yang **bergantung pada satuan pengukuran** — mengubah satuan dari mol/m² menjadi µmol/m² akan melipatgandakan variansnya secara kuadratik tanpa mengubah informasi apa pun yang terkandung di dalamnya.

Konsekuensinya, PCA pada data tanpa standardisasi akan menempatkan komponen utama pertamanya searah dengan fitur yang **kebetulan** memiliki satuan terbesar, bukan fitur yang paling informatif. Komponen yang dihasilkan menjadi artefak dari pilihan satuan, bukan cerminan struktur data.

### Solusi: Transformasi *Z-score*

`StandardScaler` menerapkan transformasi berikut pada setiap kolom fitur $j$ secara independen:

$$
z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
$$

dengan $\mu_j$ dan $\sigma_j$ masing-masing adalah rata-rata dan standar deviasi kolom tersebut. Setelah transformasi, **seluruh fitur memiliki rata-rata nol dan standar deviasi satu**, sehingga menjadi tidak berdimensi (*dimensionless*) dan berkontribusi secara setara terhadap perhitungan jarak maupun varians. Dengan demikian, yang dibandingkan antar jendela bukan lagi besaran absolut, melainkan **posisi relatif setiap jendela terhadap keseluruhan dataset**.

---

## 3.3 Mengapa Reduksi Dimensi Diperlukan?

Standardisasi menyelesaikan masalah skala, tetapi tidak menyelesaikan masalah **dimensionalitas**. Terdapat tiga alasan teknis mengapa PCA diterapkan sebelum pengelompokan.

### a. Kutukan Dimensionalitas (*Curse of Dimensionality*)

Fenomena yang paling merugikan K-Means pada ruang berdimensi tinggi adalah **konsentrasi jarak** (*distance concentration*). Seiring bertambahnya dimensi $p$, jarak Euklides antara titik terdekat dan titik terjauh dari suatu titik acuan cenderung menjadi semakin seragam:

$$
\lim_{p \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} \longrightarrow 0
$$

Ketika semua titik berjarak "hampir sama" satu sama lain, konsep kemiripan kehilangan daya bedanya. K-Means yang sepenuhnya bergantung pada jarak Euklides akan menghasilkan partisi yang nyaris arbitrer. Dengan $p = 68$, risiko ini sudah sangat nyata.

### b. Multikolinearitas Antar Fitur TSFEL

Ke-68 fitur TSFEL **tidak saling bebas**. Banyak di antaranya mengukur aspek yang secara matematis berkaitan erat, misalnya:

- *Mean* dan *Median* (keduanya ukuran pemusatan)
- *Standard Deviation*, *Variance*, dan *Root Mean Square* (secara aljabar saling terkait langsung)
- *Max*, *Min*, dan *Peak-to-peak distance*
- Berbagai statistik spektral yang diturunkan dari transformasi Fourier yang sama

Pada ruang fitur asli, redundansi ini menyebabkan **satu konsep tunggal dihitung berkali-kali**, sehingga bobotnya menjadi berlebihan dalam perhitungan jarak. PCA mengatasi hal ini secara elegan: karena komponen utama disusun saling **ortogonal**, fitur-fitur yang berkorelasi tinggi otomatis terlipat menjadi satu arah komponen yang sama.

### c. Pemisahan Sinyal dari Derau serta Kebutuhan Visualisasi

Komponen-komponen awal menangkap variasi struktural yang dominan, sementara komponen-komponen akhir umumnya memuat derau pengukuran. Membuang komponen minor karenanya berfungsi sebagai **penapis derau**. Selain itu, reduksi menjadi dua dimensi adalah satu-satunya cara agar struktur data dapat diperiksa secara visual pada bidang datar — sesuatu yang mustahil dilakukan pada ruang berdimensi 68.

---

## 3.4 Memahami Makna *Principal Component*

### PC Bukan Fitur Tunggal, Melainkan Kombinasi Linear

Kesalahpahaman yang paling umum adalah menganggap PC1 sebagai "fitur terpenting yang terpilih". Anggapan tersebut keliru. **PCA tidak melakukan seleksi fitur, melainkan konstruksi sumbu baru.**

Setiap komponen utama merupakan **kombinasi linear berbobot dari seluruh 68 fitur yang telah distandardisasi**:

$$
\text{PC}_k = w_{k1} z_1 + w_{k2} z_2 + w_{k3} z_3 + \dots + w_{k,68} z_{68} = \sum_{j=1}^{68} w_{kj} \, z_j
$$

Koefisien $w_{kj}$ disebut ***loading***, yaitu besar kontribusi fitur ke-$j$ terhadap komponen ke-$k$. Dengan demikian, setiap titik pada scatter plot tetap membawa jejak dari seluruh 68 fitur, hanya saja informasinya telah dipadatkan ke dalam dua koordinat.

### Bagaimana Sumbu Baru Ditentukan

Secara formal, PCA melakukan **dekomposisi eigen** terhadap matriks kovarians dari data terstandardisasi:

$$
\mathbf{C} = \frac{1}{n-1} \mathbf{Z}^{\top} \mathbf{Z}, \qquad \mathbf{C}\mathbf{v}_k = \lambda_k \mathbf{v}_k
$$

- **Vektor eigen** $\mathbf{v}_k$ menentukan **arah** komponen utama ke-$k$ (yaitu himpunan *loading*-nya).
- **Nilai eigen** $\lambda_k$ menyatakan **besar varians** yang dijelaskan oleh arah tersebut.

Komponen diurutkan secara menurun berdasarkan nilai eigennya, sehingga berlaku dua sifat penting:

1. **PC1 adalah arah dengan varians terbesar** dalam data, yakni sumbu di mana sampel-sampel paling terbentang dan paling mudah dibedakan satu sama lain.
2. **PC2 ortogonal terhadap PC1** ($\mathbf{v}_1 \perp \mathbf{v}_2$) dan menangkap varians terbesar berikutnya dari sisa informasi yang belum terjelaskan. Karena ortogonal, PC2 memuat informasi yang **sepenuhnya tidak tumpang tindih** dengan PC1.

### Proporsi Varians yang Dipertahankan

Ukuran seberapa banyak informasi yang berhasil dipertahankan dinyatakan oleh:

$$
\text{Explained Variance Ratio} = \frac{\lambda_1 + \lambda_2}{\sum_{k=1}^{p} \lambda_k}
$$

> **Nilai pada analisis ini:** _[isi dengan keluaran variabel `varians` dari sel kode]_ %

Angka ini merupakan **syarat validitas utama** bagi seluruh interpretasi visual yang dilakukan setelahnya, dengan pedoman umum sebagai berikut:

- **> 70%** — proyeksi dua dimensi merepresentasikan struktur data dengan baik; interpretasi visual dapat dipercaya.
- **50–70%** — representasi cukup memadai, namun sebagian struktur tidak tergambarkan pada bidang ini.
- **< 50%** — lebih dari separuh informasi hilang. Jarak visual antar titik berpotensi menyesatkan, dan kesimpulan harus disampaikan dengan kehati-hatian tinggi.

### Catatan: Arah Tanda Komponen Bersifat Arbitrer

Satu sifat PCA yang wajib diperhatikan saat menafsirkan grafik: **tanda dari vektor eigen tidak unik**. Jika $\mathbf{v}_k$ adalah solusi valid, maka $-\mathbf{v}_k$ juga solusi valid dengan nilai eigen yang identik. Implikasinya, **PC1 bernilai positif tidak secara otomatis berarti "polusi tinggi"**. Arah interpretasi hanya dapat ditetapkan dengan memeriksa *loading* (`pca.components_`) atau dengan menelusuri kembali nilai asli dari jendela-jendela yang berada di ujung sumbu.

---

## 3.5 Interpretasi Struktur Klaster

K-Means dijalankan dengan $K = 3$ pada koordinat hasil PCA. Berikut karakterisasi ketiga kelompok berdasarkan posisinya pada grafik.

### Klaster 0 (Biru) — Rezim Emisi Dasar

| Aspek | Keterangan |
|---|---|
| Posisi centroid | ≈ (−3,5 ; −1,0) |
| Sebaran PC1 | −9,6 hingga +2,1 |
| Proporsi sampel | Mayoritas dataset |

Klaster ini menempati seluruh wilayah PC1 negatif dan merupakan kelompok dengan anggota terbanyak. Dominasi jumlah anggotanya menunjukkan bahwa kelompok ini merepresentasikan **kondisi kualitas udara yang paling sering terjadi** — rezim latar (*baseline regime*) di Kecamatan Kamal.

Yang perlu dicatat, klaster ini **tidak padat**: anggotanya tersebar cukup luas pada kedua sumbu. Hal ini mengindikasikan bahwa kondisi "normal" bukanlah satu keadaan tunggal yang seragam, melainkan spektrum variasi rutin yang masih berada dalam rentang wajar. Secara lingkungan, jendela-jendela dalam kelompok ini kemungkinan besar mencerminkan periode dua mingguan dengan aktivitas transportasi reguler dan kondisi dispersi atmosfer yang normal.

### Klaster 1 (Merah) — Rezim Transisi yang Dibedakan oleh PC2

| Aspek | Keterangan |
|---|---|
| Posisi centroid | ≈ (2,6 ; 4,3) |
| Sebaran PC1 | −0,8 hingga +6,7 (lebar) |
| Sebaran PC2 | 4,0 hingga 4,6 (**sangat sempit**) |

Klaster ini memiliki karakteristik geometris yang paling menarik untuk dianalisis. Anggotanya **terbentang luas pada sumbu PC1 namun terkonsentrasi sangat rapat pada sumbu PC2** dengan rentang kurang dari satu satuan.

Pola horizontal semacam ini memiliki makna analitis yang spesifik: **faktor pembeda utama kelompok ini bukanlah PC1, melainkan PC2**. Kelima jendela ini boleh jadi berbeda satu sama lain dalam hal intensitas emisi (yang diwakili PC1), tetapi mereka berbagi satu karakteristik bersama yang kuat pada dimensi kedua.

Mengingat PC2 ortogonal terhadap PC1, karakteristik bersama tersebut merupakan dimensi yang **independen dari intensitas**. Dalam konteks polusi udara, dimensi semacam ini biasanya berkaitan dengan **bentuk dan dinamika deret**, bukan besarannya — misalnya tingkat volatilitas, keberadaan lonjakan tajam yang berulang, atau perubahan pola periodisitas. Kelompok ini karenanya lebih tepat dibaca sebagai **rezim dengan pola temporal yang khas** daripada sekadar "tingkat polusi menengah".

> **Langkah verifikasi yang disarankan:** periksa fitur dengan *loading* absolut terbesar pada PC2. Bila didominasi fitur penyebaran (standar deviasi, varians) atau fitur spektral, maka klaster ini dapat ditafsirkan sebagai periode dengan **fluktuasi emisi yang tidak stabil** — misalnya periode dengan perubahan cuaca yang sering atau aktivitas transportasi yang tidak merata.

### Klaster 2 (Hijau) — Kelompok Anomali

| Aspek | Keterangan |
|---|---|
| Posisi centroid | ≈ (14,8 ; −1,85) |
| Anggota | 3 titik: ≈ (8,8 ; −6,3), (17,9 ; +7,4), (17,9 ; −6,5) |
| Proporsi sampel | Terkecil |

Ketiga anggota kelompok ini terletak sangat jauh dari pusat sebaran utama, dengan nilai PC1 yang mencapai lebih dari tiga kali lipat titik terjauh pada Klaster 0. Jarak sebesar ini pada sumbu varians terbesar merupakan indikator kuat adanya **periode dengan karakteristik yang menyimpang secara ekstrem** dari kondisi rutin.

Namun demikian, terdapat satu aspek penting yang harus dinyatakan secara jujur dalam analisis ini: **ketiga titik tersebut tidak membentuk kelompok yang kohesif**. Rentang PC2 mereka membentang dari −6,5 hingga +7,4, dan posisi centroid di (14,8 ; −1,85) sebenarnya **tidak berdekatan dengan satu pun anggotanya**. Salah satu anggota bahkan terpisah hampir 14 satuan dari anggota lainnya pada sumbu vertikal — jarak yang lebih besar daripada keseluruhan rentang Klaster 0.

Temuan ini mengarah pada kesimpulan yang lebih berhati-hati: Klaster 2 **bukanlah satu jenis anomali yang berulang tiga kali, melainkan kumpulan tiga anomali yang masing-masing berbeda karakternya**. Ketiganya dikelompokkan bersama bukan karena saling menyerupai, tetapi karena sama-sama berada jauh dari pusat sebaran. Perilaku semacam ini merupakan konsekuensi wajar dari K-Means, yang karena mewajibkan setiap titik memperoleh keanggotaan (*hard assignment*), cenderung memperlakukan klaster terkecil sebagai **wadah penampung bagi seluruh pencilan**.

Secara lingkungan, ketiga jendela ini menandai periode dua mingguan yang layak diselidiki secara individual — bukan sebagai satu fenomena kolektif. Kemungkinan penyebabnya mencakup episode pencemaran akut, kondisi meteorologi ekstrem yang menghambat dispersi, aktivitas pembakaran terbuka musiman, atau — yang juga harus dipertimbangkan — **artefak pengolahan data** berupa jendela yang sebagian besar isinya merupakan hasil interpolasi dari Bagian 1.

> **Langkah verifikasi yang disarankan:** identifikasi indeks ketiga jendela ini, telusuri kembali rentang tanggal aktualnya, lalu periksa (a) nilai konsentrasi mentahnya pada grafik Bagian 1 dan (b) berapa banyak observasi dalam jendela tersebut yang berasal dari interpolasi. Langkah ini akan memastikan apakah anomali bersifat fenomenologis atau metodologis.

---

## 3.6 Catatan Metodologis dan Keterbatasan

Sebagai bagian dari pelaporan yang bertanggung jawab, berikut beberapa batasan yang memengaruhi kekuatan kesimpulan di atas.

1. **Pengelompokan dilakukan pada ruang tereduksi, bukan ruang fitur penuh.** K-Means dijalankan terhadap koordinat hasil PCA, bukan terhadap 68 fitur terstandardisasi. Pendekatan ini sah dan lazim digunakan karena mengurangi derau serta menghindari konsentrasi jarak, namun memiliki dua implikasi: (a) informasi pembeda yang mungkin tersimpan pada komponen ke-3 dan seterusnya tidak ikut memengaruhi penentuan klaster; dan (b) pemisahan visual yang tampak rapi pada grafik sebagian bersifat **konsekuensi logis dari desain**, karena pengelompokan dan visualisasi terjadi pada ruang yang sama persis. Grafik ini karenanya tidak dapat diperlakukan sebagai validasi independen atas hasil pengelompokan.

2. **Rasio jumlah sampel terhadap jumlah fitur sangat rendah.** Dengan sekitar dua puluhan jendela berbanding 68 fitur, dataset ini berada pada kondisi $n < p$. Pada kondisi demikian, matriks kovarians bersifat singular dan jumlah komponen utama yang bermakna terbatas maksimal pada $n - 1$. Estimasi struktur klaster pada rasio serendah ini rentan terhadap ketidakstabilan — penambahan atau penghapusan beberapa sampel saja berpotensi mengubah hasil.

3. **Nilai $K = 3$ belum divalidasi secara kuantitatif.** Jumlah klaster ditetapkan di awal tanpa pengujian formal. Untuk memperkuat justifikasi, disarankan melengkapi analisis dengan **metode Elbow** (kurva inersia) dan **Silhouette Score**. Nilai *silhouette* juga akan memberikan bukti kuantitatif atas pengamatan pada Klaster 2 — apabila skornya rendah atau negatif, dugaan bahwa kelompok tersebut merupakan wadah pencilan menjadi terkonfirmasi secara numerik.

4. **Asumsi geometris K-Means.** K-Means mengasumsikan klaster berbentuk bulat (*isotropic*) dan berukuran relatif seimbang. Sebaran Klaster 0 yang memanjang serta ukuran Klaster 2 yang sangat kecil menunjukkan bahwa asumsi tersebut tidak sepenuhnya terpenuhi. Algoritma berbasis kerapatan seperti **DBSCAN** dapat menjadi pembanding yang berguna, karena mampu menandai pencilan sebagai *noise* alih-alih memaksanya masuk ke dalam suatu klaster.

5. **Pewarisan ketidakpastian dari tahap interpolasi.** Jendela yang sebagian besar nilainya berasal dari interpolasi akan memiliki varians yang tertekan secara artifisial. Fitur-fitur penyebaran pada jendela tersebut mencerminkan proses pengisian data, bukan kondisi atmosfer yang sebenarnya.

---

## 3.7 Kesimpulan

Rangkaian analisis pada bagian ini berhasil mentransformasikan deret waktu konsentrasi polutan harian menjadi representasi terstruktur yang dapat dianalisis secara kuantitatif. Melalui *windowing* 14 harian, ekstraksi 68 fitur TSFEL, standardisasi, dan proyeksi PCA, data berhasil diringkas ke dalam dua dimensi tanpa kehilangan struktur utamanya.

Hasil pengelompokan mengungkap adanya **stratifikasi rezim kualitas udara** di Kecamatan Kamal: satu rezim dasar yang mendominasi sebagian besar periode pengamatan, satu rezim dengan karakteristik temporal khas yang dibedakan pada dimensi kedua, serta sejumlah kecil periode anomali yang menyimpang secara ekstrem dan layak ditelaah lebih lanjut secara individual.

Temuan ini menegaskan bahwa kualitas udara di wilayah studi **tidak bersifat homogen sepanjang tahun**, melainkan terdiri atas beberapa mode perilaku yang berbeda — sebuah wawasan yang tidak dapat diperoleh hanya dengan mengamati grafik deret waktu mentah pada Bagian 1.