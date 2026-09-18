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