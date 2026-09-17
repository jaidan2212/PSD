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