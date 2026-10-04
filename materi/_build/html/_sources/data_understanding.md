# 2. Data Understanding

Tahap ini berfokus pada pengenalan spesifikasi dataset hasil *crawling* dan analisis terhadap fitur-fitur yang didapatkan dari sensor satelit **Sentinel-5P**.

## 2.1 Tentang Satelit Sentinel-5P

Sentinel-5P merupakan satelit pengamat bumi yang diluncurkan oleh *European Space Agency* (ESA) pada Oktober 2017 sebagai bagian dari Program Copernicus. Satelit ini membawa instrumen utama bernama **TROPOMI** (*TROpospheric Monitoring Instrument*), yang mampu memantau berbagai gas jejak (*trace gases*) di atmosfer bumi secara harian dengan cakupan hampir global, termasuk Ozon (O3), Sulfur Dioksida (SO2), Metana (CH4), Karbon Monoksida (CO), dan Nitrogen Dioksida (NO2).

## 2.2 Deskripsi Polutan

Satelit memonitor berbagai polutan berbahaya di troposfer bumi. Dua yang paling umum menjadi indikator kualitas udara adalah **CO** dan **NO2**:

### Apa itu CO (Karbon Monoksida)?

CO adalah gas beracun yang tidak berwarna, tidak berbau, dan tidak berasa. Gas ini umumnya dihasilkan dari proses pembakaran bahan bakar fosil yang *tidak sempurna* (*incomplete combustion*). Sumber terbesarnya adalah asap knalpot kendaraan bermotor, terutama kendaraan dengan sistem pembakaran yang kurang optimal. Jika terhirup dalam jumlah banyak, CO sangat berbahaya karena dapat mengikat hemoglobin dalam darah menggantikan oksigen, sehingga menghalangi suplai oksigen ke seluruh jaringan tubuh.

### Apa itu NO2 (Nitrogen Dioksida)?

NO2 adalah gas beracun berwarna coklat kemerahan dengan bau tajam yang menyengat. Berbeda dengan CO, NO2 dihasilkan dari pembakaran bahan bakar fosil pada *suhu tinggi*, seperti pada mesin diesel kendaraan berat, kapal laut, serta aktivitas industri dan pembangkit listrik.

Pada proyek ini, **NO2 dipilih sebagai fitur (parameter) utama** karena beberapa alasan:

- Kawasan Suramadu merupakan jalur padat kendaraan logistik antarpulau, termasuk truk-truk besar bermesin diesel yang menjadi sumber utama emisi NO2.
- NO2 memiliki korelasi kuat dengan aktivitas transportasi darat dan laut, sehingga relevan untuk mengevaluasi dampak lalu lintas terhadap kualitas udara.
- Data NO2 dari instrumen TROPOMI memiliki kualitas dan resolusi yang cukup baik untuk dianalisis dalam skala kawasan seperti Suramadu, dibandingkan gas-gas lain yang konsentrasinya lebih dipengaruhi oleh faktor global/regional.

## 2.3 Eksplorasi Fitur (Dataset)

Data yang diunduh dari Copernicus Data Space Ecosystem disimpan dalam berkas `data_polutan_no2_clean.csv`, dengan dua fitur utama sebagai berikut:

1. **Tanggal (Date):** Menunjukkan waktu observasi satelit dalam rentang **1 September 2025 hingga 31 Agustus 2026**, dengan resolusi harian (*daily*).
2. **Konsentrasi NO2:** Nilai kepadatan kolom troposferik polutan NO2 di udara area Suramadu, diukur menggunakan satuan **mol/m²** (*tropospheric NO2 column density*). Semakin tinggi nilainya, semakin tinggi pula konsentrasi molekul NO2 yang berada di kolom udara pada titik pengamatan tersebut.

## 2.4 Analisis Data Anomali (Data Quality Assessment)

Dalam proses *crawling* dan eksplorasi data (*Exploratory Data Analysis*), ditemukan beberapa data "aneh" yang wajar terjadi pada observasi satelit optik. Berikut adalah tiga jenis anomali yang teridentifikasi beserta strategi penanganannya:

### a. Missing Values (Data Kosong/NaN)

**Deskripsi:** Terdapat beberapa hari di mana satelit gagal merekam nilai konsentrasi NO2 yang valid.

**Penyebab:** Fenomena alam, yaitu instrumen satelit optik terhalang oleh formasi **awan tebal**, sehingga sensor tidak dapat menembus atmosfer hingga ke permukaan tanah/laut.

**Penanganan:** Dilakukan proses **interpolasi linear** (*linear interpolation*) untuk mengisi nilai yang hilang berdasarkan tren data pada hari-hari di sekitarnya.

### b. Nilai Negatif

**Deskripsi:** Pada ringkasan statistik awal, ditemukan beberapa nilai konsentrasi NO2 yang berada di bawah angka nol (negatif).

**Penyebab:** Secara fisika, polutan tidak mungkin memiliki nilai konsentrasi negatif. Anomali ini muncul akibat *noise* (gangguan sinyal) pada spektrometer satelit ketika membaca level konsentrasi yang sangat rendah (mendekati nol).

**Penanganan:** Dilakukan proses ***clipping*** (pemotongan nilai) sehingga seluruh nilai negatif diubah menjadi 0, karena secara fisis nilai konsentrasi minimum yang mungkin adalah nol.

### c. Spike Ekstrem (Lonjakan Tajam)

**Deskripsi:** Terdapat hari-hari tertentu dengan lonjakan nilai polusi yang sangat ekstrem dibandingkan hari-hari biasanya.

**Penyebab yang Mungkin:**
- Kondisi kemacetan lalu lintas luar biasa di sekitar jembatan.
- Kondisi cuaca dengan angin minim, sehingga polutan terperangkap di suatu area (*stagnasi atmosfer*).
- Asap kiriman dari kebakaran lahan atau hutan musiman di wilayah sekitar.

**Penanganan:** Berbeda dengan dua anomali sebelumnya, nilai spike ekstrem **tidak dihapus** dari dataset karena berpotensi mencerminkan kejadian nyata (bukan galat sensor). Nilai-nilai ini tetap dipertahankan untuk keperluan analisis, namun diberi perhatian khusus melalui perhitungan **rata-rata bergerak (*Moving Average*)** pada tahap visualisasi, agar tren jangka panjang tetap dapat terbaca dengan jelas tanpa terdistorsi oleh fluktuasi harian yang tajam.

## 2.5 Visualisasi Akhir (Time Series)

Berikut adalah grafik deret waktu (*time-series*) yang merepresentasikan fluktuasi kualitas udara di kawasan Jembatan Suramadu selama satu tahun terakhir:

![Grafik Tren Polusi NO2 Suramadu](output.png)

*(Garis merah muda menunjukkan data mentah harian, sedangkan garis biru menunjukkan data rata-rata bergerak (Moving Average 7 Hari) untuk memperjelas tren utama).*
