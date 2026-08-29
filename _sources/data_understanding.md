# 2. Data Understanding

Tahap ini berfokus pada pengenalan spesifikasi dataset hasil *crawling* dan analisis terhadap fitur-fitur yang didapatkan dari sensor satelit Sentinel-5P.

## Deskripsi Polutan
Satelit memonitor berbagai polutan berbahaya di troposfer bumi. Dua yang paling umum menjadi indikator kualitas udara adalah **CO** dan **NO2**:

*   **Apa itu CO (Karbon Monoksida)?** 
    CO adalah gas beracun yang tidak berwarna, tidak berbau, dan tidak berasa. Gas ini utamanya dihasilkan dari proses pembakaran bahan bakar fosil yang *tidak sempurna*. Sumber terbesarnya adalah asap knalpot kendaraan bermotor. Jika terhirup dalam jumlah banyak, CO sangat berbahaya karena akan mengikat hemoglobin dalam darah dan menghalangi suplai oksigen ke tubuh.
*   **Apa itu NO2 (Nitrogen Dioksida)?** 
    NO2 adalah gas beracun berwarna coklat kemerahan dengan bau yang tajam dan menyengat. Berbeda dengan CO, NO2 dihasilkan dari pembakaran bahan bakar fosil pada *suhu tinggi*, seperti mesin diesel kendaraan berat, kapal laut, dan aktivitas industri/pembangkit listrik. Pada proyek ini, **NO2 dipilih sebagai fitur utama** karena kawasan Suramadu merupakan jalur padat kendaraan logistik antar-pulau.

## Eksplorasi Fitur (Dataset)
Data yang diunduh (disimpan dalam `data_polutan_no2_clean.csv`) memiliki dua fitur utama:
1.  **Tanggal (Date):** Menunjukkan waktu observasi satelit dalam rentang 1 September 2025 hingga 31 Agustus 2026 (resolusi harian).
2.  **Konsentrasi NO2:** Nilai kepadatan partikel polutan NO2 di udara area Suramadu, diukur menggunakan satuan ukur **mol/m²**.

## Apakah Ada Data Aneh (Anomali)?
Dalam proses *crawling* dan eksplorasi data (*Exploratory Data Analysis*), ditemukan beberapa data "aneh" yang wajar terjadi pada observasi satelit optik:
1.  **Missing Values (Data Kosong/NaN):** Terdapat beberapa hari di mana satelit gagal mengambil nilai konsentrasi NO2 yang valid. Hal ini disebabkan oleh fenomena alam, yaitu instrumen satelit tertutup oleh formasi **awan tebal** sehingga sensor tidak bisa menembus hingga ke permukaan tanah. (Solusi: Dilakukan *Linear Interpolation*).
2.  **Nilai Negatif (-0.000001):** Pada ringkasan statistik awal, ditemukan nilai konsentrasi NO2 yang berada di bawah angka nol (negatif). Secara fisika, polutan tidak mungkin bernilai minus. Data aneh ini muncul akibat *noise* (gangguan sinyal) pada spektrometer satelit saat pembacaan level konsentrasi yang sangat rendah. (Solusi: Dilakukan proses *clipping* menjadi nilai 0).
3.  **Spike Extreme (Lonjakan Tajam):** Terdapat hari-hari tertentu dengan grafik lonjakan polusi yang sangat ekstrem dibanding hari biasanya. Ini bisa diakibatkan oleh kondisi kemacetan luar biasa, cuaca dengan minim angin (polutan terperangkap), atau asap dari kebakaran lahan musiman.

## Visualisasi Akhir (Time Series)
Berikut adalah grafik deret waktu (*time-series*) yang merepresentasikan fluktuasi kualitas udara di kawasan Jembatan Suramadu selama 1 tahun terakhir:

![Grafik Tren Polusi NO2 Suramadu](grafik_no2.png)

*(Garis merah muda menunjukkan data mentah harian, sedangkan garis biru menunjukkan data rata-rata bergerak (Moving Average 7 Hari) untuk memperjelas tren utama).*