# Analisis Time Series: Persiapan Data Polutan NO2 Suramadu

Dalam proyek ini, kita menggunakan data historis kualitas udara, khususnya konsentrasi gas Nitrogen Dioksida (NO2), yang direkam berdasarkan urutan waktu. Sebelum melakukan pemodelan dan peramalan tren time series, data mentah dikelola di dalam cloud database agar proses penarikan data ke sistem analitik menjadi lebih efisien dan terpusat.

Dokumen ini mencakup alur lengkap mulai dari migrasi data awal hingga pengolahan dasar menggunakan KNIME Analytics Platform.

## 1. Migrasi Data ke Aiven PostgreSQL (via DBeaver)

Tahap pertama bertujuan untuk memindahkan data historis NO2 ke dalam layanan cloud database Aiven (project `zaidannabil2212-b12`, service `pg-3c8d907f`) agar siap diakses secara daring dari berbagai platform, termasuk DBeaver dan KNIME.

### 1.1. Pembuatan Struktur Tabel di Aiven

Langkah pertama sebelum memasukkan data adalah membuat penampung datanya, yaitu sebuah tabel. Karena ini adalah analisis runtun waktu, kolom waktu (`Tanggal`) wajib didefinisikan dengan tipe data yang mendukung informasi zona waktu, yaitu `TIMESTAMPTZ`.

Melalui fitur PG Studio di dashboard Aiven, tabel dibuat dengan menjalankan query SQL berikut pada source `defaultdb` dan schema `public`:

```sql
CREATE TABLE kualitas_udara_no2_lengkap (
    id SERIAL PRIMARY KEY,
    Tanggal TIMESTAMPTZ,
    NO2 DOUBLE PRECISION,
    NO2_Clean DOUBLE PRECISION,
    NO2_Moving_Avg DOUBLE PRECISION
);
```

Tabel ini dirancang tidak hanya menyimpan nilai NO2 mentah, tapi juga dua kolom turunan yang sudah disiapkan sejak tahap awal:

- `NO2_Clean`: nilai NO2 setelah melalui proses pembersihan data (penanganan outlier/nilai kosong).
- `NO2_Moving_Avg`: nilai rata-rata bergerak (moving average) dari NO2, digunakan untuk menghaluskan tren jangka pendek.

Selain tabel `kualitas_udara_no2_lengkap` ini, terdapat juga tabel `kualitas_udara_no2` yang menyimpan data mentah sebelum diperkaya dengan kolom-kolom turunan tersebut.

![Pembuatan Tabel di Aiven](./gambar1-pembuatan-tabel-aiven.jpeg)
*Keterangan: Tampilan PG Studio di Aiven saat query `CREATE TABLE kualitas_udara_no2_lengkap` dijalankan.*

### 1.2. Menghubungkan DBeaver dengan Aiven

Agar data dapat dikelola dan diimpor dengan mudah dari komputer lokal, DBeaver dihubungkan ke database Aiven di cloud.

Langkah-langkah koneksi:

1. Di DBeaver, klik **New Database Connection** dan pilih **PostgreSQL**.
2. Masukkan parameter koneksi yang didapatkan dari halaman **Overview** di dashboard Aiven, meliputi:
   - **Host**: `pg-3c8d907f-zaidannabil2212-b12.d.aivencloud.com`
   - **Port**: Port PostgreSQL dari Aiven.
   - **Database**: `defaultdb`
   - **Username & Password**: kredensial `avnadmin` dari Aiven.
3. Penting: buka tab SSL atau Driver Properties, pastikan parameter `sslmode` diatur menjadi `require`. Ini wajib dilakukan agar DBeaver dapat terhubung ke Aiven yang mewajibkan koneksi aman menggunakan SSL.
4. Klik **Test Connection** untuk memastikan koneksi berhasil, lalu klik **Finish**.

### 1.3. Import Data ke DBeaver

Setelah DBeaver berhasil terhubung ke Aiven, tabel `kualitas_udara_no2_lengkap` sudah terlihat di skema `public` (defaultdb > Schemas > public > Tables), berdampingan dengan tabel mentah `kualitas_udara_no2`.

![Data di DBeaver](./gambar2-data-dbeaver.jpeg)
*Keterangan: Tampilan DBeaver dengan data `kualitas_udara_no2_lengkap` yang berhasil ditarik, memperlihatkan kolom `id`, `tanggal`, `no2`, dan `no2_clean`.*

Verifikasi jumlah baris data dapat dilakukan dengan menjalankan query SQL berikut:

```sql
SELECT COUNT(*) FROM kualitas_udara_no2_lengkap;
```

Hasilnya menunjukkan total **185 baris data** telah berhasil diunggah dan tersimpan dengan aman di cloud database Aiven.

## 2. Integrasi dan Pengolahan Time Series di KNIME

Setelah data siap dan tersimpan dengan aman di cloud database Aiven, tahapan selanjutnya adalah menarik data tersebut ke ruang kerja lokal (KNIME Analytics Platform) untuk pra-pemrosesan dan eksplorasi data.

### Menghubungkan Aiven ke KNIME

Proses menghubungkan KNIME ke Aiven pada prinsipnya mirip dengan menghubungkan DBeaver. Alur koneksinya (workflow) dibangun menggunakan empat node utama secara berurutan:

**PostgreSQL Connector**: Diatur menggunakan detail host, database, dan kredensial server Aiven:

- Hostname: `pg-3c8d907f-zaidannabil2212-b12.d.aivencloud.com`
- Database name: `defaultdb`
- Authentication type: Username and Password, dengan kredensial `avnadmin`

Sama seperti pada DBeaver, opsi `sslmode=require` perlu diaktifkan (melalui **Show advanced settings**) agar node dapat terhubung, karena Aiven menuntut koneksi SSL.

![Konfigurasi PostgreSQL Connector](./gambar3-postgresql-connector.jpeg)
*Keterangan: Jendela konfigurasi node PostgreSQL Connector di KNIME, tempat hostname dan kredensial database Aiven dimasukkan.*

**DB Table Selector**: Diarahkan ke skema `public` untuk menyeleksi tabel `kualitas_udara_no2_lengkap`. Preview data langsung menampilkan 5 kolom (`id`, `tanggal`, `no2`, `no2_clean`, `no2_moving_avg`), dengan kolom `tanggal` sudah otomatis dikenali KNIME sebagai tipe **Date&time (Zoned)** — bukan sekadar teks — berkat tipe `TIMESTAMPTZ` yang sudah didefinisikan sejak dari database.

![DB Table Selector](./gambar4-db-table-selector.jpeg)
*Keterangan: Keseluruhan workflow KNIME beserta cuplikan hasil pemilihan tabel `kualitas_udara_no2_lengkap`.*

**DB Reader**: Mengeksekusi penarikan data dari database ke dalam memori KNIME agar siap diolah lebih lanjut. Node ini tidak memerlukan konfigurasi tambahan ("This node requires no configuration") dan berhasil menarik seluruh **185 baris, 5 kolom** data NO2 ke dalam KNIME.

![Hasil DB Reader](./gambar5-db-reader.jpeg)
*Keterangan: Cuplikan data NO2 yang berhasil ditarik ke dalam KNIME melalui node DB Reader. Nilai NO2 pada preview ini tertampil sebagai "0" karena pembulatan tampilan default KNIME — nilai aslinya berada pada orde 10⁻⁵, sebagaimana terlihat pada node sebelumnya.*

**Statistics**: Node ini ditambahkan setelah DB Reader dan berfungsi sebagai langkah awal yang krusial untuk Eksplorasi Data (EDA). Alih-alih mengecek kolom satu per satu, node Statistics secara otomatis memproses seluruh kolom numerik (`id`, `no2`, `no2_clean`, `no2_moving_avg`) secara bersamaan dan menghasilkan ringkasan statistik deskriptif. Beberapa temuan penting dari output node ini:

- **No. missings & No. NaNs = 0** pada seluruh kolom NO2, mengonfirmasi bahwa data historis NO2 sudah lengkap tanpa kekosongan data.
- **Min, Max, Mean, Std. deviation** pada kolom `no2`, `no2_clean`, dan `no2_moving_avg` tertampil sebagai "0" — bukan berarti nilainya benar-benar nol, melainkan efek pembulatan tampilan karena konsentrasi NO2 memang berada pada skala yang sangat kecil (orde 10⁻⁵). Hal ini juga tercermin dari **Overall sum** yang hanya sebesar 0,01 untuk ketiga kolom tersebut.
- **Skewness dan Kurtosis** (bersifat tanpa satuan) tetap menampilkan nilai yang informatif: `no2` dan `no2_clean` sama-sama memiliki skewness 2,577 dan kurtosis 8,787 — identik satu sama lain, menandakan tidak ada outlier yang perlu dikoreksi lebih lanjut pada proses cleaning. Sementara itu, `no2_moving_avg` memiliki skewness 0,711 dan kurtosis -0,37, jauh lebih landai, sesuai fungsinya untuk menghaluskan fluktuasi tren jangka pendek.
- Kolom `id` menunjukkan statistik sebagai penomor baris biasa: rentang 1–185, mean 93, dengan total keseluruhan (overall sum) 17.205.

![Ringkasan Statistik](./gambar6-statistics-summary.jpeg)
*Keterangan: Output tabel dari node Statistics yang merangkum perhitungan statistik deskriptif untuk seluruh kolom NO2 secara bersamaan dalam satu tampilan.*

## 3. Kesimpulan & Hasil Pre-processing

Dari tahapan pengumpulan data ke database cloud (Aiven) hingga proses integrasi di dalam KNIME, kita telah berhasil mempersiapkan data mentah menjadi himpunan data (dataset) runtun waktu NO2 yang berkualitas tinggi. Berikut adalah rangkuman dari hasil pre-processing ini:

- **Sentralisasi Data yang Aman**: Data historis NO2 Gresik kini tersimpan dengan aman di Aiven PostgreSQL (project `zaidannabil2212-b12`) dan diakses menggunakan enkripsi SSL, memungkinkan kolaborasi atau penarikan data dari berbagai platform (DBeaver, KNIME) kapan saja tanpa harus memindahkan file CSV secara manual.
- **Kualitas Data Terjamin (Bebas Missing Value)**: Berkat node Statistics di KNIME, terkonfirmasi bahwa tidak ada nilai yang hilang (missing) maupun NaN pada seluruh 185 baris data NO2, sehingga rangkaian waktu (time series) sekarang menjadi utuh dan tidak terputus.
- **Kolom Turunan Sudah Tersedia**: Tabel `kualitas_udara_no2_lengkap` sudah dilengkapi kolom `NO2_Clean` (hasil pembersihan data) dan `NO2_Moving_Avg` (rata-rata bergerak) sejak tahap penyimpanan di database, sehingga workflow KNIME saat ini berfokus pada integrasi dan validasi kualitas data, bukan pembersihan ulang dari awal.
- **Format Waktu yang Valid**: Penggunaan tipe `TIMESTAMPTZ` pada kolom `Tanggal` di database membuat KNIME secara otomatis mengenalinya sebagai tipe **Date&time (Zoned)**, memastikan urutan kronologis data tetap konsisten tanpa perlu konversi manual dari string.
- **Siap untuk Analisis Lanjutan**: Dengan selesainya tahap integrasi dan validasi kualitas data ini, dataset NO2 sudah berada dalam kondisi yang bersih, konsisten, dan terstruktur. Data ini kini sepenuhnya siap digunakan untuk tugas machine learning seperti pemodelan peramalan (forecasting) konsentrasi NO2 di periode mendatang.