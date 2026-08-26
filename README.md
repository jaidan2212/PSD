#  Pemantauan & Analisis Tren Spasial-Temporal Polutan Udara ($NO_2$) di Area Jembatan Suramadu Berbasis Data Satelit Sentinel-5P

**Mata Kuliah:** Pengumpulan dan Sains Data (PSD)  
**Program Studi:** Sains Data / Teknik Informatika  
**Oleh:** Muhammad Zaidan Nabil Rafi (240411100068)  
**Dosen Pengampu:** [Nama Dosen Kamu]  
**Repositori:** `jaidan2212/PSD`

---

##  1. Business Understanding (Pemahaman Bisnis & Domain)

### 1.1 Latar Belakang
Kualitas udara merupakan indikator vital dalam kesehatan lingkungan. Koridor Selat Madura, khususnya kawasan sekeliling **Jembatan Suramadu** (menghubungkan Kota Surabaya dan Kabupaten Bangkalan), memiliki karakteristik mobilitas yang unik:
* **Tingginya Volume Kendaraan:** Merupakan urat nadi transportasi harian logistik dan kendaraan pribadi antar-pulau.
* **Wilayah Pesisir & Industri:** Berada dekat dengan zona pelabuhan dan kawasan industri pesisir yang berpotensi menghasilkan akumulasi emisi.

Nitrogen Dioksida ($NO_2$) adalah salah satu gas polutan beracun utama (*Criteria Air Pollutants*) yang utamanya dihasilkan dari proses pembakaran bahan bakar fosil pada suhu tinggi (mesin kendaraan bermotor dan aktivitas industri).

### 1.2 Tujuan Proyek
1. **Ekstraksi Data Otomatis:** Membangun *pipeline* otomatisasi penarikan (*crawling*) data kualitas udara dari platform *Earth Observation cloud*.
2. **Analisis Spasial-Temporal:** Mengamati pola fluktuasi konsentrasi $NO_2$ di troposfer area Suramadu sepanjang tahun **2025**.
3. **Penyediaan Dataset Standar:** Menyediakan *raw dataset* yang siap untuk tahap *Data Preprocessing* dan ekstraksi fitur (*feature engineering*).

---

##  2. Data Understanding (Spesifikasi & Akuisisi Data Satelit)

### 2.1 Sumber Data & Sensor
Data tidak dikumpulkan dari stasioner pemantau darat, melainkan diekstraksi dari **Satelit Sentinel-5 Precursor (Sentinel-5P)** milik *European Space Agency* (ESA) yang diakses melalui ekosistem **Copernicus Data Space Ecosystem (CDSE)**.

| Parameter | Spesifikasi / Keterangan |
| :--- | :--- |
| **Satelit / Sensor** | Sentinel-5P / TROPOMI (*TROPOspheric Monitoring Instrument*) |
| **Koleksi Dataset** | `SENTINEL_5P_L2` (Level-2 Data Product) |
| **Variabel / Band** | `NO2` (Nitrogen Dioxide Tropospheric Vertical Column) |
| **Satuan Pengukuran** | $mol/m^2$ (Mol per meter persegi) |
| **Rentang Waktu** | `2025-01-01` s.d. `2025-12-31` (1 Tahun) |
| **Cakupan Spasial** | Bounding Box Suramadu & Selat Madura (`[112.65, -7.22]` s.d. `[112.78, -7.14]`) |

### 2.2 Alur Pemrosesan Data (openEO API)
Penarikan data dilakukan secara *server-side processing* memanfaatkan **openEO API** untuk meminimalisir beban unduh citra satelit yang berukuran *Gigabyte*:
1. **Authentication:** Otorisasi identitas via *OIDC Device Code Flow* (`connection.authenticate_oidc()`).
2. **Collection Querying:** Memanggil datacube `SENTINEL_5P_L2` sesuai *spatial* & *temporal extent*.
3. **Spatial Aggregation:** Melakukan operasi reduksi nilai piksel satelit menggunakan metode rata-rata (`reducer="mean"`) pada geometri Polygon target.
4. **Execution & Fetching:** Menjalankan pemrosesan di *cloud server* Copernicus (`timeseries.execute()`) dan menerima hasil akhir berupa deret waktu (*time-series*).

---

##  3. Temuan Awal & Analisis Visual (Exploratory Data)

Berdasarkan eksekusi pada file `rawling_data.ipynb`, grafik *time-series* yang dihasilkan menunjukkan beberapa fenomena:
* **Fluktuasi Harian:** Konsentrasi $NO_2$ bervariasi secara dinamis antara $0.00000\ mol/m^2$ hingga puncak tertinggi mencapai sekitar $0.00025\ mol/m^2$.
* **Lonjakan Polusi (Spike Event):** Terdapat lonjakan signifikan pada periode **September – Oktober 2025**, yang berpotensi berkorelasi dengan puncak musim kemarau (kecepatan angin rendah / dispersi polutan lambat) atau peningkatan aktivitas mobilitas.
* **Tutupan Awan (*Cloud Coverage*):** Terdapat beberapa titik hari kosong yang merupakan fenomena wajar pada instrumen optik/spektrometer satelit akibat tutupan awan tebal.

---

##  4. Struktur Repositori

```text
TugasPSD/
├── .ipynb_checkpoints/
├── data_polutan_no2.csv      # Dataset hasil ekstraksi (Tanggal vs Konsentrasi NO2)
├── rawling_data.ipynb        # Notebook Python (Otensifikasi, Data Extraction & Visualization)
└── README.md                 # Dokumentasi komprehensif proyek
