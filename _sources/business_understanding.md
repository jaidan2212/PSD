# 1. Business Understanding

## 1.1 Latar Belakang

Kualitas udara merupakan salah satu indikator penting dalam menentukan tingkat kelayakan huni suatu wilayah. Di era pembangunan infrastruktur yang masif, kawasan-kawasan dengan mobilitas transportasi tinggi kerap menghadapi tantangan berupa penurunan kualitas udara akibat emisi gas buang kendaraan bermotor.

Jembatan Suramadu, yang menghubungkan Kota Surabaya (Pulau Jawa) dan Kabupaten Bangkalan (Pulau Madura), merupakan salah satu infrastruktur strategis dengan intensitas lalu lintas logistik antarpulau yang sangat tinggi. Sejak beroperasi, jembatan ini menjadi jalur utama distribusi barang dan mobilitas masyarakat, yang secara tidak langsung turut berkontribusi terhadap emisi gas buang, khususnya dari kendaraan berat bermesin diesel.

Sayangnya, pemantauan kualitas udara menggunakan stasiun pengukuran berbasis darat (*ground station*) di area sekitar jembatan maupun perairan Selat Madura masih sangat terbatas, baik dari segi jumlah maupun cakupan wilayah. Keterbatasan ini menyulitkan proses pemantauan kualitas udara secara berkelanjutan dan menyeluruh.

Perkembangan teknologi penginderaan jauh (*remote sensing*) melalui satelit **Sentinel-5P**, yang membawa instrumen **TROPOMI** (*TROpospheric Monitoring Instrument*), menghadirkan solusi alternatif yang efisien. Melalui data satelit ini, konsentrasi gas polutan seperti Nitrogen Dioksida (NO2) dapat dipantau secara harian tanpa memerlukan instalasi sensor fisik di lapangan.

Atas dasar inilah, proyek analisis data ini disusun untuk mengeksplorasi, membersihkan, dan memvisualisasikan data kualitas udara di kawasan Jembatan Suramadu menggunakan data historis Sentinel-5P selama periode satu tahun.

## 1.2 Tujuan Proyek

Tujuan utama dari proyek eksplorasi data ini adalah untuk **mengetahui dan memetakan tingkat kualitas udara**, khususnya fluktuasi konsentrasi gas beracun Nitrogen Dioksida (NO2), di wilayah Surabaya dan Madura yang berpusat di koridor Jembatan Suramadu, selama satu tahun terakhir (September 2025 – Agustus 2026).

Secara lebih rinci, tujuan proyek ini meliputi:

1. Mengumpulkan data konsentrasi NO2 dari satelit Sentinel-5P melalui platform Copernicus Data Space Ecosystem.
2. Melakukan proses pembersihan data (*data cleaning*) terhadap nilai-nilai anomali yang ditemukan selama observasi.
3. Mengidentifikasi pola dan tren temporal konsentrasi NO2 sepanjang periode pengamatan.
4. Menyajikan hasil analisis dalam bentuk visualisasi deret waktu (*time-series*) yang informatif dan mudah dipahami.

## 1.3 Manfaat Proyek

Mengapa kita perlu mengetahui kualitas udara di area ini? Berikut adalah beberapa manfaat utama dari proyek ini:

1. **Bagi Kesehatan Masyarakat (*Early Warning*):** Memberikan informasi kepada masyarakat sekitar pesisir dan pengguna jalan tol Suramadu mengenai tingkat risiko paparan polusi. Paparan NO2 jangka panjang sangat berbahaya karena dapat memicu masalah pernapasan akut seperti asma dan bronkitis.

2. **Evaluasi Kebijakan Lingkungan & Transportasi:** Jembatan Suramadu adalah urat nadi logistik dengan mobilitas kendaraan yang sangat tinggi. Data ini dapat dimanfaatkan oleh pemerintah daerah (Pemkot Surabaya & Pemkab Bangkalan) untuk mengevaluasi apakah emisi kendaraan bermotor di area tersebut sudah melampaui batas aman, sehingga bisa menjadi dasar pembuatan kebijakan (misal: uji emisi ketat atau pembatasan kendaraan berat di jam tertentu).

3. **Pemanfaatan Teknologi Big Data & Remote Sensing:** Membuktikan bahwa pemantauan lingkungan kini dapat dilakukan secara *remote* dan murah dengan memanfaatkan data satelit *cloud-based*, tanpa harus membangun stasiun sensor fisik yang mahal di tengah jembatan atau laut.

4. **Kontribusi Akademis:** Menjadi referensi awal bagi penelitian lanjutan terkait dampak infrastruktur transportasi terhadap kualitas udara di wilayah kepulauan, khususnya yang menggunakan pendekatan penginderaan jauh.

## 1.4 Batasan Masalah

Agar proyek ini memiliki fokus yang jelas dan terarah, berikut adalah batasan-batasan yang ditetapkan:

- **Parameter Utama:** Analisis kuantitatif difokuskan hanya pada satu polutan, yaitu **Nitrogen Dioksida (NO2)**. Polutan Karbon Monoksida (CO) dibahas secara konseptual sebagai pembanding pada bagian *Data Understanding*, namun tidak dianalisis secara mendalam pada tahap eksplorasi data.
- **Cakupan Wilayah:** Observasi dibatasi pada koridor Jembatan Suramadu berdasarkan *bounding box* koordinat 112.65, -7.22 s.d. 112.78, -7.14, sehingga hasil analisis tidak dapat digeneralisasi untuk seluruh wilayah Jawa Timur.
- **Sumber Data:** Data yang digunakan merupakan data sekunder hasil observasi satelit (Sentinel-5P), bukan hasil pengukuran langsung di lapangan (*ground-truth*). Oleh karena itu, akurasi data bergantung pada kondisi atmosfer, resolusi spasial sensor, dan tutupan awan pada saat perekaman.
- **Rentang Waktu:** Data yang dianalisis terbatas pada periode 1 September 2025 hingga 31 Agustus 2026 (satu tahun observasi), sehingga belum dapat merepresentasikan tren jangka panjang (multi-tahun).
- **Sifat Analisis:** Proyek ini bersifat deskriptif dan eksploratif (*Exploratory Data Analysis*), berfokus pada identifikasi pola dan anomali data, serta belum mencakup pemodelan prediktif (*forecasting*) maupun analisis korelasi kausal terhadap faktor lain.
- **Faktor Eksternal:** Variabel meteorologis seperti kecepatan angin, arah angin, dan curah hujan hanya disinggung secara kualitatif sebagai konteks penjelas anomali data, tanpa dilakukan analisis statistik kuantitatif terhadap variabel-variabel tersebut.
