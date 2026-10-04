Tahap 2 - Reduksi Dimensi: 204 -> 203 -> 74 -> 37

Pada tahap ini dilakukan reduksi dimensi secara bertahap untuk memperoleh representasi data yang lebih ringkas, informatif, dan siap digunakan pada tahap pemodelan berikutnya.

Urutan reduksi yang digunakan adalah:

204 fitur -> 203 fitur -> 74 fitur -> 37 komponen

Pendekatan ini tidak langsung menggunakan PCA sejak awal. Reduksi awal dilakukan menggunakan seleksi berbasis korelasi untuk mengurangi fitur yang redundan terlebih dahulu. Setelah jumlah fitur menjadi lebih kecil dan sesuai dengan batas data, PCA digunakan pada tahap akhir untuk menghasilkan representasi yang lebih ringkas.

Mengapa tidak PCA di setiap tahap?

PCA sangat berguna untuk mereduksi dimensi, tetapi penggunaannya harus memperhatikan ukuran data. Pada dataset ini hanya terdapat 37 sampel/wilayah, sedangkan jumlah fitur awal mencapai 204 fitur.

⚠️ Warning: PCA tidak dapat digunakan secara langsung untuk menghasilkan lebih banyak komponen daripada batas dimensi yang ditentukan oleh jumlah sampel dan jumlah fitur. Dengan hanya 37 sampel, PCA yang langsung diterapkan pada 204 fitur akan mengalami keterbatasan dimensi dan dapat memunculkan error ketika jumlah komponen yang diminta melebihi batas yang diperbolehkan.

Oleh karena itu, reduksi dilakukan secara bertahap:

204 -> 203 fitur menggunakan seleksi korelasi awal.

203 -> 74 fitur dengan menghapus fitur yang redundan atau memiliki korelasi sangat tinggi.

74 -> 37 komponen menggunakan PCA pada tahap akhir.

📝 Note: Strategi ini dipilih agar proses reduksi tetap valid terhadap jumlah sampel yang tersedia. Seleksi korelasi digunakan terlebih dahulu untuk mengurangi redundansi fitur, kemudian PCA digunakan setelah dimensi data berada pada rentang yang sesuai.

Dengan pendekatan tersebut, setiap tahap mempunyai fungsi yang jelas: seleksi korelasi mengurangi redundansi fitur, sedangkan PCA melakukan kompresi representasi menjadi komponen yang lebih ringkas.

2.1 Import Library & Load Data Tahap 1

Tahap pertama dimulai dengan mengimpor library yang dibutuhkan untuk pengolahan data dan reduksi dimensi. Data yang digunakan merupakan hasil dari Tahap 1, yaitu dua representasi data:

Linear

Polynomial (Poly)

Kedua representasi tersebut masih memiliki bentuk awal 37 sampel dan 204 fitur.

# Kode Python dimasukkan di sini

=== 2.1 Memuat Data Tahap 1 ===
Linear: (37, 204), Poly: (37, 204)

Berdasarkan keluaran tersebut, dapat diketahui bahwa kedua representasi memiliki jumlah sampel yang sama, yaitu 37 wilayah, dengan masing-masing memiliki 204 fitur.

2.2 Standardisasi Data (StandardScaler)

Sebelum proses seleksi fitur dan PCA dilakukan, data perlu melalui proses standardisasi. Standardisasi bertujuan menyetarakan skala setiap fitur sehingga fitur dengan rentang nilai yang besar tidak mendominasi fitur lainnya.

Metode yang digunakan adalah StandardScaler, yang mengubah data sehingga setiap fitur memiliki nilai rata-rata mendekati 0 dan standar deviasi mendekati 1.

# Kode Python dimasukkan di sini



📝 Note: Standardisasi sangat penting terutama sebelum PCA karena PCA bekerja berdasarkan varians data. Dengan skala yang telah diseragamkan, kontribusi masing-masing fitur dapat dibandingkan secara lebih adil.

Setelah standardisasi dilakukan, data siap digunakan untuk menentukan batas dimensi yang dapat diproses oleh PCA.

2.3 Batas PCA: Berapa Dimensi yang Benar-benar Informatif?

Pada dataset ini terdapat 37 sampel. Secara umum, jumlah komponen PCA yang dapat dipertahankan dibatasi oleh nilai minimum antara jumlah sampel dan jumlah fitur. Karena jumlah sampel hanya 37, maka jumlah komponen PCA yang dapat diambil tidak boleh melebihi 37 komponen.

Hal tersebut menjadi alasan utama mengapa PCA tidak digunakan sejak tahap awal ketika data masih memiliki 204 fitur. Sebelum PCA diterapkan, fitur terlebih dahulu diseleksi agar redundansi berkurang dan struktur data menjadi lebih efisien.

# Kode Python dimasukkan di sini



⚠️ Warning: Pada kondisi 37 sampel dan 204 fitur, meminta PCA menghasilkan komponen yang melebihi batas yang diperbolehkan dapat menyebabkan error. Oleh sebab itu, PCA ditempatkan pada tahap akhir setelah seleksi fitur.

Hasil analisis pada bagian ini menunjukkan bahwa 37 komponen merupakan batas maksimum yang relevan untuk representasi PCA pada dataset ini.

2.4 Fungsi Bantu

Untuk menjaga kode tetap terstruktur dan menghindari pengulangan proses, dibuat beberapa fungsi bantu. Fungsi-fungsi tersebut digunakan untuk menangani proses seperti seleksi fitur berbasis korelasi, pemeriksaan bentuk data, dan proses transformasi pada setiap representasi.

Dengan pendekatan ini, proses pada data Linear dan Poly dapat dilakukan menggunakan alur yang konsisten.

# Kode Python dimasukkan di sini

📝 Note: Fungsi bantu dibuat agar setiap tahap reduksi mempunyai prosedur yang sama pada kedua representasi data. Hal ini membantu menjaga konsistensi hasil dan memudahkan proses validasi.

2.5 Tahap 1: 204 -> 203 Fitur

Pada tahap pertama reduksi, dilakukan penghapusan awal terhadap satu fitur yang tidak diperlukan sehingga jumlah fitur berkurang dari:

204 fitur -> 203 fitur

Reduksi ini merupakan langkah awal sebelum dilakukan seleksi fitur yang lebih ketat pada tahap berikutnya.

# Kode Python dimasukkan di sini

Tahap ini menghasilkan dua representasi data dengan struktur:

Representasi

Sebelum

Sesudah

Linear

204 fitur

203 fitur

Poly

204 fitur

203 fitur

📝 Note: Pengurangan dari 204 menjadi 203 fitur bukanlah tahap PCA. Reduksi dilakukan sebagai bagian dari proses seleksi fitur awal agar struktur data menjadi lebih sederhana sebelum masuk ke seleksi redundansi.

2.6 Tahap 2: 203 -> 74 Fitur

Setelah tahap awal, dilakukan seleksi fitur berdasarkan korelasi antarfitur. Tujuannya adalah menghapus fitur-fitur yang memberikan informasi yang sangat mirip atau redundant.

Apabila dua fitur memiliki korelasi yang sangat tinggi, salah satu fitur dapat dipertahankan sementara fitur lainnya dihapus. Dengan demikian, jumlah fitur dapat dikurangi secara signifikan tanpa mempertahankan terlalu banyak informasi yang berulang.

# Kode Python dimasukkan di sini



Hasil seleksi menunjukkan bahwa kedua representasi data berhasil direduksi menjadi 74 fitur.

=== 2.5 & 2.6 Seleksi Fitur (204 -> 203 -> 74) ===
Sisa fitur Linear: 74, Poly: 74

Ringkasan reduksi fitur dapat dilihat pada tabel berikut:

Representasi

Awal

Setelah Seleksi Awal

Hasil Seleksi Korelasi

Linear

204

203

74

Poly

204

203

74

📝 Note: Seleksi korelasi pada tahap ini berfungsi untuk mengurangi redundansi, bukan untuk membentuk komponen baru. Oleh karena itu, fitur yang tersisa masih merupakan fitur asli dari hasil ekstraksi sebelumnya.

Setelah jumlah fitur menjadi 74, data sudah berada pada ukuran yang lebih sesuai untuk dilanjutkan ke proses PCA.

2.7 Tahap 3: 74 -> 37 Komponen (PCA)

Tahap terakhir menggunakan Principal Component Analysis (PCA) untuk mengubah 74 fitur menjadi 37 komponen utama.

PCA bekerja dengan mencari kombinasi linear dari fitur-fitur yang mampu mempertahankan variasi data sebanyak mungkin. Hasilnya bukan lagi fitur asli, melainkan komponen baru yang mewakili informasi utama dari seluruh fitur.

# Kode Python dimasukkan di sini



=== 2.7 PCA (74 -> 37 Komponen) ===
[Linear] 74 dimensi -> 37 komponen
[Poly] 74 dimensi -> 37 komponen

Hasil tersebut menunjukkan bahwa:

Representasi Linear berhasil direduksi dari 74 fitur menjadi 37 komponen.

Representasi Poly berhasil direduksi dari 74 fitur menjadi 37 komponen.

📝 Note: Berbeda dari seleksi fitur pada tahap sebelumnya, PCA menghasilkan representasi baru berupa komponen utama. Setiap komponen merupakan kombinasi dari fitur-fitur sebelumnya.

Dengan demikian, alur reduksi dimensi telah selesai:

204 -> 203 -> 74 -> 37

2.8 Ringkasan Semua Representasi Data

Setelah seluruh proses reduksi selesai, tersedia beberapa representasi data yang menggambarkan kondisi dataset pada setiap tahap.

Representasi

Jumlah Sampel

Jumlah Fitur/Komponen

Keterangan

Linear Tahap 1

37

204

Data awal hasil Tahap 1

Poly Tahap 1

37

204

Data awal hasil Tahap 1

Linear Setelah Reduksi Awal

37

203

Hasil seleksi awal

Poly Setelah Reduksi Awal

37

203

Hasil seleksi awal

Linear Setelah Seleksi Korelasi

37

74

Fitur redundan telah dikurangi

Poly Setelah Seleksi Korelasi

37

74

Fitur redundan telah dikurangi

Linear PCA

37

37

Hasil PCA tahap akhir

Poly PCA

37

37

Hasil PCA tahap akhir

Secara keseluruhan terdapat 8 representasi data yang dapat digunakan untuk analisis pada tahap berikutnya.

📝 Note: Penyimpanan beberapa representasi sekaligus memungkinkan proses eksperimen pada Tahap 3 dilakukan secara lebih fleksibel. Setiap representasi dapat dibandingkan untuk melihat pengaruh reduksi dimensi terhadap performa clustering.

Alur lengkap reduksi dapat diringkas sebagai berikut:

Tahap 1
204 fitur
   ↓
Seleksi awal
203 fitur
   ↓
Seleksi korelasi
74 fitur
   ↓
PCA
37 komponen
   ↓
Tahap 3
Eksperimen Clustering & Silhouette Analysis

2.9 Simpan Hasil untuk Tahap 3

Seluruh representasi data hasil reduksi kemudian disimpan agar dapat digunakan kembali pada Tahap 3, tanpa perlu mengulang proses preprocessing dan reduksi dimensi.

# Kode Python dimasukkan di sini

Data (8 representasi) berhasil disimpan di: /content/output/tahap2_pca.pkl
-> Lanjut ke Tahap 3: Eksperimen Clustering & Silhouette Analysis

📝 Note: File hasil pada tahap ini menjadi input utama untuk eksperimen clustering pada Tahap 3. Dengan menyimpan seluruh representasi, setiap skenario clustering dapat dijalankan secara konsisten menggunakan data yang sama.