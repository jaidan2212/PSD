# Bagian 3: Pemodelan Machine Learning & Evaluasi K-Means

## 1. Penggabungan dan Standarisasi Data

### Penggabungan Data Polutan

Pada tahap ini, tiga dataset hasil ekstraksi fitur TSFEL, yaitu **CO**, **NO2**, dan **SO2**, digabungkan menjadi satu dataset menggunakan proses *concatenation*. Penggabungan dilakukan agar seluruh data dari satu kelas analisis dapat diproses dan dianalisis secara bersamaan menggunakan algoritma *clustering*.

Ketiga dataset tersebut memiliki karakteristik fitur yang sama, yaitu berupa fitur-fitur statistik dan karakteristik *time series* yang telah diekstraksi menggunakan TSFEL. Dengan menggabungkannya, data dapat dipandang sebagai satu kesatuan observasi yang memiliki **68 fitur** yang sama untuk setiap sampel.

Proses penggabungan juga memungkinkan K-Means untuk mencari pola atau struktur kelompok berdasarkan **kesamaan karakteristik fitur**, bukan berdasarkan asal polutan. Dengan demikian, proses clustering tetap bersifat **unsupervised learning**, karena label polutan tidak digunakan sebagai variabel yang menentukan pembentukan cluster.

Pada implementasinya, kolom non-numerik maupun kolom identitas seperti nomor atau NIM tidak digunakan sebagai fitur clustering. Label polutan juga hanya dapat digunakan sebagai informasi pendukung atau visualisasi dan **tidak menjadi input utama K-Means**.

### Standarisasi Data

Sebelum melakukan PCA maupun K-Means, data dinormalisasi menggunakan **StandardScaler**. Tahap ini penting karena 68 fitur yang dihasilkan dari TSFEL dapat memiliki **skala, satuan, dan rentang nilai yang berbeda**.

Standardisasi mengubah setiap fitur sehingga memiliki karakteristik statistik yang relatif seragam, yaitu:

* **Mean mendekati 0**
* **Standar deviasi mendekati 1**

Secara konseptual, proses standardisasi dilakukan dengan persamaan:

$$
z = \frac{x-\mu}{\sigma}
$$

dengan:

* $x$ = nilai asli suatu fitur,
* $\mu$ = rata-rata fitur,
* $\sigma$ = standar deviasi fitur,
* $z$ = nilai fitur setelah standardisasi.

Standardisasi menjadi penting karena **PCA dan K-Means sama-sama sensitif terhadap skala data**. PCA menentukan arah komponen berdasarkan variasi data, sedangkan K-Means menentukan kedekatan antar-observasi berdasarkan jarak. Apabila fitur tertentu mempunyai rentang nilai jauh lebih besar dibandingkan fitur lainnya, fitur tersebut dapat memberikan pengaruh yang terlalu besar terhadap hasil analisis.

Oleh karena itu, data yang telah digabungkan terlebih dahulu distandarisasi sebelum digunakan pada kedua skenario clustering.

---

## 2. Skenario 1: Reduksi Dimensi (PCA 37) & K-Means

### Penerapan Principal Component Analysis (PCA)

Pada skenario pertama, data yang semula memiliki **68 fitur** direduksi menggunakan **Principal Component Analysis (PCA)** menjadi **37 komponen utama**.

PCA merupakan metode reduksi dimensi yang bertujuan mengubah sejumlah fitur asli menjadi sejumlah komponen baru yang lebih ringkas. Komponen-komponen tersebut dibentuk sebagai kombinasi linear dari fitur awal dan diurutkan berdasarkan jumlah variasi data yang dapat dijelaskannya.

Dalam penelitian ini, penggunaan PCA dari **68 fitur menjadi 37 komponen** bertujuan untuk mengurangi kompleksitas dimensi data sebelum dilakukan clustering.

Beberapa tujuan penerapan PCA adalah:

* Mengurangi jumlah dimensi data sehingga proses clustering menjadi lebih sederhana.
* Mengurangi redundansi informasi dari fitur-fitur yang memiliki korelasi.
* Mempertahankan informasi atau variasi utama yang terdapat pada data.
* Membantu mengurangi pengaruh fitur yang kurang informatif terhadap proses clustering.
* Membentuk representasi data yang lebih efisien untuk digunakan oleh K-Means.

Dengan demikian, data hasil transformasi PCA tetap merepresentasikan karakteristik utama dari 68 fitur awal, tetapi menggunakan ruang dimensi yang lebih rendah, yaitu **37 komponen utama**.

### Clustering Menggunakan K-Means

Setelah proses PCA selesai, hasil transformasi 37 komponen digunakan sebagai input algoritma **K-Means Clustering**.

K-Means bekerja dengan cara membagi data ke dalam sejumlah cluster berdasarkan kedekatan data terhadap pusat cluster (*centroid*). Setiap observasi akan ditempatkan pada cluster yang centroid-nya memiliki jarak paling dekat terhadap observasi tersebut.

Namun, salah satu persoalan penting dalam K-Means adalah menentukan **jumlah cluster ($k$)** yang sesuai. Oleh karena itu, pada penelitian ini digunakan beberapa nilai $k$, yaitu **2 sampai 10**, kemudian setiap nilai dievaluasi menggunakan **Inertia** dan **Silhouette Score**.

### Hasil Evaluasi Skenario 1 untuk Setiap Nilai k

Hasil pengujian K-Means pada data PCA dilakukan untuk sembilan nilai jumlah cluster, yaitu **$k=2$ hingga $k=10$**. Setiap nilai $k$ menghasilkan nilai **Inertia** dan **Silhouette Score** yang berbeda.

| $k$ | Inertia PCA (37 Komponen) | Silhouette Score PCA | Keterangan     |
| --: | ------------------------: | -------------------: | -------------- |
|   2 |                     [isi] |                [isi] | [interpretasi] |
|   3 |                     [isi] |                [isi] | [interpretasi] |
|   4 |                     [isi] |                [isi] | [interpretasi] |
|   5 |                     [isi] |                [isi] | [interpretasi] |
|   6 |                     [isi] |                [isi] | [interpretasi] |
|   7 |                     [isi] |                [isi] | [interpretasi] |
|   8 |                     [isi] |                [isi] | [interpretasi] |
|   9 |                     [isi] |                [isi] | [interpretasi] |
|  10 |                     [isi] |                [isi] | [interpretasi] |

Berdasarkan tabel tersebut, **nilai Inertia secara umum akan semakin menurun seiring bertambahnya jumlah cluster**. Hal ini terjadi karena semakin banyak centroid yang digunakan, semakin dekat data terhadap centroid masing-masing.

Meskipun demikian, penurunan inertia tersebut tidak dapat digunakan secara langsung untuk menyatakan bahwa jumlah cluster yang lebih besar selalu lebih baik. Oleh sebab itu, diperlukan analisis terhadap bentuk grafik Elbow serta Silhouette Score.

Nilai **Silhouette Score tertinggi** pada Skenario 1 diperoleh pada:

> **$k = [isi nilai k]$ dengan Silhouette Score = [isi nilai]**

Nilai tersebut digunakan sebagai salah satu dasar untuk menentukan jumlah cluster yang memiliki pemisahan relatif paling baik pada ruang PCA.

### Interpretasi Metode Elbow Skenario 1

Pada grafik Elbow Skenario 1, nilai inertia diamati dari $k=2$ hingga $k=10$. Penentuan titik *elbow* dilakukan dengan mengamati lokasi ketika penurunan inertia mulai melandai.

Berdasarkan hasil pengamatan grafik, titik *elbow* diperkirakan berada pada:

> **$k = [isi nilai k]$**

Hasil ini kemudian dibandingkan dengan nilai Silhouette Score. Apabila titik *elbow* dan nilai silhouette tertinggi menunjuk pada nilai $k$ yang sama atau berdekatan, maka hasil evaluasi dari kedua metode tersebut memberikan indikasi yang lebih konsisten mengenai struktur cluster.

---

## 3. Skenario 2: K-Means pada 68 Fitur Asli

### Clustering pada Data Berdimensi Tinggi

Pada skenario kedua, proses clustering dilakukan langsung menggunakan seluruh **68 fitur asli** setelah data distandarisasi, tanpa melalui proses PCA.

Skenario ini digunakan sebagai pembanding terhadap skenario pertama. Dengan cara ini, dapat diamati apakah reduksi dimensi menggunakan PCA memberikan perubahan terhadap kualitas hasil clustering.

Pada skenario ini, K-Means bekerja langsung pada ruang fitur berdimensi tinggi. Setiap data direpresentasikan menggunakan seluruh 68 fitur dan jarak antar-data dihitung berdasarkan keseluruhan dimensi tersebut.

Pendekatan ini memiliki keuntungan karena **tidak melakukan reduksi terhadap representasi fitur**, sehingga seluruh fitur asli tetap digunakan dalam proses clustering.

Namun, penggunaan dimensi yang lebih tinggi juga dapat menimbulkan beberapa tantangan.

### Tantangan High-Dimensional Data

Salah satu konsep penting pada data berdimensi tinggi adalah **curse of dimensionality**. Ketika jumlah dimensi bertambah, ruang data menjadi semakin kompleks dan hubungan jarak antar-observasi dapat menjadi kurang informatif.

Dalam konteks K-Means, kondisi tersebut dapat menyebabkan beberapa permasalahan, seperti:

* Perhitungan jarak menjadi lebih kompleks karena melibatkan banyak dimensi.
* Fitur yang redundan atau kurang informatif tetap ikut memengaruhi proses clustering.
* Perbedaan jarak antar-observasi dapat menjadi kurang jelas pada ruang berdimensi tinggi.
* Struktur cluster yang sebenarnya dapat lebih sulit ditemukan dibandingkan pada representasi dengan dimensi yang lebih rendah.

Oleh karena itu, skenario 2 digunakan untuk mengetahui apakah mempertahankan seluruh **68 fitur asli** memberikan struktur cluster yang lebih baik atau justru menghasilkan cluster yang lebih sulit dipisahkan dibandingkan data hasil PCA.

Penting untuk diperhatikan bahwa penggunaan 68 fitur asli **tidak otomatis berarti hasil clustering lebih buruk**. Kualitas cluster tetap perlu ditentukan berdasarkan metrik evaluasi yang diperoleh dari data.

### Hasil Evaluasi Skenario 2 untuk Setiap Nilai k

Sama seperti Skenario 1, pada Skenario 2 dilakukan pengujian K-Means untuk nilai $k=2$ hingga $k=10$. Setiap konfigurasi dievaluasi menggunakan Inertia dan Silhouette Score.

| $k$ | Inertia Asli (68 Fitur) | Silhouette Score Asli | Keterangan     |
| --: | ----------------------: | --------------------: | -------------- |
|   2 |                   [isi] |                 [isi] | [interpretasi] |
|   3 |                   [isi] |                 [isi] | [interpretasi] |
|   4 |                   [isi] |                 [isi] | [interpretasi] |
|   5 |                   [isi] |                 [isi] | [interpretasi] |
|   6 |                   [isi] |                 [isi] | [interpretasi] |
|   7 |                   [isi] |                 [isi] | [interpretasi] |
|   8 |                   [isi] |                 [isi] | [interpretasi] |
|   9 |                   [isi] |                 [isi] | [interpretasi] |
|  10 |                   [isi] |                 [isi] | [interpretasi] |

Berdasarkan hasil tersebut, **nilai Inertia pada Skenario 2 juga diharapkan menurun ketika jumlah cluster bertambah**. Penentuan jumlah cluster tidak hanya didasarkan pada inertia, tetapi juga mempertimbangkan Silhouette Score dan bentuk grafik Elbow.

Nilai **Silhouette Score tertinggi** pada Skenario 2 diperoleh pada:

> **$k = [isi nilai k]$ dengan Silhouette Score = [isi nilai]**

Nilai tersebut menjadi salah satu indikator jumlah cluster dengan struktur internal terbaik pada representasi 68 fitur asli.

### Interpretasi Metode Elbow Skenario 2

Berdasarkan grafik Elbow pada Skenario 2, titik ketika penurunan inertia mulai melandai berada pada sekitar:

> **$k = [isi nilai k]$**

Hasil ini dibandingkan dengan nilai Silhouette Score pada masing-masing $k$ untuk melihat konsistensi hasil evaluasi.

---

## 4. Perbandingan Hasil Kedua Skenario

Untuk memberikan gambaran yang lebih jelas, hasil seluruh percobaan dapat dirangkum dalam satu tabel perbandingan berikut.

| $k$ | Inertia PCA | Silhouette PCA | Inertia 68 Fitur | Silhouette 68 Fitur |
| --: | ----------: | -------------: | ---------------: | ------------------: |
|   2 |       [isi] |          [isi] |            [isi] |               [isi] |
|   3 |       [isi] |          [isi] |            [isi] |               [isi] |
|   4 |       [isi] |          [isi] |            [isi] |               [isi] |
|   5 |       [isi] |          [isi] |            [isi] |               [isi] |
|   6 |       [isi] |          [isi] |            [isi] |               [isi] |
|   7 |       [isi] |          [isi] |            [isi] |               [isi] |
|   8 |       [isi] |          [isi] |            [isi] |               [isi] |
|   9 |       [isi] |          [isi] |            [isi] |               [isi] |
|  10 |       [isi] |          [isi] |            [isi] |               [isi] |

Dari tabel tersebut dapat dilakukan analisis **per nilai $k$**, bukan hanya membandingkan nilai terbaik. Analisis ini penting karena dapat menunjukkan bagaimana perubahan jumlah cluster memengaruhi kualitas clustering pada kedua representasi data.

Sebagai contoh, apabila pada $k=[x]$ Skenario PCA mempunyai Silhouette Score lebih tinggi daripada Skenario fitur asli, maka pada konfigurasi jumlah cluster tersebut data hasil PCA menunjukkan pemisahan cluster yang lebih baik berdasarkan metrik silhouette.

Sebaliknya, apabila pada $k=[y]$ Skenario fitur asli menghasilkan Silhouette Score yang lebih tinggi, maka pada jumlah cluster tersebut representasi 68 fitur asli memberikan struktur cluster yang lebih baik menurut metrik yang sama.

### Tabel Ringkasan Nilai Terbaik

| Parameter Evaluasi              | PCA 37 Komponen | 68 Fitur Asli |
| ------------------------------- | --------------: | ------------: |
| $k$ dengan Silhouette tertinggi |           [isi] |         [isi] |
| Silhouette Score tertinggi      |           [isi] |         [isi] |
| $k$ berdasarkan Elbow           |           [isi] |         [isi] |
| Inertia pada $k$ terpilih       |           [isi] |         [isi] |

Tabel tersebut digunakan untuk merangkum hasil utama sebelum memasuki tahap pembahasan.

---

## 5. Metrik Evaluasi & Interpretasi

### Metode Elbow

Metode **Elbow** digunakan dengan mengamati nilai **Inertia** atau *Within-Cluster Sum of Squares (WCSS)*.

Inertia menggambarkan total jarak kuadrat antara setiap data dengan centroid cluster tempat data tersebut berada. Secara umum, semakin besar jumlah cluster, nilai inertia akan semakin kecil.

Namun, penambahan cluster tidak selalu memberikan peningkatan yang berarti. Oleh karena itu, metode Elbow mencari titik perubahan atau **"siku" (*elbow*)**, yaitu ketika penurunan inertia mulai tidak terlalu signifikan.

Secara konseptual:

> Titik *elbow* dapat digunakan sebagai indikasi jumlah cluster yang relatif sesuai karena penambahan cluster setelah titik tersebut memberikan pengurangan inertia yang semakin kecil.

Metode Elbow digunakan sebagai **indikator struktur internal cluster**, bukan sebagai satu-satunya dasar penentuan nilai $k$.

### Silhouette Score

Selain Inertia, penelitian ini menggunakan **Silhouette Score** sebagai metrik evaluasi kualitas cluster.

Silhouette Score mengukur seberapa baik sebuah observasi berada di dalam cluster-nya sendiri dibandingkan dengan cluster lainnya. Nilai silhouette berada pada rentang:

$$
-1 \leq S \leq 1
$$

Interpretasinya secara umum adalah:

* Nilai mendekati **1** menunjukkan data berada cukup dekat dengan cluster-nya sendiri dan cukup jauh dari cluster lain.
* Nilai mendekati **0** menunjukkan adanya kedekatan atau tumpang tindih antar-cluster.
* Nilai negatif menunjukkan bahwa suatu data berpotensi lebih dekat dengan cluster lain dibandingkan cluster tempat data tersebut ditempatkan.

Dalam penelitian ini, **nilai Silhouette Score yang lebih tinggi menunjukkan struktur cluster yang secara internal lebih kompak dan lebih terpisah**.

Oleh karena itu, nilai $k$ dapat dibandingkan berdasarkan skor silhouette tertinggi, kemudian hasil tersebut dapat dianalisis bersama grafik Elbow untuk memperoleh gambaran yang lebih lengkap mengenai struktur cluster.

---

## 6. Kesimpulan

### Template Kesimpulan

> Berdasarkan hasil clustering menggunakan algoritma K-Means, dilakukan perbandingan antara dua skenario, yaitu **Skenario 1 menggunakan PCA dengan 37 komponen** dan **Skenario 2 menggunakan seluruh 68 fitur asli**.
>
> Pada Skenario 1, pengujian dilakukan pada $k=2$ hingga $k=10$. Berdasarkan hasil evaluasi, nilai Silhouette Score tertinggi diperoleh pada **$k=[isi]$** dengan nilai **[isi]**, sedangkan titik *elbow* berdasarkan grafik Inertia berada pada sekitar **$k=[isi]$**.
>
> Pada Skenario 2, pengujian juga dilakukan pada $k=2$ hingga $k=10$. Nilai Silhouette Score tertinggi diperoleh pada **$k=[isi]$** dengan nilai **[isi]**, sedangkan titik *elbow* berdasarkan grafik Inertia berada pada sekitar **$k=[isi]$**.
>
> Berdasarkan perbandingan kedua skenario, Silhouette Score maksimum pada Skenario 1 adalah **[isi]**, sedangkan pada Skenario 2 adalah **[isi]**. Perbedaan tersebut menunjukkan bahwa **[isi berdasarkan hasil: representasi PCA / 68 fitur asli]** menghasilkan struktur cluster yang lebih baik berdasarkan metrik silhouette.
>
> Apabila Silhouette Score pada Skenario 1 lebih tinggi dibandingkan Skenario 2, maka secara akademis dapat dikatakan bahwa **representasi data setelah reduksi dimensi menggunakan PCA menghasilkan struktur cluster yang lebih baik menurut metrik silhouette**. Hal tersebut menunjukkan bahwa pada data hasil penelitian ini, 37 komponen PCA mampu memberikan representasi yang menghasilkan **cluster yang relatif lebih kompak di dalam kelompok dan lebih terpisah antar-kelompok** dibandingkan penggunaan langsung 68 fitur asli.
>
> Sebaliknya, apabila Silhouette Score pada Skenario 2 lebih tinggi, maka hasil tersebut menunjukkan bahwa penggunaan seluruh fitur asli memberikan struktur cluster yang lebih baik berdasarkan metrik silhouette, sehingga reduksi dimensi ke 37 komponen tidak memberikan peningkatan kualitas cluster pada dataset tersebut.
>
> Dengan demikian, kualitas clustering dalam penelitian ini ditentukan berdasarkan **hasil evaluasi empiris**, bukan hanya berdasarkan jumlah fitur. PCA dapat membantu menyederhanakan representasi data, tetapi efektivitasnya terhadap clustering tetap harus dibuktikan melalui metrik seperti **Inertia dan Silhouette Score**.

### Makna Akademis jika Silhouette Skenario PCA Lebih Tinggi

Apabila hasil penelitian menunjukkan:

$$
Silhouette_{PCA} > Silhouette_{Asli}
$$

maka interpretasinya adalah bahwa **data hasil reduksi PCA memiliki kualitas pemisahan cluster yang lebih baik menurut Silhouette Score**.

Secara akademis, kondisi tersebut dapat menunjukkan bahwa reduksi dimensi berhasil mempertahankan atau menonjolkan struktur utama data yang relevan untuk clustering, sekaligus mengurangi pengaruh variasi atau fitur yang kurang membantu proses pemisahan kelompok.

Dengan kata lain, PCA pada dataset tersebut tidak hanya mengurangi jumlah dimensi dari **68 menjadi 37**, tetapi juga menghasilkan representasi yang pada evaluasi K-Means memiliki **kohesi intra-cluster yang lebih baik dan separasi antar-cluster yang lebih jelas**.

Namun, kesimpulan tersebut tetap harus dibatasi pada **dataset, proses preprocessing, dan metode evaluasi yang digunakan dalam penelitian ini**. Nilai Silhouette yang lebih tinggi menunjukkan kualitas struktur cluster yang lebih baik berdasarkan metrik tersebut, tetapi tidak secara otomatis membuktikan bahwa PCA selalu lebih baik untuk seluruh dataset atau seluruh permasalahan clustering.

---

## 7. Ringkasan Alur Analisis

Secara keseluruhan, tahapan analisis pada Bagian 3 dapat diringkas sebagai berikut:

**Penggabungan Data CO + NO2 + SO2**
↓
**Seleksi Fitur Numerik (68 Fitur)**
↓
**Standardisasi Menggunakan StandardScaler**
↓
**Data Dibagi Menjadi Dua Skenario**

**Skenario 1:**
68 Fitur → **PCA 37 Komponen** → K-Means → Evaluasi $k=2$ s.d. $10$

**Skenario 2:**
68 Fitur → **K-Means Langsung** → Evaluasi $k=2$ s.d. $10$

↓

**Perbandingan Inertia dan Silhouette Score pada Setiap Nilai $k$**

↓

**Identifikasi $k$ berdasarkan Elbow dan Silhouette Score**

↓

**Perbandingan Kualitas Cluster Skenario PCA vs. 68 Fitur Asli**
