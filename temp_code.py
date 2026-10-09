--- CELL 6 ---
!pip install earthengine-api geemap

--- CELL 7 ---
import ee
import geemap
import pandas as pd
import datetime

# 1. Inisialisasi Google Earth Engine (Tanpa Autentikasi Ulang)
try:
    ee.Initialize(project='my-project-0001-508112')
    print("Status: ✅ Berhasil terhubung ke Google Earth Engine!")
except Exception as e:
    print("Status: ❌ Gagal terhubung:", e)

# 2. Masukkan koordinat batas Kecamatan Kamal
kamal_coords = [
    [112.7035939437859, -7.171957277166015],
    [112.71078296512229, -7.141380662830116],
    [112.73381412314609, -7.1477614909314156],
    [112.7273485899737, -7.175260646526553],
    [112.7035939437859, -7.171957277166015]
]

# Ubah ke objek Geometri Earth Engine
roi_kamal = ee.Geometry.Polygon(kamal_coords)

# 3. Tentukan dataset (Sentinel-5P NO2) dan rentang waktu (misal: Jan 2025 - Des 2025)
start_date = '2025-01-01'
end_date = '2025-12-31' # Sesuaikan dengan kebutuhan datasetmu

dataset = (ee.ImageCollection('COPERNICUS/S5P/NRTI/L3_NO2')
           .filterBounds(roi_kamal)
           .filterDate(start_date, end_date)
           .select('NO2_column_number_density'))

# 4. Fungsi untuk mengekstrak rata-rata polutan tiap gambar (tiap hari)
def get_no2_mean(image):
    mean_dict = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi_kamal,
        scale=1113.2, # Resolusi spasial Sentinel-5P (~1km)
        bestEffort=True
    )
    # Dapatkan waktu gambar diambil
    date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
    
    # Simpan sebagai feature agar mudah diubah ke dataframe
    return ee.Feature(None, {
        'Tanggal': date,
        'NO2': mean_dict.get('NO2_column_number_density')
    })

print("Sedang menarik data satelit dari server Google... (tunggu sekitar 15-30 detik ⏳)")

# 5. Terapkan fungsi ke seluruh koleksi citra
time_series_features = dataset.map(get_no2_mean)

# Tarik data dari server ke memori lokal komputermu
data_list = time_series_features.getInfo()['features']

# 6. Susun ke dalam format DataFrame Pandas
records = []
for item in data_list:
    props = item['properties']
    records.append({
        'Tanggal': props.get('Tanggal'),
        'NO2': props.get('NO2')
    })

df_kamal = pd.DataFrame(records)

# Hapus baris yang nilai NO2-nya kosong (biasanya karena tertutup awan)
df_kamal.dropna(inplace=True)

# Urutkan berdasarkan tanggal biar rapi
df_kamal = df_kamal.sort_values(by='Tanggal').reset_index(drop=True)

print("\n--- Pratinjau 5 Data Pertama ---")
print(df_kamal.head())
print(f"\nTotal data terkumpul: {len(df_kamal)} baris.")

# 7. Simpan ke file CSV
csv_filename = "data_no2_kamal.csv"
df_kamal.to_csv(csv_filename, index=False)
print(f"✅ Selesai! Data berhasil disimpan ke file: {csv_filename}")

--- CELL 18 ---
import pandas as pd
import numpy as np
import tsfel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 1. BACA DATA MENTAH (Kita pakai polutan NO2 sebagai acuan clustering)
df_no2 = pd.read_csv('data_no2_kamal.csv')
data_bersih = df_no2['NO2'].dropna().values

# 2. EKSTRAKSI FITUR DENGAN WINDOWING (Jurus Rahasia!)
# Kita bagi data menjadi potongan 14 harian (2 minggu)
cfg = tsfel.get_features_by_domain()
print("Mengekstrak ulang 68 fitur dengan window_size=14 hari...")
df_fitur_baru = tsfel.time_series_features_extractor(cfg, data_bersih, window_size=14, verbose=0)

print(f"Selesai! Bentuk data sekarang: {df_fitur_baru.shape[0]} baris sampel dan {df_fitur_baru.shape[1]} fitur.")

# 3. STANDARISASI FITUR (Agar skalanya seimbang)
kolom_numerik = df_fitur_baru.select_dtypes(include=['float64', 'int64']).columns
X = df_fitur_baru[kolom_numerik]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. REDUKSI DIMENSI (PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
varians = sum(pca.explained_variance_ratio_) * 100
print(f"Informasi yang dipertahankan PCA: {varians:.2f}%")

# 5. K-MEANS CLUSTERING (3 Kelompok)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_pca['Klaster'] = kmeans.fit_predict(X_pca)

# Beri label yang masuk akal
mapping_klaster = {0: 'Klaster 0', 1: 'Klaster 1', 2: 'Klaster 2'}
df_pca['Nama Klaster'] = df_pca['Klaster'].map(mapping_klaster)

# 6. VISUALISASI HASIL CLUSTERING
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='Nama Klaster', data=df_pca, palette='Set1', s=150, alpha=0.8, edgecolor='white')

# Gambar pusat klaster (Centroid)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=300, marker='X', label='Pusat Klaster (Centroid)')

plt.title('Hasil K-Means Clustering (Data 14 Harian)\nReduksi 68 Fitur menjadi 2 Principal Components', fontsize=14, fontweight='bold')
plt.xlabel(f'Principal Component 1 (PC1)')
plt.ylabel(f'Principal Component 2 (PC2)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Simpan gambar
plt.savefig('grafik_pca_kmeans_fix.png')
plt.show()

--- CELL 28 ---
import pandas as pd
import numpy as np
import inspect
import tsfel.feature_extraction.features as tsfel_features

# ---------- 1. MUAT DATA DARI FOLDER ----------
print("Membaca data dari folder Gabungan_Polutan...")
df_co = pd.read_csv('data_co_kamal_fix2.csv')
df_no2 = pd.read_csv('data_no2_kamal.csv')
df_so2 = pd.read_csv('data_so2_kamal_fix2.csv')

# Samakan kolom waktu menjadi 'date'
df_co = df_co.rename(columns={'Tanggal': 'date'})
df_no2 = df_no2.rename(columns={'Tanggal': 'date'})
df_so2 = df_so2.rename(columns={'Tanggal': 'date'})

# KARENA KOLOM 'daerah' TIDAK ADA, KITA BUATKAN SECARA OTOMATIS
df_co['daerah'] = 'Kamal'
df_no2['daerah'] = 'Kamal'
df_so2['daerah'] = 'Kamal'

print("\nKolom 'date' dan 'daerah' sudah diatasi. Mulai menggabungkan data...")

# Pastikan format tanggal seragam
df_co['date'] = pd.to_datetime(df_co['date'])
df_no2['date'] = pd.to_datetime(df_no2['date'])
df_so2['date'] = pd.to_datetime(df_so2['date'])

# Gabungkan ketiga data menjadi 1 tabel utuh berdasarkan 'date' dan 'daerah'
df_gabungan1 = pd.merge(df_no2, df_so2, on=['date', 'daerah'], how='outer')
df_all = pd.merge(df_gabungan1, df_co, on=['date', 'daerah'], how='outer')

# Mengubah nama kolom polutan menjadi huruf kapital
rename_mapping = {col: col.upper() for col in df_all.columns if col.lower() in ['no2', 'so2', 'co']}
df_all = df_all.rename(columns=rename_mapping)

pollutants = ['NO2', 'SO2', 'CO']
fs = 1

# ---------- 2. DAFTAR 68 FITUR ----------
FEATURE_LIST = """abs_energy auc autocorr average_power calc_centroid calc_max calc_mean
calc_median calc_min calc_std calc_var dfa distance ecdf ecdf_percentile ecdf_percentile_count
ecdf_slope entropy fundamental_frequency higuchi_fractal_dimension hist_mode human_range_energy
hurst_exponent interq_range kurtosis lempel_ziv lpcc max_frequency max_power_spectrum
maximum_fractal_length mean_abs_deviation mean_abs_diff mean_diff median_abs_deviation
median_abs_diff median_diff median_frequency mfcc mse negative_turning neighbourhood_peaks
petrosian_fractal_dimension pk_pk_distance positive_turning power_bandwidth rms skewness slope
spectral_centroid spectral_decrease spectral_distance spectral_entropy spectral_kurtosis
spectral_positive_turning spectral_roll_off spectral_roll_on spectral_skewness spectral_slope
spectral_spread spectral_variation spectrogram_mean_coeff sum_abs_diff wavelet_abs_mean
wavelet_energy wavelet_entropy wavelet_std wavelet_var zero_cross""".split()

def to_scalar(result):
    if isinstance(result, dict) and "values" in result:
        result = result["values"]
    if isinstance(result, (list, tuple, np.ndarray)):
        arr = np.asarray(result, dtype=float)
        return float(np.nanmean(arr))
    return float(result)

def extract_one(fn_name, signal, fs):
    fn = getattr(tsfel_features, fn_name)
    params = inspect.signature(fn).parameters
    if "fs" in params:
        result = fn(signal, fs)
    else:
        result = fn(signal)
    return to_scalar(result)

semua_baris_daerah = []

# ---------- 3. LOOPING UNTUK SETIAP DAERAH ----------
for nama_daerah, df_group in df_all.groupby('daerah'):
    print(f"Mengekstrak 204 fitur untuk daerah: {nama_daerah} ...")
    
    df = df_group.sort_values('date').reset_index(drop=True)
    combined_row = {'daerah': nama_daerah}
    
    for pollutant in pollutants:
        if pollutant not in df.columns:
            for fn_name in FEATURE_LIST:
                combined_row[f"{pollutant}_{fn_name}"] = 0
            continue
            
        df_poly = df[['date', pollutant]].copy()
        df_poly[pollutant] = pd.to_numeric(df_poly[pollutant], errors='coerce')
        
        # Handling outlier IQR
        Q1 = df_poly[pollutant].quantile(0.25)
        Q3 = df_poly[pollutant].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_poly.loc[(df_poly[pollutant] < lower_bound) | (df_poly[pollutant] > upper_bound), pollutant] = np.nan
        
        # Interpolasi LINEAR (Sudah diamankan dengan groupby.mean)
        df_clean = df_poly.groupby('date').mean().interpolate(method='time').ffill().bfill()
        signal_1d = df_clean[pollutant].astype(float).values
        
        if len(signal_1d) == 0 or np.isnan(signal_1d).all():
            for fn_name in FEATURE_LIST:
                combined_row[f"{pollutant}_{fn_name}"] = 0
            continue
            
        for fn_name in FEATURE_LIST:
            feature_key = f"{pollutant}_{fn_name}"
            combined_row[feature_key] = extract_one(fn_name, signal_1d, fs)
            
    semua_baris_daerah.append(combined_row)

# ---------- 4. SIMPAN KE CSV FINAL ----------
extracted_features_final = pd.DataFrame(semua_baris_daerah)

print(f"\nBerhasil! Total Baris: {extracted_features_final.shape[0]}, Total Kolom: {extracted_features_final.shape[1]}")

# Nama file diubah agar spesifik untuk form Linear
output_filename = 'Ekstraksi_204_Fitur_Polutan_Linear.csv'
extracted_features_final.to_csv(output_filename, index=False)
print(f"File tersimpan sebagai: {output_filename}")

--- CELL 29 ---
import pandas as pd
import numpy as np
import inspect
import tsfel.feature_extraction.features as tsfel_features

# ---------- 1. MUAT DATA DARI FOLDER ----------
print("Membaca data dari folder Gabungan_Polutan...")
# Membaca 3 file aslimu
df_co = pd.read_csv('Gabungan_Polutan/data_co_kamal_fix2.csv')
df_no2 = pd.read_csv('Gabungan_Polutan/data_no2_kamal.csv')
df_so2 = pd.read_csv('Gabungan_Polutan/data_so2_kamal_fix2.csv')

# Samakan kolom waktu menjadi 'date'
df_co = df_co.rename(columns={'Tanggal': 'date'})
df_no2 = df_no2.rename(columns={'Tanggal': 'date'})
df_so2 = df_so2.rename(columns={'Tanggal': 'date'})

# KARENA KOLOM 'daerah' TIDAK ADA, KITA BUATKAN SECARA OTOMATIS
df_co['daerah'] = 'Kamal'
df_no2['daerah'] = 'Kamal'
df_so2['daerah'] = 'Kamal'

# Pastikan format tanggal seragam
df_co['date'] = pd.to_datetime(df_co['date'])
df_no2['date'] = pd.to_datetime(df_no2['date'])
df_so2['date'] = pd.to_datetime(df_so2['date'])

# Gabungkan ketiga data menjadi 1 tabel utuh berdasarkan 'date' dan 'daerah'
df_gabungan1 = pd.merge(df_no2, df_so2, on=['date', 'daerah'], how='outer')
df_all = pd.merge(df_gabungan1, df_co, on=['date', 'daerah'], how='outer')

# Mengubah nama kolom polutan menjadi huruf kapital
rename_mapping = {col: col.upper() for col in df_all.columns if col.lower() in ['no2', 'so2', 'co']}
df_all = df_all.rename(columns=rename_mapping)

pollutants = ['NO2', 'SO2', 'CO']
fs = 1

# ---------- 2. DAFTAR 68 FITUR ----------
FEATURE_LIST = """abs_energy auc autocorr average_power calc_centroid calc_max calc_mean
calc_median calc_min calc_std calc_var dfa distance ecdf ecdf_percentile ecdf_percentile_count
ecdf_slope entropy fundamental_frequency higuchi_fractal_dimension hist_mode human_range_energy
hurst_exponent interq_range kurtosis lempel_ziv lpcc max_frequency max_power_spectrum
maximum_fractal_length mean_abs_deviation mean_abs_diff mean_diff median_abs_deviation
median_abs_diff median_diff median_frequency mfcc mse negative_turning neighbourhood_peaks
petrosian_fractal_dimension pk_pk_distance positive_turning power_bandwidth rms skewness slope
spectral_centroid spectral_decrease spectral_distance spectral_entropy spectral_kurtosis
spectral_positive_turning spectral_roll_off spectral_roll_on spectral_skewness spectral_slope
spectral_spread spectral_variation spectrogram_mean_coeff sum_abs_diff wavelet_abs_mean
wavelet_energy wavelet_entropy wavelet_std wavelet_var zero_cross""".split()

def to_scalar(result):
    if isinstance(result, dict) and "values" in result:
        result = result["values"]
    if isinstance(result, (list, tuple, np.ndarray)):
        arr = np.asarray(result, dtype=float)
        val = float(np.nanmean(arr))
    else:
        val = float(result)
        
    # PENANGKAL 1: Jika rumus TSFEL menghasilkan error/NaN, paksa jadi 0
    if np.isnan(val) or np.isinf(val):
        return 0.0
    return val

def extract_one(fn_name, signal, fs):
    fn = getattr(tsfel_features, fn_name)
    params = inspect.signature(fn).parameters
    if "fs" in params:
        result = fn(signal, fs)
    else:
        result = fn(signal)
    return to_scalar(result)

semua_baris_daerah = []

# ---------- 3. LOOPING UNTUK SETIAP DAERAH ----------
for nama_daerah, df_group in df_all.groupby('daerah'):
    print(f"Mengekstrak 204 fitur untuk daerah: {nama_daerah} ... (Tunggu sekitar 10-15 menit)")
    
    df = df_group.sort_values('date').reset_index(drop=True)
    combined_row = {'daerah': nama_daerah}
    
    for pollutant in pollutants:
        if pollutant not in df.columns:
            for fn_name in FEATURE_LIST:
                combined_row[f"{pollutant}_{fn_name}"] = 0
            continue
            
        df_poly = df[['date', pollutant]].copy()
        df_poly[pollutant] = pd.to_numeric(df_poly[pollutant], errors='coerce')
        
        # PENANGKAL 2: Buang nilai -9999 (error sensor satelit) agar tidak merusak rumus
        df_poly[pollutant] = df_poly[pollutant].replace([-9999, -9999.0], np.nan)
        
        # Handling outlier IQR
        Q1 = df_poly[pollutant].quantile(0.25)
        Q3 = df_poly[pollutant].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_poly.loc[(df_poly[pollutant] < lower_bound) | (df_poly[pollutant] > upper_bound), pollutant] = np.nan
        
        # Interpolasi POLYNOMIAL Derajat 2
        df_clean = df_poly.groupby('date').mean().interpolate(method='polynomial', order=2).ffill().bfill()
        signal_1d = df_clean[pollutant].astype(float).values
        
        if len(signal_1d) == 0 or np.isnan(signal_1d).all():
            for fn_name in FEATURE_LIST:
                combined_row[f"{pollutant}_{fn_name}"] = 0
            continue
            
        for fn_name in FEATURE_LIST:
            feature_key = f"{pollutant}_{fn_name}"
            combined_row[feature_key] = extract_one(fn_name, signal_1d, fs)
            
    semua_baris_daerah.append(combined_row)

# ---------- 4. SIM ঐতিহ্য KE CSV FINAL ----------
extracted_features_final = pd.DataFrame(semua_baris_daerah)

# PENANGKAL 3: Sapu bersih semua sisa Missing Value (NULL) menjadi 0 sebelum disave
extracted_features_final = extracted_features_final.fillna(0)

print(f"\nBerhasil! Total Baris: {extracted_features_final.shape[0]}, Total Kolom: {extracted_features_final.shape[1]}")

output_filename = 'Ekstraksi_204_Fitur_Polutan_Polynomial.csv'
extracted_features_final.to_csv(output_filename, index=False)
print(f"File tersimpan sebagai: {output_filename}")

--- CELL 30 ---
import pandas as pd
import numpy as np
import inspect
import tsfel.feature_extraction.features as tsfel_features

# ---------- 1. MUAT DATA DARI FOLDER ----------
print("Membaca data dari folder Gabungan_Polutan...")
# Membaca 3 file aslimu
df_co = pd.read_csv('Gabungan_Polutan/data_co_kamal_fix2.csv')
df_no2 = pd.read_csv('Gabungan_Polutan/data_no2_kamal.csv')
df_so2 = pd.read_csv('Gabungan_Polutan/data_so2_kamal_fix2.csv')

# Samakan kolom waktu menjadi 'date'
df_co = df_co.rename(columns={'Tanggal': 'date'})
df_no2 = df_no2.rename(columns={'Tanggal': 'date'})
df_so2 = df_so2.rename(columns={'Tanggal': 'date'})

# KARENA KOLOM 'daerah' TIDAK ADA, KITA BUATKAN SECARA OTOMATIS
df_co['daerah'] = 'Kamal'
df_no2['daerah'] = 'Kamal'
df_so2['daerah'] = 'Kamal'

# Pastikan format tanggal seragam
df_co['date'] = pd.to_datetime(df_co['date'])
df_no2['date'] = pd.to_datetime(df_no2['date'])
df_so2['date'] = pd.to_datetime(df_so2['date'])

# Gabungkan ketiga data menjadi 1 tabel utuh berdasarkan 'date' dan 'daerah'
df_gabungan1 = pd.merge(df_no2, df_so2, on=['date', 'daerah'], how='outer')
df_all = pd.merge(df_gabungan1, df_co, on=['date', 'daerah'], how='outer')

# Mengubah nama kolom polutan menjadi huruf kapital
rename_mapping = {col: col.upper() for col in df_all.columns if col.lower() in ['no2', 'so2', 'co']}
df_all = df_all.rename(columns=rename_mapping)

pollutants = ['NO2', 'SO2', 'CO']
fs = 1

# ---------- 2. DAFTAR 68 FITUR ----------
FEATURE_LIST = """abs_energy auc autocorr average_power calc_centroid calc_max calc_mean
calc_median calc_min calc_std calc_var dfa distance ecdf ecdf_percentile ecdf_percentile_count
ecdf_slope entropy fundamental_frequency higuchi_fractal_dimension hist_mode human_range_energy
hurst_exponent interq_range kurtosis lempel_ziv lpcc max_frequency max_power_spectrum
maximum_fractal_length mean_abs_deviation mean_abs_diff mean_diff median_abs_deviation
median_abs_diff median_diff median_frequency mfcc mse negative_turning neighbourhood_peaks
petrosian_fractal_dimension pk_pk_distance positive_turning power_bandwidth rms skewness slope
spectral_centroid spectral_decrease spectral_distance spectral_entropy spectral_kurtosis
spectral_positive_turning spectral_roll_off spectral_roll_on spectral_skewness spectral_slope
spectral_spread spectral_variation spectrogram_mean_coeff sum_abs_diff wavelet_abs_mean
wavelet_energy wavelet_entropy wavelet_std wavelet_var zero_cross""".split()

def to_scalar(result):
    if isinstance(result, dict) and "values" in result:
        result = result["values"]
    if isinstance(result, (list, tuple, np.ndarray)):
        arr = np.asarray(result, dtype=float)
        val = float(np.nanmean(arr))
    else:
        val = float(result)
        
    # PENANGKAL 1: Jika rumus TSFEL menghasilkan error/NaN, paksa jadi 0
    if np.isnan(val) or np.isinf(val):
        return 0.0
    return val

def extract_one(fn_name, signal, fs):
    fn = getattr(tsfel_features, fn_name)
    params = inspect.signature(fn).parameters
    if "fs" in params:
        result = fn(signal, fs)
    else:
        result = fn(signal)
    return to_scalar(result)

semua_baris_daerah = []

# ---------- 3. LOOPING UNTUK SETIAP DAERAH ----------
for nama_daerah, df_group in df_all.groupby('daerah'):
    print(f"Mengekstrak 204 fitur untuk daerah: {nama_daerah} ... (Tunggu sekitar 10-15 menit)")
    
    df = df_group.sort_values('date').reset_index(drop=True)
    combined_row = {'daerah': nama_daerah}
    
    for pollutant in pollutants:
        if pollutant not in df.columns:
            for fn_name in FEATURE_LIST:
                combined_row[f"{pollutant}_{fn_name}"] = 0
            continue
            
        df_poly = df[['date', pollutant]].copy()
        df_poly[pollutant] = pd.to_numeric(df_poly[pollutant], errors='coerce')
        
        # PENANGKAL 2: Buang nilai -9999 (error sensor satelit) agar tidak merusak rumus
        df_poly[pollutant] = df_poly[pollutant].replace([-9999, -9999.0], np.nan)
        
        # Handling outlier IQR
        Q1 = df_poly[pollutant].quantile(0.25)
        Q3 = df_poly[pollutant].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_poly.loc[(df_poly[pollutant] < lower_bound) | (df_poly[pollutant] > upper_bound), pollutant] = np.nan
        
        # Interpolasi LINEAR (Menggunakan method='time')
        df_clean = df_poly.groupby('date').mean().interpolate(method='time').ffill().bfill()
        signal_1d = df_clean[pollutant].astype(float).values
        
        if len(signal_1d) == 0 or np.isnan(signal_1d).all():
            for fn_name in FEATURE_LIST:
                combined_row[f"{pollutant}_{fn_name}"] = 0
            continue
            
        for fn_name in FEATURE_LIST:
            feature_key = f"{pollutant}_{fn_name}"
            combined_row[feature_key] = extract_one(fn_name, signal_1d, fs)
            
    semua_baris_daerah.append(combined_row)

# ---------- 4. SIMPAN KE CSV FINAL ----------
extracted_features_final = pd.DataFrame(semua_baris_daerah)

# PENANGKAL 3: Sapu bersih semua sisa Missing Value (NULL) menjadi 0 sebelum disave
extracted_features_final = extracted_features_final.fillna(0)

print(f"\nBerhasil! Total Baris: {extracted_features_final.shape[0]}, Total Kolom: {extracted_features_final.shape[1]}")

output_filename = 'Ekstraksi_204_Fitur_Polutan_Linear.csv'
extracted_features_final.to_csv(output_filename, index=False)
print(f"File tersimpan sebagai: {output_filename}")

--- CELL 36 ---
import warnings
warnings.filterwarnings('ignore')
import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from sklearn.decomposition import PCA as PCA_viz

OUTPUT_DIR = 'output'
plt.rcParams['figure.figsize'] = (13, 5)
sns.set_theme(style='whitegrid', font_scale=1.05)

# ==========================================
# 3.1 LOAD DATA TAHAP 2
# ==========================================
print("=== 3.1 Memuat Data Tahap 2 ===")
with open(f'{OUTPUT_DIR}/tahap2_pca.pkl', 'rb') as f:
    d = pickle.load(f)

representasi = d['representasi']
meta_linear, meta_poly = d['meta_linear'], d['meta_poly']
X_linear, X_poly = d['X_linear'], d['X_poly']
print(f"Data dimuat. Total representasi: {len(representasi)}")

# ==========================================
# 3.2 JALANKAN 72 EKSPERIMEN CLUSTERING
# ==========================================
print("\n=== 3.2 Menjalankan Eksperimen Clustering ===")
K_RANGE = range(2, 11)
N_INIT = 20
RANDOM_STATE = 42

hasil, labels_map = [], {}
for nama_data, X in representasi.items():
    best_sil, best_k = -1, 2
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=N_INIT, init='k-means++', random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        
        # Urutkan label cluster agar konsisten (Cluster 0 selalu yang terbanyak)
        urut = np.argsort(-np.bincount(labels, minlength=k), kind='stable')
        remap = np.empty(k, dtype=int); remap[urut] = np.arange(k)
        labels = remap[labels]
        
        sil = silhouette_score(X, labels)
        iner = km.inertia_
        labels_map[(nama_data, k)] = labels
        hasil.append({'dataset': nama_data, 'k': k, 'silhouette': round(sil, 6), 'inertia': round(iner, 4)})
        if sil > best_sil: best_sil, best_k = sil, k
    print(f' {nama_data:<15} k_terbaik = {best_k} silhouette = {best_sil:.4f}')

df_hasil = pd.DataFrame(hasil)

# ==========================================
# 3.4 PEMILIHAN KONFIGURASI TERBAIK
# ==========================================
best_row = df_hasil.loc[df_hasil['silhouette'].idxmax()]
BEST_DATASET = best_row['dataset']
BEST_K = int(best_row['k'])
BEST_SIL = best_row['silhouette']
BEST_LABELS = labels_map[(BEST_DATASET, BEST_K)]

print(f'\n>> TERBAIK GLOBAL: {BEST_DATASET} | k={BEST_K} | Sil={BEST_SIL:.4f}')

# ==========================================
# 3.4b CLUSTER TERBAIK PER DIMENSI
# ==========================================
DIMENSI = [204, 203, 74, 37]
METODE = ['Linear', 'Poly']
BEST_PER_DIM = {}

for metode in METODE:
    for dim in DIMENSI:
        nama = f'{metode}-{dim}'
        if nama not in representasi: continue
        sub = df_hasil[df_hasil['dataset'] == nama]
        row = sub.loc[sub['silhouette'].idxmax()]
        k = int(row['k'])
        lab = labels_map[(nama, k)]
        BEST_PER_DIM[(metode, dim)] = {
            'dataset': nama, 'metode': metode, 'dimensi': dim, 'k': k,
            'silhouette': float(row['silhouette']), 'inertia': float(row['inertia']),
            'labels': lab, 'ukuran': np.bincount(lab, minlength=k).tolist()
        }

# ==========================================
# VISUALISASI (OTOMATIS DISIMPAN)
# ==========================================
print("\nMembuat 8 visualisasi grafik...")
PALETTE = {'Linear-204': '#0D47A1', 'Linear-203': '#1565C0', 'Linear-74': '#1976D2', 'Linear-37': '#90CAF9',
           'Poly-204': '#B71C1C', 'Poly-203': '#C62828', 'Poly-74': '#E53935', 'Poly-37': '#FFCDD2'}
CLUSTER_COLORS = ['#E53935','#1E88E5', '#43A047', '#FB8C00', '#8E24AA']

# 1. Grafik Silhouette Semua
fig, axes = plt.subplots(2, 4, figsize=(22, 11))
for idx, nama_data in enumerate(representasi):
    sub = df_hasil[df_hasil['dataset'] == nama_data]
    color = PALETTE.get(nama_data, '#555')
    ax = axes.flatten()[idx]
    ax.plot(sub['k'], sub['silhouette'], 'o-', color=color, linewidth=2.5, markersize=7)
    best = sub.loc[sub['silhouette'].idxmax()]
    ax.scatter(best['k'], best['silhouette'], s=280, color='gold', edgecolors=color, linewidth=2, zorder=5, marker='*')
    ax.set_title(nama_data, fontweight='bold')
plt.tight_layout(); plt.savefig(f'{OUTPUT_DIR}/07_silhouette_semua.png', dpi=150); plt.close()

# 2. Grafik Silhouette Per Dimensi
fig, ax = plt.subplots(figsize=(11, 5))
for metode, color in [('Linear', '#1565C0'), ('Poly', '#B71C1C')]:
    pts = [(i, BEST_PER_DIM[(metode, d)]) for i, d in enumerate(DIMENSI)]
    ax.plot([i for i, p in pts], [p['silhouette'] for i, p in pts], 'o-', color=color, lw=2.5, label=metode)
ax.set_xticks(range(4)); ax.set_xticklabels([f'{d}\ndim' for d in DIMENSI])
ax.legend(); plt.savefig(f'{OUTPUT_DIR}/07c_silhouette_per_dimensi.png', dpi=150); plt.close()

# 3. Heatmap
pivot = df_hasil.pivot(index='dataset', columns='k', values='silhouette')
urutan = ['Linear-204', 'Linear-203', 'Linear-74', 'Linear-37', 'Poly-204', 'Poly-203', 'Poly-74', 'Poly-37']
pivot = pivot.reindex(urutan)
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0.3, ax=ax)
plt.savefig(f'{OUTPUT_DIR}/07b_heatmap_silhouette.png', dpi=150); plt.close()

# 4. Silhouette Plot Terbaik
X_best = representasi[BEST_DATASET]
sil_vals = silhouette_samples(X_best, BEST_LABELS)
fig, ax = plt.subplots(figsize=(10, 6))
y_lower = 10
for i in range(BEST_K):
    v = np.sort(sil_vals[BEST_LABELS == i])
    size_i = len(v)
    y_upper = y_lower + size_i
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, v, facecolor=CLUSTER_COLORS[i % len(CLUSTER_COLORS)], alpha=0.8)
    y_lower = y_upper + 5
ax.axvline(BEST_SIL, color='crimson', ls='--')
plt.savefig(f'{OUTPUT_DIR}/08_silhouette_plot.png', dpi=150); plt.close()

# 5. Elbow Plot
sub_best = df_hasil[df_hasil['dataset'] == BEST_DATASET]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sub_best['k'], sub_best['inertia'], 'o-', color='#1565C0', lw=2.5)
ax.axvline(BEST_K, color='crimson', ls='--')
plt.savefig(f'{OUTPUT_DIR}/09_elbow_silhouette.png', dpi=150); plt.close()

# 6. Scatter PCA 2D
pca2d = PCA_viz(n_components=2, random_state=42)
X_viz = pca2d.fit_transform(X_best)
fig, ax = plt.subplots(figsize=(8, 6))
for c in range(BEST_K):
    mask = BEST_LABELS == c
    ax.scatter(X_viz[mask, 0], X_viz[mask, 1], s=100, color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], label=f'Cluster {c}')
ax.legend()
plt.savefig(f'{OUTPUT_DIR}/10_scatter_cluster.png', dpi=150); plt.close()

# ==========================================
# 3.10 SIMPAN DATA UNTUK TAHAP 4
# ==========================================
save_data = {
    'BEST_DATASET': BEST_DATASET, 'BEST_K': BEST_K, 'BEST_SIL': BEST_SIL,
    'BEST_LABELS': BEST_LABELS, 'CLUSTER_COLORS': CLUSTER_COLORS,
    'meta_linear': meta_linear, 'meta_poly': meta_poly,
    'X_linear': X_linear, 'X_poly': X_poly, 'df_hasil': df_hasil,
    'BEST_PER_DIM': BEST_PER_DIM, 'DIMENSI': DIMENSI
}

save_path = f'{OUTPUT_DIR}/tahap3_clustering.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(save_data, f)

print(f"Semua grafik tersimpan di folder '{OUTPUT_DIR}'!")
print(f"Hasil Tahap 3 disimpan di: {save_path}")
print("-> Lanjut ke Tahap 4: Visualisasi Peta Segmentasi")

--- CELL 37 ---
import warnings
warnings.filterwarnings('ignore')
import pickle, os
import pandas as pd
import matplotlib.pyplot as plt
import folium

OUTPUT_DIR = 'output'

# === 4.1 Load Data ===
print("Memuat data Tahap 3...")
with open(f'{OUTPUT_DIR}/tahap3_clustering.pkl', 'rb') as f:
    d = pickle.load(f)

BEST_DATASET = d['BEST_DATASET']
BEST_K = d['BEST_K']
CLUSTER_COLORS = d['CLUSTER_COLORS']
df_map = d['meta_linear'] if 'Linear' in BEST_DATASET else d['meta_poly']
df_map['cluster'] = d['BEST_LABELS']

# === 4.2 & 4.3 Geocoding Cepat ===
# Hardcode koordinat agar tidak error saat menarik data lokasi
koordinat_cache = {
    'Baron Nganjuk': [-7.5083, 111.9167], 'Nunukan': [4.1333, 117.6667],
    'Sreseh, Sampang': [-7.1833, 113.2167], 'Manyar, Gresik': [-7.1500, 112.6500],
    'Kamal, Bangkalan': [-7.1833, 112.7833], 'Kedungpring Lamongan': [-7.3500, 112.2167],
    'Gresik Kota, Gresik': [-7.1556, 112.6527], 'Waru, Pamekasan': [-7.1667, 113.4833],
    'Paciran, Lamongan': [-6.8667, 112.3333], 'Kertosono, Nganjuk': [-7.5833, 112.1000],
    'Banyu Ajuh, Perumnas, Kamal': [-7.1833, 112.7833], 'Bandung Jogoroto, Jombang': [-7.5467, 112.2331],
    'Kec. Kalianget, Sumenep': [-7.0583, 113.9333], 'Jabon, Sidoarjo': [-7.5351, 112.8107],
    'Menganti, Gresik': [-7.3029, 112.5829], 'Widang, Tuban': [-7.0851, 112.1708],
    'Kwanyar, Bangkalan': [-7.1639, 112.8510], 'sambeng, lamongan': [-7.2973, 112.2709],
    'Cerme, Gresik': [-7.2243, 112.5708], 'Tikala, Manado': [1.4680, 124.8625],
    'Kerek, Tuban': [-6.8971, 111.8855], 'Wonokromo, Surabaya': [-7.3021, 112.7392],
    'Asemrowo, Surabaya': [-7.2417, 112.6888], 'Kota Sumenep, Sumenep': [-7.0067, 113.8599],
    'Socah, Bangkalan': [-7.0909, 112.7055], 'Pilangkenceng, Madiun': [-7.4996, 111.6443],
    'Tanah Merah, Bangkalan': [-7.0883, 112.8853], 'Sidoarjo, Wonoayu': [-7.4456, 112.6644],
    'Labang, Bangkalan': [-7.1396, 112.7731], 'Widodaren, Ngawi': [-7.4029, 111.2239],
    'Bangkalan, Bangkalan': [-7.0295, 112.7473], 'Warudoyong, Kota Sukabumi': [-6.9341, 106.9209],
    'Banyuajuh kamal, Bangkalan': [-7.1833, 112.7833], 'Dukun, Gresik': [-6.9964, 112.5098],
    'Kecamatan Bangkalan, Bangkalan': [-7.0295, 112.7473]
}

df_map['lat'] = df_map['daerah'].apply(lambda x: koordinat_cache.get(x, [-7.5, 112.7])[0])
df_map['lon'] = df_map['daerah'].apply(lambda x: koordinat_cache.get(x, [-7.5, 112.7])[1])

# === 4.4 Peta Interaktif (Folium) ===
print("Membuat Peta Interaktif Folium...")
m = folium.Map(location=[-2.5, 118.0], zoom_start=5)
folium.TileLayer('OpenStreetMap').add_to(m)

fg = folium.FeatureGroup(name=f"Cluster {BEST_DATASET}")
for _, row in df_map.iterrows():
    c = int(row['cluster'])
    col = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
    folium.CircleMarker(
        location=[row['lat'], row['lon']], radius=12, color=col, fill=True,
        fill_opacity=0.8, tooltip=f"Cluster {c}: {row['daerah']}"
    ).add_to(fg)

fg.add_to(m)
folium.LayerControl().add_to(m)
html_path = f'{OUTPUT_DIR}/peta_clustering_interaktif.html'
m.save(html_path)
print(f"Peta HTML Interaktif tersimpan di: {html_path}")

# === 4.5 Peta Statis (Matplotlib) ===
print("Membuat Peta Statis Matplotlib...")
fig, ax = plt.subplots(figsize=(14, 8))
for c in range(BEST_K):
    mask = df_map['cluster'] == c
    sub = df_map[mask]
    ax.scatter(sub['lon'], sub['lat'], s=200, color=CLUSTER_COLORS[c],
               edgecolors='white', linewidth=1.5, label=f'Cluster {c} ({len(sub)} wil)')
    for _, row in sub.iterrows():
        ax.annotate(row['daerah'][:12], (row['lon'], row['lat']), xytext=(4,4),
                    textcoords='offset points', fontsize=7.5)

ax.set_facecolor('#E8F4FD')
ax.grid(True, linestyle='-', alpha=0.4, color='white')
ax.set_title(f'Peta Segmentasi Wilayah Polutan\nDataset: {BEST_DATASET} | k={BEST_K}', fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
png_path = f'{OUTPUT_DIR}/12_peta_statis.png'
plt.savefig(png_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Peta Gambar Statis tersimpan di: {png_path}")
print("\n[SELESAI] Tahap 4 Berhasil!")

