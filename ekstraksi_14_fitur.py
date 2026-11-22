"""
================================================================================
MODUL EKSTRAKSI 14 FITUR CITRA DIGITAL (WARNA RGB & TEKSTUR GLCM HARALICK)
Praktikum Pengolahan Sinyal Digital (PSD) - Teknik Informatika UTM
Penulis: Muhammad Zaidan Nabil Rafi (NIM: 240411100068)
================================================================================

Spesifikasi 14 Fitur Citra:
1. Domain Warna (9 Fitur):
   - Kanal Merah (Red)   : mean_r, std_r, skew_r
   - Kanal Hijau (Green) : mean_g, std_g, skew_g
   - Kanal Biru (Blue)   : mean_b, std_b, skew_b
2. Domain Tekstur Spasial GLCM Grayscale 8-bit (5 Fitur Haralick):
   - glcm_energy      : Ukuran keseragaman tekstur (Angular Second Moment)
   - glcm_entropy     : Ukuran keacakan / kompleksitas informasi spasial
   - glcm_contrast    : Variasi intensitas lokal antar pasangan piksel
   - glcm_correlation : Ketergantungan linier nilai keabuan antar piksel
   - glcm_homogeneity : Tingkat kemiripan intensitas lokal (Inverse Difference Moment)
================================================================================
"""

import os
import cv2
import numpy as np
import pandas as pd
from scipy import stats

def hitung_glcm_5_fitur(gray_img, n_levels=64, distance=1):
    """
    Menghitung 5 Fitur Tekstur Haralick dari Gray-Level Co-occurrence Matrix (GLCM).
    
    Parameter:
    - gray_img : numpy 2D array (citra derajat keabuan uint8)
    - n_levels : jumlah level kuantisasi derajat keabuan (default: 64 tingkat)
    - distance : jarak spasial antar pasangan piksel (default: 1 piksel)
    
    Output:
    - dictionary berisi 5 nilai fitur GLCM (rata-rata dari 4 arah sudut: 0°, 45°, 90°, 135°)
    """
    # 1. Kuantisasi citra ke n_levels untuk efisiensi dan kestabilan statistik
    scaled = (gray_img.astype(np.float32) / 256.0 * n_levels).astype(np.uint32)
    scaled = np.clip(scaled, 0, n_levels - 1)
    H, W = scaled.shape
    
    # 2. Empat arah orientasi spasial (dx, dy)
    #    (0, 1) = 0 derajat (horizontal)
    #    (-1, 1) = 45 derajat (diagonal kanan)
    #    (-1, 0) = 90 derajat (vertikal)
    #    (-1, -1) = 135 derajat (diagonal kiri)
    directions = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]
    d = distance
    
    all_contrasts, all_homogeneities = [], []
    all_energies, all_correlations, all_entropies = [], [], []
    
    for dy, dx in directions:
        y1_s = max(0, -dy * d); y1_e = min(H, H - dy * d)
        x1_s = max(0, -dx * d); x1_e = min(W, W - dx * d)
        y2_s = max(0, dy * d);  y2_e = min(H, H + dy * d)
        x2_s = max(0, dx * d);  x2_e = min(W, W + dx * d)
        
        p1 = scaled[y1_s:y1_e, x1_s:x1_e].ravel()
        p2 = scaled[y2_s:y2_e, x2_s:x2_e].ravel()
        
        # Hitung matriks kemunculan bersama (co-occurrence)
        mat, _, _ = np.histogram2d(p1, p2, bins=n_levels, range=[[0, n_levels], [0, n_levels]])
        mat = mat + mat.T  # Matriks simetris
        tot = mat.sum()
        p = mat / tot if tot > 0 else mat  # Normalisasi menjadi nilai probabilitas P(i,j)
        
        # Koordinat matriks
        i_idx, j_idx = np.indices((n_levels, n_levels))
        diff = np.abs(i_idx - j_idx)
        diff_sq = diff ** 2
        
        # Rumus Matematis Haralick:
        # A. Kontras: sum((i - j)^2 * P(i, j))
        contrast = np.sum(diff_sq * p)
        
        # B. Homogenitas: sum(P(i, j) / (1 + |i - j|))
        homogeneity = np.sum(p / (1.0 + diff))
        
        # C. Energi: sum(P(i, j)^2)
        energy = np.sum(p ** 2)
        
        # D. Entropi: -sum(P(i, j) * log2(P(i, j)))
        nonzero_p = p[p > 0]
        entropy = -np.sum(nonzero_p * np.log2(nonzero_p)) if len(nonzero_p) > 0 else 0.0
        
        # E. Korelasi: sum((i - mu_i)*(j - mu_j)*P(i, j)) / (std_i * std_j)
        mu_i = np.sum(i_idx * p)
        mu_j = np.sum(j_idx * p)
        var_i = np.sum(((i_idx - mu_i) ** 2) * p)
        var_j = np.sum(((j_idx - mu_j) ** 2) * p)
        std_ij = np.sqrt(var_i * var_j)
        corr = np.sum((i_idx - mu_i) * (j_idx - mu_j) * p) / std_ij if std_ij > 1e-9 else 1.0
        
        all_contrasts.append(contrast)
        all_homogeneities.append(homogeneity)
        all_energies.append(energy)
        all_correlations.append(corr)
        all_entropies.append(entropy)
        
    return {
        'glcm_energy': float(np.mean(all_energies)),
        'glcm_entropy': float(np.mean(all_entropies)),
        'glcm_contrast': float(np.mean(all_contrasts)),
        'glcm_correlation': float(np.mean(all_correlations)),
        'glcm_homogeneity': float(np.mean(all_homogeneities))
    }

def hitung_9_momen_warna(img_rgb):
    """
    Menghitung 9 Momen Warna RGB: Mean, Standar Deviasi, dan Skewness tiap kanal.
    
    Formula Statistik per Kanal c in [R, G, B]:
    - Mean     : mu_c = (1/N) * sum(x_i)
    - Std Dev  : sigma_c = sqrt((1/N) * sum((x_i - mu_c)^2))
    - Skewness : skew_c = (1/N) * sum(((x_i - mu_c) / sigma_c)^3)
    """
    feats = {}
    channels = ['r', 'g', 'b']
    for idx, c in enumerate(channels):
        channel_data = img_rgb[:, :, idx].astype(np.float64)
        feats[f'mean_{c}'] = float(np.mean(channel_data))
        feats[f'std_{c}']  = float(np.std(channel_data))
        feats[f'skew_{c}'] = float(stats.skew(channel_data.ravel()))
    return feats

def ekstraksi_14_fitur(citra_input):
    """
    Fungsi Pipeline Lengkap Ekstraksi 14 Fitur Citra:
    Input  : Path file citra (string) ATAU numpy array citra (BGR uint8)
    Output : Dictionary berisi 14 fitur (9 Momen Warna + 5 GLCM Tekstur)
    """
    # 1. Pengecekan input (apakah string path atau array numpy)
    if isinstance(citra_input, str):
        if not os.path.exists(citra_input):
            raise FileNotFoundError(f"File citra tidak ditemukan: {citra_input}")
        img_bgr = cv2.imread(citra_input)
        if img_bgr is None:
            raise ValueError(f"Gagal membaca format citra dari: {citra_input}")
    else:
        img_bgr = citra_input

    # 2. Preprocessing: Standarisasi resolusi spasial ke 512x512
    img_std = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_AREA)
    
    # 3. Preprocessing: Konversi warna dari BGR ke RGB murni
    img_rgb = cv2.cvtColor(img_std, cv2.COLOR_BGR2RGB)
    
    # 4. Preprocessing: Konversi RGB ke Grayscale 8-bit (ITU-R BT.601)
    #    Formula: Y = 0.299*R + 0.587*G + 0.114*B
    gray = cv2.cvtColor(img_std, cv2.COLOR_BGR2GRAY)
    
    # 5. Ekstraksi 9 Momen Warna RGB
    warna_feats = hitung_9_momen_warna(img_rgb)
    
    # 6. Ekstraksi 5 Fitur Tekstur GLCM Haralick
    glcm_feats = hitung_glcm_5_fitur(gray, n_levels=64, distance=1)
    
    # 7. Penggabungan seluruh 14 fitur
    fitur_lengkap = {
        # Warna RGB (9 Fitur)
        'mean_r': warna_feats['mean_r'],
        'std_r': warna_feats['std_r'],
        'skew_r': warna_feats['skew_r'],
        'mean_g': warna_feats['mean_g'],
        'std_g': warna_feats['std_g'],
        'skew_g': warna_feats['skew_g'],
        'mean_b': warna_feats['mean_b'],
        'std_b': warna_feats['std_b'],
        'skew_b': warna_feats['skew_b'],
        # Tekstur GLCM (5 Fitur)
        'glcm_energy': glcm_feats['glcm_energy'],
        'glcm_entropy': glcm_feats['glcm_entropy'],
        'glcm_contrast': glcm_feats['glcm_contrast'],
        'glcm_correlation': glcm_feats['glcm_correlation'],
        'glcm_homogeneity': glcm_feats['glcm_homogeneity']
    }
    
    return fitur_lengkap

def ekstrak_folder_ke_csv(dirs_dict, output_csv_path):
    """
    Mengekstrak seluruh citra dalam folder dataset dan mengekspornya ke CSV.
    
    Parameter:
    - dirs_dict: dictionary dengan format {'Nama_Kelas': 'path/ke/folder'}
    - output_csv_path: path file output CSV
    """
    records = []
    print("=" * 80)
    print(f"MEMULAI EKSTRAKSI BATCH DATASET MENUJU: {output_csv_path}")
    print("=" * 80)
    
    for label, folder in dirs_dict.items():
        if not os.path.exists(folder):
            print(f"[PERINGATAN] Folder tidak ditemukan: {folder}")
            continue
            
        file_list = sorted([
            f for f in os.listdir(folder) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ])
        print(f"-> Memproses Kelas '{label}': {len(file_list)} citra...")
        
        for idx, fname in enumerate(file_list, 1):
            fpath = os.path.join(folder, fname)
            try:
                feats = ekstraksi_14_fitur(fpath)
                feats['filename'] = fname
                feats['label'] = label
                records.append(feats)
            except Exception as e:
                print(f"   [ERROR] Gagal memproses {fname}: {e}")
                
            if idx % 25 == 0 or idx == len(file_list):
                print(f"   Progress: [{idx:3d}/{len(file_list):3d}] citra selesai diekstrak.")
                
    kolom_urut = [
        'filename',
        'mean_r', 'std_r', 'skew_r',
        'mean_g', 'std_g', 'skew_g',
        'mean_b', 'std_b', 'skew_b',
        'glcm_energy', 'glcm_entropy', 'glcm_contrast', 'glcm_correlation', 'glcm_homogeneity',
        'label'
    ]
    
    df = pd.DataFrame(records)[kolom_urut]
    df.to_csv(output_csv_path, index=False)
    print(f"\n[SUKSES] {len(df)} baris data berhasil disimpan ke: {output_csv_path}")
    return df

if __name__ == '__main__':
    # Contoh penggunaan langsung jika file ini dieksekusi mandiri
    print("Modul Ekstraksi 14 Fitur berhasil dimuat.")
    print("Silakan jalankan file testing: `python test_modul_14_fitur.py` untuk demonstrasi lengkap.")
