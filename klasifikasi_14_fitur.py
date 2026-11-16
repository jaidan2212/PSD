"""
=============================================================================
PRAKTIKUM PENGOLAHAN SINYAL DIGITAL (PSD) - TEKNIK INFORMATIKA UTM
MODUL: EKSTRAKSI 14 FITUR CITRA DIGITAL, PEMBUATAN CSV & KLASIFIKASI (ALUR AUDIO)
PENULIS: Muhammad Zaidan Nabil Rafi (240411100068)
=============================================================================
Spesifikasi Tugas Dosen:
1. Momen Warna RGB (9 Fitur): 
   - Mean, Std, Skewness per kanal (R, G, B)
2. Konversi Grayscale 8-bit & GLCM Tekstur Haralick (5 Fitur): 
   - Energy, Entropy, Contrast, Correlation, Homogeneity
3. Ekspor ke Berkas CSV: 
   - hasil_ekstraksi_fitur_citra.csv (194 citra: 97 Rapi vs 97 Berantakan)
4. Matriks Akurasi (Confusion Matrix) & Klasifikasi Machine Learning (Alur Audio):
   - Stratified Train-Test Split (80:20), StandardScaler, SVM RBF, Random Forest, 5-Fold CV
=============================================================================
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# =============================================================================
# 1. FUNGSI EKSTRAKSI FITUR CITRA DIGITAL (FITUR WARNA & FITUR TEKSTUR GLCM)
# =============================================================================

def compute_glcm_features(gray_img, n_levels=64, distance=1):
    """
    Menghitung 5 Fitur Tekstur Haralick dari Gray-Level Co-occurrence Matrix (GLCM):
    1. Kontras (Contrast)     : Ukuran variasi intensitas lokal antar piksel bertetangga
    2. Homogenitas (Homog)    : Ukuran kemiripan derajat keabuan (Inverse Difference Moment)
    3. Energi (Energy/ASM)    : Ukuran keseragaman tekstur (Angular Second Moment)
    4. Korelasi (Correlation) : Ketergantungan linier derajat keabuan antar piksel
    5. Entropi (Entropy)      : Ukuran keacakan / kompleksitas informasi spasial
    """
    # Kuantisasi citra ke n_levels derajat keabuan untuk stabilitas statistik
    scaled = (gray_img.astype(np.float32) / 256.0 * n_levels).astype(np.uint32)
    scaled = np.clip(scaled, 0, n_levels - 1)
    H, W = scaled.shape
    
    # 4 Arah spasial utama: 0 derajat, 45 derajat, 90 derajat, 135 derajat
    directions = [(0, 1), (-1, 1), (-1, 0), (-1, -1)]
    d = distance
    
    all_contrasts, all_homogeneities = [], []
    all_energies, all_correlations, all_entropies = [], []
    
    for dy, dx in directions:
        y1_s = max(0, -dy * d); y1_e = min(H, H - dy * d)
        x1_s = max(0, -dx * d); x1_e = min(W, W - dx * d)
        y2_s = max(0, dy * d);  y2_e = min(H, H + dy * d)
        x2_s = max(0, dx * d);  x2_e = min(W, W + dx * d)
        
        p1 = scaled[y1_s:y1_e, x1_s:x1_e].ravel()
        p2 = scaled[y2_s:y2_e, x2_s:x2_e].ravel()
        
        # Bentuk matriks frekuensi kemunculan bersama (co-occurrence)
        mat, _, _ = np.histogram2d(p1, p2, bins=n_levels, range=[[0, n_levels], [0, n_levels]])
        mat = mat + mat.T  # Simetriskan matriks
        tot = mat.sum()
        p = mat / tot if tot > 0 else mat  # Normalisasi menjadi probabilitas P(i, j)
        
        # Koordinat matriks i dan j
        i_indices, j_indices = np.indices((n_levels, n_levels))
        diff = np.abs(i_indices - j_indices)
        diff_sq = diff ** 2
        
        # Rumus Matematis Haralick
        contrast = np.sum(diff_sq * p)
        homogeneity = np.sum(p / (1.0 + diff))
        energy = np.sum(p ** 2)
        
        nonzero_p = p[p > 0]
        entropy = -np.sum(nonzero_p * np.log2(nonzero_p)) if len(nonzero_p) > 0 else 0.0
        
        mu_i = np.sum(i_indices * p)
        mu_j = np.sum(j_indices * p)
        var_i = np.sum(((i_indices - mu_i) ** 2) * p)
        var_j = np.sum(((j_indices - mu_j) ** 2) * p)
        std_ij = np.sqrt(var_i * var_j)
        corr = np.sum((i_indices - mu_i) * (j_indices - mu_j) * p) / std_ij if std_ij > 1e-9 else 1.0
        
        all_contrasts.append(contrast)
        all_homogeneities.append(homogeneity)
        all_energies.append(energy)
        all_correlations.append(corr)
        all_entropies.append(entropy)
        
    # Rata-rata dari 4 arah sudut pandang rotasional
    return {
        'glcm_energy': float(np.mean(all_energies)),
        'glcm_entropy': float(np.mean(all_entropies)),
        'glcm_contrast': float(np.mean(all_contrasts)),
        'glcm_correlation': float(np.mean(all_correlations)),
        'glcm_homogeneity': float(np.mean(all_homogeneities))
    }

def extract_14_features(img_bgr):
    """
    Fungsi Utama Ekstraksi 14 Fitur Citra Digital:
    Input  : Citra BGR mentah (dibaca menggunakan cv2.imread)
    Output : Dictionary berisi persis 14 fitur sesuai permintaan dosen:
             - 9 Momen Warna RGB: mean_r, std_r, skew_r, mean_g, std_g, skew_g, mean_b, std_b, skew_b
             - 5 Fitur GLCM Tekstur: glcm_energy, glcm_entropy, glcm_contrast, glcm_correlation, glcm_homogeneity
    """
    # 1. Standarisasi resolusi spasial citra (512x512) agar representasi statistik adil
    img_std = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_AREA)
    
    # 2. Konversi warna dari BGR (default OpenCV) ke RGB
    img_rgb = cv2.cvtColor(img_std, cv2.COLOR_BGR2RGB)
    
    # 3. Konversi citra ke Grayscale 8-bit sesuai standar ITU-R BT.601:
    #    Y = 0.299*R + 0.587*G + 0.114*B
    gray = cv2.cvtColor(img_std, cv2.COLOR_BGR2GRAY)
    
    features = {}
    
    # -------------------------------------------------------------
    # A. EKSTRAKSI 9 FITUR MOMEN WARNA RGB (Mean, Std, Skewness)
    # -------------------------------------------------------------
    channels = ['r', 'g', 'b']
    for idx, c in enumerate(channels):
        channel_data = img_rgb[:, :, idx].astype(np.float64)
        
        # Mean (Rata-rata intensitas kanal)
        features[f'mean_{c}'] = float(np.mean(channel_data))
        
        # Standard Deviation (Standar deviasi variasi warna)
        features[f'std_{c}'] = float(np.std(channel_data))
        
        # Skewness (Kemiringan distribusi derajat warna)
        features[f'skew_{c}'] = float(stats.skew(channel_data.ravel()))
        
    # -------------------------------------------------------------
    # B. EKSTRAKSI 5 FITUR GLCM TEKSTUR HARALICK PADA CITRA GRAYSCALE
    # -------------------------------------------------------------
    glcm_res = compute_glcm_features(gray, n_levels=64, distance=1)
    features['glcm_energy'] = glcm_res['glcm_energy']
    features['glcm_entropy'] = glcm_res['glcm_entropy']
    features['glcm_contrast'] = glcm_res['glcm_contrast']
    features['glcm_correlation'] = glcm_res['glcm_correlation']
    features['glcm_homogeneity'] = glcm_res['glcm_homogeneity']
    
    return features

def extract_dataset_to_csv(dirs_dict, output_csv_path):
    """
    Membaca seluruh berkas citra mentah dari folder dataset,
    mengekstrak 14 fitur untuk setiap citra, dan mengekspornya ke CSV.
    """
    print(f"\n[PROSES EKSTRAKSI FITUR DARI CITRA MENTAH KE CSV]")
    print(f"Target Output CSV: {output_csv_path}")
    
    records = []
    for label, folder in dirs_dict.items():
        if not os.path.exists(folder):
            print(f"PERINGATAN: Folder {folder} tidak ditemukan!")
            continue
            
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        print(f"-> Memproses Kelas '{label}': {len(files)} citra...")
        
        for i, fname in enumerate(files, 1):
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
                
            # Jalankan fungsi ekstraksi 14 fitur
            feats = extract_14_features(img)
            feats['filename'] = fname
            feats['label'] = label
            records.append(feats)
            
            if i % 25 == 0 or i == len(files):
                print(f"   [{i:3d}/{len(files):3d}] {fname} berhasil diekstrak.")
                
    # Urutkan kolom sesuai permintaan dosen
    ordered_cols = [
        'filename',
        'mean_r', 'std_r', 'skew_r',
        'mean_g', 'std_g', 'skew_g',
        'mean_b', 'std_b', 'skew_b',
        'glcm_energy', 'glcm_entropy', 'glcm_contrast', 'glcm_correlation', 'glcm_homogeneity',
        'label'
    ]
    df_out = pd.DataFrame(records)[ordered_cols]
    df_out.to_csv(output_csv_path, index=False)
    print(f"\n[SUKSES] Seluruh {len(df_out)} citra berhasil diekstrak dan disimpan ke: {output_csv_path}")
    return df_out

# =============================================================================
# 2. PEMODELAN KLASIFIKASI MACHINE LEARNING & MATRIKS AKURASI (ALUR AUDIO)
# =============================================================================

def run_classification_pipeline(df):
    """
    Melakukan klasifikasi Machine Learning persis seperti alur tugas audio:
    - Stratified Train-Test Split (80:20)
    - Normalisasi Z-Score (StandardScaler)
    - Pelatihan Model: SVM (RBF), Random Forest (100 Trees), Logistic Regression
    - Evaluasi 5-Fold Cross Validation
    - Pembentukan Matriks Akurasi (Confusion Matrix)
    """
    features = [
        'mean_r', 'std_r', 'skew_r',
        'mean_g', 'std_g', 'skew_g',
        'mean_b', 'std_b', 'skew_b',
        'glcm_energy', 'glcm_entropy', 'glcm_contrast', 'glcm_correlation', 'glcm_homogeneity'
    ]
    
    print("\n" + "=" * 80)
    print("  [1] INFORMASI DATASET & 14 FITUR INTI")
    print("=" * 80)
    print(f"Total Sampel    : {len(df)} citra")
    print(f"Distribusi Kelas: {df['label'].value_counts().to_dict()}")
    print("\nDaftar 14 Fitur:")
    for i, col in enumerate(features, 1):
        print(f"  {i:2d}. {col}")
        
    X = df[features].values
    y = (df['label'] == 'Berantakan').astype(int).values
    
    # Stratified Train-Test Split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nPartisi Data (Stratified Split 80:20):")
    print(f"- Data Latih (Train Set) : {len(y_train)} sampel ({np.sum(y_train==0)} Rapi, {np.sum(y_train==1)} Berantakan)")
    print(f"- Data Uji (Test Set)    : {len(y_test)} sampel ({np.sum(y_test==0)} Rapi, {np.sum(y_test==1)} Berantakan)")
    
    # Standarisasi Z-score (Fit HANYA pada data latih)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    X_all_sc = scaler.fit_transform(X)
    
    models = {
        'Support Vector Machine (RBF)': SVC(kernel='rbf', C=5.0, probability=True, random_state=42),
        'Random Forest (100 Trees)': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n" + "=" * 80)
    print("  [2] HASIL EVALUASI AKURASI DARI KODE (MODEL BENCHMARK 14 FITUR)")
    print("=" * 80)
    
    best_clf = None
    best_acc = -1
    best_name = ""
    
    for name, clf in models.items():
        # 5-Fold Cross Validation
        cv_scores = cross_val_score(clf, X_all_sc, y, cv=cv, scoring='accuracy')
        
        # Latih pada Training Set
        clf.fit(X_train_sc, y_train)
        
        y_train_pred = clf.predict(X_train_sc)
        y_test_pred = clf.predict(X_test_sc)
        
        train_acc = accuracy_score(y_train, y_train_pred) * 100
        test_acc = accuracy_score(y_test, y_test_pred) * 100
        prec = precision_score(y_test, y_test_pred) * 100
        rec = recall_score(y_test, y_test_pred) * 100
        f1 = f1_score(y_test, y_test_pred) * 100
        cm = confusion_matrix(y_test, y_test_pred)
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_clf = clf
            best_name = name
            
        print(f"\n>>> MODEL: {name}")
        print(f"    * Akurasi Data Latih (Train Acc)     : {train_acc:.2f}%")
        print(f"    * AKURASI DATA UJI (TEST ACCURACY)   : {test_acc:.2f}% ({cm[0,0]+cm[1,1]}/{len(y_test)} Citra Benar)")
        print(f"    * Validasi Silang (5-Fold CV)        : {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
        print(f"    * Presisi (Precision)                : {prec:.2f}%")
        print(f"    * Sensitivitas / Recall              : {rec:.2f}%")
        print(f"    * F1-Score                           : {f1:.2f}%")
        print(f"    * Matriks Akurasi (Confusion Matrix) :")
        print(f"        [TN={cm[0,0]:2d} (Rapi benar)     FP={cm[0,1]:2d} (Rapi salah)]")
        print(f"        [FN={cm[1,0]:2d} (Berantakan salah) TP={cm[1,1]:2d} (Berantakan benar)]")
        
    print("\n" + "=" * 80)
    print(f"  [3] LAPORAN KLASIFIKASI LENGKAP ({best_name})")
    print("=" * 80)
    y_best_pred = best_clf.predict(X_test_sc)
    print(classification_report(y_test, y_best_pred, target_names=['Rapi (0)', 'Berantakan (1)']))
    print("=" * 80)

# =============================================================================
# 3. TITIK MASUK UTAMA PROGRAM (MAIN FUNCTION)
# =============================================================================

def main():
    csv_file = 'hasil_ekstraksi_fitur_citra.csv'
    if not os.path.exists(csv_file):
        csv_file = os.path.join(r'c:\laragon\www\TugasPSD', 'hasil_ekstraksi_fitur_citra.csv')
        
    dataset_dirs = {
        'Berantakan': r'c:\laragon\www\TugasPSD\dataset_baru\berantakan',
        'Rapi': r'c:\laragon\www\TugasPSD\dataset_baru\rapi'
    }
    
    # Cek apakah user ingin melakukan ekstraksi ulang dari gambar citra mentah
    force_extract = '--extract' in sys.argv or '--reextract' in sys.argv
    
    if not os.path.exists(csv_file) or force_extract:
        print("[INFO] Menjalankan ekstraksi fitur langsung dari gambar mentah...")
        df = extract_dataset_to_csv(dataset_dirs, csv_file)
    else:
        print(f"[INFO] File CSV fitur '{csv_file}' sudah tersedia.")
        print("       (Gunakan flag `python klasifikasi_14_fitur.py --extract` jika ingin mengekstrak ulang dari citra mentah)")
        df = pd.read_csv(csv_file)
        
    # Jalankan pipeline pemodelan klasifikasi
    run_classification_pipeline(df)

if __name__ == '__main__':
    main()
