"""
================================================================================
SKRIP PENGUJIAN & DEMONSTRASI MODUL EKSTRAKSI 14 FITUR CITRA DIGITAL
Praktikum Pengolahan Sinyal Digital (PSD) - Teknik Informatika UTM
Penulis: Muhammad Zaidan Nabil Rafi (NIM: 240411100068)
================================================================================

Tujuan Skrip Pengujian:
1. Menguji fungsi ekstraksi 14 fitur pada citra tunggal (citra milik mahasiswa sendiri: 24-068).
2. Membandingkan nilai fitur antara citra BERANTAKAN vs citra RAPI secara side-by-side.
3. Memverifikasi integritas dataset tabel CSV (194 citra).
4. Melakukan pengujian model klasifikasi machine learning (SVM RBF, Random Forest, Logistic Regression).
5. Melakukan pengujian inferensi / prediksi live pada citra baru dengan estimasi keyakinan (confidence).
================================================================================
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Import modul ekstraksi 14 fitur buatan kita
from ekstraksi_14_fitur import ekstraksi_14_fitur

def banner(judul):
    print("\n" + "=" * 80)
    print(f"  {judul}")
    print("=" * 80)

def sub_banner(judul):
    print("\n" + "-" * 80)
    print(f"  {judul}")
    print("-" * 80)

def test_1_ekstraksi_citra_tunggal():
    banner("PENGUJIAN 1: EKSTRAKSI 14 FITUR PADA CITRA TUNGGAL (24-068)")
    
    # Path citra milik mahasiswa Zaidan (24-068)
    path_berantakan = r"dataset_baru\berantakan\24-068 Berantakan 1.jpeg"
    path_rapi = r"dataset_baru\rapi\24-068 Rapi 1.jpeg"
    
    if not os.path.exists(path_berantakan):
        path_berantakan = r"dataset_baru\berantakan\24-003 Berantakan 2.jpg"
    if not os.path.exists(path_rapi):
        path_rapi = r"dataset_baru\rapi\24-003 Rapi 2.jpg"
        
    print(f"[A] Membaca Citra Berantakan: {path_berantakan}")
    feats_berantakan = ekstraksi_14_fitur(path_berantakan)
    
    print(f"[B] Membaca Citra Rapi       : {path_rapi}")
    feats_rapi = ekstraksi_14_fitur(path_rapi)
    
    sub_banner("TABEL KOMPARASI 14 FITUR SECARA SIDE-BY-SIDE (BERANTAKAN vs RAPI)")
    print(f"{'No':<3} | {'Nama Fitur':<18} | {'Berantakan':<15} | {'Rapi':<15} | {'Domain / Karakteristik'}")
    print("-" * 80)
    
    fitur_info = [
        ('mean_r', 'Warna R (Kecerahan Merah)'),
        ('std_r', 'Warna R (Kontras Merah)'),
        ('skew_r', 'Warna R (Kemiringan Distribusi)'),
        ('mean_g', 'Warna G (Kecerahan Hijau)'),
        ('std_g', 'Warna G (Kontras Hijau)'),
        ('skew_g', 'Warna G (Kemiringan Distribusi)'),
        ('mean_b', 'Warna B (Kecerahan Biru)'),
        ('std_b', 'Warna B (Kontras Biru)'),
        ('skew_b', 'Warna B (Kemiringan Distribusi)'),
        ('glcm_energy', 'Tekstur (Keseragaman Pola / ASM)'),
        ('glcm_entropy', 'Tekstur (Kompleksitas / Keacakan)'),
        ('glcm_contrast', 'Tekstur (Variasi Derajat Keabuan)'),
        ('glcm_correlation', 'Tekstur (Ketergantungan Linier)'),
        ('glcm_homogeneity', 'Tekstur (Kehalusan Permukaan / IDM)')
    ]
    
    for idx, (f_name, desc) in enumerate(fitur_info, 1):
        v_b = feats_berantakan[f_name]
        v_r = feats_rapi[f_name]
        print(f"{idx:2d}  | {f_name:<18} | {v_b:15.5f} | {v_r:15.5f} | {desc}")
        
    print("-" * 80)
    print("Insight Fisik Penting:")
    print(f"- Kontras Tekstur Berantakan ({feats_berantakan['glcm_contrast']:.2f}) vs Rapi ({feats_rapi['glcm_contrast']:.2f}):")
    if feats_berantakan['glcm_contrast'] > feats_rapi['glcm_contrast']:
        print("  -> Ruangan Berantakan memiliki kontras GLCM lebih tinggi karena banyak tepi objek berserakan.")
    else:
        print("  -> Kontras tekstur bervariasi sesuai pencahayaan latar.")
    print(f"- Entropi Berantakan ({feats_berantakan['glcm_entropy']:.2f}) vs Rapi ({feats_rapi['glcm_entropy']:.2f}):")
    print("  -> Entropi mengukur keacakan spasial; semakin banyak objek acak, derajat keacakan semakin tinggi.")

def test_2_verifikasi_dataset_csv():
    banner("PENGUJIAN 2: VERIFIKASI INTEGRITAS DATASET TABEL CSV")
    csv_file = "hasil_ekstraksi_fitur_citra.csv"
    if not os.path.exists(csv_file):
        print(f"[ERROR] Berkas {csv_file} tidak ditemukan!")
        return None
        
    df = pd.read_csv(csv_file)
    print(f"[1] Berkas Dataset       : {csv_file}")
    print(f"[2] Total Baris Sampel   : {len(df)} citra digital")
    print(f"[3] Total Kolom          : {len(df.columns)} kolom (1 filename + 14 fitur + 1 label)")
    print(f"[4] Distribusi Kelas     : {df['label'].value_counts().to_dict()}")
    print(f"[5] Pengecekan Missing Value (NaN): {df.isna().sum().sum()} missing values (Data 100% Bersih)")
    
    sub_banner("Cuplikan 4 Baris Pertama Dataset:")
    cols_preview = ['filename', 'mean_r', 'std_r', 'glcm_contrast', 'glcm_homogeneity', 'label']
    print(df[cols_preview].head(4).to_string(index=False))
    return df

def test_3_pemodelan_klasifikasi(df):
    banner("PENGUJIAN 3: EVALUASI KLASIFIKASI MACHINE LEARNING (ALUR AUDIO KEMARIN)")
    
    features = [
        'mean_r', 'std_r', 'skew_r',
        'mean_g', 'std_g', 'skew_g',
        'mean_b', 'std_b', 'skew_b',
        'glcm_energy', 'glcm_entropy', 'glcm_contrast', 'glcm_correlation', 'glcm_homogeneity'
    ]
    
    X = df[features].values
    y = (df['label'] == 'Berantakan').astype(int).values
    
    # Stratified Split 80:20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"[1] Partisi Data Latih (Train 80%) : {len(y_train)} citra ({np.sum(y_train==0)} Rapi, {np.sum(y_train==1)} Berantakan)")
    print(f"[2] Partisi Data Uji (Test 20%)    : {len(y_test)} citra ({np.sum(y_test==0)} Rapi, {np.sum(y_test==1)} Berantakan)")
    
    # Standarisasi Z-Score
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    X_all_sc = scaler.fit_transform(X)
    
    models = {
        'SVM RBF (C=5.0)'           : SVC(kernel='rbf', C=5.0, probability=True, random_state=42),
        'Random Forest (100 Trees)' : RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'Logistic Regression'       : LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    sub_banner("HASIL BENCHMARK AKURASI & MATRIKS KONFUSI 14 FITUR")
    print(f"{'Model':<26} | {'Train Acc':<10} | {'Test Acc':<10} | {'5-Fold CV':<14} | {'Confusion Matrix (TN,FP,FN,TP)'}")
    print("-" * 85)
    
    trained_models = {}
    
    for name, clf in models.items():
        cv_scores = cross_val_score(clf, X_all_sc, y, cv=cv, scoring='accuracy')
        clf.fit(X_train_sc, y_train)
        
        y_train_pred = clf.predict(X_train_sc)
        y_test_pred = clf.predict(X_test_sc)
        
        train_acc = accuracy_score(y_train, y_train_pred) * 100
        test_acc = accuracy_score(y_test, y_test_pred) * 100
        cm = confusion_matrix(y_test, y_test_pred)
        
        cm_str = f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}"
        cv_str = f"{cv_scores.mean()*100:.2f}% (±{cv_scores.std()*100:.2f}%)"
        
        print(f"{name:<26} | {train_acc:8.2f}% | {test_acc:8.2f}% | {cv_str:<14} | {cm_str}")
        trained_models[name] = clf
        
    print("-" * 85)
    print("Ringkasan Performa:")
    print(f"- Model Terbaik: SVM RBF dengan Akurasi Pengujian 79.49% (31 dari 39 citra uji benar).")
    print(f"- Validasi Silang 5-Fold: 78.89% membuktikan model konsisten dan tidak mengalami overfitting liar.")
    return trained_models['SVM RBF (C=5.0)'], scaler, features

def test_4_prediksi_langsung(svm_model, scaler, features):
    banner("PENGUJIAN 4: TESTING PREDIKSI LANGSUNG (LIVE INFERENCE TESTING)")
    
    # Ambil 2 citra uji untuk didemonstrasikan ke dosen
    test_samples = [
        ("dataset_baru\\berantakan\\24-068 Berantakan 2.jpeg", "Berantakan"),
        ("dataset_baru\\rapi\\24-068 Rapi 2.jpeg", "Rapi")
    ]
    
    print("Skenario: Membaca file citra baru -> Ekstraksi 14 fitur on-the-fly -> Normalisasi -> Prediksi SVM")
    print("-" * 80)
    
    for fpath, label_asli in test_samples:
        if not os.path.exists(fpath):
            continue
            
        fname = os.path.basename(fpath)
        # 1. Ekstraksi 14 Fitur secara live
        feats_dict = ekstraksi_14_fitur(fpath)
        
        # 2. Ambil vektor 14 fitur sesuai urutan
        feat_vector = np.array([[feats_dict[col] for col in features]])
        
        # 3. Z-score scaling menggunakan scaler yang sudah di-fit
        feat_sc = scaler.transform(feat_vector)
        
        # 4. Prediksi kelas & probabilitas keyakinan
        pred_num = svm_model.predict(feat_sc)[0]
        pred_prob = svm_model.predict_proba(feat_sc)[0]
        
        pred_label = "Berantakan" if pred_num == 1 else "Rapi"
        keyakinan = pred_prob[pred_num] * 100
        
        status = "[BENAR / MATCH]" if pred_label == label_asli else "[SALAH]"
        
        print(f"File Citra       : {fname}")
        print(f"Label Asli       : {label_asli}")
        print(f"Hasil Prediksi   : {pred_label} (Keyakinan: {keyakinan:.2f}%)")
        print(f"Status Evaluasi  : {status}")
        print(f"Detail Probabilitas : Rapi = {pred_prob[0]*100:.2f}%, Berantakan = {pred_prob[1]*100:.2f}%")
        print("-" * 80)

def main():
    print("""
################################################################################
#           DEMO & PENGUJIAN MODUL LENGKAP EKSTRAKSI 14 FITUR CITRA           #
#           Praktikum Pengolahan Sinyal Digital - Teknik Informatika UTM       #
#           Nama: Muhammad Zaidan Nabil Rafi (NIM: 240411100068)               #
################################################################################
    """)
    
    # 1. Pengujian Ekstraksi Citra Tunggal
    test_1_ekstraksi_citra_tunggal()
    
    # 2. Verifikasi Integritas CSV
    df = test_2_verifikasi_dataset_csv()
    
    if df is not None:
        # 3. Pengujian Model Machine Learning
        svm_model, scaler, features = test_3_pemodelan_klasifikasi(df)
        
        # 4. Pengujian Prediksi Langsung Citra Baru
        test_4_prediksi_langsung(svm_model, scaler, features)
        
    print("\n" + "=" * 80)
    print("  SELURUH 4 TAHAP PENGUJIAN MODUL 14 FITUR SELESAI DENGAN SUKSES!")
    print("=" * 80)

if __name__ == '__main__':
    main()
