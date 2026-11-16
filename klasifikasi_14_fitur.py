"""
=============================================================================
PRAKTIKUM PENGOLAHAN SINYAL DIGITAL (PSD) - TEKNIK INFORMATIKA UTM
MODUL: EKSTRAKSI 14 FITUR CITRA DIGITAL & PEMODELAN KLASIFIKASI (ALUR AUDIO)
PENULIS: Muhammad Zaidan Nabil Rafi (240411100068)
=============================================================================
Spesifikasi Tugas Dosen:
1. Momen Warna RGB (9 Fitur): mean_r, std_r, skew_r, mean_g, std_g, skew_g, mean_b, std_b, skew_b
2. Konversi Grayscale 8-bit & GLCM Tekstur (5 Fitur): glcm_energy, glcm_entropy, glcm_contrast, glcm_correlation, glcm_homogeneity
3. Dataset CSV: hasil_ekstraksi_fitur_citra.csv (194 citra: 97 Rapi vs 97 Berantakan)
4. Matriks Akurasi (Confusion Matrix) & Klasifikasi Machine Learning (Alur Audio Kemarin)
=============================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def main():
    csv_file = 'hasil_ekstraksi_fitur_citra.csv'
    if not os.path.exists(csv_file):
        csv_file = os.path.join(r'c:\laragon\www\TugasPSD', 'hasil_ekstraksi_fitur_citra.csv')
        
    print("=" * 80)
    print("  KLASIFIKASI 14 FITUR INTI CITRA DIGITAL (SESUAI SPESIFIKASI DOSEN)")
    print("=" * 80)
    print(f"[1] Memuat Dataset CSV: {csv_file}")
    df = pd.read_csv(csv_file)
    print(f"    Total Sampel: {len(df)} citra")
    print(f"    Distribusi Kelas: {df['label'].value_counts().to_dict()}")
    
    # 14 Fitur Inti yang diminta Dosen
    features = [
        'mean_r', 'std_r', 'skew_r',
        'mean_g', 'std_g', 'skew_g',
        'mean_b', 'std_b', 'skew_b',
        'glcm_energy', 'glcm_entropy', 'glcm_contrast', 'glcm_correlation', 'glcm_homogeneity'
    ]
    
    print("\n[2] Daftar 14 Fitur yang Digunakan:")
    for i, col in enumerate(features, 1):
        print(f"    {i:2d}. {col}")
        
    X = df[features].values
    y = (df['label'] == 'Berantakan').astype(int).values
    
    # Stratified Train-Test Split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n[3] Partisi Dataset (Stratified Split 80:20):")
    print(f"    - Training Set : {len(y_train)} sampel ({np.sum(y_train==0)} Rapi, {np.sum(y_train==1)} Berantakan)")
    print(f"    - Testing Set  : {len(y_test)} sampel ({np.sum(y_test==0)} Rapi, {np.sum(y_test==1)} Berantakan)")
    
    # Standardisasi Z-score (Fit HANYA pada data latih)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    X_all_sc = scaler.fit_transform(X)
    
    # Model yang digunakan (identik dengan tugas audio kemarin)
    models = {
        'Support Vector Machine (RBF)': SVC(kernel='rbf', C=5.0, probability=True, random_state=42),
        'Random Forest (100 Trees)': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n" + "=" * 80)
    print("  HASIL EVALUASI AKURASI DARI KODE (MODEL BENCHMARK 14 FITUR)")
    print("=" * 80)
    
    for name, clf in models.items():
        # Validasi Silang 5-Fold
        cv_scores = cross_val_score(clf, X_all_sc, y, cv=cv, scoring='accuracy')
        
        # Latih pada Training Set
        clf.fit(X_train_sc, y_train)
        
        # Prediksi Data Latih & Data Uji
        y_train_pred = clf.predict(X_train_sc)
        y_test_pred = clf.predict(X_test_sc)
        
        train_acc = accuracy_score(y_train, y_train_pred) * 100
        test_acc = accuracy_score(y_test, y_test_pred) * 100
        prec = precision_score(y_test, y_test_pred) * 100
        rec = recall_score(y_test, y_test_pred) * 100
        f1 = f1_score(y_test, y_test_pred) * 100
        cm = confusion_matrix(y_test, y_test_pred)
        
        print(f"\n>>> MODEL: {name}")
        print(f"    * Akurasi Data Latih (Train Acc)     : {train_acc:.2f}%")
        print(f"    * AKURASI DATA UJI (TEST ACCURACY)   : {test_acc:.2f}% ({cm[0,0]+cm[1,1]}/{len(y_test)} Citra Benar)")
        print(f"    * Validasi Silang (5-Fold CV)        : {np.mean(cv_scores)*100:.2f}% (+/- {np.std(cv_scores)*100:.2f}%)")
        print(f"    * Presisi (Precision)                : {prec:.2f}%")
        print(f"    * Sensitivitas / Recall              : {rec:.2f}%")
        print(f"    * F1-Score                           : {f1:.2f}%")
        print(f"    * Matriks Akurasi (Confusion Matrix) :")
        print(f"        [TN={cm[0,0]:2d} (Rapi benar)     FP={cm[0,1]:2d} (Rapi salah)]")
        print(f"        [FN={cm[1,0]:2d} (Berantakan salah) TP={cm[1,1]:2d} (Berantakan benar)]")
        
    print("\n" + "=" * 80)
    print("  LAPORAN KLASIFIKASI LENGKAP (SVM RBF - MODEL TERBAIK 14 FITUR)")
    print("=" * 80)
    svm_best = models['Support Vector Machine (RBF)']
    y_test_pred_svm = svm_best.predict(X_test_sc)
    print(classification_report(y_test, y_test_pred_svm, target_names=['Rapi (0)', 'Berantakan (1)']))
    print("=" * 80)

if __name__ == '__main__':
    main()

