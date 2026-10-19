"""
Pipeline Analisis Fitur Audio dan Klasifikasi Berisik vs Non-Berisik
Mata Kuliah: Pengolahan Sinyal Digital (PSD)
Penulis: Muhammad Zaidan Nabil Rafi (240411100068)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Pastikan ffmpeg tersedia di PATH
try:
    import imageio_ffmpeg
    ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
    print(f"[INFO] FFmpeg path didaftarkan: {ffmpeg_dir}")
except Exception as e:
    print(f"[WARN] imageio_ffmpeg warning: {e}")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

warnings.filterwarnings('ignore')

# Konfigurasi Matplotlib
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(WORKSPACE_DIR, 'dataset_audio')
MATERI_OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'materi', 'output')
os.makedirs(MATERI_OUTPUT_DIR, exist_ok=True)

TARGET_SR = 22050
N_MFCC = 13

def scan_dataset(dataset_dir):
    supported_ext = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    file_list = []
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    for cls in classes:
        subdir = os.path.join(dataset_dir, cls)
        for f in sorted(os.listdir(subdir)):
            ext = os.path.splitext(f)[1].lower()
            if ext in supported_ext:
                file_list.append({
                    'path': os.path.join(subdir, f),
                    'filename': f,
                    'label': cls,
                    'ext': ext
                })
    return file_list, classes

def extract_features(filepath, filename, label, target_sr=TARGET_SR, n_mfcc=N_MFCC):
    try:
        y, sr = librosa.load(filepath, sr=target_sr, mono=True)
        if y is None or len(y) == 0:
            return None, None
        duration = float(librosa.get_duration(y=y, sr=sr))
        
        rms = librosa.feature.rms(y=y)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        feats = {
            'filename': filename,
            'label': label,
            'durasi_detik': round(duration, 2),
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms)),
            'centroid_mean': float(np.mean(centroid)),
            'centroid_std': float(np.std(centroid)),
            'zcr_mean': float(np.mean(zcr)),
            'zcr_std': float(np.std(zcr))
        }
        for j in range(n_mfcc):
            feats[f'mfcc_{j+1}_mean'] = float(np.mean(mfcc[j]))
            feats[f'mfcc_{j+1}_std'] = float(np.std(mfcc[j]))
            
        raw_signals = {
            'y': y,
            'sr': sr,
            'duration': duration,
            'rms': rms,
            'centroid': centroid,
            'zcr': zcr,
            'mfcc': mfcc
        }
        return feats, raw_signals
    except Exception as e:
        print(f"  [ERROR LOAD] {filename}: {e}")
        return None, None

def main():
    print("=" * 70)
    print("PIPELINE ANALISIS FITUR AUDIO: BERISIK vs NON-BERISIK")
    print("=" * 70)
    
    file_list, classes = scan_dataset(DATASET_DIR)
    print(f"Ditemukan {len(file_list)} file audio dalam kelas: {classes}")
    for cls in classes:
        cnt = sum(1 for f in file_list if f['label'] == cls)
        print(f"  - {cls}: {cnt} file")
        
    csv_path = os.path.join(MATERI_OUTPUT_DIR, 'hasil_ekstraksi_fitur_audio.csv')
    exemplars = {}
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"[1/5] Memuat hasil ekstraksi dari {csv_path} ({len(df)} file)...")
        # Load 1 exemplar per class
        for item in file_list:
            if item['label'] not in exemplars:
                _, raw = extract_features(item['path'], item['filename'], item['label'])
                if raw is not None:
                    exemplars[item['label']] = {'filename': item['filename'], 'raw': raw}
            if len(exemplars) == len(classes):
                break
    else:
        print("\n[1/5] Mengekstraksi Fitur dari seluruh file audio...")
        feature_rows = []
        for i, item in enumerate(file_list):
            feats, raw = extract_features(item['path'], item['filename'], item['label'])
            if feats is not None:
                feature_rows.append(feats)
                if item['label'] not in exemplars:
                    exemplars[item['label']] = {'filename': item['filename'], 'raw': raw}
            if (i + 1) % 40 == 0 or (i + 1) == len(file_list):
                print(f"   Diproses: {i+1}/{len(file_list)} file...")

        df = pd.DataFrame(feature_rows)
        print(f"Ekstraksi selesai! Berhasil: {len(df)} dari {len(file_list)} file.")
        df.to_csv(csv_path, index=False)
        df.to_csv(os.path.join(WORKSPACE_DIR, 'hasil_ekstraksi_fitur_audio.csv'), index=False)
        print(f"CSV Fitur disimpan di: {csv_path}")
    
    # -------------------------------------------------------------
    # [2/5] Perhitungan Statistik per Kelas
    # -------------------------------------------------------------
    print("\n[2/5] Menghitung Statistik Deskriptif per Kelas...")
    stats_summary = {}
    main_cols = ['rms_mean', 'rms_std', 'centroid_mean', 'centroid_std', 'zcr_mean', 'zcr_std',
                 'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean', 'mfcc_4_mean']
                 
    for cls in classes:
        sub = df[df['label'] == cls]
        cls_stat = {
            'n_sampel': len(sub),
            'durasi_mean': float(sub['durasi_detik'].mean()),
            'durasi_min': float(sub['durasi_detik'].min()),
            'durasi_max': float(sub['durasi_detik'].max()),
        }
        for col in main_cols:
            cls_stat[f"{col}_mean"] = float(sub[col].mean())
            cls_stat[f"{col}_std"] = float(sub[col].std())
        stats_summary[cls] = cls_stat
        print(f"\n--- {cls} ({len(sub)} sampel) ---")
        print(f"  Durasi: {cls_stat['durasi_mean']:.2f}s (range: {cls_stat['durasi_min']:.2f} - {cls_stat['durasi_max']:.2f}s)")
        print(f"  RMS Mean           : {cls_stat['rms_mean_mean']:.6f} ± {cls_stat['rms_mean_std']:.6f}")
        print(f"  Centroid Mean (Hz) : {cls_stat['centroid_mean_mean']:.2f} ± {cls_stat['centroid_mean_std']:.2f}")
        print(f"  ZCR Mean           : {cls_stat['zcr_mean_mean']:.6f} ± {cls_stat['zcr_mean_std']:.6f}")
        print(f"  MFCC-1 Mean        : {cls_stat['mfcc_1_mean_mean']:.2f} ± {cls_stat['mfcc_1_mean_std']:.2f}")

    # -------------------------------------------------------------
    # [3/5] Visualisasi Komparatif Berisik vs Non-Berisik
    # -------------------------------------------------------------
    print("\n[3/5] Memproduksi Visualisasi Publikasi...")
    
    # 1. Waveform dan Spektrogram Komparatif
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for col_idx, cls in enumerate(classes):
        ex = exemplars[cls]
        y_sig = ex['raw']['y']
        sr_sig = ex['raw']['sr']
        time_axis = np.linspace(0, len(y_sig)/sr_sig, len(y_sig))
        
        # Waveform
        axes[0, col_idx].plot(time_axis, y_sig, color='#2563eb' if cls=='Berisik' else '#059669', lw=0.6, alpha=0.85)
        axes[0, col_idx].set_title(f"Waveform: Kelas {cls}\n({ex['filename']})", fontweight='bold')
        axes[0, col_idx].set_xlabel('Waktu (detik)')
        axes[0, col_idx].set_ylabel('Amplitudo')
        axes[0, col_idx].grid(True, alpha=0.3)
        axes[0, col_idx].set_ylim([-1.05, 1.05])
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y_sig)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr_sig, x_axis='time', y_axis='hz', ax=axes[1, col_idx], cmap='magma')
        axes[1, col_idx].set_title(f"Spektrogram Frekuensi: Kelas {cls}", fontweight='bold')
        axes[1, col_idx].set_xlabel('Waktu (detik)')
        axes[1, col_idx].set_ylabel('Frekuensi (Hz)')
        fig.colorbar(img, ax=axes[1, col_idx], format="%+2.0f dB")
        
    plt.tight_layout()
    wf_path = os.path.join(MATERI_OUTPUT_DIR, 'audio_waveform_spectrogram.png')
    plt.savefig(wf_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Waveform & Spektrogram: {wf_path}")
    
    # 2. Kurva Fitur Temporal (RMS, Centroid, ZCR)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    for cls, color in [('Berisik', '#dc2626'), ('NonBerisik', '#16a34a')]:
        ex = exemplars[cls]
        raw = ex['raw']
        sr = raw['sr']
        
        t_rms = librosa.times_like(raw['rms'], sr=sr)
        axes[0].plot(t_rms, raw['rms'], label=f"{cls} ({ex['filename'][:20]}...)", color=color, lw=1.5)
        
        t_cent = librosa.times_like(raw['centroid'], sr=sr)
        axes[1].plot(t_cent, raw['centroid'], label=f"{cls}", color=color, lw=1.2)
        
        t_zcr = librosa.times_like(raw['zcr'], sr=sr)
        axes[2].plot(t_zcr, raw['zcr'], label=f"{cls}", color=color, lw=1.2)
        
    axes[0].set_title("Perbandingan Energi Temporal (RMS)", fontweight='bold')
    axes[0].set_ylabel("RMS (Amplitudo)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title("Perbandingan Kecerahan Spektral (Spectral Centroid)", fontweight='bold')
    axes[1].set_ylabel("Centroid (Hz)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title("Perbandingan Zero-Crossing Rate (ZCR)", fontweight='bold')
    axes[2].set_xlabel("Waktu (detik)")
    axes[2].set_ylabel("Laju Perpotongan Nol")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    tf_path = os.path.join(MATERI_OUTPUT_DIR, 'audio_temporal_features.png')
    plt.savefig(tf_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Fitur Temporal: {tf_path}")
    
    # 3. Heatmap MFCC Komparatif
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, cls in enumerate(classes):
        ex = exemplars[cls]
        mfcc_mat = ex['raw']['mfcc']
        sr = ex['raw']['sr']
        img = librosa.display.specshow(mfcc_mat, sr=sr, x_axis='time', ax=axes[idx], cmap='coolwarm')
        axes[idx].set_title(f"Heatmap MFCC (1-13): {cls}\n({ex['filename']})", fontweight='bold')
        axes[idx].set_xlabel('Waktu (detik)')
        axes[idx].set_ylabel('Indeks MFCC')
        fig.colorbar(img, ax=axes[idx], format="%+2.0f")
    plt.tight_layout()
    mfcc_path = os.path.join(MATERI_OUTPUT_DIR, 'audio_mfcc_heatmaps.png')
    plt.savefig(mfcc_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  [OK] MFCC Heatmaps: {mfcc_path}")
    
    # 4. Boxplots Distribusi Fitur Utama Antar-Kelas
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    feats_to_plot = [
        ('rms_mean', 'RMS Mean (Energi)'),
        ('centroid_mean', 'Spectral Centroid Mean (Hz)'),
        ('zcr_mean', 'ZCR Mean (Perpotongan Nol)'),
        ('mfcc_1_mean', 'MFCC-1 Mean (Energi Spektral Makro)'),
        ('mfcc_2_mean', 'MFCC-2 Mean (Distribusi Low-High)'),
        ('mfcc_3_mean', 'MFCC-3 Mean (Bentuk Formant)')
    ]
    
    for ax, (f_name, f_title) in zip(axes.flatten(), feats_to_plot):
        data_berisik = df[df['label'] == 'Berisik'][f_name].dropna()
        data_non = df[df['label'] == 'NonBerisik'][f_name].dropna()
        bp = ax.boxplot([data_berisik, data_non], labels=['Berisik', 'NonBerisik'], patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('#ef4444')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('#10b981')
        bp['boxes'][1].set_alpha(0.6)
        ax.set_title(f_title, fontweight='bold', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
    plt.suptitle("Distribusi Perbandingan Fitur Audio: Berisik vs Non-Berisik (N=229)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    bp_path = os.path.join(MATERI_OUTPUT_DIR, 'audio_boxplots_distribusi.png')
    plt.savefig(bp_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Boxplots Distribusi: {bp_path}")

    # -------------------------------------------------------------
    # [4/5] Klasifikasi Machine Learning (SVM & Random Forest)
    # -------------------------------------------------------------
    print("\n[4/5] Melakukan Klasifikasi Supervised Machine Learning...")
    meta_cols = ['filename', 'label', 'durasi_detik']
    feat_cols = sorted([c for c in df.columns if c not in meta_cols])
    
    X = df[feat_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)  # 0: Berisik, 1: NonBerisik (atau sebaliknya)
    class_labels = list(le.classes_)
    
    # Standardisasi Fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train-test split (80:20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )
    
    models = {
        'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    }
    
    eval_results = {}
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax_idx, (m_name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        rep = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        eval_results[m_name] = {
            'accuracy': round(float(acc), 4),
            'classification_report': rep,
            'confusion_matrix': cm.tolist()
        }
        
        print(f"\nModel: {m_name}")
        print(f"  Accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, y_pred, target_names=class_labels, digits=4))
        print(f"  Confusion Matrix:\n{cm}")
        
        disp = ConfusionMatrixDisplay(cm, display_labels=class_labels)
        disp.plot(ax=axes[ax_idx], cmap='Blues' if m_name=='SVM' else 'Greens', values_format='d')
        axes[ax_idx].set_title(f"Confusion Matrix — {m_name}\n(Akurasi: {acc*100:.1f}%)", fontweight='bold')
        
    plt.tight_layout()
    cm_path = os.path.join(MATERI_OUTPUT_DIR, 'audio_cm_models.png')
    plt.savefig(cm_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Confusion Matrix Models: {cm_path}")
    
    # Feature Importance dari Random Forest
    rf_model = models['Random Forest']
    importances = rf_model.feature_importances_
    feat_series = pd.Series(importances, index=feat_cols).sort_values(ascending=False)
    
    eval_results['top_features'] = {k: round(float(v), 5) for k, v in feat_series.head(15).items()}
    
    plt.figure(figsize=(10, 6))
    feat_series.head(15).iloc[::-1].plot(kind='barh', color='#0284c7', edgecolor='black', linewidth=0.5)
    plt.title("Top 15 Feature Importance — Random Forest (Berisik vs Non-Berisik)", fontweight='bold', fontsize=12)
    plt.xlabel("Tingkat Kepentingan Fitur (Gini Importance)")
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    fi_path = os.path.join(MATERI_OUTPUT_DIR, 'audio_feature_importance.png')
    plt.savefig(fi_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Feature Importance: {fi_path}")
    
    # 5-Fold Stratified Cross Validation
    print("\n[5-Fold Stratified Cross Validation]")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    for m_name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
        cv_results[m_name] = {
            'mean': round(float(scores.mean()), 4),
            'std': round(float(scores.std()), 4),
            'scores': [round(float(s), 4) for s in scores]
        }
        print(f"  {m_name}: Mean Accuracy = {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}% | Folds: {[round(s*100, 2) for s in scores]}")
    eval_results['cross_validation'] = cv_results
    
    # Simpan summary hasil dalam JSON
    full_output = {
        'total_files': len(df),
        'classes': {cls: sum(1 for f in file_list if f['label'] == cls) for cls in classes},
        'stats_per_class': stats_summary,
        'evaluation': eval_results
    }
    json_path = os.path.join(MATERI_OUTPUT_DIR, 'audio_analysis_summary.json')
    with open(json_path, 'w') as f:
        json.dump(full_output, f, indent=2)
    print(f"\nRingkasan komprehensif JSON disimpan di: {json_path}")
    print("=" * 70)
    print("ANALISIS SELESAI DENGAN SUKSES!")
    print("=" * 70)

if __name__ == '__main__':
    main()
