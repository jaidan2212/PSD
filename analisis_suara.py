"""
Identifikasi dan Klasifikasi Suara Menggunakan Fitur Audio
===========================================================
Skrip lengkap untuk ekstraksi fitur (RMS, Spectral Centroid, ZCR, MFCC),
visualisasi, dan klasifikasi suara.

Penggunaan:
    1. Letakkan file audio (.wav/.mp3/.flac/.ogg) di folder 'dataset_audio/'.
       - Dengan label: buat subfolder per kelas (misal: dataset_audio/kelas_a/)
       - Tanpa label: letakkan langsung di dataset_audio/
    2. Jalankan: python analisis_suara.py
    3. Hasil CSV & grafik akan disimpan di folder yang sama.

Dependensi:
    pip install librosa numpy pandas matplotlib scikit-learn soundfile
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 4)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11

# ============================================================
# KONFIGURASI — Sesuaikan jika diperlukan
# ============================================================
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_audio')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output_audio')
TARGET_SR = 22050
N_MFCC = 13

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. PEMINDAIAN DATASET
# ============================================================
def scan_dataset(dataset_dir):
    """Memindai folder dataset dan mendeteksi label kelas."""
    supported_ext = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    file_list = []
    has_labels = False
    class_names = []

    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] Folder '{dataset_dir}' tidak ditemukan.")
        print("Buat folder 'dataset_audio/' dan masukkan file audio.")
        sys.exit(1)

    subdirs = sorted([d for d in os.listdir(dataset_dir)
                      if os.path.isdir(os.path.join(dataset_dir, d))])

    for subdir in subdirs:
        subdir_path = os.path.join(dataset_dir, subdir)
        audio_files = sorted([f for f in os.listdir(subdir_path)
                              if os.path.splitext(f)[1].lower() in supported_ext])
        if audio_files:
            has_labels = True
            class_names.append(subdir)
            for f in audio_files:
                file_list.append({
                    'path': os.path.join(subdir_path, f),
                    'label': subdir
                })

    if not has_labels:
        root_files = sorted([f for f in os.listdir(dataset_dir)
                             if os.path.isfile(os.path.join(dataset_dir, f))
                             and os.path.splitext(f)[1].lower() in supported_ext])
        for f in root_files:
            file_list.append({
                'path': os.path.join(dataset_dir, f),
                'label': None
            })

    if not file_list:
        print("[ERROR] Tidak ada file audio ditemukan di 'dataset_audio/'.")
        print("Masukkan file audio (.wav/.mp3/.flac/.ogg) lalu jalankan ulang.")
        sys.exit(1)

    return file_list, has_labels, class_names


# ============================================================
# 2. MEMUAT AUDIO
# ============================================================
def load_audio(filepath, target_sr=TARGET_SR):
    """Membaca file audio dan mengembalikan sinyal + metadata."""
    filename = os.path.basename(filepath)
    try:
        y, sr = librosa.load(filepath, sr=target_sr, mono=True)
        if y is None or len(y) == 0:
            return {'filename': filename, 'signal': None, 'sr': None,
                    'duration': 0, 'n_samples': 0, 'success': False,
                    'error': 'Sinyal kosong'}
        return {'filename': filename, 'signal': y, 'sr': sr,
                'duration': librosa.get_duration(y=y, sr=sr),
                'n_samples': len(y), 'success': True, 'error': None}
    except Exception as e:
        return {'filename': filename, 'signal': None, 'sr': None,
                'duration': 0, 'n_samples': 0, 'success': False,
                'error': str(e)}


# ============================================================
# 3. EKSTRAKSI FITUR
# ============================================================
def extract_all_features(signal, sr, n_mfcc=N_MFCC):
    """Mengekstrak seluruh fitur dan merangkumnya dengan mean & std."""
    features = {}

    # RMS
    rms = librosa.feature.rms(y=signal)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    features['centroid_mean'] = float(np.mean(centroid))
    features['centroid_std'] = float(np.std(centroid))

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y=signal)[0]
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std'] = float(np.std(zcr))

    # MFCC
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    for j in range(n_mfcc):
        features[f'mfcc_{j+1}_mean'] = float(np.mean(mfcc[j]))
        features[f'mfcc_{j+1}_std'] = float(np.std(mfcc[j]))

    return features


# ============================================================
# 4. VISUALISASI INDIVIDUAL (WAVEFORM, RMS, CENTROID, ZCR, MFCC)
# ============================================================
def plot_individual_analysis(audio_data_list, output_dir, max_display=6):
    """Membuat grafik waveform, RMS, Spectral Centroid, ZCR, dan MFCC."""
    n = min(len(audio_data_list), max_display)
    if n == 0:
        return

    # --- Waveform ---
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))
    if n == 1:
        axes = [axes]
    for i, a in enumerate(audio_data_list[:n]):
        lbl = f" [{a['label']}]" if a.get('label') else ""
        librosa.display.waveshow(a['signal'], sr=a['sr'], ax=axes[i], alpha=0.7)
        axes[i].set_title(f"Waveform: {a['filename']}{lbl}", fontweight='bold')
        axes[i].set_xlabel('Waktu (detik)')
        axes[i].set_ylabel('Amplitudo')
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'waveforms.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ waveforms.png")

    # --- RMS ---
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))
    if n == 1:
        axes = [axes]
    for i, a in enumerate(audio_data_list[:n]):
        rms = librosa.feature.rms(y=a['signal'])[0]
        times = librosa.times_like(rms, sr=a['sr'])
        lbl = f" [{a['label']}]" if a.get('label') else ""
        axes[i].plot(times, rms, color='tab:red', linewidth=1.2)
        axes[i].fill_between(times, rms, alpha=0.2, color='tab:red')
        axes[i].set_title(f"RMS: {a['filename']}{lbl}", fontweight='bold')
        axes[i].set_xlabel('Waktu (detik)')
        axes[i].set_ylabel('RMS')
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rms.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ rms.png")

    # --- Spectral Centroid ---
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))
    if n == 1:
        axes = [axes]
    for i, a in enumerate(audio_data_list[:n]):
        cent = librosa.feature.spectral_centroid(y=a['signal'], sr=a['sr'])[0]
        times = librosa.times_like(cent, sr=a['sr'])
        lbl = f" [{a['label']}]" if a.get('label') else ""
        axes[i].plot(times, cent, color='tab:blue', linewidth=1.2)
        axes[i].set_title(f"Spectral Centroid: {a['filename']}{lbl}", fontweight='bold')
        axes[i].set_xlabel('Waktu (detik)')
        axes[i].set_ylabel('Spectral Centroid (Hz)')
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectral_centroid.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ spectral_centroid.png")

    # --- ZCR ---
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))
    if n == 1:
        axes = [axes]
    for i, a in enumerate(audio_data_list[:n]):
        zcr = librosa.feature.zero_crossing_rate(y=a['signal'])[0]
        times = librosa.times_like(zcr, sr=a['sr'])
        lbl = f" [{a['label']}]" if a.get('label') else ""
        axes[i].plot(times, zcr, color='tab:green', linewidth=1.2)
        axes[i].set_title(f"ZCR: {a['filename']}{lbl}", fontweight='bold')
        axes[i].set_xlabel('Waktu (detik)')
        axes[i].set_ylabel('Zero-Crossing Rate')
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'zcr.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ zcr.png")

    # --- MFCC Heatmap ---
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n))
    if n == 1:
        axes = [axes]
    for i, a in enumerate(audio_data_list[:n]):
        mfcc = librosa.feature.mfcc(y=a['signal'], sr=a['sr'], n_mfcc=N_MFCC)
        lbl = f" [{a['label']}]" if a.get('label') else ""
        img = librosa.display.specshow(mfcc, x_axis='time', sr=a['sr'],
                                       ax=axes[i], cmap='coolwarm')
        axes[i].set_title(f"MFCC: {a['filename']}{lbl}", fontweight='bold')
        axes[i].set_ylabel('Koefisien MFCC')
        axes[i].set_xlabel('Waktu (detik)')
        fig.colorbar(img, ax=axes[i], format='%+2.0f')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mfcc_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ mfcc_heatmap.png")


# ============================================================
# 5. VISUALISASI PERBANDINGAN
# ============================================================
def plot_feature_comparison(df, output_dir, has_labels, class_names, n_mfcc=N_MFCC):
    """Membuat grafik perbandingan fitur antar-sampel dan antar-kelas."""
    if len(df) < 2:
        print("  [INFO] Hanya 1 sampel, grafik perbandingan dilewati.")
        return

    # --- Bar chart: fitur utama ---
    main_feats = ['rms_mean', 'centroid_mean', 'zcr_mean']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, feat in enumerate(main_feats):
        colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
        axes[idx].bar(range(len(df)), df[feat], color=colors, edgecolor='black', lw=0.5)
        axes[idx].set_title(f'Perbandingan {feat}', fontweight='bold')
        axes[idx].set_xlabel('Sampel Audio')
        axes[idx].set_ylabel(feat)
        axes[idx].set_xticks(range(len(df)))
        axes[idx].set_xticklabels(df['filename'], rotation=45, ha='right', fontsize=7)
        axes[idx].grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'perbandingan_fitur.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ perbandingan_fitur.png")

    # --- MFCC comparison ---
    mfcc_cols = [f'mfcc_{j+1}_mean' for j in range(n_mfcc)]
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(1, n_mfcc + 1)
    width = 0.8 / len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[c] for c in mfcc_cols]
        lbl = f"{row['filename']}"
        if row['label'] != '-':
            lbl += f" [{row['label']}]"
        offset = (i - len(df) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width, label=lbl)
    ax.set_title('Perbandingan Rata-rata MFCC', fontsize=14, fontweight='bold')
    ax.set_xlabel('Koefisien MFCC')
    ax.set_ylabel('Nilai Mean')
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{j}' for j in range(1, n_mfcc + 1)])
    ax.legend(fontsize=7, loc='best')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mfcc_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ mfcc_comparison.png")

    # --- Boxplot per kelas ---
    if has_labels and len(class_names) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        comparisons = [
            ('rms_mean', 'RMS Mean', 'tab:red'),
            ('centroid_mean', 'Spectral Centroid Mean (Hz)', 'tab:blue'),
            ('zcr_mean', 'ZCR Mean', 'tab:green'),
            ('mfcc_1_mean', 'MFCC-1 Mean', 'tab:purple')
        ]
        for ax, (feat, title, color) in zip(axes.flatten(), comparisons):
            data = [df[df['label'] == c][feat].values for c in class_names]
            bp = ax.boxplot(data, labels=class_names, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            ax.set_title(f'{title} per Kelas', fontweight='bold')
            ax.set_ylabel(feat)
            ax.grid(axis='y', alpha=0.3)
        plt.suptitle('Distribusi Fitur per Kelas', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fitur_per_kelas.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ fitur_per_kelas.png")


# ============================================================
# 6. KLASIFIKASI
# ============================================================
def run_classification(df, has_labels, class_names, output_dir):
    """Menjalankan klasifikasi atau clustering sesuai ketersediaan label."""
    from sklearn.preprocessing import StandardScaler

    feature_cols = [c for c in df.columns if c not in ['filename', 'label', 'durasi_detik', 'cluster']]
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if has_labels:
        from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (classification_report, confusion_matrix,
                                     ConfusionMatrixDisplay, accuracy_score)

        y = df['label'].values
        class_counts = pd.Series(y).value_counts()
        min_count = class_counts.min()

        print("\n" + "=" * 60)
        print("KLASIFIKASI TERAWASI (SUPERVISED)")
        print("=" * 60)
        print(f"\nDistribusi kelas:\n{class_counts.to_string()}")

        if len(df) >= 10 and min_count >= 3:
            print("\nStrategi: Train-Test Split (80:20)")
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            print(f"Data latih: {len(X_train)} | Data uji: {len(X_test)}")

            for name, model in [("SVM", SVC(kernel='rbf', random_state=42)),
                                ("Random Forest", RandomForestClassifier(
                                    n_estimators=100, random_state=42, n_jobs=-1))]:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                print(f"\n--- {name} ---")
                print(f"Accuracy: {acc:.4f}")
                print(classification_report(y_test, y_pred, zero_division=0))

                fig, ax = plt.subplots(figsize=(8, 6))
                cm = confusion_matrix(y_test, y_pred, labels=class_names)
                ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
                    ax=ax, cmap='Blues' if name == 'SVM' else 'Greens', values_format='d')
                ax.set_title(f'Confusion Matrix — {name}', fontweight='bold')
                plt.tight_layout()
                fname = f'confusion_matrix_{name.lower().replace(" ", "_")}.png'
                plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  ✓ {fname}")

                if name == "Random Forest":
                    imp = pd.Series(model.feature_importances_, index=feature_cols)
                    imp = imp.sort_values(ascending=True)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    imp.tail(15).plot(kind='barh', ax=ax, color='tab:olive')
                    ax.set_title('Top 15 Feature Importance — RF', fontweight='bold')
                    ax.set_xlabel('Importance')
                    ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'feature_importance.png'),
                                dpi=150, bbox_inches='tight')
                    plt.close()
                    print("  ✓ feature_importance.png")

        elif len(df) >= 6:
            n_splits = min(min_count, 5)
            print(f"\nStrategi: Stratified {n_splits}-Fold Cross-Validation")
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            for name, model in [("SVM", SVC(kernel='rbf', random_state=42)),
                                ("Random Forest", RandomForestClassifier(
                                    n_estimators=100, random_state=42))]:
                scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
                print(f"\n{name} — Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
                print(f"  Per fold: {scores}")

            print(f"\n[CATATAN] Dengan {len(df)} sampel, hasil memiliki varians tinggi.")
        else:
            print("\n[INFO] Data terlalu sedikit untuk klasifikasi yang bermakna.")

    elif len(df) >= 3:
        from sklearn.cluster import KMeans

        print("\n" + "=" * 60)
        print("PENGELOMPOKAN TANPA LABEL (K-MEANS CLUSTERING)")
        print("=" * 60)
        print("\n[INFO] Label tidak tersedia. Hasil cluster bukan kelas sebenarnya.")

        max_k = min(len(df) - 1, 5)
        max_k = max(max_k, 2)
        K_range = range(2, max_k + 1)
        inertias = []
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(list(K_range), inertias, 'bo-', linewidth=2, markersize=8)
        ax.set_title('Elbow Method', fontweight='bold')
        ax.set_xlabel('K')
        ax.set_ylabel('Inertia')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'elbow_plot.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ elbow_plot.png")

        km_final = KMeans(n_clusters=2, random_state=42, n_init=10)
        df['cluster'] = km_final.fit_predict(X_scaled)
        print("\nHasil cluster:")
        print(df[['filename', 'cluster']].to_string(index=False))


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("IDENTIFIKASI DAN KLASIFIKASI SUARA")
    print("Menggunakan Fitur: RMS, Spectral Centroid, ZCR, MFCC")
    print("=" * 60)

    # 1. Scan dataset
    print("\n[1/6] Memindai dataset...")
    file_list, has_labels, class_names = scan_dataset(DATASET_DIR)
    print(f"  Total file: {len(file_list)}")
    print(f"  Label: {'Ya — ' + str(class_names) if has_labels else 'Tidak'}")

    # 2. Load audio
    print("\n[2/6] Memuat audio...")
    audio_data = []
    for item in file_list:
        meta = load_audio(item['path'])
        meta['label'] = item['label']
        meta['filepath'] = item['path']
        audio_data.append(meta)

    valid = [a for a in audio_data if a['success']]
    failed = [a for a in audio_data if not a['success']]
    print(f"  Berhasil: {len(valid)} | Gagal: {len(failed)}")
    if failed:
        for f in failed:
            print(f"  [GAGAL] {f['filename']}: {f['error']}")

    if not valid:
        print("[ERROR] Tidak ada file audio yang berhasil dimuat.")
        sys.exit(1)

    # Metadata
    print("\n  --- Metadata ---")
    meta_df = pd.DataFrame([{
        'File': a['filename'], 'Durasi': f"{a['duration']:.2f}s",
        'SR': a['sr'], 'Sampel': a['n_samples'],
        'Label': a['label'] or '-'
    } for a in valid])
    print(meta_df.to_string(index=False))

    # 3. Visualisasi individual
    print("\n[3/6] Membuat visualisasi individual...")
    plot_individual_analysis(valid, OUTPUT_DIR)

    # 4. Ekstraksi fitur
    print("\n[4/6] Mengekstrak fitur...")
    rows = []
    for a in valid:
        try:
            feats = extract_all_features(a['signal'], a['sr'])
            feats['filename'] = a['filename']
            feats['label'] = a['label'] if a['label'] else '-'
            feats['durasi_detik'] = round(a['duration'], 2)
            rows.append(feats)
        except Exception as e:
            print(f"  [WARNING] {a['filename']}: {e}")

    df = pd.DataFrame(rows)
    meta_cols = ['filename', 'label', 'durasi_detik']
    feat_cols = sorted([c for c in df.columns if c not in meta_cols])
    df = df[meta_cols + feat_cols]

    print(f"  Tabel: {df.shape[0]} baris x {df.shape[1]} kolom ({len(feat_cols)} fitur)")

    # Simpan CSV
    csv_path = os.path.join(OUTPUT_DIR, 'hasil_ekstraksi_fitur_audio.csv')
    df.to_csv(csv_path, index=False)
    print(f"  ✓ {csv_path}")

    # Tampilkan ringkasan
    print("\n  --- Ringkasan Fitur ---")
    summary_cols = ['filename', 'label', 'rms_mean', 'centroid_mean', 'zcr_mean',
                    'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean']
    print(df[summary_cols].to_string(index=False))

    # 5. Visualisasi perbandingan
    print("\n[5/6] Membuat visualisasi perbandingan...")
    plot_feature_comparison(df, OUTPUT_DIR, has_labels, class_names)

    # 6. Klasifikasi
    print("\n[6/6] Klasifikasi/Clustering...")
    run_classification(df, has_labels, class_names, OUTPUT_DIR)

    # Simpan CSV final (dengan cluster jika ada)
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 60)
    print("SELESAI!")
    print(f"Hasil tersimpan di: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
