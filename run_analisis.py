"""
Script untuk menjalankan analisis fitur audio dan klasifikasi
pada dataset Berisik vs NonBerisik.
"""
import os, sys, warnings, json
import numpy as np
import pandas as pd
import librosa

warnings.filterwarnings('ignore')

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_audio')
TARGET_SR = 22050
N_MFCC = 13

def scan_dataset(dataset_dir):
    supported_ext = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    file_list = []
    class_names = []
    subdirs = sorted([d for d in os.listdir(dataset_dir)
                      if os.path.isdir(os.path.join(dataset_dir, d))])
    for subdir in subdirs:
        subdir_path = os.path.join(dataset_dir, subdir)
        audio_files = sorted([f for f in os.listdir(subdir_path)
                              if os.path.splitext(f)[1].lower() in supported_ext])
        if audio_files:
            class_names.append(subdir)
            for f in audio_files:
                file_list.append({'path': os.path.join(subdir_path, f), 'label': subdir})
    return file_list, class_names

def extract_features(filepath, label, target_sr=TARGET_SR, n_mfcc=N_MFCC):
    filename = os.path.basename(filepath)
    try:
        y, sr = librosa.load(filepath, sr=target_sr, mono=True)
        if y is None or len(y) == 0:
            return None
        duration = librosa.get_duration(y=y, sr=sr)
        feats = {'filename': filename, 'label': label, 'durasi_detik': round(duration, 2)}
        rms = librosa.feature.rms(y=y)[0]
        feats['rms_mean'] = float(np.mean(rms))
        feats['rms_std'] = float(np.std(rms))
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        feats['centroid_mean'] = float(np.mean(centroid))
        feats['centroid_std'] = float(np.std(centroid))
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        feats['zcr_mean'] = float(np.mean(zcr))
        feats['zcr_std'] = float(np.std(zcr))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        for j in range(n_mfcc):
            feats[f'mfcc_{j+1}_mean'] = float(np.mean(mfcc[j]))
            feats[f'mfcc_{j+1}_std'] = float(np.std(mfcc[j]))
        return feats
    except Exception as e:
        print(f"  [GAGAL] {filename}: {e}")
        return None

def main():
    print("=" * 60)
    print("ANALISIS FITUR AUDIO: BERISIK vs NONBERISIK")
    print("=" * 60)

    file_list, class_names = scan_dataset(DATASET_DIR)
    print(f"Kelas: {class_names}")
    for cls in class_names:
        cnt = sum(1 for f in file_list if f['label'] == cls)
        print(f"  {cls}: {cnt} file")
    print(f"Total: {len(file_list)} file\n")

    print("Mengekstrak fitur...")
    rows = []
    failed = 0
    for i, item in enumerate(file_list):
        if (i+1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(file_list)}] ...")
        feats = extract_features(item['path'], item['label'])
        if feats:
            rows.append(feats)
        else:
            failed += 1

    print(f"\nBerhasil: {len(rows)} | Gagal: {failed}")

    df = pd.DataFrame(rows)
    meta_cols = ['filename', 'label', 'durasi_detik']
    feat_cols = sorted([c for c in df.columns if c not in meta_cols])
    df = df[meta_cols + feat_cols]

    # Save CSV
    os.makedirs('output_audio', exist_ok=True)
    df.to_csv('output_audio/hasil_ekstraksi_fitur_audio.csv', index=False)
    print(f"CSV disimpan: output_audio/hasil_ekstraksi_fitur_audio.csv")

    # === Statistics per class ===
    print("\n" + "=" * 60)
    print("RINGKASAN FITUR PER KELAS")
    print("=" * 60)
    main_feats = ['rms_mean', 'rms_std', 'centroid_mean', 'centroid_std',
                  'zcr_mean', 'zcr_std', 'mfcc_1_mean', 'mfcc_2_mean', 'mfcc_3_mean']
    for cls in class_names:
        subset = df[df['label'] == cls]
        print(f"\n--- {cls} ({len(subset)} sampel) ---")
        print(f"  Durasi: mean={subset['durasi_detik'].mean():.2f}s, "
              f"min={subset['durasi_detik'].min():.2f}s, max={subset['durasi_detik'].max():.2f}s")
        for feat in main_feats:
            print(f"  {feat}: mean={subset[feat].mean():.6f}, std={subset[feat].std():.6f}")

    # === Classification ===
    print("\n" + "=" * 60)
    print("KLASIFIKASI: SVM dan Random Forest")
    print("=" * 60)

    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    X = df[feat_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)
    label_names = le.classes_

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nData latih: {len(X_train)} | Data uji: {len(X_test)}")

    results = {}
    for name, model in [("SVM", SVC(kernel='rbf', random_state=42)),
                        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))]:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=label_names, zero_division=0, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=label_names, zero_division=0))
        print(f"Confusion Matrix:\n{cm}")

        results[name] = {
            'accuracy': round(acc, 4),
            'report': {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                           for kk, vv in v.items()} if isinstance(v, dict) else round(v, 4)
                       for k, v in report.items()},
            'confusion_matrix': cm.tolist()
        }

        if name == "Random Forest":
            importances = model.feature_importances_
            imp_series = pd.Series(importances, index=feat_cols).sort_values(ascending=False)
            print(f"\nTop 10 Feature Importance:")
            for feat_name, imp_val in imp_series.head(10).items():
                print(f"  {feat_name}: {imp_val:.4f}")
            results['feature_importance'] = {k: round(v, 4) for k, v in imp_series.head(15).items()}

    # Cross-validation
    print("\n--- 5-Fold Stratified Cross-Validation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for name, model in [("SVM", SVC(kernel='rbf', random_state=42)),
                        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42))]:
        scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
        print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f} (per fold: {[round(s,4) for s in scores]})")
        results[f'{name}_cv'] = {'mean': round(scores.mean(), 4), 'std': round(scores.std(), 4),
                                  'per_fold': [round(s, 4) for s in scores]}

    # Save results
    with open('output_audio/classification_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nHasil disimpan: output_audio/classification_results.json")

    # Save per-class summary
    summary = {}
    for cls in class_names:
        subset = df[df['label'] == cls]
        cls_summary = {'count': len(subset), 'durasi_mean': round(subset['durasi_detik'].mean(), 2)}
        for feat in main_feats:
            cls_summary[f'{feat}_mean'] = round(subset[feat].mean(), 6)
            cls_summary[f'{feat}_std'] = round(subset[feat].std(), 6)
        summary[cls] = cls_summary
    with open('output_audio/class_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nSELESAI!")

if __name__ == '__main__':
    main()
