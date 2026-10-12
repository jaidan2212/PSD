import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

os.makedirs('output', exist_ok=True)

with open('output_markdown.md', 'w') as f:
    df = pd.read_csv('dataset_sentinel2_jatim.csv')
    
    # 2.4 Head
    f.write('## 2.4 Head\n')
    f.write(df.head().to_markdown(index=False) + '\n\n')

    # 4.1 Jumlah data per kelas plot
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x='kelas', palette='viridis')
    plt.title('Jumlah Data per Kelas Lahan')
    plt.tight_layout()
    plt.savefig('output/jumlah_data_per_kelas.png')
    plt.close()

    # 4.2 Statistik Deskriptif Fitur
    FITUR = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 
             'NDVI', 'NDWI', 'MNDWI', 'NDBI', 'NDRE', 'EVI', 'SAVI', 'BSI']
    
    f.write('## 4.2 Statistik\n')
    f.write(df[FITUR].describe().T.round(4).to_markdown() + '\n\n')
    f.write('Rata-rata fitur per kelas:\n')
    f.write(df.groupby("kelas")[FITUR].mean().round(3).to_markdown() + '\n\n')

    # 4.3 Tanda Tangan Spektral
    BAND_ASLI = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
    panjang_gelombang = [492, 560, 665, 704, 740, 783, 833, 865, 1614, 2202]
    rata = df.groupby("kelas")[BAND_ASLI].mean()

    plt.figure(figsize=(10, 5))
    for k in rata.index:
        plt.plot(panjang_gelombang, rata.loc[k].values, marker="o", lw=2, label=k)
    plt.xlabel("Panjang gelombang (nm)")
    plt.ylabel("Reflektansi permukaan")
    plt.title("Tanda tangan spektral rata-rata per kelas (Sentinel-2A)")
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/spectral_signature.png')
    plt.close()

    # 4.4 Korelasi Antar Fitur
    plt.figure(figsize=(11, 9))
    sns.heatmap(df[FITUR].corr(), cmap="coolwarm", center=0, annot=True, fmt=".1f",
                annot_kws={"size": 7}, linewidths=.3)
    plt.title("Matriks korelasi antar fitur")
    plt.tight_layout()
    plt.savefig('output/korelasi_fitur.png')
    plt.close()

    # 5. Pemodelan
    X = df[FITUR]
    y = df['kelas_id']
    kelas_map = dict(zip(df['kelas_id'], df['kelas']))
    NAMA_KELAS = [kelas_map[i] for i in sorted(kelas_map.keys())]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    f.write('## 5. Classification Report\n')
    report = classification_report(y_test, y_pred, target_names=NAMA_KELAS)
    f.write('```\n' + report + '\n```\n\n')

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=NAMA_KELAS, yticklabels=NAMA_KELAS)
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.title('Confusion Matrix - Random Forest')
    plt.tight_layout()
    plt.savefig('output/confusion_matrix.png')
    plt.close()

    # 6.1 Eksperimen Rasio
    hasil_rasio = []
    for ts in [0.10, 0.20, 0.30, 0.40]:
        skor = []
        for s in range(10): # reduced to 10 for speed
            Xa, Xb, ya, yb = train_test_split(X, y, test_size=ts, stratify=y, random_state=s)
            skor.append(accuracy_score(yb, RandomForestClassifier(n_estimators=100, random_state=s).fit(Xa, ya).predict(Xb)))
        hasil_rasio.append({"Rasio train:test": f"{int((1-ts)*100)}:{int(ts*100)}",
                            "Akurasi rata-rata": np.mean(skor), "Std": np.std(skor)})
    f.write('## 6.1 Rasio\n')
    f.write(pd.DataFrame(hasil_rasio).round(4).to_markdown(index=False) + '\n\n')

    # 6.2 Kombinasi Fitur
    INDEKS = ['NDVI', 'NDWI', 'MNDWI', 'NDBI', 'NDRE', 'EVI', 'SAVI', 'BSI']
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    skenario_fitur = {
        "A. 4 band 10 m (B2,B3,B4,B8)": ["B2", "B3", "B4", "B8"],
        "B. 10 band Sentinel-2A": BAND_ASLI,
        "C. 8 indeks spektral": INDEKS,
        "D. Band + indeks (18 fitur)": FITUR,
    }

    hasil_fitur = []
    for nama, cols in skenario_fitur.items():
        s = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), df[cols], y, cv=cv, scoring="accuracy")
        hasil_fitur.append({"Skenario": nama, "Jumlah fitur": len(cols), "Akurasi CV": s.mean(), "Std": s.std()})
    f.write('## 6.2 Kombinasi Fitur\n')
    f.write(pd.DataFrame(hasil_fitur).sort_values("Akurasi CV", ascending=False).reset_index(drop=True).round(4).to_markdown(index=False) + '\n\n')

    # 6.3 Eksperimen Parameter
    grid = GridSearchCV(RandomForestClassifier(random_state=42), 
                        {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20]},
                        cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(df[FITUR], y)
    
    cv_res = pd.DataFrame(grid.cv_results_)[["param_n_estimators", "param_max_depth", "mean_test_score", "std_test_score"]]
    f.write('## 6.3 Grid Search\n')
    f.write(cv_res.round(4).to_markdown(index=False) + '\n\n')
