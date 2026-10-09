import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score

df = pd.read_csv('../dataset_sentinel2_jatim.csv')
FITUR = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 
         'NDVI', 'NDWI', 'MNDWI', 'NDBI', 'NDRE', 'EVI', 'SAVI', 'BSI']
BAND_ASLI = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
INDEKS = ['NDVI', 'NDWI', 'MNDWI', 'NDBI', 'NDRE', 'EVI', 'SAVI', 'BSI']
X = df[FITUR]
y = df['kelas_id']
kelas_map = dict(zip(df['kelas_id'], df['kelas']))
NAMA_KELAS = [kelas_map[i] for i in sorted(kelas_map.keys())]

# 6.1 Rasio
hasil_rasio = []
for ts in [0.10, 0.20, 0.30, 0.40]:
    skor = []
    for s in range(30):
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=ts, stratify=y, random_state=s)
        skor.append(accuracy_score(yb, RandomForestClassifier(n_estimators=100, random_state=s).fit(Xa, ya).predict(Xb)))
    hasil_rasio.append({"Rasio train:test": f"{int((1-ts)*100)}:{int(ts*100)}",
                        "n training": int(len(y)*(1-ts)), "n testing": int(round(len(y)*ts)),
                        "Akurasi rata-rata": np.mean(skor), "Std": np.std(skor)})
tab_rasio = pd.DataFrame(hasil_rasio)
print("=== RASIO ===")
print(tab_rasio.round(4).to_markdown())

# 6.2 Kombinasi
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
skenario_fitur = {
    "A. 4 band 10 m (B2,B3,B4,B8)": ["B2", "B3", "B4", "B8"],
    "B. 10 band Sentinel-2A": BAND_ASLI,
    "C. 8 indeks spektral": INDEKS,
    "D. Indeks pilihan (NDVI,MNDWI,NDBI,BSI)": ["NDVI", "MNDWI", "NDBI", "BSI"],
    "E. Band + indeks (18 fitur)": FITUR,
}
hasil_fitur = []
for nama, cols in skenario_fitur.items():
    s = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), df[cols], y, cv=cv, scoring="accuracy")
    f = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), df[cols], y, cv=cv, scoring="f1_macro")
    hasil_fitur.append({"Skenario": nama, "Jumlah fitur": len(cols),
                        "Akurasi CV": s.mean(), "Std": s.std(), "F1 macro CV": f.mean()})
tab_fitur = pd.DataFrame(hasil_fitur).sort_values("Akurasi CV", ascending=False).reset_index(drop=True)
print("\n=== FITUR ===")
print(tab_fitur.round(4).to_markdown())
FITUR_FINAL_NAMA = tab_fitur.loc[0, "Skenario"]
FITUR_FINAL = skenario_fitur[FITUR_FINAL_NAMA]

# 6.3 Eksperimen parameter n_estimators & max_depth
grid = GridSearchCV(RandomForestClassifier(random_state=42), 
                    {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10, 20]},
                    cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(df[FITUR_FINAL], y)
tab_vs = (pd.DataFrame(grid.cv_results_)[["param_n_estimators", "param_max_depth", "mean_test_score", "std_test_score"]]
          .rename(columns={"param_n_estimators": "n_estimators", "param_max_depth": "max_depth",
                           "mean_test_score": "Akurasi CV", "std_test_score": "Std"}))
print("\n=== PARAMETER ===")
print(tab_vs.sort_values("Akurasi CV", ascending=False).head(5).round(4).to_markdown())
