import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

DATA_DIR = "shapefiles"
SEED = 42

KELAS = {
    1: dict(nama="Sawah",                   file="sawah",       n=100, warna="#FFD92F"),
    2: dict(nama="Bangunan",                file="bangunan",    n=100, warna="#E41A1C"),
    3: dict(nama="Mangrove",                file="mangrove",    n=80,  warna="#8E44AD"),
    4: dict(nama="Lahan Hijau",             file="lahan_hijau", n=80,  warna="#2E7D32"),
    5: dict(nama="Perairan Terbuka (Laut)", file="lautan",      n=80,  warna="#0D47A1"),
    6: dict(nama="Danau",                   file="danau",       n=80,  warna="#4FC3F7"),
}

def baca_shp(path):
    g = gpd.read_file(path)
    g = g.set_crs(4326) if g.crs is None else g.to_crs(4326)
    g = g[g.geometry.notnull() & ~g.geometry.is_empty].copy()
    g["geometry"] = g.geometry.apply(lambda x: x if x.is_valid else x.buffer(0))
    return g[["geometry"]].reset_index(drop=True)

def cari_shp(nama):
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower().endswith(".shp") and f[:-4].lower().replace(" ", "_") == nama.lower():
                return os.path.join(root, f)
    raise FileNotFoundError(f"File {nama}.shp not found.")

jalur_shp = {k: cari_shp(v["file"]) for k, v in KELAS.items()}
poligon = {k: baca_shp(p) for k, p in jalur_shp.items()}

rng = np.random.default_rng(SEED)

def titik_acak(poly, rng, maks=300):
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(maks):
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if poly.contains(p):
            return p
    return poly.representative_point()

def ambil_sampel(gdf, n, rng):
    polys = list(gdf.geometry)
    if len(polys) == 0:
        return []
    urutan = rng.permutation(len(polys))
    hasil = []
    if len(polys) >= n:
        for i in urutan[:n]:
            hasil.append((polys[i].representative_point(), int(i)))
    else:
        for i in urutan:
            hasil.append((polys[i].representative_point(), int(i)))
        while len(hasil) < n:
            i = int(rng.integers(len(polys)))
            hasil.append((titik_acak(polys[i], rng), i))
    return hasil

baris = []
for k, v in KELAS.items():
    for p, pid in ambil_sampel(poligon[k], v["n"], rng):
        baris.append({"kelas_id": k, "kelas": v["nama"], "poligon_id": pid, "lon": p.x, "lat": p.y})
df_titik = pd.DataFrame(baris).reset_index(drop=True)

# Generate mock bands based on class
# B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
mock_data = []
for idx, row in df_titik.iterrows():
    c = row["kelas_id"]
    if c == 1: # Sawah
        b = [0.05, 0.08, 0.07, 0.13, 0.23, 0.27, 0.27, 0.29, 0.21, 0.12]
    elif c == 2: # Bangunan
        b = [0.10, 0.12, 0.14, 0.16, 0.17, 0.18, 0.18, 0.20, 0.26, 0.24]
    elif c == 3: # Mangrove
        b = [0.03, 0.06, 0.03, 0.10, 0.26, 0.30, 0.30, 0.33, 0.11, 0.05]
    elif c == 4: # Lahan Hijau
        b = [0.04, 0.06, 0.04, 0.10, 0.28, 0.35, 0.34, 0.38, 0.18, 0.09]
    elif c == 5: # Lautan
        b = [0.05, 0.05, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01]
    elif c == 6: # Danau
        b = [0.03, 0.05, 0.03, 0.04, 0.03, 0.03, 0.02, 0.03, 0.02, 0.01]
        
    # add some noise
    b = [max(0.001, min(0.999, val + rng.normal(0, 0.015))) for val in b]
    mock_data.append(b)

df_bands = pd.DataFrame(mock_data, columns=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"])
df = df_titik.join(df_bands)

# Calculate indices
e = 1e-9
nd = lambda a, b: (a - b) / (a + b + e)
df["NDVI"]  = nd(df["B8"], df["B4"])
df["NDWI"]  = nd(df["B3"], df["B8"])
df["MNDWI"] = nd(df["B3"], df["B11"])
df["NDBI"]  = nd(df["B11"], df["B8"])
df["NDRE"]  = nd(df["B8"], df["B5"])
df["EVI"]   = 2.5 * (df["B8"] - df["B4"]) / (df["B8"] + 6 * df["B4"] - 7.5 * df["B2"] + 1)
df["SAVI"]  = 1.5 * (df["B8"] - df["B4"]) / (df["B8"] + df["B4"] + 0.5)
df["BSI"]   = (((df["B11"] + df["B4"]) - (df["B8"] + df["B2"])) /
              ((df["B11"] + df["B4"]) + (df["B8"] + df["B2"]) + e))

df.to_csv("dataset_sentinel2_jatim.csv", index=False)
print("Mock dataset created successfully.")
